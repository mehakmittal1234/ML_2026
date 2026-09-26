"""Expected-F0.5 decoder: replaces the global threshold with a per-entity choice that maximises expected F0.5.

The official score is F0.5 per S1 entity, so the value of one more match depends on what the entity already has:
the first correct match of an entity lifts it from 0 to 1.25/(0.25n+1) (0.625 for n = 4 true copies), the fifth adds ~0.05,
and any match on a singleton costs the whole 1.0. A single threshold ignores this.

Model: each S2/S3 record goes to its best S1 (one owner per record, as in common.decide). For an entity with candidate
probabilities q_1 >= q_2 >= ... (records that picked it), truth indicators are independent Bernoulli(q_i); true copies not
among its candidates (retrieval misses, lost competitions, tail) are Poisson(lam + sum of tail q). The prediction is the
top-k prefix; k is chosen to maximise E[F0.5] = E[1.25 TP / (0.25 N + k)] (k >= 1) or P(N = 0) (k = 0), computed exactly
with Poisson-binomial DPs over the prefix (TP) and the rest (N - TP).
usage: from efd import efd_decide ; pred = efd_decide(S, lam, a, b)   (S: id1 id2 src p)"""
import numpy as np
import polars as pl
from scipy.special import expit, logit
from scipy.stats import poisson

KY = ['id1', 'id2', 'src']


def _gain_table(K, C):
    k = np.arange(K + 1)[:, None, None].astype(float); a = np.arange(C)[None, :, None]; b = np.arange(C)[None, None, :]
    G = np.where(k > 0, 1.25 * a / np.maximum(1e-12, 0.25 * (a + b) + k), 0.0)
    G[0] = 0.0; G[0, 0, 0] = 1.0                                # empty prediction scores 1 only if the entity has no true copy
    G[:, :, :] *= (a <= k)                                      # TP cannot exceed the prefix length
    return G


def _expected_f(Q, lam, G, C):
    """Q: (E, K) sorted-desc probabilities (0-padded); lam: (E,) Poisson mass of true copies outside the prefix window.
    returns EF (E, K+1): expected F0.5 of predicting the top-k prefix."""
    E, K = Q.shape
    pf = np.zeros((E, K + 1, C)); pf[:, 0, 0] = 1.0
    for i in range(K):
        q = Q[:, i:i + 1]
        pf[:, i + 1] = pf[:, i] * (1 - q); pf[:, i + 1, 1:] += pf[:, i, :-1] * q
    pb = np.zeros((E, K + 1, C))
    pb[:, K] = poisson.pmf(np.arange(C)[None, :], lam[:, None]); pb[:, K, -1] += 1 - pb[:, K].sum(1)
    for i in range(K - 1, -1, -1):
        q = Q[:, i:i + 1]
        pb[:, i] = pb[:, i + 1] * (1 - q); pb[:, i, 1:] += pb[:, i + 1, :-1] * q; pb[:, i, -1] += pb[:, i + 1, -1] * q[:, 0]
    T = np.einsum('eka,kab->ekb', pf, G)
    return (T * pb).sum(-1)


def efd_decide(S, lam=0.08, a=1.0, b=0.0, K=12, pmin=0.01, score='p', chunk=100_000, return_k=False):
    """S: DataFrame with id1 id2 src <score>. a, b: logit calibration q = sigmoid(a*logit(p)+b). Returns predicted (id1,id2,src)."""
    B = S.select(*KY, pl.col(score).alias('p')).filter(pl.col('p') >= pmin).sort('p', descending=True).unique(['id2', 'src'], keep='first')
    p = np.clip(B['p'].to_numpy().astype(np.float64), 1e-6, 1 - 1e-6)
    B = B.with_columns(pl.Series('q', expit(a * logit(p) + b)))
    B = B.sort(['id1', 'q'], descending=[False, True]).with_columns(pl.int_range(pl.len()).over('id1').alias('r'))
    tail = B.filter(pl.col('r') >= K).group_by('id1').agg(pl.col('q').sum().alias('tq'))
    H = B.filter(pl.col('r') < K)
    ent = H.select('id1').unique(maintain_order=True).with_row_index('e').join(tail, on='id1', how='left').with_columns(pl.col('tq').fill_null(0.0))
    H = H.join(ent.select('id1', 'e'), on='id1')
    E = ent.height; Q = np.zeros((E, K)); Q[H['e'].to_numpy(), H['r'].to_numpy()] = H['q'].to_numpy()
    lamv = lam + ent['tq'].to_numpy()
    C = K + 6; G = _gain_table(K, C)
    kbest = np.zeros(E, dtype=np.int64)
    for s in range(0, E, chunk):
        EF = _expected_f(Q[s:s + chunk], lamv[s:s + chunk], G, C)
        kbest[s:s + chunk] = EF.argmax(1)
    ent = ent.with_columns(pl.Series('k', kbest))
    pred = H.join(ent.select('e', 'k'), on='e').filter(pl.col('r') < pl.col('k')).select(*KY)
    return (pred, ent.select('id1', 'k')) if return_k else pred
