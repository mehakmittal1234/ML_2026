"""Validation fold 0: (1) ceilings: oracle that predicts exactly the true pairs present in the stage-2 pair set (loss = retrieval),
and the same oracle abstaining on no-address copies whose owner's exact core name is shared in S1 (class A floor);
(2) non-exclusive decoding: the metric is per S1 entity and the rules only forbid duplicates within a list, so each entity can take the
expected-F-optimal prefix of ALL its candidates using marginal probabilities (record-normalised: q = p / max(1, sum over S1 of p)),
instead of first forcing every record onto its single best S1.
RESULT (m4it fold 0): oracle with every true pair of the pair set 0.99577 (retrieval loss 0.0042, recall 0.9860); abstaining on
class-A pairs 0.99189 (class A 0.0039). Exclusive expected-F decoder 0.98951, non-exclusive 0.98947 (never lists a record twice): rejected.
usage: python s60_hedge.py <val score tag>"""
import sys
from common import *
from efd import _expected_f, _gain_table
tag = sys.argv[1]
gt = load_gt_pairs(); v0 = pl.read_parquet(wp('data', 'train_s1ids.parquet')).filter(fold_expr('id1') == 0)
S = pl.read_parquet(wp('scores', tag, 'train_s2_c0.parquet'), columns=[*KY, 'p'])
# (1) ceilings
inset = gt.join(S.select(KY), on=KY, how='semi')
print(f'oracle (all true pairs in the pair set): {macro_f05(inset, gt, v0)["macro"]:.5f}   (stage-2 set recall on fold 0: '
      f'{inset.join(v0, on="id1", how="semi").height / gt.join(v0, on="id1", how="semi").height:.4f})')
nk = pl.read_parquet(wp('norm2', 'train_s1.parquet'), columns=['id', 'ctry', 'nc']); nk = nk.join(nk.group_by('ctry', 'nc').len(), on=['ctry', 'nc']).select(pl.col('id').alias('id1'), pl.col('len').alias('k'))
na = pl.concat([pl.read_parquet(wp('norm2', f'train_s{s}.parquet'), columns=['id', 'at']).select(pl.col('id').alias('id2'), pl.lit(s, pl.Int8).alias('src'), (pl.col('at') == '').alias('na')) for s in (2, 3)])
cA = inset.join(nk, on='id1').join(na, on=['id2', 'src']).filter(pl.col('na') & (pl.col('k') > 1)).select(KY)
print(f'oracle abstaining on class-A pairs: {macro_f05(inset.join(cA, on=KY, how="anti"), gt, v0)["macro"]:.5f}')

def decode(S, exclusive, lam=0.08, K=14, pmin=0.01):
    B = S.filter(pl.col('p') >= pmin)
    if exclusive:
        B = B.sort('p', descending=True).unique(['id2', 'src'], keep='first').with_columns(pl.col('p').alias('q'))
    else:
        B = B.with_columns((pl.col('p') / pl.col('p').sum().over(['id2', 'src']).clip(1.0)).alias('q'))
    B = B.sort(['id1', 'q'], descending=[False, True]).with_columns(pl.int_range(pl.len()).over('id1').alias('r'))
    tail = B.filter(pl.col('r') >= K).group_by('id1').agg(pl.col('q').sum().alias('tq')); H = B.filter(pl.col('r') < K)
    ent = H.select('id1').unique().with_row_index('e').join(tail, on='id1', how='left').with_columns(pl.col('tq').fill_null(0.0)); H = H.join(ent.select('id1', 'e'), on='id1')
    Q = np.zeros((ent.height, K)); Q[H['e'].to_numpy(), H['r'].to_numpy()] = H['q'].to_numpy()
    C = K + 6; G = _gain_table(K, C); kb = np.zeros(ent.height, dtype=np.int64); lamv = lam + ent['tq'].to_numpy()
    for s in range(0, ent.height, 100_000):
        kb[s:s + 100_000] = _expected_f(Q[s:s + 100_000], lamv[s:s + 100_000], G, C).argmax(1)
    return H.join(ent.select('e', pl.Series('k', kb)), on='e').filter(pl.col('r') < pl.col('k')).select(KY)

S0 = S.join(v0.select('id1'), on='id1', how='semi')              # fold-0 entities' pairs; records shared with other folds keep their full competition
base = macro_f05(decide(S, 0.7), gt, v0)['macro']
print(f'threshold 0.7 (exclusive): {base:.5f}')
for ex in (True, False):
    P = decode(S, ex)
    m = macro_f05(P, gt, v0, by='ctry')
    dup = P.join(v0, on='id1', how='semi').group_by(['id2', 'src']).len().filter(pl.col('len') > 1).height
    print(f'expected-F decoder, exclusive={ex}: {m["macro"]:.5f} ({m["macro"] - base:+.5f}) ' + str([(c, round(v, 5)) for c, v, _ in m['by'].rows()]) + f'  records listed under >1 fold-0 entity: {dup}')
