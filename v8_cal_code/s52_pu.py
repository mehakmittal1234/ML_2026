"""Test-adapting posterior by density ratio (PU learning under generator invariance).
The copy generator is identical in train and test, so the feature density of TRUE pairs is invariant; the distractor mix is not.
A classifier D separates P = known-true pairs (fold 0; its scores are two-model averages like test) from U = the target pairs;
then P_target(true | f) = odds_D(f) * S_target / S_P  (S = number of S1 entities; count invariance).
S1 composition (same-name group size k, capped at 4) is matched by weighting P entities to the target's composition, so a different
share of unique names is not mistaken for a distractor shift. U is cross-fitted in 2 halves by S1 entity.
modes:
  sanity  : P = true pairs of fold-0 half A, U = all pairs of fold-0 half B (no shift) -> macro F on half B, posterior vs base p
  test    : P = fold-0 true pairs, U = test pairs, per country (US, India) -> scores/<out>/test_s2_c0.parquet (France keeps base p)
usage: python s52_pu.py sanity | python s52_pu.py test <base test tag> <out tag>"""
import sys, glob, time
import lightgbm as lgb
from common import *
mode = sys.argv[1]
FEATS = ['p', 'p1', 'x_rank', 'x_gap', 'k_s1', 'g_src', 'g_oth', 'n_tset', 'n_ratio', 'n_part', 'nt_ratio', 'a_tset', 'a_ratio', 'n_inter', 'n_extra2', 'n_extra1',
         'name_rel', 'legal_rel', 'hn_rel', 'hn_ldiff', 'street_inter', 'street_extra2', 'city_eq', 'dg_jac', 'dg_extra2', 'x_noaddr', 'c_src', 'c_oth', 'anchor',
         'anchor_oth', 'c_at_xh', 'c_at_h1', 'c_src_at_xh']
PAR = dict(objective='binary', learning_rate=0.05, num_leaves=63, min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
           lambda_l2=10.0, num_threads=4, verbose=-1)
R = 400
t0 = time.time()
load = lambda split, filt: pl.scan_parquet(sorted(glob.glob(wp('feat', 'f50', f'{split}_c*.parquet')))).filter(filt).collect()
kc = pl.col('k_s1').clip(1, 4).alias('kc')

def posterior(P, U, S_P, S_U, comp_P, comp_U):
    """P: true pairs (with kc), U: target pairs (with kc, half). comp_*: S1 entity counts by kc. Returns U posteriors (cross-fitted)."""
    w = (comp_U / comp_U.sum()) / (comp_P / comp_P.sum())                  # entity-composition weight for P rows, by kc
    wP = P['kc'].replace_strict({k: float(w[k - 1]) for k in range(1, 5)}, return_dtype=pl.Float64).to_numpy()
    out = np.zeros(U.height)
    for h in (0, 1):
        Ut = U.filter(pl.col('half') != h)
        X = np.vstack([P.select(FEATS).to_numpy(), Ut.select(FEATS).to_numpy()]).astype(np.float32)
        y = np.r_[np.ones(P.height), np.zeros(Ut.height)]
        sw = np.r_[wP, np.ones(Ut.height)]
        m = lgb.train(PAR, lgb.Dataset(X, y, weight=sw, feature_name=FEATS, free_raw_data=True), R)
        te = (U['half'] == h).to_numpy()
        d = m.predict(U.filter(pl.col('half') == h).select(FEATS).to_numpy().astype(np.float32), num_threads=4)
        # D/(1-D) = sum(wP) p_true(f) / (|Ut| p_U(f)); expected true pairs in U = sum(wP) * S_U / S_P (composition-matched count invariance)
        # => P_U(true | f) = odds_D(f) * (S_U / S_P) * (|Ut| / |U|)
        out[te] = np.clip(d / (1 - d) * (S_U / S_P) * (Ut.height / U.height), 0, 1)
    return out, m

if mode == 'sanity':
    F0 = load('train', pl.col('fold') == 0).with_columns(kc, (pl.col('id1').hash(seed=11) % 2).alias('ab'))
    ents = pl.read_parquet(wp('data', 'train_s1ids.parquet')).filter(fold_expr('id1') == 0).with_columns((pl.col('id1').hash(seed=11) % 2).alias('ab'))
    k_all = pl.read_parquet(wp('norm2', 'train_s1.parquet'), columns=['id', 'ctry', 'nc']); k_all = k_all.join(k_all.group_by('ctry', 'nc').len(), on=['ctry', 'nc']).select(pl.col('id').alias('id1'), pl.col('len').clip(1, 4).alias('kc'))
    ents = ents.join(k_all, on='id1')
    gt = load_gt_pairs(); BASE = pl.read_parquet(wp('scores', 'm4it', 'train_s2_c0.parquet'), columns=[*KY, 'p'])
    evalB = ents.filter(pl.col('ab') == 1).select('id1')
    newp = []
    for c in ('US', 'India'):
        P = F0.filter((pl.col('ab') == 0) & (pl.col('ctry') == c) & (pl.col('y') == 1))
        U = F0.filter((pl.col('ab') == 1) & (pl.col('ctry') == c)).with_columns((pl.col('id1').hash(seed=3) % 2).alias('half'))
        eA = ents.filter((pl.col('ab') == 0) & (pl.col('ctry') == c)); eB = ents.filter((pl.col('ab') == 1) & (pl.col('ctry') == c))
        comp = lambda e: np.array([e.filter(pl.col('kc') == k).height for k in range(1, 5)], dtype=float)
        q, _ = posterior(P, U, eA.height, eB.height, comp(eA), comp(eB))
        Uq = U.select(*KY, 'p', 'y').with_columns(pl.Series('q', q))
        print(f'{c}: U pairs {U.height}; sum q {Uq["q"].sum():.0f} vs true {Uq["y"].sum()} vs sum p {Uq["p"].sum():.0f}; '
              f'mean |q-p| {(Uq["q"] - Uq["p"]).abs().mean():.4f}; logloss p {(-(Uq["y"]*np.log(np.clip(Uq["p"],1e-6,1))+(1-Uq["y"])*np.log(np.clip(1-Uq["p"],1e-6,1)))).mean():.4f} '
              f'q {(-(Uq["y"]*np.log(np.clip(Uq["q"],1e-6,1))+(1-Uq["y"])*np.log(np.clip(1-Uq["q"],1e-6,1)))).mean():.4f}  ({time.time()-t0:.0f}s)', flush=True)
        newp.append(Uq.select(*KY, pl.col('q').cast(pl.Float32).alias('p')))
    newp = pl.concat(newp)
    V = BASE.join(newp.select(KY), on=KY, how='anti').vstack(newp)
    for nm, S in (('base p', BASE), ('PU posterior', V)):
        sc = {t: macro_f05(decide(S, t), gt, evalB)['macro'] for t in (0.5, 0.6, 0.7, 0.8)}
        print(f'half B macro F, {nm}: ' + ' '.join(f'{t}:{v:.5f}' for t, v in sc.items()))
