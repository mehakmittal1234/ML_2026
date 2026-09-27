"""Density-ratio shift correction of calibrated scores (generator-invariant true class).
If the true-pair feature density is the same in source (validation) and target (test), then
  P_target(true | f) = P_source(true | f) * (pi_t / pi_s) * p_s(f) / p_t(f)
and with count invariance (true pairs per S1 entity equal) this is  p(f) * (S_t / S_s) * odds_D(f) * (|U_fit| / |U|),
where D separates ALL source pairs (label 1, S1-composition weighted to the target) from target pairs (label 0). Where nothing shifted
the factor is ~1 (q ~ p); where the target has extra distractors the factor drops. Target pairs are cross-fitted in 2 halves.
modes:  sanity                       : source = fold-0 half A, target = fold-0 half B (no shift) -> macro F on half B, p vs q
        test <base test tag> <out>   : source = fold 0 (m4it), target = test, per country US / India (France keeps p)
        [--feats minimal|full]"""
import sys, glob, time
import lightgbm as lgb
from common import *
mode = sys.argv[1]
FULL = ['p', 'kc', 'g_src', 'g_oth', 'n_tset', 'n_ratio', 'nt_ratio', 'a_tset', 'a_ratio', 'n_extra2', 'n_extra1', 'name_rel', 'legal_rel', 'hn_rel', 'hn_ldiff',
        'street_inter', 'street_extra2', 'city_eq', 'dg_jac', 'dg_extra2', 'x_noaddr', 'c_src', 'c_oth', 'anchor', 'anchor_oth', 'c_at_xh', 'c_at_h1', 'c_src_at_xh']
INV = ['kc', 'n_tset', 'n_ratio', 'nt_ratio', 'a_tset', 'a_ratio', 'n_extra2', 'n_extra1', 'name_rel', 'legal_rel', 'hn_rel', 'hn_ldiff', 'street_inter', 'street_extra2',
       'city_eq', 'dg_jac', 'dg_extra2', 'x_noaddr', 'c_src', 'c_oth', 'anchor', 'anchor_oth', 'c_at_xh', 'c_at_h1', 'c_src_at_xh',
       'sv_s_min', 'sv_o_min', 'sv_s_eq', 'sv_s_city', 'sv_s_hn', 'sv_x_extra']
FEATS = INV if '--inv' in sys.argv else ([f for f in FULL if f != 'p'] if '--nop' in sys.argv else FULL)
MINLEAF = 5000 if ('--nop' in sys.argv or '--inv' in sys.argv) else 2000
PAR = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_data_in_leaf=MINLEAF, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
           lambda_l2=50.0, num_threads=4, verbose=-1)
R = 300
t0 = time.time()
load = lambda split, filt: pl.scan_parquet(sorted(glob.glob(wp('feat', 'f50', f'{split}_c*.parquet')))).join(pl.scan_parquet(sorted(glob.glob(wp('feat', 'f56', f'{split}_c*.parquet')))), on=KY, how='left') \
    .filter(filt).collect().with_columns(pl.col('k_s1').clip(1, 4).alias('kc'))
comp = lambda e: np.array([e.filter(pl.col('kc') == k).height for k in range(1, 5)], dtype=float)

def correct(Src, Tgt, eS, eT):
    """Src/Tgt: pair frames (with kc; Tgt with half). eS/eT: S1 entity frames with kc. Returns target factor and q = min(1, p*factor)."""
    cS, cT = comp(eS), comp(eT)
    w = (cT / cT.sum()) / (cS / cS.sum())
    wS = Src['kc'].replace_strict({k: float(w[k - 1]) for k in range(1, 5)}, return_dtype=pl.Float64).to_numpy()
    fac = np.zeros(Tgt.height)
    for h in (0, 1):
        Tt = Tgt.filter(pl.col('half') != h)
        X = np.vstack([Src.select(FEATS).to_numpy(), Tt.select(FEATS).to_numpy()]).astype(np.float32)
        y = np.r_[np.ones(Src.height), np.zeros(Tt.height)]
        m = lgb.train(PAR, lgb.Dataset(X, y, weight=np.r_[wS, np.ones(Tt.height)], feature_name=FEATS, free_raw_data=True), R)
        te = (Tgt['half'] == h).to_numpy()
        d = m.predict(Tgt.filter(pl.col('half') == h).select(FEATS).to_numpy().astype(np.float32), num_threads=4)
        # odds = sum(wS) p_s / (|Tt| p_t);  p_s/p_t = odds * |Tt| / sum(wS);  pi_t/pi_s = (N_true_s S_t/S_s / N_t) / (N_true_s / N_s) with N_s -> sum(wS)
        fac[te] = d / (1 - d) * (len(eT) / len(eS)) * (Tt.height / Tgt.height)
    q = np.minimum(1.0, Tgt['p'].to_numpy() * fac)
    return fac, q, m

if mode in ('sanity', 'sanity2'):
    F0 = load('train', pl.col('fold') >= 0).with_columns(pl.when(pl.col('fold') == 0).then(pl.col('id1').hash(seed=11) % 2).otherwise(0).alias('ab'))
    ka = pl.read_parquet(wp('norm2', 'train_s1.parquet'), columns=['id', 'ctry', 'nc'])
    ka = ka.join(ka.group_by('ctry', 'nc').len(), on=['ctry', 'nc']).select(pl.col('id').alias('id1'), pl.col('len').clip(1, 4).alias('kc'))
    ents = pl.read_parquet(wp('data', 'train_s1ids.parquet')).with_columns(pl.when(fold_expr('id1') == 0).then(pl.col('id1').hash(seed=11) % 2).otherwise(0).alias('ab')).join(ka, on='id1')
    if mode == 'sanity':                       # source = fold-0 half A only
        F0 = F0.filter(fold_expr('id1') == 0); ents = ents.filter(fold_expr('id1') == 0)
    gt = load_gt_pairs(); BASE = pl.read_parquet(wp('scores', 'm4it', 'train_s2_c0.parquet'), columns=[*KY, 'p'])
    new = []
    for c in ('US', 'India'):
        Src = F0.filter((pl.col('ab') == 0) & (pl.col('ctry') == c)); Tgt = F0.filter((pl.col('ab') == 1) & (pl.col('ctry') == c)).with_columns((pl.col('id1').hash(seed=3) % 2).alias('half'))
        fac, q, _ = correct(Src, Tgt, ents.filter((pl.col('ab') == 0) & (pl.col('ctry') == c)), ents.filter((pl.col('ab') == 1) & (pl.col('ctry') == c)))
        y = Tgt['y'].to_numpy(); p = Tgt['p'].to_numpy()
        ll = lambda z: float(-(y * np.log(np.clip(z, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - z, 1e-6, 1))).mean())
        print(f'{c}: factor median {np.median(fac):.3f} p5 {np.percentile(fac, 5):.3f} p95 {np.percentile(fac, 95):.3f}; sum q {q.sum():.0f} / true {y.sum()} / sum p {p.sum():.0f}; logloss p {ll(p):.4f} q {ll(q):.4f} ({time.time()-t0:.0f}s)', flush=True)
        new.append(Tgt.select(*KY, 'p').with_columns(pl.Series('fac', fac)))
        print(f'   share of pairs with factor < 0.3/0.5/0.7: {(fac < 0.3).mean():.5f} {(fac < 0.5).mean():.5f} {(fac < 0.7).mean():.5f}', flush=True)
    new = pl.concat(new); evalB = ents.filter(pl.col('ab') == 1).select('id1')
    print(f'half B macro F, base p: ' + ' '.join(f'{t}:{macro_f05(decide(BASE, t), gt, evalB)["macro"]:.5f}' for t in (0.6, 0.7, 0.8)))
    for tau in (0.3, 0.5, 0.7):
        nq = new.select(*KY, pl.when(pl.col('fac') < tau).then(pl.col('p') * pl.col('fac')).otherwise(pl.col('p')).cast(pl.Float32).alias('p'))
        V = BASE.join(nq.select(KY), on=KY, how='anti').vstack(nq)
        print(f'half B macro F, corrected only where factor < {tau}: ' + ' '.join(f'{t}:{macro_f05(decide(V, t), gt, evalB)["macro"]:.5f}' for t in (0.6, 0.7, 0.8)), flush=True)

if mode == 'test':
    base, out = sys.argv[2], sys.argv[3]
    Src = load('train', pl.col('fold') >= 0)
    ks = lambda split: (lambda a: a.join(a.group_by('ctry', 'nc').len(), on=['ctry', 'nc']).select(pl.col('id').alias('id1'), 'ctry', pl.col('len').clip(1, 4).alias('kc')))(
        pl.read_parquet(wp('norm2', f'{split}_s1.parquet'), columns=['id', 'ctry', 'nc']))
    eS_all, eT_all = ks('train'), ks('test')
    res = []
    for c in ('US', 'India'):
        Tgt = load('test', pl.col('ctry') == c).with_columns((pl.col('id1').hash(seed=3) % 2).alias('half'))
        fac, _, m = correct(Src.filter(pl.col('ctry') == c), Tgt, eS_all.filter(pl.col('ctry') == c), eT_all.filter(pl.col('ctry') == c))
        print(f'{c}: target pairs {Tgt.height}; factor median {np.median(fac):.3f} p5 {np.percentile(fac, 5):.3f}; share < 0.3/0.5/0.7: '
              f'{(fac < 0.3).mean():.5f} {(fac < 0.5).mean():.5f} {(fac < 0.7).mean():.5f} ({time.time()-t0:.0f}s)', flush=True)
        print('   top gain:', sorted(zip(FEATS, m.feature_importance('gain').round()), key=lambda x: -x[1])[:10], flush=True)
        res.append(Tgt.select(*KY).with_columns(pl.Series('fac', fac).cast(pl.Float32)))
    F = pl.concat(res)
    os.makedirs(wp('scores', out), exist_ok=True); F.write_parquet(wp('scores', out, 'test_factor.parquet'))
    print('factors written ->', out)
