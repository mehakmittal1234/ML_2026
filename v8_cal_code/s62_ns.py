"""Near-twin specialist trained on REAL generator neighbours: train pairs whose record belongs to ANOTHER S1 entity with the same core
name (a same-name twin, mostly India S1 neighbours) vs true pairs, restricted to different-house-number pairs (hn_rel 3 trunc / 4 near /
5 far) and name_rel 0-3. Structural features only (no score p: in train the twin is in S1 and wins the competition, in test the neighbour
is an orphan). Cross-validated by fold (AUC per country); final model on all folds gives a likelihood ratio for test pairs:
    LR(f) = odds_model(f) / (n_true / n_neigh)   and, inside a hybrid cell with count-invariant share s,
    q = s*LR / (s*LR + 1 - s)
RESULT: cross-validated AUC true vs neighbour copies India 0.9978 (10,219 neighbours), US 0.9881 (264), nearby-number 0.9977.
Test (ns1): 17,323 accepted pairs removed (US 11,619, India 5,704). Cross-truth: ns1 worst case +0.00092 vs hyb4 +0.00076;
blend ens1 (mean of hyb4 and ns1 scores) worst case +0.00120 -> outputs/v13_ens_efd.
usage: python s62_ns.py cv  |  python s62_ns.py test <raw tag> <calibrated tag> <thr> <out tag> [hybrid cell source tag, default hyb4]"""
import sys, glob, time
import lightgbm as lgb
from common import *
mode = sys.argv[1]
FE = ['name_rel', 'hn_rel', 'hn_ldiff', 'legal_rel', 'n_extra1', 'n_extra2', 'n_tset', 'n_ratio', 'nt_ratio', 'a_tset', 'a_ratio', 'street_inter', 'street_extra2',
      'city_eq', 'dg_jac', 'dg_extra2', 'c_src', 'c_oth', 'anchor', 'anchor_oth', 'c_at_xh', 'c_at_h1', 'c_src_at_xh', 'g_src', 'g_oth',
      'sv_s_min', 'sv_o_min', 'sv_s_eq', 'sv_s_city', 'sv_s_hn', 'sv_x_extra', 'is_india']
PAR = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, num_threads=4, verbose=-1)
t0 = time.time()
flt = pl.col('hn_rel').is_in([3, 4, 5]) & pl.col('name_rel').is_in([0, 1, 2, 3])
def load(split, extra=None):
    lf = pl.scan_parquet(sorted(glob.glob(wp('feat', 'f50', f'{split}_c*.parquet')))).join(pl.scan_parquet(sorted(glob.glob(wp('feat', 'f56', f'{split}_c*.parquet')))), on=KY, how='left').filter(flt)
    if extra is not None: lf = lf.filter(extra)
    return lf.collect().with_columns((pl.col('ctry') == 'India').cast(pl.Int8).alias('is_india'))
def auc(pos, neg):
    x = np.r_[pos, neg]; o = np.argsort(x, kind='mergesort'); r = np.empty(len(x)); r[o] = np.arange(1, len(x) + 1)
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

T = load('train')
gt = load_gt_pairs(); N1 = pl.read_parquet(wp('norm2', 'train_s1.parquet'), columns=['id', 'nc'])
T = T.join(gt.select('id2', 'src', pl.col('id1').alias('owner')), on=['id2', 'src'], how='left').join(N1.rename({'id': 'id1'}), on='id1') \
     .join(N1.rename({'id': 'owner', 'nc': 'nc_o'}), on='owner', how='left')
T = T.with_columns(pl.when(pl.col('y') == 1).then(1).when(pl.col('owner').is_not_null() & (pl.col('nc_o') == pl.col('nc')) & (pl.col('owner') != pl.col('id1'))).then(0).otherwise(-1).alias('lab'))
D = T.filter(pl.col('lab') >= 0)
print(f'training pairs: true {D.filter(pl.col("lab") == 1).height}, same-name neighbour copies {D.filter(pl.col("lab") == 0).height} '
      f'(India {D.filter((pl.col("lab") == 0) & (pl.col("ctry") == "India")).height}, US {D.filter((pl.col("lab") == 0) & (pl.col("ctry") == "US")).height}) ({time.time()-t0:.0f}s)', flush=True)
if mode == 'cv':
    oof = np.zeros(D.height); f = D['fold'].to_numpy()
    for k in range(0, 8, 2):
        te = (f == k) | (f == k + 1)
        m = lgb.train(PAR, lgb.Dataset(D.filter(pl.Series(~te)).select(FE).to_numpy().astype(np.float32), D['lab'].to_numpy()[~te]), 400)
        oof[te] = m.predict(D.filter(pl.Series(te)).select(FE).to_numpy().astype(np.float32), num_threads=4)
    D = D.with_columns(pl.Series('s', oof))
    for c in ('India', 'US'):
        d = D.filter(pl.col('ctry') == c); pos, neg = d.filter(pl.col('lab') == 1)['s'].to_numpy(), d.filter(pl.col('lab') == 0)['s'].to_numpy()
        if len(neg): print(f'{c}: cross-validated AUC true vs neighbour {auc(pos, neg):.4f}  (n true {len(pos)}, neighbour {len(neg)})')
    d = D.filter(pl.col('hn_rel') == 4); print(f'nearby-number only: AUC {auc(d.filter(pl.col("lab") == 1)["s"].to_numpy(), d.filter(pl.col("lab") == 0)["s"].to_numpy()):.4f}')
    print('top gain:', sorted(zip(FE, m.feature_importance('gain').round()), key=lambda x: -x[1])[:12])

if mode == 'test':
    from scipy.special import logit, expit
    rtag, ctag, thr, out = sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5]
    m = lgb.train(PAR, lgb.Dataset(D.select(FE).to_numpy().astype(np.float32), D['lab'].to_numpy()), 400)
    prior_odds = D['lab'].sum() / (D.height - D['lab'].sum())
    del T, D
    # ---- hybrid cells (same definition as s54: fine (ctry,name_rel,hn_rel,anchor) with coarse fallback; corrected if share<0.9, sum p>=200, z>=4)
    FINE = ['ctry', 'name_rel', 'hn_rel', 'anchor']; COARSE = ['ctry', 'name_rel', 'hn_rel']; MINTRUE = 200
    cols = [*KY, 'ctry', 'p', 'name_rel', 'hn_rel', 'anchor']
    V = pl.scan_parquet(sorted(glob.glob(wp('feat', 'f50', 'train_c*.parquet')))).filter(pl.col('fold') == 0).select(*cols, 'y').collect()
    TT = pl.scan_parquet(sorted(glob.glob(wp('feat', 'f50', 'test_c*.parquet')))).select(cols).collect().filter(pl.col('ctry') != 'France')
    nv = dict(pl.read_parquet(wp('data', 'train_s1ids.parquet')).filter(fold_expr('id1') == 0).group_by('ctry').len().rows())
    nt = dict(pl.read_parquet(wp('data', 'test_s1.parquet'), columns=['country']).group_by('country').len().rows())
    sc = lambda c: pl.col('ctry').replace_strict(c, return_dtype=pl.Float64)
    EF = V.group_by(FINE).agg(pl.col('y').sum().alias('vt_f')).with_columns((pl.col('vt_f') / sc(nv) * sc(nt)).alias('exp_f'))
    EC = V.group_by(COARSE).agg(pl.col('y').sum().alias('vt_c')).with_columns((pl.col('vt_c') / sc(nv) * sc(nt)).alias('exp_c'))
    TT = TT.join(EF, on=FINE, how='left').join(EC, on=COARSE, how='left').with_columns(pl.col('vt_f', 'vt_c', 'exp_c').fill_null(0))
    fine = pl.col('vt_f') >= MINTRUE
    TT = TT.with_columns(pl.when(fine).then(pl.concat_str([pl.col(c).cast(pl.Utf8) for c in FINE], separator='|')).otherwise(pl.concat_str([pl.col(c).cast(pl.Utf8) for c in COARSE], separator='|')).alias('cell'),
                         pl.when(fine).then(pl.col('vt_f')).otherwise(pl.col('vt_c')).alias('vt_cell'))
    fin = TT.filter(fine).select(*COARSE, 'cell', 'exp_f').unique().group_by(COARSE).agg(pl.col('exp_f').sum().alias('_fin'))
    TT = TT.join(fin, on=COARSE, how='left').with_columns(pl.when(fine).then(pl.col('exp_f')).otherwise(pl.col('exp_c') - pl.col('_fin').fill_null(0)).clip(0).alias('exp_cell')).drop('_fin')
    G = TT.group_by('cell').agg(pl.col('exp_cell').first(), pl.col('vt_cell').first(), pl.col('p').sum().alias('sum_p'))
    G = G.with_columns((pl.col('exp_cell') / pl.col('sum_p')).alias('share'), ((pl.col('sum_p') - pl.col('exp_cell')) / (pl.col('exp_cell') / pl.col('vt_cell').clip(1).sqrt()).clip(1e-9)).alias('z'))
    fix = G.filter((pl.col('share') < 0.9) & (pl.col('sum_p') >= 200) & (pl.col('z') >= 4))
    print(f'corrected cells {fix.height} ({time.time()-t0:.0f}s)', flush=True)
    # ---- specialist likelihood ratio for test pairs in its domain
    X = load('test').filter(pl.col('ctry') != 'France')
    d = m.predict(X.select(FE).to_numpy().astype(np.float32), num_threads=4)
    LR = X.select(KY).with_columns(pl.Series('llr', np.log(np.clip(d, 1e-9, 1 - 1e-9) / np.clip(1 - d, 1e-9, 1)) - np.log(prior_odds)).cast(pl.Float32))
    TT = TT.join(fix.select('cell'), on='cell', how='semi').join(LR, on=KY, how='left')
    HY = pl.read_parquet(wp('scores', sys.argv[6] if len(sys.argv) > 6 else 'hyb4', 'test_s2_c0.parquet'), columns=[*KY, 'p']).rename({'p': 'p_hyb'})
    CAL = pl.read_parquet(wp('scores', ctag, 'test_s2_c0.parquet'), columns=[*KY, 'p']).rename({'p': 'p_cal'})
    TT = TT.join(HY, on=KY, how='left').join(CAL, on=KY, how='left')
    Q = []
    for r in fix.iter_rows(named=True):
        c = TT.filter(pl.col('cell') == r['cell'])
        lp = logit(np.clip(c['p'].to_numpy().astype(np.float64), 1e-6, 1 - 1e-6)); ll = c['llr'].to_numpy()
        has = ~np.isnan(ll.astype(float)) if ll.dtype != object else np.array([v is not None for v in ll])
        ll = np.where(has, np.nan_to_num(ll.astype(float)), 0.0)
        # pairs outside the specialist's domain keep the hybrid's within-cell ratio; domain pairs: logit p + llr - b, b fitted to the count
        other = (c['p_hyb'] / c['p_cal']).fill_null(1.0).to_numpy() * c['p'].to_numpy()
        lo, hi = -20.0, 20.0
        for _ in range(60):
            b = (lo + hi) / 2
            q = np.where(has, expit(lp + ll - b), np.minimum(other, 1.0))
            if q.sum() > r['exp_cell']: lo = b
            else: hi = b
        Q.append(c.select(KY).with_columns(pl.Series('ratio', q / np.maximum(c['p'].to_numpy(), 1e-9)).cast(pl.Float32)))
    Q = pl.concat(Q)
    full = pl.read_parquet(wp('scores', ctag, 'test_s2_c0.parquet')).join(Q, on=KY, how='left')
    full = full.with_columns(pl.when(pl.col('ratio').is_not_null()).then((pl.col('p') * pl.col('ratio')).clip(0, 1)).otherwise(pl.col('p')).cast(pl.Float32).alias('p')).drop('ratio')
    os.makedirs(wp('scores', out), exist_ok=True); full.write_parquet(wp('scores', out, 'test_s2_c0.parquet'))
    M0 = decide(pl.read_parquet(wp('scores', ctag, 'test_s2_c0.parquet'), columns=[*KY, 'p']), thr); M1 = decide(full, thr)
    Mh = decide(pl.read_parquet(wp('scores', 'hyb4', 'test_s2_c0.parquet'), columns=[*KY, 'p']), thr)
    tc = pl.read_parquet(wp('data', 'test_s1.parquet'), columns=['id', 'country']).rename({'id': 'id1', 'country': 'ctry'})
    print(f'matches {M0.height} -> {M1.height}; removed {M0.join(M1, on=KY, how="anti").height} (by country {M0.join(M1, on=KY, how="anti").join(tc, on="id1").group_by("ctry").len().sort("ctry").rows()}), '
          f'added {M1.join(M0, on=KY, how="anti").height}; vs hyb4: removed-in-both {M0.join(M1, on=KY, how="anti").join(M0.join(Mh, on=KY, how="anti"), on=KY, how="semi").height}, hyb4 removed {M0.join(Mh, on=KY, how="anti").height}')
    print('written scores/' + out, f'({time.time()-t0:.0f}s)')
