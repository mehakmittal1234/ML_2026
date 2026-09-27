"""Shift-robust re-scorer: LightGBM on pair evidence only (s80 f80 pair features + raw-name relations + record-level gaps, the
pair-intrinsic part of f50, s67 signed house numbers, s74 street agreement). No input is derived from the base score chain or from
counts of other candidates / distractors, so the model cannot inherit the test-only failures of the chain.
modes:  cv               : train folds 1-7, score fold 0; official macro F0.5 of p (m4it), q, and logit blends of p and q
        test <out tag>   : fit on all folds, score test -> scores/<out>/test_s2_c0.parquet (KY, p1, p = q) and <out>_q.parquet
usage: python s81_robust.py cv | test <out tag>    (env R = rounds, default 900)"""
import sys, glob, time
import lightgbm as lgb
from common import *
mode = sys.argv[1]; t0 = time.time()
R = int(os.environ.get('R', '900'))
F80T = os.environ.get('F80T', '0') == '1'      # test features built with train IDF tables (s80 --trainidf)
fdir = lambda split: 'f80t' if (F80T and split == 'test') else 'f80'
F50 = ['n_tset', 'n_ratio', 'n_part', 'nt_ratio', 'a_tset', 'a_ratio', 'n_inter', 'n_extra2', 'n_extra1', 'name_rel', 'legal_rel', 'hn_rel',
       'hn_ldiff', 'street_inter', 'street_extra2', 'city_eq', 'dg_jac', 'dg_extra2', 'x_noaddr', 'k_s1']
PAR = dict(objective='binary', learning_rate=0.06, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1,
           lambda_l2=1.0, num_threads=4, verbose=-1, max_bin=255)

def load(split, filt):
    F = pl.scan_parquet(sorted(glob.glob(wp('feat', 'f50', f'{split}_c*.parquet')))).filter(filt) \
          .select(*KY, 'ctry', *F50, 'p', 'p1', *(['y', 'fold'] if split == 'train' else [])).collect()
    ids = F.select(KY)
    F = F.join(pl.scan_parquet(sorted(glob.glob(wp('feat', fdir(split), f'{split}_c*.parquet')))).join(ids.lazy(), on=KY, how='semi').collect(), on=KY, how='left')
    F = F.join(pl.read_parquet(wp('feat', fdir(split), f'{split}_rec.parquet')).join(ids, on=KY, how='semi'), on=KY, how='left')
    F = F.join(pl.read_parquet(wp('feat', 'f67', f'{split}.parquet')).join(ids, on=KY, how='semi'), on=KY, how='left')
    F = F.join(pl.read_parquet(wp('feat', 'f74', f'{split}.parquet')).join(ids, on=KY, how='semi'), on=KY, how='left')
    return F.with_columns((pl.col('ctry') == 'India').cast(pl.Int8).alias('is_india2'), (pl.col('ctry') == 'France').cast(pl.Int8).alias('is_france2'))

def feats_of(F):
    drop = {'id1', 'id2', 'src', 'ctry', 'p', 'p1', 'y', 'fold'}
    return [c for c in F.columns if c not in drop and F[c].dtype != pl.Utf8]

def mat(F, FE):
    return F.select([pl.col(c).cast(pl.Float32) for c in FE]).to_numpy()

if mode == 'cv':
    gt = load_gt_pairs(); v0 = pl.read_parquet(wp('data', 'train_s1ids.parquet')).filter(fold_expr('id1') == 0)
    BASE = pl.read_parquet(wp('scores', 'm4it', 'train_s2_c0.parquet'), columns=[*KY, 'p'])
    Tr = load('train', pl.col('fold') != 0); FE = feats_of(Tr)
    print(f'train rows {Tr.height}, features {len(FE)} ({time.time()-t0:.0f}s)', flush=True)
    X = mat(Tr, FE); y = Tr['y'].to_numpy(); del Tr
    m = lgb.train(PAR, lgb.Dataset(X, y, feature_name=FE, free_raw_data=True), R); del X
    m.save_model(wp('models', 'robust_cv.txt'))
    Va = load('train', pl.col('fold') == 0)
    q = m.predict(mat(Va, FE), num_threads=4); yv = Va['y'].to_numpy(); p = Va['p'].to_numpy()
    ll = lambda z: float(-(yv * np.log(np.clip(z, 1e-6, 1)) + (1 - yv) * np.log(np.clip(1 - z, 1e-6, 1))).mean())
    print(f'fold 0 logloss: p {ll(p):.5f}  q {ll(q):.5f} ({time.time()-t0:.0f}s)', flush=True)
    print('top gain:', sorted(zip(FE, m.feature_importance('gain').round()), key=lambda x: -x[1])[:20], flush=True)
    lg = lambda z: np.log(np.clip(z, 1e-6, 1 - 1e-6) / np.clip(1 - z, 1e-6, 1 - 1e-6))
    for w in (0.0, 0.3, 0.5, 0.7, 1.0):
        z = 1 / (1 + np.exp(-(w * lg(q) + (1 - w) * lg(p))))
        V = BASE.join(Va.select(KY), on=KY, how='anti').vstack(Va.select(*KY, pl.Series('p', z).cast(pl.Float32)))
        print(f'weight on q {w}: ' + ' '.join(f'{t}:{macro_f05(decide(V, t), gt, v0)["macro"]:.5f}' for t in (0.5, 0.6, 0.7, 0.8)) + f'  logloss {ll(z):.5f}', flush=True)
    Va.select(*KY, 'ctry', 'y', 'p').with_columns(pl.Series('q', q).cast(pl.Float32)).write_parquet(wp('analysis', 'robust_fold0.parquet'))

if mode == 'score':                                  # score test with a saved model: score <model name> <out tag>
    m = lgb.Booster(model_file=wp('models', f'{sys.argv[2]}.txt')); FE = m.feature_name(); out = sys.argv[3]
    Te = load('test', pl.col('p') >= 0)
    q = m.predict(mat(Te, FE), num_threads=4)
    os.makedirs(wp('scores', out), exist_ok=True)
    Te.select(*KY, 'p1', pl.Series('p', q).cast(pl.Float32)).write_parquet(wp('scores', out, 'test_s2_c0.parquet'))
    print('scored test ->', out, f'({time.time()-t0:.0f}s)')

if mode == 'test':
    out = sys.argv[2]
    Tr = load('train', pl.col('fold') >= 0); FE = feats_of(Tr)
    X = mat(Tr, FE); y = Tr['y'].to_numpy(); del Tr
    m = lgb.train(PAR, lgb.Dataset(X, y, feature_name=FE, free_raw_data=True), R); del X
    m.save_model(wp('models', f'{out}.txt'))
    Te = load('test', pl.col('p') >= 0)
    q = m.predict(mat(Te, FE), num_threads=4)
    os.makedirs(wp('scores', out), exist_ok=True)
    Te.select(*KY, 'p1', pl.Series('p', q).cast(pl.Float32)).write_parquet(wp('scores', out, 'test_s2_c0.parquet'))
    print('robust test scores ->', out, f'({time.time()-t0:.0f}s)')
