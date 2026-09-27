"""Structural model: LightGBM on pair structure only (f50 without any score-derived column, f56 source versions, f67 signed numbers),
so it cannot inherit test-time failures of the p1 -> p chain. Motivation: US test pairs whose name swaps one word for 'service' at the
same address get p ~0.3 (p1 ~0.56) although every pair/ranker feature matches the train distribution, where such pairs are 99.9% true
(p 0.999); count invariance says they are true (their S1 entities have 3.21 other confident copies, true-copy value 3.20).
modes:  cv          : train folds 1-7, fold 0 held out: logloss/AUC of struct vs p, and macro F0.5 of p, struct, and blends
        test <out>  : fit on all folds, score every test pair -> scores/<out>/test_struct.parquet (KY, ctry, p, q)
usage: python s68_struct.py cv | test <out tag>"""
import sys, glob, time
import lightgbm as lgb
from common import *
mode = sys.argv[1]; t0 = time.time()
SCORE = {'p', 'p1', 'x_rank', 'x_gap', 'x_ncand', 'a_rank', 'a_ncand'}
PAR = dict(objective='binary', learning_rate=0.05, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
           lambda_l2=1.0, num_threads=4, verbose=-1, max_bin=255)
R = int(os.environ.get('R', '800'))

def load(split, filt):
    F = pl.scan_parquet(sorted(glob.glob(wp('feat', 'f50', f'{split}_c*.parquet')))).filter(filt) \
          .join(pl.scan_parquet(sorted(glob.glob(wp('feat', 'f56', f'{split}_c*.parquet')))), on=KY, how='left').collect()
    F = F.join(pl.read_parquet(wp('feat', 'f67', f'{split}.parquet')), on=KY, how='left')
    return F.with_columns((pl.col('ctry') == 'India').cast(pl.Int8).alias('is_india'), (pl.col('ctry') == 'France').cast(pl.Int8).alias('is_france'))

def feats_of(F):
    return [c for c in F.columns if c not in SCORE | {'id1', 'id2', 'src', 'ctry', 'y', 'fold'} and F[c].dtype != pl.Utf8]

def auc(y, s):
    o = np.argsort(s, kind='mergesort'); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

if mode == 'cv':
    gt = load_gt_pairs(); v0 = pl.read_parquet(wp('data', 'train_s1ids.parquet')).filter(fold_expr('id1') == 0)
    BASE = pl.read_parquet(wp('scores', 'm4it', 'train_s2_c0.parquet'), columns=[*KY, 'p'])
    Tr = load('train', pl.col('fold') != 0); FE = feats_of(Tr)
    m = lgb.train(PAR, lgb.Dataset(Tr.select(FE).to_numpy().astype(np.float32), Tr['y'].to_numpy(), feature_name=FE, free_raw_data=True), R)
    del Tr
    Va = load('train', pl.col('fold') == 0)
    q = m.predict(Va.select(FE).to_numpy().astype(np.float32), num_threads=4); y = Va['y'].to_numpy(); p = Va['p'].to_numpy()
    ll = lambda z: float(-(y * np.log(np.clip(z, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - z, 1e-6, 1))).mean())
    print(f'fold 0: logloss p {ll(p):.5f} struct {ll(q):.5f} | AUC p {auc(y, p):.5f} struct {auc(y, q):.5f} ({time.time()-t0:.0f}s)', flush=True)
    print('top gain:', sorted(zip(FE, m.feature_importance('gain').round()), key=lambda x: -x[1])[:15])
    dis = (np.abs(q - p) > 0.5)
    print(f'strong disagreement |q-p|>0.5: {dis.mean():.5f} of pairs; among them true share {y[dis].mean():.3f}; p right {((p[dis] > 0.5) == y[dis]).mean():.3f}')
    for name, z in (('p', p), ('struct', q), ('mean', (p + q) / 2), ('geo-logit', 1 / (1 + np.exp(-(np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1)) + np.log(np.clip(q, 1e-6, 1 - 1e-6) / np.clip(1 - q, 1e-6, 1))) / 2)))):
        V = BASE.join(Va.select(KY), on=KY, how='anti').vstack(Va.select(*KY, pl.Series('p', z).cast(pl.Float32)))
        print(f'{name:9s} ' + ' '.join(f'{t}:{macro_f05(decide(V, t), gt, v0)["macro"]:.5f}' for t in (0.6, 0.7, 0.8)), flush=True)
    m.save_model(wp('models', 'struct_cv.txt'))
    Va.select(*KY, 'ctry', 'y', 'p').with_columns(pl.Series('q', q).cast(pl.Float32)).write_parquet(wp('analysis', 'struct_fold0.parquet'))

if mode == 'test':
    out = sys.argv[2]
    Tr = load('train', pl.col('fold') >= 0); FE = feats_of(Tr)
    m = lgb.train(PAR, lgb.Dataset(Tr.select(FE).to_numpy().astype(np.float32), Tr['y'].to_numpy(), feature_name=FE, free_raw_data=True), R)
    del Tr; m.save_model(wp('models', 'struct_all.txt'))
    Te = load('test', pl.col('p') >= 0)
    q = m.predict(Te.select(FE).to_numpy().astype(np.float32), num_threads=4)
    os.makedirs(wp('scores', out), exist_ok=True)
    Te.select(*KY, 'ctry', 'p').with_columns(pl.Series('q', q).cast(pl.Float32)).write_parquet(wp('scores', out, 'test_struct.parquet'))
    print('struct scores written ->', out, f'({time.time()-t0:.0f}s)')
