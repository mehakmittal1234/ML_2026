"""Round 2 of the shift-robust re-scorer: competition / S1-context features recomputed from the robust scores themselves.
Step oof : two half-data robust models (A on folds 1-4, B on folds 5-7, same features as s81) give out-of-fold q for every train pair
           (fold 0 = mean of A and B) and test q = mean of A and B -> scores/robA_B/{train,test}_q.parquet
Step ctx : per split, from q: record rank / gap to best rival / #candidates q>=0.5 / sum of q, and the S1 entity's confident copies
           (best owner by q, q >= 0.9) in the same and the other source (excluding the pair's own record) -> feat/f83/<split>.parquet
Step cv  : LightGBM on s81 features + ctx + logit q, train folds 1-7, fold 0 macro F0.5 (and logit blends with m4it p)
Step test <out>: fit on all folds, score test.
usage: python s83_round2.py oof | ctx | cv | test <out>     (env R rounds, default 700)"""
import sys, glob, time
import lightgbm as lgb
from common import *
sys.argv = [sys.argv[0], 'noop'] + sys.argv[1:]      # import s81 helpers without running its modes
import s81_robust as S81
sys.argv = [sys.argv[0]] + sys.argv[2:]
mode = sys.argv[1]; t0 = time.time()
R = int(os.environ.get('R', '700'))
PAR = S81.PAR
lg = lambda z: np.log(np.clip(z, 1e-6, 1 - 1e-6) / np.clip(1 - z, 1e-6, 1 - 1e-6))

if mode == 'oof':
    Tr = S81.load('train', pl.col('fold') >= 0); FE = S81.feats_of(Tr)
    fo = Tr['fold'].to_numpy(); y = Tr['y'].to_numpy(); X = S81.mat(Tr, FE); ky = Tr.select(KY); del Tr
    Te = S81.load('test', pl.col('p') >= 0); XT = S81.mat(Te, FE); kt = Te.select(KY); del Te
    qtr = np.zeros(len(y), dtype=np.float32); q0 = np.zeros(len(y), dtype=np.float32); qte = np.zeros(XT.shape[0], dtype=np.float32)
    for name, trf in (('A', (1, 2, 3, 4)), ('B', (5, 6, 7))):
        tr = np.isin(fo, trf)
        m = lgb.train(PAR, lgb.Dataset(X[tr], y[tr], feature_name=FE), R)
        oth = ~np.isin(fo, trf) & (fo != 0)
        qtr[oth] = m.predict(X[oth], num_threads=4)
        z = fo == 0; q0[z] += m.predict(X[z], num_threads=4) / 2
        qte += m.predict(XT, num_threads=4) / 2
        print(f'  model {name} on folds {trf} ({time.time()-t0:.0f}s)', flush=True)
    qtr[fo == 0] = q0[fo == 0]
    os.makedirs(wp('scores', 'robAB'), exist_ok=True)
    ky.with_columns(pl.Series('q', qtr)).write_parquet(wp('scores', 'robAB', 'train_q.parquet'))
    kt.with_columns(pl.Series('q', qte)).write_parquet(wp('scores', 'robAB', 'test_q.parquet'))
    print('OOF_DONE', f'{time.time()-t0:.0f}s')

if mode == 'ctx':
    os.makedirs(wp('feat', 'f83'), exist_ok=True)
    for split in ('train', 'test'):
        Q = pl.read_parquet(wp('scores', 'robAB', f'{split}_q.parquet'))
        Q = Q.with_columns(pl.col('q').rank('ordinal', descending=True).over(['id2', 'src']).cast(pl.Float32).alias('c_rrank'),
                           (pl.col('q') >= 0.5).sum().over(['id2', 'src']).cast(pl.Float32).alias('c_rn05'),
                           pl.col('q').sum().over(['id2', 'src']).alias('c_rsum'))
        top2 = Q.group_by(['id2', 'src']).agg(pl.col('q').top_k(2).alias('_t')).with_columns(pl.col('_t').list.get(0).alias('_b1'), pl.col('_t').list.get(1, null_on_oob=True).fill_null(0).alias('_b2')).drop('_t')
        Q = Q.join(top2, on=['id2', 'src']).with_columns(pl.when(pl.col('c_rrank') == 1).then(pl.col('q') - pl.col('_b2')).otherwise(pl.col('q') - pl.col('_b1')).alias('c_rgap')).drop('_b1', '_b2')
        conf = Q.filter((pl.col('c_rrank') == 1) & (pl.col('q') >= 0.9)).group_by('id1', 'src').len().rename({'len': '_n'})
        Q = Q.join(conf, on=['id1', 'src'], how='left').join(conf.with_columns((5 - pl.col('src')).cast(pl.Int8).alias('src')).rename({'_n': '_no'}), on=['id1', 'src'], how='left')
        own = ((pl.col('c_rrank') == 1) & (pl.col('q') >= 0.9)).cast(pl.Float32)
        Q = Q.with_columns((pl.col('_n').fill_null(0) - own).alias('c_s1_src'), pl.col('_no').fill_null(0).cast(pl.Float32).alias('c_s1_oth'),
                           pl.col('q').rank('ordinal', descending=True).over(['id1', 'src']).cast(pl.Float32).alias('c_s1rank')).drop('_n', '_no')
        Q = Q.with_columns(pl.Series('lq', lg(Q['q'].to_numpy())).cast(pl.Float32)).drop('q')
        Q.write_parquet(wp('feat', 'f83', f'{split}.parquet')); print(split, Q.height, f'({time.time()-t0:.0f}s)', flush=True)

def load2(split, filt):
    F = S81.load(split, filt)
    return F.join(pl.read_parquet(wp('feat', 'f83', f'{split}.parquet')).join(F.select(KY), on=KY, how='semi'), on=KY, how='left')

if mode == 'cv':
    gt = load_gt_pairs(); v0 = pl.read_parquet(wp('data', 'train_s1ids.parquet')).filter(fold_expr('id1') == 0)
    BASE = pl.read_parquet(wp('scores', 'm4it', 'train_s2_c0.parquet'), columns=[*KY, 'p'])
    Tr = load2('train', pl.col('fold') != 0); FE = S81.feats_of(Tr)
    X = S81.mat(Tr, FE); y = Tr['y'].to_numpy(); del Tr
    m = lgb.train(PAR, lgb.Dataset(X, y, feature_name=FE, free_raw_data=True), R); del X
    m.save_model(wp('models', 'robust2_cv.txt'))
    Va = load2('train', pl.col('fold') == 0)
    q = m.predict(S81.mat(Va, FE), num_threads=4); yv = Va['y'].to_numpy(); p = Va['p'].to_numpy()
    ll = lambda z: float(-(yv * np.log(np.clip(z, 1e-6, 1)) + (1 - yv) * np.log(np.clip(1 - z, 1e-6, 1))).mean())
    print(f'fold 0 logloss: p {ll(p):.5f}  q2 {ll(q):.5f} ({time.time()-t0:.0f}s)', flush=True)
    print('top gain:', sorted(zip(FE, m.feature_importance('gain').round()), key=lambda x: -x[1])[:15], flush=True)
    for w in (0.0, 0.3, 0.5, 0.7, 1.0):
        z = 1 / (1 + np.exp(-(w * lg(q) + (1 - w) * lg(p))))
        V = BASE.join(Va.select(KY), on=KY, how='anti').vstack(Va.select(*KY, pl.Series('p', z).cast(pl.Float32)))
        print(f'weight on q2 {w}: ' + ' '.join(f'{t}:{macro_f05(decide(V, t), gt, v0)["macro"]:.5f}' for t in (0.5, 0.6, 0.7, 0.8)) + f'  logloss {ll(z):.5f}', flush=True)
    Va.select(*KY, 'ctry', 'y', 'p').with_columns(pl.Series('q2', q).cast(pl.Float32)).write_parquet(wp('analysis', 'robust2_fold0.parquet'))

if mode == 'test':
    out = sys.argv[2]
    Tr = load2('train', pl.col('fold') >= 0); FE = S81.feats_of(Tr)
    X = S81.mat(Tr, FE); y = Tr['y'].to_numpy(); del Tr
    m = lgb.train(PAR, lgb.Dataset(X, y, feature_name=FE, free_raw_data=True), R); del X
    m.save_model(wp('models', f'{out}.txt'))
    Te = load2('test', pl.col('p') >= 0)
    q = m.predict(S81.mat(Te, FE), num_threads=4)
    os.makedirs(wp('scores', out), exist_ok=True)
    Te.select(*KY, 'p1', pl.Series('p', q).cast(pl.Float32)).write_parquet(wp('scores', out, 'test_s2_c0.parquet'))
    print('round-2 test scores ->', out, f'({time.time()-t0:.0f}s)')
