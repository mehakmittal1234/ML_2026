"""Borderline specialist: a larger LightGBM trained only on decision-relevant pairs (LO < p1 < HI) with ALL features
(stage-1 pair features + ranker/context + stage-2 competition/siblings + raw-name + record share + tells + name support).
Validation: specialist trained on folds 1-7 band pairs, scores fold-0 band pairs; all other pairs keep m2t scores (baseline 0.98857).
usage: python s19_specialist.py eval <out tag> [LO HI]      |   python s19_specialist.py final <out tag> [LO HI]  (train all folds, score test)"""
import sys, time, glob
import lightgbm as lgb
from common import *
LO, HI = 0.01, 0.99
PAR = dict(objective='binary', learning_rate=0.03, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
           lambda_l2=3.0, num_threads=6, verbose=-1, max_bin=255)
ROUNDS = 1500
t0 = time.time()
SRC = os.environ.get('S19_SRC', 'm1')                 # stage-2 feature dir (m1 = round 1; iteration dirs recompute context from better scores)
BASE = os.environ.get('S19_BASE', 'm2t')              # validation baseline scores for pairs outside the band
def band(split):
    B = pl.read_parquet(wp('feat', f'{SRC}_s2', f'{split}.parquet'))
    for e in ('x', 'sup'): B = B.join(pl.read_parquet(wp('feat', f'{SRC}_s2', f'{split}_{e}.parquet')), on=KY, how='left')
    B = B.join(pl.read_parquet(wp('feat', os.environ.get('S19_TELLS', 'm1') + '_s2', f'{split}_tells.parquet')), on=KY, how='left')
    for e in [x for x in os.environ.get('S19_EXTRA', '').split(',') if x]:
        B = B.join(pl.read_parquet(wp('feat', e, f'{split}.parquet')), on=KY, how='left')
    if SRC != 'm1' and os.environ.get('S19_NOP1S') != '1': B = B.join(pl.read_parquet(wp('feat', 'm1_s2', f'{split}.parquet'), columns=[*KY, 'p1']).rename({'p1': 'p1_stage1'}), on=KY, how='left')
    B = B.filter((pl.col('p1') > LO) & (pl.col('p1') < HI))
    F1 = pl.concat([pl.scan_parquet(wp('feat', os.environ.get('S19_FTAG', 'v6c'), f'{split}_s{s}_c*.parquet')).join(B.lazy().filter(pl.col('src') == s).select(KY), on=KY, how='semi')
                    for s in (2, 3)], how='vertical_relaxed').collect()
    F1 = F1.join(pl.scan_parquet(wp('feat', 'v6c', f'{split}_pairctx.parquet')).join(B.lazy().select(KY), on=KY, how='semi').collect(), on=KY, how='left')
    dup = [c for c in F1.columns if c in B.columns and c not in KY]
    X = B.join(F1.drop(dup), on=KY, how='left')
    X = X.with_columns(*[((pl.col('kmask') // (1 << i)) % 2).cast(pl.Float32).alias(f'k{i}') for i in range(19)]).drop('kmask')
    return X
def labelled(split='train'):
    return band(split).join(load_gt_pairs().with_columns(pl.lit(1, pl.Int8).alias('y')), on=KY, how='left').with_columns(pl.col('y').fill_null(0), fold_expr('id1').alias('fold'))

if __name__ == '__main__':
    mode, out = sys.argv[1], sys.argv[2]
    if len(sys.argv) > 4: LO, HI = float(sys.argv[3]), float(sys.argv[4])
    X = band('train').join(load_gt_pairs().with_columns(pl.lit(1, pl.Int8).alias('y')), on=KY, how='left').with_columns(pl.col('y').fill_null(0), fold_expr('id1').alias('fold'))
    C = [c for c in X.columns if c not in ('id1', 'id2', 'y', 'fold') and X[c].dtype != pl.Utf8]
    print(f'band pairs {X.height}, features {len(C)}, true rate {X["y"].mean():.3f} ({time.time()-t0:.0f}s)', flush=True)
    if mode == 'eval':
        tr = X.filter(pl.col('fold') != 0)
        WNT = float(os.environ.get('S19_WNT', '1'))     # up-weight near-twin pairs (similar name, same address) toward their test share
        wt = np.where(((tr['n_tset'] >= 60) & (tr['n_exact'] == 0) & (tr['a_tset'] >= 90)).fill_null(False).to_numpy(), WNT, 1.0)
        m = lgb.train(PAR, lgb.Dataset(tr.select(C).to_numpy().astype(np.float32), tr['y'].to_numpy(), weight=wt, feature_name=C), ROUNDS); del tr
        print(f'trained ({time.time()-t0:.0f}s); top gain:', sorted(zip(C, m.feature_importance('gain').round()), key=lambda x: -x[1])[:20], flush=True)
        V = X.filter(pl.col('fold') == 0)
        pv = m.predict(V.select(C).to_numpy().astype(np.float32), num_threads=6)
        from sklearn.metrics import log_loss
        base = pl.read_parquet(wp('scores', BASE, 'train_s2_c0.parquet'))
        bv = V.select(KY).join(base.select(*KY, 'p'), on=KY, how='left')['p'].to_numpy()
        print(f'fold-0 band logloss: {BASE} {log_loss(V["y"].to_numpy(), np.clip(bv, 1e-6, 1-1e-6)):.4f} -> specialist {log_loss(V["y"].to_numpy(), np.clip(pv, 1e-6, 1-1e-6)):.4f}', flush=True)
        new = pl.concat([base.join(V.select(KY), on=KY, how='anti'), V.select(*KY, 'p1', pl.Series('p', pv).cast(pl.Float32))], how='vertical_relaxed')
        os.makedirs(wp('scores', out), exist_ok=True); new.write_parquet(wp('scores', out, 'train_s2_c0.parquet'))
    else:
        WNT = float(os.environ.get('S19_WNT', '1'))
        wt = np.where(((X['n_tset'] >= 60) & (X['n_exact'] == 0) & (X['a_tset'] >= 90)).fill_null(False).to_numpy(), WNT, 1.0)
        m = lgb.train(PAR, lgb.Dataset(X.select(C).to_numpy().astype(np.float32), X['y'].to_numpy(), weight=wt, feature_name=C), ROUNDS); del X
        m.save_model(wp('models', f'{out}_specialist.txt'))
        T = band('test'); pt = m.predict(T.select(C).to_numpy().astype(np.float32), num_threads=6)
        base = pl.read_parquet(wp('scores', sys.argv[5] if len(sys.argv) > 5 else 'v7t', 'test_s2_c0.parquet'))
        new = pl.concat([base.join(T.select(KY), on=KY, how='anti'), T.select(*KY, 'p1', pl.Series('p', pt).cast(pl.Float32))], how='vertical_relaxed')
        os.makedirs(wp('scores', out), exist_ok=True); new.write_parquet(wp('scores', out, 'test_s2_c0.parquet'))
    print('DONE', f'{time.time()-t0:.0f}s')
