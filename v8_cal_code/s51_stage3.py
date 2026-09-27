"""Stage-3 model on the f50 features + existing scores, evaluated on fold 0 (official macro F0.5, one owner per record, full competition:
fold-0 pairs get the new score, all other pairs keep the base score).
  E1: trained inside fold 0 (2-way split by S1 entity; each half scored by the model trained on the other half)
  E2: E1's training half + ALL of folds 1-7, with a domain flag d0 (1 = fold 0 / test, whose scores average two models)
usage: python s51_stage3.py <E1|E2> [rounds] [out tag: also fit on all of fold 0 (+ folds 1-7 for E2) and score test]"""
import sys, glob, time
import lightgbm as lgb
from common import *
mode = sys.argv[1]; R = int(sys.argv[2]) if len(sys.argv) > 2 else 600; out = sys.argv[3] if len(sys.argv) > 3 else None
DROP = {'id1', 'id2', 'src', 'ctry', 'y', 'fold', 'half'}
PAR = dict(objective='binary', learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
           lambda_l2=1.0, num_threads=4, verbose=-1, max_bin=255)
t0 = time.time()
def load(split, filt=None):
    lf = pl.scan_parquet(sorted(glob.glob(wp('feat', 'f50', f'{split}_c*.parquet'))))
    if filt is not None: lf = lf.filter(filt)
    d = lf.collect()
    return d.with_columns((pl.col('ctry') == 'India').cast(pl.Int8).alias('is_india'), (pl.col('ctry') == 'France').cast(pl.Int8).alias('is_france'))
F0 = load('train', pl.col('fold') == 0).with_columns((pl.col('id1').hash(seed=11) % 2).alias('half'), pl.lit(1, pl.Int8).alias('d0'))
feats = [c for c in F0.columns if c not in DROP]
print(f'fold-0 pairs {F0.height}, features {len(feats)} ({time.time()-t0:.0f}s)', flush=True)
OTH = load('train', pl.col('fold') != 0).with_columns(pl.lit(0, pl.Int8).alias('d0')) if mode == 'E2' else None
if OTH is not None: print(f'folds 1-7 pairs {OTH.height} ({time.time()-t0:.0f}s)', flush=True)
gt = load_gt_pairs(); v0 = pl.read_parquet(wp('data', 'train_s1ids.parquet')).filter(fold_expr('id1') == 0)
BASE = pl.read_parquet(wp('scores', 'm4it', 'train_s2_c0.parquet'), columns=[*KY, 'p'])
def fit(parts):
    X = pl.concat([p.select(feats + ['y']) for p in parts])
    return lgb.train(PAR, lgb.Dataset(X.select(feats).to_numpy().astype(np.float32), X['y'].to_numpy(), feature_name=feats, free_raw_data=True), R)
q = np.zeros(F0.height)
for h in (0, 1):
    parts = [F0.filter(pl.col('half') != h)] + ([OTH] if OTH is not None else [])
    m = fit(parts)
    te = (F0['half'] == h).to_numpy()
    q[te] = m.predict(F0.filter(pl.col('half') == h).select(feats).to_numpy().astype(np.float32), num_threads=4)
    print(f'  half {h} done ({time.time()-t0:.0f}s)', flush=True)
V = BASE.join(F0.select(KY), on=KY, how='anti').vstack(F0.select(KY).with_columns(pl.Series('p', q).cast(pl.Float32)))
sc = {t: macro_f05(decide(V, t), gt, v0, by='ctry') for t in (0.5, 0.6, 0.65, 0.7, 0.75, 0.8)}
bt = max(sc, key=lambda t: sc[t]['macro'])
print(f'{mode}: ' + ' '.join(f'{t}:{v["macro"]:.5f}' for t, v in sc.items()) + f' | best {bt}: {sc[bt]["macro"]:.5f} ' + str([(c, round(v, 5)) for c, v, _ in sc[bt]['by'].rows()]) + ' (m4it 0.98941)')
print('top gain:', sorted(zip(feats, m.feature_importance('gain').round()), key=lambda x: -x[1])[:15])
os.makedirs(wp('scores', f's3_{mode}_val'), exist_ok=True)
V.write_parquet(wp('scores', f's3_{mode}_val', 'train_s2_c0.parquet'))
if out:
    m = fit([F0] + ([OTH] if OTH is not None else []))
    T = load('test').with_columns(pl.lit(1, pl.Int8).alias('d0'))
    pt = m.predict(T.select(feats).to_numpy().astype(np.float32), num_threads=4)
    os.makedirs(wp('scores', out), exist_ok=True)
    T.select(*KY, 'p1').with_columns(pl.Series('p', pt).cast(pl.Float32)).write_parquet(wp('scores', out, 'test_s2_c0.parquet'))
    print('test scored ->', out)
print(f'DONE {time.time()-t0:.0f}s')
