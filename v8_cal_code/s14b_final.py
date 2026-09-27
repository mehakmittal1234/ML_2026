"""Final stage-2 model with extra feature files (sup = record name support, tells = edit-type tells), trained on ALL train folds;
scores test. Validation of this feature set (7-fold protocol): 0.98857 vs 0.98787 without the extras.
usage: python s14b_final.py <out tag> <extras e.g. sup,tells> [rounds]"""
import sys, time
import lightgbm as lgb
from common import *
import s11_stage2 as S2
out, extras = sys.argv[1], [e for e in sys.argv[2].split(',') if e]; R = int(sys.argv[3]) if len(sys.argv) > 3 else 700
t0 = time.time()
def load(split):
    T = pl.read_parquet(wp('feat', 'm1_s2', f'{split}.parquet')).join(pl.read_parquet(wp('feat', 'm1_s2', f'{split}_x.parquet')), on=KY, how='left')
    for e in extras: T = T.join(pl.read_parquet(wp('feat', 'm1_s2', f'{split}_{e}.parquet')), on=KY, how='left')
    return T
T = load('train').join(load_gt_pairs().with_columns(pl.lit(1, pl.Int8).alias('y')), on=KY, how='left').with_columns(pl.col('y').fill_null(0))
C = [c for c in T.columns if c not in ('id1', 'id2', 'y')]
m = lgb.train(S2.PAR, lgb.Dataset(T.select(C).to_numpy().astype(np.float32), T['y'].to_numpy(), feature_name=C), R)
m.save_model(wp('models', f'{out}_stage2_all.txt')); del T
print(f'stage 2 trained on all train rows ({time.time()-t0:.0f}s)', flush=True)
Te = load('test')
p = m.predict(Te.select(C).to_numpy().astype(np.float32), num_threads=6)
os.makedirs(wp('scores', out), exist_ok=True)
Te.select(*KY, 'p1', pl.Series('p', p).cast(pl.Float32)).write_parquet(wp('scores', out, 'test_s2_c0.parquet'))
print(f'test pairs scored {Te.height} ({time.time()-t0:.0f}s)')
