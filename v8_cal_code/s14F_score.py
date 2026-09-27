"""Score test stage-2 features of a feature source (default m1F) with the SAVED all-fold stage-2 model (v7t: extras sup,tells). No training.
usage: python s14F_score.py <src s2 tag, e.g. m1F> <model tag, e.g. v7t> <out score tag>"""
import sys, time
import lightgbm as lgb
from common import *
src, mtag, out = sys.argv[1], sys.argv[2], sys.argv[3]; t0 = time.time()
m = lgb.Booster(model_file=wp('models', f'{mtag}_stage2_all.txt')); C = m.feature_name()
T = pl.read_parquet(wp('feat', f'{src}_s2', 'test.parquet')).join(pl.read_parquet(wp('feat', f'{src}_s2', 'test_x.parquet')), on=KY, how='left')
for e in ('sup', 'tells'): T = T.join(pl.read_parquet(wp('feat', f'{src}_s2', f'test_{e}.parquet')), on=KY, how='left')
miss = [c for c in C if c not in T.columns]; assert not miss, miss
p = m.predict(T.select(C).to_numpy().astype(np.float32), num_threads=6)
os.makedirs(wp('scores', out), exist_ok=True)
T.select(*KY, 'p1', pl.Series('p', p).cast(pl.Float32)).write_parquet(wp('scores', out, 'test_s2_c0.parquet'))
print('S14F_DONE', T.height, f'{time.time()-t0:.0f}s')
