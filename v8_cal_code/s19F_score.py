"""Score the test borderline band with a SAVED specialist model (no training). Band/features from S19_SRC / S19_TELLS / S19_FTAG env.
usage: python s19F_score.py <specialist model file tag, e.g. v8s> <base test score tag, e.g. v7tF> <out tag>"""
import sys, time, os
import lightgbm as lgb
from common import *
import s19_specialist as SP
mt, base_tag, out = sys.argv[1], sys.argv[2], sys.argv[3]; t0 = time.time()
m = lgb.Booster(model_file=wp('models', f'{mt}_specialist.txt')); C = m.feature_name()
T = SP.band('test')
miss = [c for c in C if c not in T.columns]; assert not miss, miss
pt = m.predict(T.select(C).to_numpy().astype(np.float32), num_threads=6)
base = pl.read_parquet(wp('scores', base_tag, 'test_s2_c0.parquet'))
new = pl.concat([base.join(T.select(KY), on=KY, how='anti'), T.select(*KY, 'p1', pl.Series('p', pt).cast(pl.Float32))], how='vertical_relaxed')
os.makedirs(wp('scores', out), exist_ok=True); new.write_parquet(wp('scores', out, 'test_s2_c0.parquet'))
print('S19F_DONE band', T.height, 'total', new.height, f'{time.time()-t0:.0f}s')
