"""Stage-1 scoring (existing m1_A / m1_B models) of the train-scale-IDF test features feat/v6cF -> scores/m1F (same format as s09 score)."""
import glob, time
import lightgbm as lgb
from common import *
NB = 19; t0 = time.time()
mA = lgb.Booster(model_file=wp('models', 'm1_A.txt')); mB = lgb.Booster(model_file=wp('models', 'm1_B.txt')); C = mA.feature_name()
os.makedirs(wp('scores', 'm1F'), exist_ok=True)
ctxf = wp('feat', 'v6c', 'test_pairctx.parquet')            # pair context comes from the ranker (unchanged by the IDF fix)
for f in sorted(glob.glob(wp('feat', 'v6cF', 'test_s[23]_c*.parquet'))):
    out = wp('scores', 'm1F', os.path.basename(f))
    if os.path.exists(out): continue
    d = pl.read_parquet(f)
    ctx = pl.scan_parquet(ctxf).join(d.select(*KY).lazy(), on=KY, how='semi').collect()
    d = d.join(ctx, on=KY, how='left').with_columns(*[((pl.col('kmask') // (1 << i)) % 2).cast(pl.Float32).alias(f'k{i}') for i in range(NB)],
            pl.col('wsum').log().alias('lwsum'), pl.col('wmax').log().alias('lwmax'), (pl.col('s1_rsum_src') - pl.col('r')).alias('s1_rsum_other'))
    X = d.select(C).to_numpy().astype(np.float32)
    pA = mA.predict(X, num_threads=6); pB = mB.predict(X, num_threads=6)
    d.select(*KY, 'r', 'rr', 'a2_empty', 'is_india', 'is_france').with_columns(pl.Series('p', (pA + pB) / 2).cast(pl.Float32), pl.Series('pA', pA).cast(pl.Float32),
                                                                 pl.Series('pB', pB).cast(pl.Float32)).write_parquet(out)
print('M1F_DONE', f'{time.time()-t0:.0f}s')
