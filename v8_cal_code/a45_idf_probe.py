"""Proof test for the IDF fix: recompute pair features with train-scale IDF (CtxFixed) for (a) US 'S1 name + service/services' same-address
pairs on test, (b) a random sample of test pairs, and re-score with the existing stage-1 models (m1_A / m1_B). Compare p before/after, and
with the same pattern on train."""
import lightgbm as lgb, glob
from common import *
import feats2, idf_fix
NB = 19
mA = lgb.Booster(model_file=wp('models', 'm1_A.txt')); mB = lgb.Booster(model_file=wp('models', 'm1_B.txt')); C = mA.feature_name()
RKC = ['nk', 'ntyp', 'kmask', 'wsum', 'wmax', 'rrank', 'c_ntset', 'c_nratio', 'c_natset', 'c_atset', 'c_apart', 'c_hneq', 'c_hn2e', 'c_a2e',
       'npool', 'wrel', 'ntset_gap', 'atset_gap', 'c_sum', 'sum_gap', 'r', 'rr', 'nkept', 'r_gap_next', 'r_gap_top', 'r_n05']
def load_rows(split, K):
    parts = [pl.scan_parquet(f).join(K.lazy().filter(pl.col('src') == s), on=KY, how='semi') for s in (2, 3) for f in sorted(glob.glob(wp('feat', 'v6c', f'{split}_s{s}_c*.parquet')))]
    return pl.concat(parts, how='diagonal_relaxed').collect()
def score(split, D):
    ctx = pl.scan_parquet(wp('feat', 'v6c', f'{split}_pairctx.parquet')).join(D.select(KY).lazy(), on=KY, how='semi').collect()
    d = D.join(ctx, on=KY, how='left').with_columns(*[((pl.col('kmask') // (1 << i)) % 2).cast(pl.Float32).alias(f'k{i}') for i in range(NB)],
            pl.col('wsum').log().alias('lwsum'), pl.col('wmax').log().alias('lwmax'), (pl.col('s1_rsum_src') - pl.col('r')).alias('s1_rsum_other'))
    X = d.select(C).to_numpy().astype(np.float32)
    return d.select(KY).with_columns(pl.Series('p', (mA.predict(X, num_threads=6) + mB.predict(X, num_threads=6)) / 2))
svc = pl.read_parquet(wp('analysis', 'deco_service_test.parquet')).select(KY)
rnd = pl.read_parquet(wp('scores', 'm1', 'test_s2_c0.parquet'), columns=KY).sample(20000, seed=3) if os.path.exists(wp('scores', 'm1', 'test_s2_c0.parquet')) else None
K = pl.concat([svc.with_columns(pl.lit('service').alias('g')), rnd.with_columns(pl.lit('random').alias('g'))])
old = load_rows('test', K.select(KY))
p_old = score('test', old)
cf = idf_fix.CtxFixed('test')
print('country factors log(Nc_train(ref)/Nc_test):', {k: round(v, 3) for k, v in cf.fac.items()}, flush=True)
new = feats2.pair_features(old.select(*KY, *[c for c in RKC if c in old.columns]), cf)
p_new = score('test', new)
R = K.join(p_old.rename({'p': 'p_old'}), on=KY).join(p_new.rename({'p': 'p_new'}), on=KY)
print(R.group_by('g').agg(pl.len(), pl.col('p_old').mean().round(4), pl.col('p_new').mean().round(4), (pl.col('p_old') >= 0.5).mean().round(3).alias('old>=.5'),
                          (pl.col('p_new') >= 0.5).mean().round(3).alias('new>=.5'), (pl.col('p_new') - pl.col('p_old')).abs().mean().round(4).alias('mean_abs_change')))
tr = pl.read_parquet(wp('analysis', 'deco_service_train.parquet'))
print('train same pattern: mean stage-1 p1', round(tr['p1'].mean(), 4))
chk = old.select(*KY, 'ni_idf_miss2').join(new.select(*KY, pl.col('ni_idf_miss2').alias('miss2_new')), on=KY).join(svc, on=KY, how='semi')
print('service pairs ni_idf_miss2 old -> new:', round(chk['ni_idf_miss2'].mean(), 4), '->', round(chk['miss2_new'].mean(), 4), '(train 5.0207)')
