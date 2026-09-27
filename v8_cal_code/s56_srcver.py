"""Source-version features (generator-invariant: they depend only on the pair and the S1 entity's own confident copies).
A neighbour's copy carries ITS source version (other city alias, PO box, NULL part...), a jittered true copy carries A's source version
and differs only in the number. For pair (A, x), y = A's confident copies (best owner A, p >= 0.9), y != x; alpha(r) = non-digit address
tokens of r (norm2 canonical):
  sv_s_min    : min |alpha(x) sym-diff alpha(y)| over same-source y (null if none)      sv_o_min: same over other-source y
  sv_s_eq     : # same-source y with alpha(y) == alpha(x)                                sv_s_city: any same-source y with x's city
  sv_s_hn     : any same-source y with x's house number                                  sv_x_extra: min |alpha(x) - alpha(y)| over all y
usage: python s56_srcver.py <split> <score tag>   -> feat/f56/<split>_c<k>.parquet"""
import sys, time
from common import *
split, tag = sys.argv[1], sys.argv[2]
OUT = wp('feat', 'f56'); os.makedirs(OUT, exist_ok=True)
t0 = time.time()
comps = pl.col('at').str.split(' , ')
R = pl.concat([pl.read_parquet(wp('norm2', f'{split}_s{s}.parquet'), columns=['id', 'at', 'dz']).select(
    pl.col('id').alias('id2'), pl.lit(s, pl.Int8).alias('src'),
    pl.col('at').str.split(' ').list.eval(pl.element().filter(pl.element().str.contains(r'^[a-z]+$'))).list.unique().alias('al'),
    comps.list.slice(0, comps.list.len() - 1).list.eval(pl.element().filter(~pl.element().str.contains(r'\d') & (pl.element() != ''))).list.last().fill_null('').alias('city'),
    pl.col('dz').str.split(' ').list.first().fill_null('').alias('h')) for s in (2, 3)])
S = pl.read_parquet(wp('scores', tag, f'{split}_s2_c0.parquet'), columns=[*KY, 'p'])
best = S.sort('p', descending=True).unique(['id2', 'src'], keep='first').filter(pl.col('p') >= 0.9).select('id1', pl.col('id2').alias('yid'), pl.col('src').alias('ysrc'))
Y = best.join(R.rename({'id2': 'yid', 'src': 'ysrc', 'al': 'aly', 'city': 'cy', 'h': 'hy'}), on=['yid', 'ysrc'])
print(f'pairs {S.height}, confident copies {Y.height} ({time.time()-t0:.0f}s)', flush=True)
for k in range(8):
    P = S.filter(pl.col('id1') % 8 == k).select(KY).join(R, on=['id2', 'src'])
    Q = P.join(Y.filter(pl.col('id1') % 8 == k), on='id1').filter(~((pl.col('yid') == pl.col('id2')) & (pl.col('ysrc') == pl.col('src'))))
    Q = Q.filter(pl.col('al').list.len() + pl.col('aly').list.len() > 0).with_columns(
        (pl.col('al').list.set_difference('aly').list.len() + pl.col('aly').list.set_difference('al').list.len()).alias('sd'),
        pl.col('al').list.set_difference('aly').list.len().alias('xe'), (pl.col('ysrc') == pl.col('src')).alias('same'))
    A = Q.group_by(KY).agg(pl.col('sd').filter(pl.col('same')).min().alias('sv_s_min'), pl.col('sd').filter(~pl.col('same')).min().alias('sv_o_min'),
                           (pl.col('same') & (pl.col('sd') == 0)).sum().alias('sv_s_eq'),
                           (pl.col('same') & (pl.col('cy') == pl.col('city')) & (pl.col('city') != '')).any().cast(pl.Int8).alias('sv_s_city'),
                           (pl.col('same') & (pl.col('hy') == pl.col('h')) & (pl.col('h') != '')).any().cast(pl.Int8).alias('sv_s_hn'),
                           pl.col('xe').min().alias('sv_x_extra'))
    X = P.select(KY).join(A, on=KY, how='left').with_columns(pl.col('sv_s_eq', 'sv_s_city', 'sv_s_hn').fill_null(0))
    X.with_columns(pl.col(pl.UInt32).cast(pl.Int32)).write_parquet(os.path.join(OUT, f'{split}_c{k}.parquet'))
    print(f'chunk {k}: {X.height} ({time.time()-t0:.0f}s)', flush=True)
print('F56_DONE')
