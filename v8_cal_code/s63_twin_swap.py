"""Same-name S1 twins competing for one record: the record's best owner A has a different house number, while a same-name S1 twin B that is
also a candidate has EXACTLY the record's house number. Rule: move the record to B. Train (labelled, India/US twins): precision of the
swap. Test: how often it fires per country (France has 0.36 same-name same-street S1 pairs per entity).
RESULT: rejected. Train swap candidates: model right (A) 462, swap right (B) 34, neither 165; test fires on 260 France pairs.
usage: python s63_twin_swap.py <val tag> <test tag>"""
import sys, glob
from common import *
def swaps(split, tag):
    S = pl.read_parquet(wp('scores', tag, f'{split}_s2_c0.parquet'), columns=[*KY, 'p'])
    A = pl.read_parquet(wp('norm2', f'{split}_s1.parquet'), columns=['id', 'ctry', 'nc', 'dz']).select(pl.col('id').alias('id1'), 'ctry', 'nc', pl.col('dz').str.split(' ').list.first().fill_null('').alias('h1'))
    R = pl.concat([pl.read_parquet(wp('norm2', f'{split}_s{s}.parquet'), columns=['id', 'dz']).select(pl.col('id').alias('id2'), pl.lit(s, pl.Int8).alias('src'),
                   pl.col('dz').str.split(' ').list.first().fill_null('').alias('hx')) for s in (2, 3)])
    X = S.join(A, on='id1').join(R, on=['id2', 'src']).filter(pl.col('hx') != '')
    X = X.with_columns(pl.col('p').rank('ordinal', descending=True).over(['id2', 'src']).alias('rk'))
    best = X.filter(pl.col('rk') == 1).select('id2', 'src', pl.col('id1').alias('a'), pl.col('p').alias('pa'), pl.col('nc').alias('nca'), pl.col('h1').alias('ha'), 'hx', 'ctry')
    alt = X.filter(pl.col('rk') > 1).select('id2', 'src', pl.col('id1').alias('b'), pl.col('p').alias('pb'), pl.col('nc').alias('ncb'), pl.col('h1').alias('hb'))
    J = best.join(alt, on=['id2', 'src']).filter((pl.col('nca') == pl.col('ncb')) & (pl.col('hb') == pl.col('hx')) & (pl.col('ha') != pl.col('hx')))
    return J.sort('pb', descending=True).unique(['id2', 'src'], keep='first')
gt = load_gt_pairs()
J = swaps('train', sys.argv[1]).join(gt.select('id2', 'src', pl.col('id1').alias('owner')), on=['id2', 'src'], how='left')
J = J.with_columns(pl.when(pl.col('owner') == pl.col('b')).then(pl.lit('B (swap right)')).when(pl.col('owner') == pl.col('a')).then(pl.lit('A (swap wrong)')).otherwise(pl.lit('neither')).alias('truth'))
print('TRAIN swap candidates (all folds):'); print(J.group_by('ctry', 'truth').agg(pl.len(), pl.col('pa').mean().alias('mean_pa'), pl.col('pb').mean().alias('mean_pb')).sort('ctry', 'truth'))
print('   by best-owner p (pa>=0.7 = currently accepted for A):', J.filter(pl.col('pa') >= 0.7).group_by('truth').len().sort('truth').rows())
T = swaps('test', sys.argv[2])
nt = dict(pl.read_parquet(wp('data', 'test_s1.parquet'), columns=['country']).group_by('country').len().rows())
print('TEST swap candidates:', T.group_by('ctry').agg(pl.len(), (pl.col('pa') >= 0.7).sum().alias('A_accepted'), pl.col('pb').mean().alias('mean_pb')).sort('ctry').rows())
T.write_parquet(wp('analysis', 'twin_swaps_test.parquet'))
