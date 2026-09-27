"""France decoration fill (no France labels). On test, the single token a France record adds to its S1 name is one of a few decoration
words; accepted + rejected-but-structurally-true (s68 q >= 0.9) pairs per S1 entity sit at a common level for six of them
(groupe 0.0307, france 0.0281, developpement 0.0288, associes 0.0292, services 0.0290, cie 0.0282) while the accepted level alone ranges
0.005-0.031: the score chain rejects the France copies decorated with 'cie', 'services', 'fils', ... (as it does with US 'service').
Cap test on the pool (S2 records whose S1 already holds 5 accepted S2 copies): 4 / 12,637 vs ~0.3% for distractors.
mode budget: fill each word up to TARGET per S1 (x FRAC), highest q first ('fils' would reach 0.055 with its full pool)
mode full  : add the whole pool
Only records not assigned to any S1; a record flagged for several S1 goes to the highest q.
usage: python s72_frdeco.py <base output dir> <struct tag> <final score tag> <budget|full> <TARGET> <FRAC> <out dir>"""
import sys
from common import *
base, stag, ftag, mode, TARGET, FRAC, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]), float(sys.argv[6]), sys.argv[7]
if os.path.exists(os.path.join(OUT, 'matching_results.tsv')): sys.exit(f'{OUT} exists; refusing to overwrite')
WF = ['groupe', 'france', 'developpement', 'associes', 'services', 'fils', 'cie']
n1 = pl.read_parquet(wp('norm2', 'test_s1.parquet'), columns=['id', 'ctry', 'nc']).filter(pl.col('ctry') == 'France').rename({'id': 'id1', 'nc': 'nc1'})
n2 = pl.concat([pl.read_parquet(wp('norm2', f'test_s{s}.parquet'), columns=['id', 'nc']).with_columns(pl.lit(s, pl.Int8).alias('src')) for s in (2, 3)]).rename({'id': 'id2', 'nc': 'nc2'})
def added(P):
    X = P.join(n1, on='id1').join(n2, on=['id2', 'src'])
    X = X.with_columns(pl.col('nc2').str.split(' ').list.set_difference(pl.col('nc1').str.split(' ')).alias('a')).filter(pl.col('a').list.len() == 1)
    return X.with_columns(pl.col('a').list.first().alias('w')).filter(pl.col('w').is_in(WF)).drop('a', 'nc1', 'nc2', 'ctry')
P = pl.read_parquet(os.path.join(base, 'pairs.parquet')).select(KY)
cur = added(P).group_by('w').len().rename({'len': 'cur'})
S = pl.read_parquet(wp('scores', ftag, 'test_s2_c0.parquet'), columns=[*KY, 'p'])
T = pl.read_parquet(wp('scores', stag, 'test_struct.parquet')).select(*KY, 'q')
pool = added(S.join(P, on=KY, how='anti')).join(T, on=KY).join(P.select('id2', 'src'), on=['id2', 'src'], how='anti') \
        .filter(pl.col('q') >= 0.9).sort('q', descending=True).unique(['id2', 'src'], keep='first')
B = pl.DataFrame({'w': WF}).join(cur, on='w', how='left').with_columns(pl.col('cur').fill_null(0)).with_columns(
    (FRAC * (TARGET * n1.height - pl.col('cur'))).floor().clip(0, None).cast(pl.Int64).alias('budget') if mode == 'budget' else pl.lit(10 ** 9).alias('budget'))
pool = pool.join(B.select('w', 'budget'), on='w').sort('q', descending=True).with_columns(pl.int_range(pl.len()).over('w').alias('r'))
A = pool.filter(pl.col('r') < pl.col('budget'))
print(B.join(pool.group_by('w').len().rename({'len': 'pool'}), on='w', how='left').join(A.group_by('w').len().rename({'len': 'added'}), on='w', how='left')
       .with_columns((pl.col('cur') / n1.height).alias('cur_per_S1'), ((pl.col('cur') + pl.col('added').fill_null(0)) / n1.height).alias('new_per_S1')))
M = pl.concat([P, A.select(KY)])
assert M.height == M.unique(['id2', 'src']).height
os.makedirs(OUT, exist_ok=True)
s1 = pl.read_parquet(wp('data', 'test_s1.parquet'), columns=['entity_id', 'id', 'country'])
print('matches by country:', M.join(s1.rename({'id': 'id1'}), on='id1').group_by('country').len().sort('country').rows(), 'total', M.height, 'added', A.height)
e2 = pl.concat_str([pl.lit('S'), pl.col('src').cast(pl.Utf8), pl.lit('-'), pl.col('id2').cast(pl.Utf8)])
agg = M.sort(['id1', 'src', 'id2']).with_columns(e2.alias('e2')).group_by('id1', maintain_order=True).agg(pl.col('e2').str.join(',').alias('matched_entity_ids'))
res = s1.join(agg, left_on='id', right_on='id1', how='left', maintain_order='left').with_columns(pl.col('matched_entity_ids').fill_null(''))
res.select(pl.col('entity_id').alias('source1_entity_id'), 'matched_entity_ids').write_csv(os.path.join(OUT, 'matching_results.tsv'), separator='\t', quote_style='never')
M.write_parquet(os.path.join(OUT, 'pairs.parquet'))
V9 = pl.read_csv('/home/user/w/v9_cal_rebuilt.tsv', separator='\t', infer_schema=False).with_columns(pl.col('matched_entity_ids').str.split(',')).explode('matched_entity_ids') \
       .filter(pl.col('matched_entity_ids').is_not_null() & (pl.col('matched_entity_ids') != '')).select('source1_entity_id', pl.col('matched_entity_ids').alias('entity_id'))
N = M.join(s1.select(pl.col('id').alias('id1'), 'entity_id'), on='id1').select(pl.col('entity_id').alias('source1_entity_id'), e2.alias('entity_id'))
D = pl.concat([V9.join(N, on=['source1_entity_id', 'entity_id'], how='anti').select(pl.lit('remove').alias('action'), pl.all()),
               N.join(V9, on=['source1_entity_id', 'entity_id'], how='anti').select(pl.lit('add').alias('action'), pl.all())])
D.write_csv(os.path.join(OUT, 'diff_vs_v9_cal.tsv'), separator='\t')
print('diff vs v9_cal:', D.group_by('action').len().sort('action').rows(), '->', OUT)
