"""v15: same-name copies the test pipeline drops (US, India), on top of a base output. Groups (name_rel 0, record not assigned to any S1,
structural q >= 0.8): G1 nearby number with the record's number BELOW the S1 number (planted neighbours sit above: 99% US / 85% India),
G2 truncated number below, G3 no address with a unique S1 name (India only: US S2 cap violations 2/497 = distractor level),
G4 far number with a unique S1 name, G5 missing number with a unique S1 name.
Census (s73) and sign balance: v9scal accepted 12,418 lower-number same-name US pairs (expected true 12,248), v14e keeps 10,582.
Cap test on the added S2 records: 2 / 2,618.
usage: python s75_samename.py <base output dir> <out dir>"""
import sys
from common import *
base, OUT = sys.argv[1], sys.argv[2]
if os.path.exists(os.path.join(OUT, 'matching_results.tsv')): sys.exit(f'{OUT} exists; refusing to overwrite')
X = pl.read_parquet(wp('analysis', 'v15_groups.parquet'))
A = X.filter(~((pl.col('ctry') == 'US') & (pl.col('g') == 'G3 noaddr,unique'))).select(KY)
P = pl.read_parquet(os.path.join(base, 'pairs.parquet')).select(KY)
A = A.join(P.select('id2', 'src'), on=['id2', 'src'], how='anti')
M = pl.concat([P, A]); assert M.height == M.unique(['id2', 'src']).height
os.makedirs(OUT, exist_ok=True)
s1 = pl.read_parquet(wp('data', 'test_s1.parquet'), columns=['entity_id', 'id', 'country'])
print('added', A.height, 'matches by country:', M.join(s1.rename({'id': 'id1'}), on='id1').group_by('country').len().sort('country').rows(), 'total', M.height)
e2 = pl.concat_str([pl.lit('S'), pl.col('src').cast(pl.Utf8), pl.lit('-'), pl.col('id2').cast(pl.Utf8)])
agg = M.sort(['id1', 'src', 'id2']).with_columns(e2.alias('e2')).group_by('id1', maintain_order=True).agg(pl.col('e2').str.join(',').alias('matched_entity_ids'))
res = s1.join(agg, left_on='id', right_on='id1', how='left', maintain_order='left').with_columns(pl.col('matched_entity_ids').fill_null(''))
res.select(pl.col('entity_id').alias('source1_entity_id'), 'matched_entity_ids').write_csv(os.path.join(OUT, 'matching_results.tsv'), separator='\t', quote_style='never')
M.write_parquet(os.path.join(OUT, 'pairs.parquet'))
print('->', OUT)
