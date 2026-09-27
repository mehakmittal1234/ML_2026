"""Per-country mix of two outputs (candidates never cross countries): take the pairs of S1 entities of the given countries from output A
and all other S1 entities' pairs from output B. v16c = US + India from the IDF-fixed blend (v16_bl05), France from v14e_frfull.
usage: python s85_mix.py <output A> <countries from A, e.g. US,India> <output B> <out dir>"""
import sys
from common import *
A, CA, B, OUT = sys.argv[1], sys.argv[2].split(','), sys.argv[3], sys.argv[4]
if os.path.exists(os.path.join(OUT, 'matching_results.tsv')): sys.exit(f'{OUT} exists; refusing to overwrite')
s1 = pl.read_parquet(wp('data', 'test_s1.parquet'), columns=['entity_id', 'id', 'country'])
ca = s1.filter(pl.col('country').is_in(CA)).select(pl.col('id').alias('id1'))
M = pl.concat([pl.read_parquet(os.path.join(A, 'pairs.parquet')).select(KY).join(ca, on='id1', how='semi'),
               pl.read_parquet(os.path.join(B, 'pairs.parquet')).select(KY).join(ca, on='id1', how='anti')])
assert M.height == M.unique(['id2', 'src']).height
os.makedirs(OUT, exist_ok=True)
print('matches by country:', M.join(s1.rename({'id': 'id1'}), on='id1').group_by('country').len().sort('country').rows(), 'total', M.height)
e2 = pl.concat_str([pl.lit('S'), pl.col('src').cast(pl.Utf8), pl.lit('-'), pl.col('id2').cast(pl.Utf8)])
agg = M.sort(['id1', 'src', 'id2']).with_columns(e2.alias('e2')).group_by('id1', maintain_order=True).agg(pl.col('e2').str.join(',').alias('matched_entity_ids'))
res = s1.join(agg, left_on='id', right_on='id1', how='left', maintain_order='left').with_columns(pl.col('matched_entity_ids').fill_null(''))
res.select(pl.col('entity_id').alias('source1_entity_id'), 'matched_entity_ids').write_csv(os.path.join(OUT, 'matching_results.tsv'), separator='\t', quote_style='never')
M.write_parquet(os.path.join(OUT, 'pairs.parquet')); print('->', OUT)
