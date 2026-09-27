"""Build a submission from a score tag (threshold decisions, one owner per record) minus an optional drop list filtered to given
countries. Never overwrites: refuses an existing output folder.
usage: python s55_build.py <score tag> <thr> <outdir> [drop_pairs.tsv countries e.g. France]"""
import sys
from common import *
tag, thr, OUT = sys.argv[1], float(sys.argv[2]), sys.argv[3]
if os.path.exists(os.path.join(OUT, 'matching_results.tsv')): sys.exit(f'{OUT} exists; refusing to overwrite')
os.makedirs(OUT, exist_ok=True)
S = pl.read_parquet(wp('scores', tag, 'test_s2_c0.parquet'), columns=[*KY, 'p']); M = decide(S, thr)
if len(sys.argv) > 5:
    L = pl.read_csv(sys.argv[4], separator='\t').filter(pl.col('country').is_in(sys.argv[5].split(','))).select(
        pl.col('source1_entity_id').str.slice(3).cast(pl.Int64).alias('id1'), pl.col('entity_id').str.slice(1, 1).cast(pl.Int8).alias('src'), pl.col('entity_id').str.slice(3).cast(pl.Int64).alias('id2'))
    n0 = M.height; M = M.join(L, on=KY, how='anti'); print(f'drop list ({sys.argv[5]}): removed {n0 - M.height}')
s1 = pl.read_parquet(wp('data', 'test_s1.parquet'), columns=['entity_id', 'id', 'country'])
print('matches by country:', M.join(s1.rename({'id': 'id1'}), on='id1').group_by('country').len().sort('country').rows(), 'total', M.height)
agg = M.sort(['id1', 'src', 'id2']).with_columns(pl.concat_str([pl.lit('S'), pl.col('src').cast(pl.Utf8), pl.lit('-'), pl.col('id2').cast(pl.Utf8)]).alias('e2')) \
       .group_by('id1', maintain_order=True).agg(pl.col('e2').str.join(',').alias('matched_entity_ids'))
res = s1.join(agg, left_on='id', right_on='id1', how='left', maintain_order='left').with_columns(pl.col('matched_entity_ids').fill_null(''))
res.select(pl.col('entity_id').alias('source1_entity_id'), 'matched_entity_ids').write_csv(os.path.join(OUT, 'matching_results.tsv'), separator='\t', quote_style='never')
M.write_parquet(os.path.join(OUT, 'pairs.parquet'))
print('written', OUT)
