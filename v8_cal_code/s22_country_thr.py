"""Per-country threshold variants of a scored test set (leaderboard probes / final decisions). Candidates never cross countries,
so changing one country's threshold changes only that country's rows.
usage: python s22_country_thr.py <score tag> <out dir> <US thr> <India thr> <France thr>"""
import sys
from common import *
mtag, OUT = sys.argv[1], sys.argv[2]; T = {'US': float(sys.argv[3]), 'India': float(sys.argv[4]), 'France': float(sys.argv[5])}
S = pl.read_parquet(wp('scores', mtag, 'test_s2_c0.parquet'), columns=[*KY, 'p'])
s1 = pl.read_parquet(wp('data', 'test_s1.parquet'), columns=['entity_id', 'id', 'country'])
S = S.join(s1.select(pl.col('id').alias('id1'), 'country'), on='id1')
best = S.sort('p', descending=True).unique(['id2', 'src'], keep='first')
M = best.filter(pl.col('p') >= pl.col('country').replace_strict(T, return_dtype=pl.Float64)).select(KY)
agg = M.sort(['id1', 'src', 'id2']).with_columns(pl.concat_str([pl.lit('S'), pl.col('src').cast(pl.Utf8), pl.lit('-'), pl.col('id2').cast(pl.Utf8)]).alias('e2')) \
       .group_by('id1', maintain_order=True).agg(pl.col('e2').str.join(',').alias('matched_entity_ids'))
res = s1.join(agg, left_on='id', right_on='id1', how='left', maintain_order='left').with_columns(pl.col('matched_entity_ids').fill_null(''))
os.makedirs(OUT, exist_ok=True)
res.select(pl.col('entity_id').alias('source1_entity_id'), 'matched_entity_ids').write_csv(os.path.join(OUT, 'matching_results.tsv'), separator='\t', quote_style='never')
print(OUT, 'matched pairs', M.height, 'by country', M.join(S.select('id1', 'country').unique(), on='id1').group_by('country').len().sort('country').rows())
