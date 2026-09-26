"""Conservative neighbour-business drop list (s35 cells with estimated true share <= SHARE_MAX, US + France) and a v9_cal-based
submission without those pairs. The list holds EVERY test pair of those cells (raw p >= 0.3, record's best S1), so it can be
removed from any decoder's output (v9_cal threshold, v10_expf, ...).
usage: python s37_droplist.py <raw test tag> <calibrated test tag> <thr> <outdir>
  -> <outdir>/drop_pairs.tsv (source1_entity_id, entity_id, country, cell, est_true_share) and <outdir>/matching_results.tsv"""
import sys
from common import *
from s34_census import typed
from s35_near_cells import cells
SHARE_MAX = 0.45
raw, cal, thr, OUT = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
os.makedirs(OUT, exist_ok=True)
CK = ['ctry', 'anchor', 'rep_src', 'rep_oth']
sh = pl.read_parquet(wp('analysis', 'near_cells.parquet')).filter(pl.col('ctry').is_in(['US', 'France']) & (pl.col('est_share') <= SHARE_MAX)).select(*CK, 'est_share')
print(sh)
tc = pl.read_parquet(wp('data', 'test_s1.parquet'), columns=['id', 'entity_id', 'country']).rename({'id': 'id1', 'country': 'ctry'})
SR = pl.read_parquet(wp('scores', raw, 'test_s2_c0.parquet'), columns=[*KY, 'p'])
L = cells('test', SR, typed('test', SR)).join(tc.select('id1', 'ctry'), on='id1').join(sh, on=CK).select(*KY, *CK, 'est_share')
del SR
L = L.join(tc.select('id1', 'entity_id'), on='id1').with_columns(pl.concat_str([pl.lit('S'), pl.col('src').cast(pl.Utf8), pl.lit('-'), pl.col('id2').cast(pl.Utf8)]).alias('e2'),
    pl.concat_str([pl.when(pl.col('anchor')).then(pl.lit('anchor')).otherwise(pl.lit('-')), pl.when(pl.col('rep_src')).then(pl.lit('rep_src')).otherwise(pl.lit('-')),
                   pl.when(pl.col('rep_oth')).then(pl.lit('rep_oth')).otherwise(pl.lit('-'))], separator='+').alias('cell'))
L.select(pl.col('entity_id').alias('source1_entity_id'), pl.col('e2').alias('entity_id'), pl.col('ctry').alias('country'), 'cell', pl.col('est_share').round(3)) \
 .write_csv(os.path.join(OUT, 'drop_pairs.tsv'), separator='\t')
S = pl.read_parquet(wp('scores', cal, 'test_s2_c0.parquet'), columns=[*KY, 'p'])
M0 = decide(S, thr); M = M0.join(L.select(KY), on=KY, how='anti')
print(f'drop list: {L.height} pairs; v9_cal matches {M0.height} -> {M.height} ({M0.height - M.height} removed); by country:',
      M0.join(L.select(KY), on=KY, how='semi').join(tc, on='id1').group_by('ctry').len().rows())
agg = M.sort(['id1', 'src', 'id2']).with_columns(pl.concat_str([pl.lit('S'), pl.col('src').cast(pl.Utf8), pl.lit('-'), pl.col('id2').cast(pl.Utf8)]).alias('e2')) \
       .group_by('id1', maintain_order=True).agg(pl.col('e2').str.join(',').alias('matched_entity_ids'))
s1 = pl.read_parquet(wp('data', 'test_s1.parquet'), columns=['entity_id', 'id'])
res = s1.join(agg, left_on='id', right_on='id1', how='left', maintain_order='left').with_columns(pl.col('matched_entity_ids').fill_null(''))
res.select(pl.col('entity_id').alias('source1_entity_id'), 'matched_entity_ids').write_csv(os.path.join(OUT, 'matching_results.tsv'), separator='\t', quote_style='never')
print(f'matching_results.tsv: {res.height} rows, {M.height} pairs')
