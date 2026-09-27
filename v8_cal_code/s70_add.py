"""v14: add back true copies the score chain rejects on test only (US, India).
Evidence (label-free, test): pairs the score-free structural model s68 calls true (q >= QMIN) while the final score says no (p < 0.5)
  - per-source copy cap (S2 <= 5, S3 <= 6 per entity): a record attached to an S1 that already holds the cap of confident copies in its
    source cannot be a true copy. S2 violation rate of these pairs: US 1 / 10,951, India 0 / 4,910 (q >= 0.97); a distractor on a
    validation S1 violates at 0.3% (random S1 0.9%), so the distractor share is ~3% or less.
  - count invariance: accepted pairs per S1 on test are below validation (US 3.329 vs 3.372, India 3.340 vs 3.366); the additions
    bring them to 3.363 and 3.352.
  - inspection: mostly 'Service' / 'Services' / 'Partners' decorations and legal-form variants at the same address, p ~0 on test,
    while such pairs are 99.9% true in train (p 0.999).
France is excluded: its q >= 0.99 pairs include different streets and cities (the structural model is weak on unseen French addresses).
Only records not already assigned to another S1 are added; a record flagged for several S1 goes to the highest q.
usage: python s70_add.py <base output dir> <struct tag> <final score tag> <QMIN> <countries e.g. US,India> <out dir>"""
import sys
from common import *
base, stag, ftag, QMIN, CTRY, OUT = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5].split(','), sys.argv[6]
if os.path.exists(os.path.join(OUT, 'matching_results.tsv')): sys.exit(f'{OUT} exists; refusing to overwrite')
P = pl.read_parquet(os.path.join(base, 'pairs.parquet')).select(KY)
S = pl.read_parquet(wp('scores', ftag, 'test_s2_c0.parquet'), columns=[*KY, 'p'])
T = pl.read_parquet(wp('scores', stag, 'test_struct.parquet')).select(*KY, 'ctry', 'q').join(S, on=KY)
A = T.filter(pl.col('ctry').is_in(CTRY) & (pl.col('q') >= QMIN) & (pl.col('p') < 0.5)).join(P.select('id2', 'src'), on=['id2', 'src'], how='anti') \
     .sort('q', descending=True).unique(['id2', 'src'], keep='first')
print('additions by country:', A.group_by('ctry').len().sort('ctry').rows(), 'total', A.height)
M = pl.concat([P, A.select(KY)])
assert M.height == M.unique().height and M.height == M.unique(['id2', 'src']).height
os.makedirs(OUT, exist_ok=True)
s1 = pl.read_parquet(wp('data', 'test_s1.parquet'), columns=['entity_id', 'id', 'country'])
print('matches by country:', M.join(s1.rename({'id': 'id1'}), on='id1').group_by('country').len().sort('country').rows(), 'total', M.height)
e2 = pl.concat_str([pl.lit('S'), pl.col('src').cast(pl.Utf8), pl.lit('-'), pl.col('id2').cast(pl.Utf8)])
agg = M.sort(['id1', 'src', 'id2']).with_columns(e2.alias('e2')).group_by('id1', maintain_order=True).agg(pl.col('e2').str.join(',').alias('matched_entity_ids'))
res = s1.join(agg, left_on='id', right_on='id1', how='left', maintain_order='left').with_columns(pl.col('matched_entity_ids').fill_null(''))
res.select(pl.col('entity_id').alias('source1_entity_id'), 'matched_entity_ids').write_csv(os.path.join(OUT, 'matching_results.tsv'), separator='\t', quote_style='never')
M.write_parquet(os.path.join(OUT, 'pairs.parquet'))
# pair diff against the rebuilt v9_cal file (what the user holds locally)
V9 = pl.read_csv('/home/user/w/v9_cal_rebuilt.tsv', separator='\t', infer_schema=False).with_columns(pl.col('matched_entity_ids').str.split(',')).explode('matched_entity_ids') \
       .filter(pl.col('matched_entity_ids').is_not_null() & (pl.col('matched_entity_ids') != '')).select('source1_entity_id', pl.col('matched_entity_ids').alias('entity_id'))
N = M.join(s1.select(pl.col('id').alias('id1'), 'entity_id'), on='id1').select(pl.col('entity_id').alias('source1_entity_id'), e2.alias('entity_id'))
D = pl.concat([V9.join(N, on=['source1_entity_id', 'entity_id'], how='anti').select(pl.lit('remove').alias('action'), pl.all()),
               N.join(V9, on=['source1_entity_id', 'entity_id'], how='anti').select(pl.lit('add').alias('action'), pl.all())])
D.write_csv(os.path.join(OUT, 'diff_vs_v9_cal.tsv'), separator='\t')
print('diff vs v9_cal:', D.group_by('action').len().sort('action').rows(), '->', OUT)
