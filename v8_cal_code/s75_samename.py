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
import glob
S = pl.read_parquet(wp('scores', 'ens1', 'test_s2_c0.parquet'), columns=[*KY, 'p'])
P0 = pl.read_parquet(os.path.join(base, 'pairs.parquet')).select(KY)
nc = P0.group_by('id1', 'src').len().rename({'len': 'nc'})
T = pl.read_parquet(wp('scores', 'st1', 'test_struct.parquet')).select(*KY, 'ctry', 'q')
F = pl.scan_parquet(sorted(glob.glob(wp('feat', 'f50', 'test_c*.parquet')))).filter(pl.col('name_rel') == 0).select(*KY, 'hn_rel', 'k_s1', 'x_noaddr').collect() \
      .join(pl.read_parquet(wp('feat', 'f67', 'test.parquet'), columns=[*KY, 'sd']), on=KY, how='left')
X = T.join(S, on=KY).join(F, on=KY).join(P0, on=KY, how='anti').join(P0.select('id2', 'src'), on=['id2', 'src'], how='anti') \
     .filter((pl.col('q') >= 0.8) & (pl.col('ctry') != 'France'))
X = X.with_columns(pl.when((pl.col('hn_rel') == 4) & (pl.col('sd') < 0)).then(pl.lit('G1 near,lower'))
                   .when((pl.col('hn_rel') == 3) & (pl.col('sd') < 0)).then(pl.lit('G2 trunc,lower'))
                   .when((pl.col('hn_rel') == 0) & (pl.col('k_s1') == 1)).then(pl.lit('G3 noaddr,unique'))
                   .when((pl.col('hn_rel') == 5) & (pl.col('k_s1') == 1)).then(pl.lit('G4 far,unique'))
                   .when((pl.col('hn_rel') == 1) & (pl.col('k_s1') == 1)).then(pl.lit('G5 hn missing,unique')).alias('g')).filter(pl.col('g').is_not_null())
X = X.sort('q', descending=True).unique(['id2', 'src'], keep='first')
print('groups:', X.group_by('ctry', 'g').len().sort('ctry', 'g').rows())
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
