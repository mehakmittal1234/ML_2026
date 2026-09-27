"""Decoration fill (count invariance x structural ordering). The generator's name noise appends/substitutes one of a few decoration words
('service', 'services', 'center', 'partners') at a fixed rate per S1 entity (train truth, US: 0.0365 / 0.0766 / 0.1038 / 0.0286).
Expected accepted rate on test = truth rate x validation recall of that decoration type. On test v13 kept 0.0043 'service' pairs per
US entity (expected 0.0350): the score chain rejects them on test only. s70 (v14) restored part; this fills the remaining deficit per
(country, word): budget = FRAC x (expected - current) x #S1, filled with the highest structural-q candidates (q >= QMIN) whose record is
not assigned to any S1. France excluded (no labels for its decoration vocabulary).
QMIN is per country, e.g. US:0.8,India:0.97 (cap test: India decorated pairs with q < 0.97 violate the per-source copy cap at the distractor rate).
usage: python s71_deco.py <base output dir> <struct tag> <final score tag> <QMIN spec> <FRAC> <out dir>"""
import sys
from common import *
base, stag, ftag, FRAC, OUT = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[5]), sys.argv[6]
QMIN = {k: float(v) for k, v in (x.split(':') for x in sys.argv[4].split(','))}
if os.path.exists(os.path.join(OUT, 'matching_results.tsv')): sys.exit(f'{OUT} exists; refusing to overwrite')
W = ['service', 'services', 'center', 'partners']
def deco(split, P):
    n1 = pl.read_parquet(wp('norm2', f'{split}_s1.parquet'), columns=['id', 'ctry', 'nc']).rename({'id': 'id1', 'nc': 'nc1'})
    n2 = pl.concat([pl.read_parquet(wp('norm2', f'{split}_s{s}.parquet'), columns=['id', 'nc']).with_columns(pl.lit(s, pl.Int8).alias('src')) for s in (2, 3)]).rename({'id': 'id2', 'nc': 'nc2'})
    X = P.join(n1, on='id1').join(n2, on=['id2', 'src'])
    e = pl.lit(None, pl.Utf8)
    for w in W[::-1]:
        r = rf'\b{w}\b'; e = pl.when(pl.col('nc2').str.contains(r) & ~pl.col('nc1').str.contains(r)).then(pl.lit(w)).otherwise(e)
    return X.with_columns(e.alias('w')).filter(pl.col('w').is_not_null()).drop('nc1', 'nc2')
gt = load_gt_pairs(); v0 = pl.read_parquet(wp('data', 'train_s1ids.parquet')).filter(fold_expr('id1') == 0)
s1tr = pl.read_parquet(wp('norm2', 'train_s1.parquet'), columns=['ctry']).group_by('ctry').len().rename({'len': 'n1'})
Gt = deco('train', gt).group_by('ctry', 'w').len().join(s1tr, on='ctry').with_columns((pl.col('len') / pl.col('n1')).alias('truth')).select('ctry', 'w', 'truth')
Pv = decide(pl.read_parquet(wp('scores', 'm4it', 'train_s2_c0.parquet'), columns=[*KY, 'p']), 0.7).join(v0, on='id1', how='semi')
G0 = deco('train', gt.join(v0, on='id1', how='semi'))
rec = G0.join(Pv, on=KY, how='semi').group_by('ctry', 'w').len().rename({'len': 'tp'}).join(G0.group_by('ctry', 'w').len(), on=['ctry', 'w']).select('ctry', 'w', (pl.col('tp') / pl.col('len')).alias('recall'))
P = pl.read_parquet(os.path.join(base, 'pairs.parquet')).select(KY)
s1te = pl.read_parquet(wp('norm2', 'test_s1.parquet'), columns=['ctry']).group_by('ctry').len().rename({'len': 'n1'})
cur = deco('test', P).group_by('ctry', 'w').len().rename({'len': 'cur'})
B = Gt.join(rec, on=['ctry', 'w']).join(s1te, on='ctry').join(cur, on=['ctry', 'w'], how='left').with_columns(pl.col('cur').fill_null(0)) \
      .with_columns((pl.col('truth') * pl.col('recall') * pl.col('n1')).alias('expected')).with_columns(
      (FRAC * (pl.col('expected') - pl.col('cur'))).floor().clip(0, None).cast(pl.Int64).alias('budget'))
S = pl.read_parquet(wp('scores', ftag, 'test_s2_c0.parquet'), columns=[*KY, 'p'])
T = pl.read_parquet(wp('scores', stag, 'test_struct.parquet')).select(*KY, 'q')
pool = deco('test', S.join(P, on=KY, how='anti')).join(T, on=KY).join(P.select('id2', 'src'), on=['id2', 'src'], how='anti') \
        .filter(pl.col('q') >= pl.col('ctry').replace_strict(QMIN, default=2.0, return_dtype=pl.Float64)).sort('q', descending=True).unique(['id2', 'src'], keep='first')
pool = pool.join(B.select('ctry', 'w', 'budget'), on=['ctry', 'w']).sort('q', descending=True).with_columns(pl.int_range(pl.len()).over(['ctry', 'w']).alias('r'))
A = pool.filter(pl.col('r') < pl.col('budget'))
print(B.join(pool.group_by('ctry', 'w').len().rename({'len': 'pool'}), on=['ctry', 'w'], how='left').join(A.group_by('ctry', 'w').len().rename({'len': 'added'}), on=['ctry', 'w'], how='left').sort('ctry', 'w'))
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
