"""Record-record 'name support' features for stage 2 (label-free, computed on each split's own records).
A copy's noise (swapped word, suffix) is random per copy, so a variant name repeated by other records at the same address marks a
separate business (another entity, or an orphan whose owner is missing from Source-1).
For pair (A, x):  sup_addr  = # other records with x's exact (country, core name, house number)
                  sup_name  = # other records with x's exact (country, core name)
                  sup_best_A / sup_best_oth = among those same-(name, hn) records, # whose confident best S1 (p1 >= 0.5, rank 1) is A / another S1
usage: S2_DROP=<per-mille> python s16_support.py <split>   -> feat/m1_s2/<split><suffix>_sup.parquet"""
import sys, time
from common import *
split = sys.argv[1]; DROP = int(os.environ.get('S2_DROP', '0')); SUF = f'_d{DROP}' if DROP else ''
t0 = time.time()
R = pl.concat([pl.read_parquet(wp('norm2', f'{split}_s{s}.parquet'), columns=['id', 'ctry', 'nc', 'dz']).with_columns(pl.lit(s, pl.Int8).alias('src')) for s in (2, 3)])
R = R.select(pl.col('id').alias('id2'), 'src', pl.concat_str([pl.col('ctry'), pl.lit('|'), pl.col('nc')]).hash(seed=5).alias('gn'),
             pl.col('dz').str.split(' ').list.first().fill_null('').alias('hn'))
R = R.with_columns(pl.when(pl.col('hn') != '').then(pl.concat_str([pl.col('gn').cast(pl.Utf8), pl.lit('|'), pl.col('hn')]).hash(seed=6)).alias('ga')).drop('hn')
R = R.with_columns((pl.len().over('gn') - 1).cast(pl.Float32).alias('sup_name'),
                   pl.when(pl.col('ga').is_not_null()).then(pl.len().over('ga') - 1).cast(pl.Float32).alias('sup_addr'))
print(f'records {R.height} ({time.time()-t0:.0f}s)', flush=True)
SRC = os.environ.get('S16_SRC', 'm1')
S = pl.read_parquet(wp('feat', f'{SRC}_s2', f'{split}{SUF}.parquet'), columns=[*KY, 'p1', 'rec_rank'])
best = S.filter((pl.col('rec_rank') == 1) & (pl.col('p1') >= 0.5)).select('id2', 'src', pl.col('id1').alias('bid'))
G = R.filter(pl.col('ga').is_not_null()).select('id2', 'src', 'ga').join(best, on=['id2', 'src'], how='left')
tot = G.filter(pl.col('bid').is_not_null()).group_by('ga').agg(pl.len().cast(pl.Float32).alias('_conf'))
byA = G.filter(pl.col('bid').is_not_null()).group_by(['ga', 'bid']).agg(pl.len().cast(pl.Float32).alias('_confA'))
X = S.select(*KY).join(R.select('id2', 'src', 'ga', 'sup_name', 'sup_addr'), on=['id2', 'src'], how='left')
X = X.join(best.rename({'bid': 'self_bid'}), on=['id2', 'src'], how='left')
X = X.join(tot, on='ga', how='left').join(byA, left_on=['ga', 'id1'], right_on=['ga', 'bid'], how='left')
self_conf = pl.col('self_bid').is_not_null().cast(pl.Float32); self_A = (pl.col('self_bid') == pl.col('id1')).fill_null(False).cast(pl.Float32)
X = X.with_columns((pl.col('_confA').fill_null(0) - self_A).alias('sup_best_A'),
                   (pl.col('_conf').fill_null(0) - pl.col('_confA').fill_null(0) - (self_conf - self_A)).alias('sup_best_oth'))
X = X.with_columns(pl.when(pl.col('ga').is_null()).then(None).otherwise(pl.col('sup_best_A')).alias('sup_best_A'),
                   pl.when(pl.col('ga').is_null()).then(None).otherwise(pl.col('sup_best_oth')).alias('sup_best_oth'))
X.select(*KY, 'sup_name', 'sup_addr', 'sup_best_A', 'sup_best_oth').write_parquet(wp('feat', f'{SRC}_s2', f'{split}{SUF}_sup.parquet'))
print(f'SUP_DONE {X.height} ({time.time()-t0:.0f}s)', flush=True)
