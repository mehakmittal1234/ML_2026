"""Edit-type tell features for all stage-2 pairs (see a15_tells.py for their measured separation).
usage: python s17_tells.py <split>  -> feat/m1_s2/<split>_tells.parquet"""
import sys, time
from common import *
import a15_tells as T15
split = sys.argv[1]; t0 = time.time()
S = pl.read_parquet(wp('feat', 'm1_s2', f'{split}.parquet'), columns=[*KY, 'p1', 'rec_rank'])
out = []
for k in range(4):                                   # chunks keep the token explode small
    X = T15.tells(split, S.filter(pl.col('id1') % 4 == k))
    out.append(X.select(*KY,
        *[(pl.col('legal_rel') == v).cast(pl.Float32).alias(f'lg_{v.split("/")[0]}') for v in ('added/swapped', 'dropped', 'same', 'none')],
        *[(pl.col('dig_rel') == v).cast(pl.Float32).alias(f'dg_{v}') for v in ('equal', 'truncated', 'altered', 'other', 'missing')],
        pl.col('n_noise').cast(pl.Float32).alias('tk_noise'), pl.col('n_realword').cast(pl.Float32).alias('tk_realword'), pl.col('n_other').cast(pl.Float32).alias('tk_other')))
    print(f'chunk {k} ({time.time()-t0:.0f}s)', flush=True); del X
pl.concat(out).write_parquet(wp('feat', 'm1_s2', f'{split}_tells.parquet'))
print('TELLS_DONE', f'{time.time()-t0:.0f}s')
