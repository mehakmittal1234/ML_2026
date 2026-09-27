"""Cheap pair features for every pair of a candidate pool (for the stage-0 ranker that truncates the pool).
usage: python s05_cheap.py <split> <pool tag> [part letter p|q] [norm dir]   (reads cand/<tag>/<split>_s{2,3}_<l>*.parquet, writes ..._cf.parquet)"""
import sys, glob, time
from rapidfuzz import process, fuzz
from common import *

split, tag = sys.argv[1], sys.argv[2]; PL = sys.argv[3] if len(sys.argv) > 3 else 'p'; ND = sys.argv[4] if len(sys.argv) > 4 else 'norm'
t0 = time.time()
def side(s):
    dcol = 'dz' if ND == 'norm2' else 'dg'
    d = pl.read_parquet(wp(ND, f'{split}_s{s}.parquet'), columns=['id', 'nc', 'na', 'at', dcol])
    return d.select('id', 'nc', 'na', pl.col('at').str.replace_all(' , ', ' ').alias('a'), pl.col(dcol).str.split(' ').list.first().fill_null('').alias('h'))
S1 = side(1)
for s in (2, 3):
    S = side(s)
    for f in sorted(glob.glob(wp('cand', tag, f'{split}_s{s}_{PL}*.parquet'))):
        if f.endswith('_cf.parquet'): continue
        out = f.replace('.parquet', '_cf.parquet')
        if os.path.exists(out): continue
        c = pl.read_parquet(f)
        x = c.select('id1', 'id2').join(S1, left_on='id1', right_on='id', how='left').join(S, left_on='id2', right_on='id', how='left', suffix='_2')
        n1, n2, a1, a2 = x['nc'].to_list(), x['nc_2'].to_list(), x['a'].to_list(), x['a_2'].to_list()
        cf = pl.DataFrame({
            'c_ntset': process.cpdist(n1, n2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32),
            'c_nratio': process.cpdist(n1, n2, scorer=fuzz.ratio, workers=-1, dtype=np.float32),
            'c_natset': process.cpdist(n1, x['na_2'].to_list(), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32),
            'c_atset': process.cpdist(a1, a2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32),
            'c_apart': process.cpdist(a1, a2, scorer=fuzz.partial_ratio, workers=-1, dtype=np.float32)})
        cf = cf.with_columns((x['h'] == x['h_2']).cast(pl.Int8).alias('c_hneq'), (x['h_2'] == '').cast(pl.Int8).alias('c_hn2e'),
                             (x['a_2'] == '').cast(pl.Int8).alias('c_a2e'))
        pl.concat([c, cf], how='horizontal').write_parquet(out)
        print(os.path.basename(f), c.height, f'{time.time()-t0:.0f}s', flush=True)
        del c, x, cf
    del S
print('CHEAP_DONE', f'{time.time()-t0:.0f}s')
