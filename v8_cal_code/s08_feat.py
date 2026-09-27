"""Full pair features for retrieved candidates (rank <= K per record), one retrieval partition at a time.
usage: python s08_feat.py <split> <cand tag> <K> <NP>  -> feat/<cand tag>/<split>_s{s}_c{p}.parquet"""
import sys, glob, time
from common import *
import feats2

split, tag, KEEP, NP = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
OUT = wp('feat', tag); os.makedirs(OUT, exist_ok=True)
t0 = time.time()
ctx = feats2.Ctx(split)
S1 = feats2.side(ctx, 1)
print(f'ctx + S1 side {time.time()-t0:.0f}s ({S1.estimated_size("gb"):.2f} GB)', flush=True)
RKC = ['nk', 'ntyp', 'kmask', 'wsum', 'wmax', 'rrank', 'c_ntset', 'c_nratio', 'c_natset', 'c_atset', 'c_apart', 'c_hneq', 'c_hn2e', 'c_a2e',
       'npool', 'wrel', 'ntset_gap', 'atset_gap', 'c_sum', 'sum_gap', 'r', 'rr']
S1r = S1.rename({c: c + '_1' for c in S1.columns if c != 'id'})
for s in (2, 3):
    for p in range(NP):
        fn = f'{OUT}/{split}_s{s}_c{p}.parquet'
        if os.path.exists(fn): continue
        c = pl.read_parquet(wp('cand', tag, f'{split}_s{s}_c{p}.parquet')).filter(pl.col('rr') <= KEEP).with_columns(pl.lit(s, pl.Int8).alias('src'))
        # record context from the ranker (all kept candidates of the record are in this partition)
        c = c.with_columns(pl.len().over('id2').cast(pl.Float32).alias('nkept'),
                           (pl.col('r') - pl.col('r').filter(pl.col('rr') > 1).max().over('id2').fill_null(0)).alias('r_gap_next'),
                           (pl.col('r') - pl.col('r').max().over('id2')).alias('r_gap_top'),
                           (pl.col('r') > 0.5).sum().over('id2').cast(pl.Float32).alias('r_n05'))
        S = feats2.side(ctx, s, c.select(pl.col('id2').unique().alias('id')))
        pr = c.select(*KY).join(S1r, left_on='id1', right_on='id', how='left') \
              .join(S.rename({x: x + '_2' for x in S.columns if x != 'id'}), left_on='id2', right_on='id', how='left')
        F = feats2.compute(pr, ctx, s).join(c.select(*KY, *RKC, 'nkept', 'r_gap_next', 'r_gap_top', 'r_n05'), on=KY, how='left')
        F.write_parquet(fn)
        print(f'  s{s} c{p}: {F.height} pairs ({time.time()-t0:.0f}s)', flush=True)
        del c, S, pr, F
print('FEAT_DONE', f'{time.time()-t0:.0f}s', flush=True)
