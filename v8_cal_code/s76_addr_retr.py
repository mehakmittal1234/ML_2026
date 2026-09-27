"""Retrieval pass 3: address-driven candidates for pairs outside the stage-2 set. Validation fold-0 address-bearing misses (3,815) mostly
have near-identical addresses (India: 2,114 of 3,212 with address token-set >= 90) but garbled / different / crowded names (e.g.
'Shree Food Private Limited' has 62 S1 namesakes, 'Lyraonyxevo' replaces the name), so the name-driven blocking never ranks them.
Tokens: address tokens (>= 2 chars, digits included) of the normalised address; rare = S1 document frequency <= C1 and record document
frequency <= CR within the country. Pair score = sum over shared rare tokens of log(N1 / df1); a pair needs >= MINSH shared tokens.
Keep the top K S1 per record, excluding pairs already in the stage-2 set.
RESULT (train, C1 20 / CR 200 and 100 / 1000): recovers 31-77 of 3,212 India and 2-4 of 603 US fold-0 address-bearing misses: rejected.
usage: python s76_addr_retr.py <split> [C1 CR MINSH K]  -> cand/r3/<split>.parquet (id1 id2 src r3w r3n r3rank)"""
import sys, time
from common import *
split = sys.argv[1]
C1, CR, MINSH, K = (int(x) for x in sys.argv[2:6]) if len(sys.argv) > 5 else (20, 200, 2, 3)
t0 = time.time()
tok = lambda: pl.col('at').str.replace_all(' , ', ' ').str.split(' ').list.eval(pl.element().filter(pl.element().str.len_chars() >= 2)).list.unique()
A = pl.read_parquet(wp('norm2', f'{split}_s1.parquet'), columns=['id', 'ctry', 'at']).filter(pl.col('at') != '').select(pl.col('id').alias('id1'), 'ctry', tok().alias('t')).explode('t')
R = pl.concat([pl.read_parquet(wp('norm2', f'{split}_s{s}.parquet'), columns=['id', 'ctry', 'at']).filter(pl.col('at') != '')
               .select(pl.col('id').alias('id2'), pl.lit(s, pl.Int8).alias('src'), 'ctry', tok().alias('t')).explode('t') for s in (2, 3)])
d1 = A.group_by('ctry', 't').len().rename({'len': 'df1'}); dR = R.group_by('ctry', 't').len().rename({'len': 'dfR'})
n1 = A.group_by('ctry').agg(pl.col('id1').n_unique().alias('N1'))
rare = d1.join(dR, on=['ctry', 't']).filter((pl.col('df1') <= C1) & (pl.col('dfR') <= CR)).join(n1, on='ctry') \
         .with_columns((pl.col('N1') / pl.col('df1')).log().cast(pl.Float32).alias('w')).select('ctry', 't', 'w')
A = A.join(rare, on=['ctry', 't']); R = R.join(rare.select('ctry', 't'), on=['ctry', 't'])
print(f'rare tokens {rare.height}, S1 rows {A.height}, record rows {R.height} ({time.time()-t0:.0f}s)', flush=True)
S = pl.read_parquet(wp('scores', 'm4it' if split == 'train' else 'v9scal', f'{split}_s2_c0.parquet'), columns=KY)
out = []
NP = 16
for k in range(NP):
    r = R.filter(pl.col('id2').hash(seed=5) % NP == k)
    J = r.join(A, on=['ctry', 't']).group_by('id1', 'id2', 'src').agg(pl.col('w').sum().alias('r3w'), pl.len().cast(pl.Int16).alias('r3n'))
    J = J.filter(pl.col('r3n') >= MINSH).join(S, on=KY, how='anti')
    J = J.with_columns(pl.col('r3w').rank('ordinal', descending=True).over(['id2', 'src']).cast(pl.Int16).alias('r3rank')).filter(pl.col('r3rank') <= K)
    out.append(J); del J, r
    if k % 4 == 3: print(f'  part {k + 1}/{NP}: {sum(x.height for x in out)} pairs ({time.time()-t0:.0f}s)', flush=True)
N = pl.concat(out)
os.makedirs(wp('cand', 'r3'), exist_ok=True); N.write_parquet(wp('cand', 'r3', f'{split}.parquet'))
nrec = R.select('id2', 'src').unique().height
print(f'new candidate pairs {N.height} ({N.height / max(1, nrec):.2f} per address-bearing record with a rare token)')
if split == 'train':
    gt = load_gt_pairs(); v0 = pl.read_parquet(wp('data', 'train_s1ids.parquet')).filter(fold_expr('id1') == 0)
    miss = pl.read_parquet(wp('analysis', 'missed_fold0.parquet')).filter(~pl.col('noaddr')).select(*KY, 'ctry')
    got = miss.join(N, on=KY, how='semi')
    print('fold-0 address-bearing misses recovered:', got.group_by('ctry').len().sort('ctry').rows(), 'of', miss.group_by('ctry').len().sort('ctry').rows())
    n0 = N.join(v0, on='id1', how='semi'); tp = n0.join(gt, on=KY, how='semi').height
    print(f'fold-0 new pairs {n0.height}, true {tp} ({tp / max(1, n0.height):.3f}); true by rank:', n0.join(gt, on=KY, how='semi').group_by('r3rank').len().sort('r3rank').rows())
print('R3_DONE', f'{time.time()-t0:.0f}s')
