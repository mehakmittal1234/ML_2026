"""Generator-leak probe (train labels): do IDs, file row order or ground-truth list order link a copy to its S1 owner or mark distractors?"""
import numpy as np, polars as pl
W = '/home/user/w/data'
S = {s: pl.read_parquet(f'{W}/train_s{s}.parquet', columns=['row', 'id', 'country']) for s in (1, 2, 3)}
P = pl.read_parquet(f'{W}/train_pairs.parquet')
for s in (1, 2, 3):
    S[s] = S[s].with_columns((pl.col('id').rank('ordinal') / pl.len()).alias('qi'), (pl.col('row') / pl.len()).alias('qr'))
    print(f's{s}: n={S[s].height} id range {S[s]["id"].min()}..{S[s]["id"].max()}  corr(row,id)={np.corrcoef(S[s]["row"], S[s]["id"])[0,1]:+.4f}')
ov = lambda a, b: len(set(S[a]['id'].to_list()) & set(S[b]['id'].to_list()))
print('id overlaps s1&s2', ov(1, 2), 's1&s3', ov(1, 3), 's2&s3', ov(2, 3))
X = P.join(S[1].select(pl.col('id').alias('id1'), pl.col('qi').alias('qi1'), pl.col('qr').alias('qr1')), on='id1')
X = pl.concat([X.filter(pl.col('src') == s).join(S[s].select(pl.col('id').alias('id2'), pl.col('qi').alias('qi2'), pl.col('qr').alias('qr2')), on='id2') for s in (2, 3)])
print('\n[owner link] true pairs', X.height)
for a, b in (('qi1', 'qi2'), ('qr1', 'qr2')):
    d = (X[a] - X[b]).abs().to_numpy()
    print(f'  {a}~{b}: spearman {np.corrcoef(X[a], X[b])[0,1]:+.5f}  mean|d| {d.mean():.4f} (indep 0.3333)  P(|d|<0.001) {(d<0.001).mean():.5f} (indep 0.0020)')
print('  GT row order vs S1 row order spearman', np.corrcoef(X['grow'].cast(pl.Float64).rank(), X['qr1'].rank())[0, 1])
# siblings: same-owner records adjacent by id / by row?
print('\n[sibling adjacency] fraction of owned records whose next neighbour has the same owner')
for s in (2, 3):
    R = S[s].join(P.filter(pl.col('src') == s).select(pl.col('id2').alias('id'), 'id1'), on='id', how='left')
    for k in ('id', 'row'):
        r = R.sort(k).with_columns(pl.col('id1').shift(-1).alias('nx'))
        f = r.filter(pl.col('id1').is_not_null()).select((pl.col('id1') == pl.col('nx')).fill_null(False).mean()).item()
        c = P.filter(pl.col('src') == s).group_by('id1').len()['len'].to_numpy().astype(float)
        print(f'  s{s} by {k}: {f:.6f}  (chance ~ {(c*(c-1)).sum()/c.sum()/R.height:.2e})')
# cross-source siblings: S2 copy vs S3 copy of the same owner
Y = X.filter(pl.col('src') == 2).select('id1', pl.col('qi2').alias('a'), pl.col('qr2').alias('ar')).join(
    X.filter(pl.col('src') == 3).select('id1', pl.col('qi2').alias('b'), pl.col('qr2').alias('br')), on='id1')
print('  S2 copy vs S3 copy of same owner: spearman id', f"{np.corrcoef(Y['a'], Y['b'])[0,1]:+.5f}", 'row', f"{np.corrcoef(Y['ar'], Y['br'])[0,1]:+.5f}")
# digit-level mutual information between id1 and id2 of true pairs
def digits(v, n=9):
    return np.stack([(v // 10**k) % 10 for k in range(n)], 1)
d1, d2 = digits(X['id1'].to_numpy()), digits(X['id2'].to_numpy())
def mi(a, b):
    j = np.bincount(a * 10 + b, minlength=100).reshape(10, 10).astype(float); j /= j.sum()
    pa, pb = j.sum(1, keepdims=True), j.sum(0, keepdims=True); m = j > 0
    return float((j[m] * np.log2(j[m] / (pa @ pb)[m])).sum())
M = sorted(((mi(d1[:, i], d2[:, j]), i, j) for i in range(9) for j in range(9)), reverse=True)[:5]
print('\n[digit MI] top (bits, digit_i(id1), digit_j(id2)); independence noise ~', f'{81/(2*X.height*np.log(2)):.1e}', M)
for m in (2, 3, 7, 10, 97, 1000):
    for nm, v in (('diff', X['id2'].to_numpy() - X['id1'].to_numpy()), ('sum', X['id2'].to_numpy() + X['id1'].to_numpy())):
        c = np.bincount(np.mod(v, m), minlength=m); e = c.sum() / m
        print(f'  ({nm}) mod {m:4d}: chi2/dof {((c-e)**2/e).sum()/(m-1):.2f}', end='')
    print()
# distractors (unowned records) by position / id / country
print('\n[distractor rate] by decile of row and of id')
for s in (2, 3):
    R = S[s].join(P.filter(pl.col('src') == s).select(pl.col('id2').alias('id'), pl.lit(1).alias('own')), on='id', how='left').with_columns(pl.col('own').fill_null(0))
    for k in ('qr', 'qi'):
        t = R.group_by((pl.col(k) * 10).floor().alias('dec')).agg((1 - pl.col('own').mean()).round(4).alias('dist')).sort('dec')
        print(f'  s{s} {k}:', t['dist'].to_list())
    print(f'  s{s} by country:', R.group_by('country').agg((1 - pl.col('own').mean()).round(4)).rows())
S1o = S[1].join(P.group_by('id1').len().rename({'id1': 'id'}), on='id', how='left').with_columns(pl.col('len').fill_null(0))
print('  S1 singleton rate by row decile', S1o.group_by((pl.col('qr') * 10).floor().alias('d')).agg((pl.col('len') == 0).mean().round(4)).sort('d')['len'].to_list())
print('  GT list: position of S2 vs S3 ids (is list order meaningful?)', P.group_by('src').agg(pl.col('gpos').mean()).rows())
