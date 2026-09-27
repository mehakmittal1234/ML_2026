"""Label-free estimate of the true-copy share among NEAR pairs (s27) on test.
A neighbour distractor is its own hidden entity, so its several copies share its house number; a true copy with a jittered house
number is per-copy noise and its number is rarely repeated. support(x) = # other S2/S3 records with x's (country, core name, first
house number). On train: P(support>=1 | true NEAR) = a, P(support>=1 | false NEAR) = b. On test: observed m -> true share t = (m-b)/(a-b).
usage: python s28_near_support.py   (needs analysis/nearhn_{train,test}.parquet from s27)"""
from common import *

def support(split):
    R = pl.concat([pl.read_parquet(wp('norm2', f'{split}_s{s}.parquet'), columns=['id', 'ctry', 'nc', 'dz']).select(
        pl.col('id').alias('id2'), pl.lit(s, pl.Int8).alias('src'), 'ctry', 'nc', pl.col('dz').str.split(' ').list.first().fill_null('').alias('h')) for s in (2, 3)])
    R = R.filter(pl.col('h') != '').with_columns((pl.len().over('ctry', 'nc', 'h') - 1).alias('supp'))
    return R.select('id2', 'src', 'supp')

out = {}
for split in ('train', 'test'):
    X = pl.read_parquet(wp('analysis', f'nearhn_{split}.parquet')).join(support(split), on=['id2', 'src'], how='left').with_columns(pl.col('supp').fill_null(0))
    out[split] = X
T = out['train']
for c in ('US', 'India'):
    d = T.filter(pl.col('ctry') == c)
    a = (d.filter(pl.col('y') == 1)['supp'] >= 1).mean(); b = (d.filter(pl.col('y') == 0)['supp'] >= 1).mean()
    m_tr = (d['supp'] >= 1).mean()
    dt = out['test'].filter(pl.col('ctry') == c); m = (dt['supp'] >= 1).mean()
    t = (m - b) / (a - b)
    print(f'{c:6s} train: P(supp>=1|true) {a:.3f}  P(supp>=1|false) {b:.3f}  (train true share {d["y"].mean():.3f}, mixture check {(m_tr-b)/(a-b):.3f})'
          f'  | test: P(supp>=1) {m:.3f} -> estimated true share {t:.3f}  vs model mean p {dt["p"].mean():.3f}')
    for lo, hi in ((0.0, 0.3), (0.3, 0.7), (0.7, 1.01)):
        e = d.filter((pl.col('p') >= lo) & (pl.col('p') < hi)); f = dt.filter((pl.col('p') >= lo) & (pl.col('p') < hi))
        if e.height and f.height:
            a2 = (e.filter(pl.col('y') == 1)['supp'] >= 1).mean() if e['y'].sum() else float('nan'); b2 = (e.filter(pl.col('y') == 0)['supp'] >= 1).mean()
            m2 = (f['supp'] >= 1).mean()
            print(f'        p in [{lo},{hi}): train n={e.height} true {e["y"].mean():.3f} a={a2:.3f} b={b2:.3f} | test n={f.height} P(supp>=1) {m2:.3f} -> true share {(m2-b2)/(a2-b2) if a2==a2 and a2!=b2 else float("nan"):.3f}  mean p {f["p"].mean():.3f}')
dF = out['test'].filter(pl.col('ctry') == 'France')
print(f'France test: P(supp>=1) {(dF["supp"] >= 1).mean():.3f}  mean p {dF["p"].mean():.3f}  n={dF.height}')
