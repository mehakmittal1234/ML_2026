"""Near-house-number CLUSTERS: for S1 entity A, candidate records with A's core name on A's street whose house number is a nearby
different number (s27 'near') and on which >= 2 records agree. Train (labelled) vs test, per S1 entity; plus how often, on train,
S1's own house number disagrees with the consensus of its true copies (S1-side corruption).
usage: python s29_nearcluster.py <val scores> <test scores>"""
import sys
from common import *
from s27_nearhn import pairs

def clusters(split, S):
    X = pairs(split, S)                                   # all candidate pairs with equal core name (+ near/street flags)
    N = X.filter(pl.col('near') & pl.col('street'))
    c = N.group_by('id1', 'h2').agg(pl.len().alias('csize'), pl.col('src').n_unique().alias('nsrc'), pl.col('p').max().alias('pmax'))
    same = X.filter(pl.col('street') & (pl.col('h1') == pl.col('h2'))).group_by('id1').agg(pl.len().alias('n_at_own_hn'))
    return N, c.join(same, on='id1', how='left').with_columns(pl.col('n_at_own_hn').fill_null(0))

VAL, TEST = sys.argv[1], sys.argv[2]
gt = load_gt_pairs()
n1 = {sp: dict(pl.read_parquet(wp('data', f'{sp}_s1.parquet'), columns=['country']).group_by('country').len().rows()) for sp in ('train', 'test')}
ctr = {sp: pl.read_parquet(wp('data', f'{sp}_s1.parquet'), columns=['id', 'country']).rename({'id': 'id1', 'country': 'ctry'}) for sp in ('train', 'test')}
for sp, F in (('train', VAL), ('test', TEST)):
    N, C = clusters(sp, pl.read_parquet(F, columns=[*KY, 'p']))
    C = C.join(ctr[sp], on='id1')
    if sp == 'train':
        lab = N.join(gt.with_columns(pl.lit(1).alias('y')), on=KY, how='left').with_columns(pl.col('y').fill_null(0)).group_by('id1', 'h2').agg(pl.col('y').mean().alias('true_rate'))
        C = C.join(lab, on=['id1', 'h2'], how='left')
    for c in sorted(n1[sp]):
        d = C.filter(pl.col('ctry') == c); k = d.filter(pl.col('csize') >= 2)
        line = (f'{sp:5s} {c:6s}: near-hn clusters (>=2 records) per S1 {k.height / n1[sp][c]:.5f} | of which cross-source {(k["nsrc"] == 2).mean():.3f}'
                f' | A has >=1 candidate at its own hn {(k["n_at_own_hn"] > 0).mean():.3f} | max p in cluster: mean {k["pmax"].mean():.3f}, >=0.7 {(k["pmax"] >= 0.7).mean():.3f}')
        if sp == 'train': line += f' | TRUE rate of cluster records {k["true_rate"].mean():.3f}'
        print(line, flush=True)
    del N, C
# S1-side corruption on train: entities whose >=2 address-bearing true copies agree on a house number different from S1's
A = pl.read_parquet(wp('norm2', 'train_s1.parquet'), columns=['id', 'ctry', 'dz']).select(pl.col('id').alias('id1'), 'ctry', pl.col('dz').str.split(' ').list.first().fill_null('').alias('h1'))
R = pl.concat([pl.read_parquet(wp('norm2', f'train_s{s}.parquet'), columns=['id', 'dz']).select(pl.col('id').alias('id2'), pl.lit(s, pl.Int8).alias('src'),
               pl.col('dz').str.split(' ').list.first().fill_null('').alias('h2')) for s in (2, 3)])
T = gt.join(R, on=['id2', 'src']).join(A, on='id1').filter((pl.col('h1') != '') & (pl.col('h2') != ''))
g = T.group_by('id1', 'ctry', 'h1', 'h2').agg(pl.len().alias('n'), pl.col('src').n_unique().alias('ns'))
agree_own = T.filter(pl.col('h1') == pl.col('h2')).group_by('id1').len().rename({'len': 'n_own'})
k = g.filter((pl.col('h2') != pl.col('h1')) & (pl.col('n') >= 2) & (pl.col('ns') == 2)).join(agree_own, on='id1', how='left').with_columns(pl.col('n_own').fill_null(0))
for c in ('US', 'India'):
    kk = k.filter(pl.col('ctry') == c)
    print(f'train {c:6s}: true copies from BOTH sources agreeing on a house number != S1: per S1 {kk.height / n1["train"][c]:.5f}; of those, S1 has no copy at its own number {(kk["n_own"] == 0).mean():.3f}')
