"""Copy-count invariance test for near-number clusters (s29). For an S1 entity A with a cluster (>= 2 records agreeing on a nearby
house number on A's street): n_other = A's best-owner candidates with p >= 0.9 outside the cluster. If the cluster is A's own
(source-level jitter), n_other + cluster ~ generator distribution (mean 3.46); if it is an orphan neighbour, n_other alone is.
Train (labelled, by cluster truth) vs test.  usage: python s30_cluster_counts.py <val scores> <test scores>"""
import sys
from common import *
from s27_nearhn import pairs
VAL, TEST = sys.argv[1], sys.argv[2]
gt = load_gt_pairs()
for sp, F in (('train', VAL), ('test', TEST)):
    S = pl.read_parquet(F, columns=[*KY, 'p'])
    X = pairs(sp, S); N = X.filter(pl.col('near') & pl.col('street'))
    cl = N.group_by('id1', 'h2').agg(pl.len().alias('csize')).filter(pl.col('csize') >= 2)
    mem = N.join(cl, on=['id1', 'h2'], how='semi').select(*KY, 'h2', 'p')
    best = S.sort('p', descending=True).unique(['id2', 'src'], keep='first').filter(pl.col('p') >= 0.9)
    other = best.join(mem.select(KY), on=KY, how='anti').join(cl.select('id1').unique(), on='id1', how='semi').group_by('id1').agg(pl.len().alias('n_other'))
    C = cl.join(other, on='id1', how='left').with_columns(pl.col('n_other').fill_null(0)).join(
        pl.read_parquet(wp('data', f'{sp}_s1.parquet'), columns=['id', 'country']).rename({'id': 'id1', 'country': 'ctry'}), on='id1')
    if sp == 'train':
        C = C.join(mem.join(gt.with_columns(pl.lit(1).alias('y')), on=KY, how='left').with_columns(pl.col('y').fill_null(0))
                   .group_by('id1', 'h2').agg(pl.col('y').mean().round(0).alias('cl_true')), on=['id1', 'h2'])
        for c in ('US', 'India'):
            for t in (1, 0):
                d = C.filter((pl.col('ctry') == c) & (pl.col('cl_true') == t))
                print(f'train {c:6s} cluster true={t}: n={d.height:6d}  mean n_other {d["n_other"].mean():.2f}  mean n_other+csize {(d["n_other"] + d["csize"]).mean():.2f}'
                      f'  P(n_other>=3) {(d["n_other"] >= 3).mean():.3f}')
    else:
        for c in ('US', 'India', 'France'):
            d = C.filter(pl.col('ctry') == c)
            print(f'test  {c:6s}: n={d.height:6d}  mean n_other {d["n_other"].mean():.2f}  mean n_other+csize {(d["n_other"] + d["csize"]).mean():.2f}  P(n_other>=3) {(d["n_other"] >= 3).mean():.3f}')
    C.write_parquet(wp('analysis', f'nearclusters_{sp}.parquet')); mem.write_parquet(wp('analysis', f'nearcluster_members_{sp}.parquet'))
    del S, X, N
