"""Same-source house-number conflict. The generator gives each source one address version per entity, so two copies of one entity
in the same source rarely disagree on a nearby house number. conflict(A, x) = A has a confident copy y (best owner A, p >= 0.9, y != x)
in x's source whose first house number is a NEAR (s27: small, non-truncation) difference from x's.
(1) train: rate among true (A, x) vs false (A, x) pairs, same-source vs cross-source siblings;  (2) test: rate among near-cluster members.
usage: python s31_samesrc_conflict.py <val scores> <test scores>"""
import sys
from common import *
DMAX = 50
def hn(split):
    return pl.concat([pl.read_parquet(wp('norm2', f'{split}_s{s}.parquet'), columns=['id', 'dz']).select(pl.col('id').alias('id2'), pl.lit(s, pl.Int8).alias('src'),
                      pl.col('dz').str.split(' ').list.first().fill_null('').alias('h')) for s in (2, 3)]).filter(pl.col('h') != '')

def conflicts(split, S, P):
    """P: pairs (id1,id2,src) to test; returns P with conf_same, conf_cross flags"""
    H = hn(split)
    sib = S.sort('p', descending=True).unique(['id2', 'src'], keep='first').filter(pl.col('p') >= 0.9).join(H, on=['id2', 'src']) \
           .select('id1', pl.col('id2').alias('sid'), pl.col('src').alias('ssrc'), pl.col('h').alias('hs'))
    X = P.join(H, on=['id2', 'src']).join(sib.join(P.select('id1').unique(), on='id1', how='semi'), on='id1').filter(~((pl.col('sid') == pl.col('id2')) & (pl.col('ssrc') == pl.col('src'))))
    num = lambda c: pl.col(c).str.slice(0, 9).cast(pl.Int64, strict=False)
    near = ((pl.col('h') != pl.col('hs')) & ~pl.col('h').str.starts_with(pl.col('hs')) & ~pl.col('hs').str.starts_with(pl.col('h'))
            & ~pl.col('h').str.ends_with(pl.col('hs')) & ~pl.col('hs').str.ends_with(pl.col('h')) & ((num('h') - num('hs')).abs() <= DMAX))
    X = X.with_columns(near.alias('nr'), (pl.col('ssrc') == pl.col('src')).alias('same'), (pl.col('h') == pl.col('hs')).alias('eq'))
    A = X.group_by(KY).agg((pl.col('nr') & pl.col('same')).any().alias('conf_same'), (pl.col('nr') & ~pl.col('same')).any().alias('conf_cross'),
                           (pl.col('eq') & pl.col('same')).any().alias('agree_same'), pl.col('same').any().alias('has_same'))
    return P.join(A, on=KY, how='left').with_columns(pl.col('conf_same', 'conf_cross', 'agree_same', 'has_same').fill_null(False))

VAL, TEST = sys.argv[1], sys.argv[2]
gt = load_gt_pairs()
S = pl.read_parquet(VAL, columns=[*KY, 'p'])
v0 = pl.read_parquet(wp('data', 'train_s1ids.parquet')).filter(fold_expr('id1') == 0)
P = S.join(v0, on='id1').filter(pl.col('p') >= 0.05).select(*KY, 'p', 'ctry')
X = conflicts('train', S, P).join(gt.with_columns(pl.lit(1).alias('y')), on=KY, how='left').with_columns(pl.col('y').fill_null(0))
print('TRAIN fold-0 candidate pairs with p >= 0.05, x has a house number:')
for c in ('US', 'India'):
    d = X.filter((pl.col('ctry') == c))
    for y in (1, 0):
        e = d.filter(pl.col('y') == y)
        print(f'  {c:6s} y={y}: n={e.height:8d} | has same-source sibling {e["has_same"].mean():.3f} | same-source NEAR conflict {e["conf_same"].mean():.4f} | cross-source NEAR conflict {e["conf_cross"].mean():.4f}')
    e = d.filter(pl.col('conf_same'))
    print(f'  {c:6s} among pairs WITH a same-source conflict: true rate {e["y"].mean():.3f}  mean p {e["p"].mean():.3f}  n={e.height}')
del S, X
M = pl.read_parquet(wp('analysis', 'nearcluster_members_test.parquet')).select(*KY, 'p')
ST = pl.read_parquet(TEST, columns=[*KY, 'p'])
Y = conflicts('test', ST, M).join(pl.read_parquet(wp('data', 'test_s1.parquet'), columns=['id', 'country']).rename({'id': 'id1', 'country': 'ctry'}), on='id1')
Mt = pl.read_parquet(wp('analysis', 'nearcluster_members_train.parquet')).select(*KY, 'p')
Yt = conflicts('train', pl.read_parquet(VAL, columns=[*KY, 'p']), Mt).join(gt.with_columns(pl.lit(1).alias('y')), on=KY, how='left').with_columns(pl.col('y').fill_null(0)) \
       .join(pl.read_parquet(wp('data', 'train_s1ids.parquet')), on='id1')
print('NEAR-CLUSTER MEMBERS:')
for c in ('US', 'India'):
    for y in (1, 0):
        e = Yt.filter((pl.col('ctry') == c) & (pl.col('y') == y))
        print(f'  train {c:6s} y={y}: n={e.height:6d} | same-source conflict {e["conf_same"].mean():.3f} | cross-source conflict {e["conf_cross"].mean():.3f} | mean p {e["p"].mean():.3f}')
for c in ('US', 'India', 'France'):
    e = Y.filter(pl.col('ctry') == c)
    print(f'  test  {c:6s}: n={e.height:6d} | same-source conflict {e["conf_same"].mean():.3f} | cross-source conflict {e["conf_cross"].mean():.3f} | mean p {e["p"].mean():.3f}'
          f' | p>=0.7 & same-source conflict {(e.filter(pl.col("conf_same"))["p"] >= 0.7).sum()}')
Y.write_parquet(wp('analysis', 'nearcluster_conflicts_test.parquet'))
