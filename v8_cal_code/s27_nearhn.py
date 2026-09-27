"""'Near house number' pairs: same core name (legal forms dropped), same street, first house numbers differ by a small amount and
are not a truncation (neither is a prefix/suffix of the other). Frequency per S1 entity among scored candidate pairs (train vs
test by country) and, on train, their true rate and the model's mean p.
usage: python s27_nearhn.py <train|test> <score parquet>   -> analysis/nearhn_<split>.parquet"""
import sys
from common import *
DMAX = 50


def pairs(split, S):
    """memory-lean: exact core-name pairs first, token lists only for those"""
    f = lambda s, cols: pl.read_parquet(wp('norm2', f'{split}_s{s}.parquet'), columns=cols)
    A = f(1, ['id', 'ctry', 'nc']).rename({'id': 'id1', 'nc': 'nc1'})
    R = pl.concat([f(s, ['id', 'nc']).rename({'id': 'id2', 'nc': 'nc2'}).with_columns(pl.lit(s, pl.Int8).alias('src')) for s in (2, 3)])
    X = S.join(A, on='id1').join(R, on=['id2', 'src']).filter(pl.col('nc1') == pl.col('nc2')).drop('nc1', 'nc2'); del R
    g = lambda s, key: f(s, ['id', 'at', 'dz']).join(X.select(pl.col(key).alias('id')).unique(), on='id', how='semi').select(
        'id', pl.col('dz').str.split(' ').list.first().fill_null('').alias('h'),
        pl.col('at').str.split(' ').list.eval(pl.element().filter(pl.element().str.contains(r'^[a-z]{3,}$'))).list.unique().alias('w'))
    X = X.join(g(1, 'id1').rename({'id': 'id1', 'h': 'h1', 'w': 'w1'}), on='id1')
    X = X.join(pl.concat([g(s, 'id2').rename({'id': 'id2', 'h': 'h2', 'w': 'w2'}).with_columns(pl.lit(s, pl.Int8).alias('src')) for s in (2, 3)]), on=['id2', 'src'])
    h1, h2 = pl.col('h1'), pl.col('h2')
    num = lambda c: pl.col(c).str.slice(0, 9).cast(pl.Int64, strict=False)
    X = X.with_columns(((h1 != '') & (h2 != '') & (h1 != h2) & ~h1.str.starts_with(h2) & ~h2.str.starts_with(h1) & ~h1.str.ends_with(h2) & ~h2.str.ends_with(h1)
                        & ((num('h1') - num('h2')).abs() <= DMAX)).alias('near'),
                       (pl.col('w1').list.set_intersection('w2').list.len() >= 2).alias('street'),
                       (num('h1') - num('h2')).abs().alias('hdiff'))
    return X.with_columns((pl.col('near') & pl.col('street')).alias('NEAR'))


if __name__ == '__main__':
    split, F = sys.argv[1], sys.argv[2]
    X = pairs(split, pl.read_parquet(F, columns=[*KY, 'p']))
    n1 = dict(pl.read_parquet(wp('data', f'{split}_s1.parquet'), columns=['country']).group_by('country').len().rows())
    if split == 'train':
        gt = load_gt_pairs()
        X = X.join(gt.with_columns(pl.lit(1).alias('y')), on=KY, how='left').join(gt.select('id2', 'src', pl.lit(1).alias('owned')), on=['id2', 'src'], how='left') \
             .with_columns(pl.col('y', 'owned').fill_null(0))
    for c in sorted(n1):
        d = X.filter(pl.col('NEAR') & (pl.col('ctry') == c))
        line = f'{split} {c:6s}: NEAR pairs per S1 {d.height / n1[c]:.4f} | mean p {d["p"].mean():.3f} | p>=0.7 share {(d["p"] >= 0.7).mean():.3f} | hdiff median {d["hdiff"].median()}'
        if split == 'train':
            line += f' | true rate {d["y"].mean():.3f} | record owned by someone {d["owned"].mean():.3f}'
        print(line, flush=True)
    X.filter(pl.col('NEAR')).drop('w1', 'w2').write_parquet(wp('analysis', f'nearhn_{split}.parquet'))
