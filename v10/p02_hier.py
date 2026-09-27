"""Generator hierarchy check (train labels): copies of one hidden entity in the same source share a source-level version.
dev(x) = tokens of copy x (name+address, lowercased alnum) that are absent from its S1 owner. For two copies x, y we test
whether dev(x) & dev(y) is non-empty: same entity same source / same entity other source / different entity (same S1 name)."""
import numpy as np, polars as pl
W = '/home/user/w/data'
tok = lambda c: pl.col(c).fill_null('').str.to_lowercase().str.replace_all(r'[^a-z0-9\s]', ' ').str.split(' ').list.eval(pl.element().filter(pl.element().str.len_chars() >= 2)).list.unique()
P = pl.read_parquet(f'{W}/train_pairs.parquet')
S1 = pl.read_parquet(f'{W}/train_s1.parquet').select(pl.col('id').alias('id1'), 'country', pl.concat_list(tok('name'), tok('addr')).alias('t1'),
                                                     pl.col('name').str.to_lowercase().str.replace_all(r'[^a-z0-9]', '').alias('k'))
R = pl.concat([pl.read_parquet(f'{W}/train_s{s}.parquet').select(pl.col('id').alias('id2'), pl.lit(s, pl.Int8).alias('src'), 'country',
               tok('name').alias('tn'), tok('addr').alias('ta'), pl.col('addr').is_null().alias('noaddr')) for s in (2, 3)])
X = P.join(R, on=['id2', 'src']).join(S1.select('id1', 't1', 'k'), on='id1')
X = X.with_columns(pl.col('tn').list.set_difference('t1').alias('dn'), pl.col('ta').list.set_difference('t1').alias('da'))
sub = X.filter(pl.col('id1').hash(3) % 20 == 0)   # 5% of owners
def pairs(A, B, same_owner=True):
    j = A.join(B, on='id1', suffix='_y') if same_owner else A
    return j.filter(~((pl.col('id2') == pl.col('id2_y')) & (pl.col('src') == pl.col('src_y'))))
Y = sub.select('id1', 'id2', 'src', 'dn', 'da', 'noaddr')
J = Y.join(Y, on='id1', suffix='_y').filter(~((pl.col('id2') == pl.col('id2_y')) & (pl.col('src') == pl.col('src_y'))))
J = J.with_columns((pl.col('src') == pl.col('src_y')).alias('same_src'),
                   (pl.col('da').list.set_intersection('da_y').list.len() > 0).alias('share_addr_dev'),
                   (pl.col('dn').list.set_intersection('dn_y').list.len() > 0).alias('share_name_dev'),
                   (pl.col('da').list.len() > 0).alias('x_has_addr_dev'))
print('copy pairs of the SAME owner (5% sample)')
print(J.filter(~pl.col('noaddr') & ~pl.col('noaddr_y')).group_by('same_src').agg(pl.len(), pl.col('x_has_addr_dev').mean().round(3),
      pl.col('share_addr_dev').mean().round(4), pl.col('share_name_dev').mean().round(4)).sort('same_src'))
# different owners with the same S1 name (twins): copy of A vs copy of B, dev computed against own S1
T = sub.select('id1', 'k').unique().join(S1.select(pl.col('id1').alias('b'), 'k', 'country'), on='k').filter(pl.col('b') != pl.col('id1'))
Yb = X.select(pl.col('id1').alias('b'), pl.col('id2').alias('id2_y'), pl.col('src').alias('src_y'), pl.col('da').alias('da_y'), pl.col('dn').alias('dn_y'), pl.col('noaddr').alias('noaddr_y'))
K = Y.join(T.select('id1', 'b'), on='id1').join(Yb, on='b')
K = K.with_columns((pl.col('src') == pl.col('src_y')).alias('same_src'), (pl.col('da').list.set_intersection('da_y').list.len() > 0).alias('share_addr_dev'),
                   (pl.col('dn').list.set_intersection('dn_y').list.len() > 0).alias('share_name_dev'))
print('copy pairs of DIFFERENT owners whose S1 names are identical (twins)')
print(K.filter(~pl.col('noaddr') & ~pl.col('noaddr_y')).group_by('same_src').agg(pl.len(), pl.col('share_addr_dev').mean().round(4), pl.col('share_name_dev').mean().round(4)).sort('same_src'))
print('name-dev sharing when x has NO address:')
print('  same owner', J.filter(pl.col('noaddr')).group_by('same_src').agg(pl.len(), pl.col('share_name_dev').mean().round(4)).sort('same_src').rows())
print('  twin owner', K.filter(pl.col('noaddr')).group_by('same_src').agg(pl.len(), pl.col('share_name_dev').mean().round(4)).sort('same_src').rows())
