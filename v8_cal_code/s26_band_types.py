"""Where does the extra test uncertainty sit? Borderline best-owner pairs (LO < p < HI; each S2/S3 record's best S1 only) per S1
entity, split by label-free pair type, validation fold 0 vs every test country. Types (from the raw records):
  noaddr  : record has no address
  nexact  : normalised name (lowercase alnum) equals the S1 name
  twin    : the S1 name is shared by another S1 entity of the same country
  hn      : first digit group of the record's address equals the S1's (both present)
usage: python s26_band_types.py <val score parquet> <test score parquet>"""
import sys
from common import *
LO, HI = 0.05, 0.95
nk = pl.col('name').str.to_lowercase().str.replace_all(r'[^a-z0-9]', '')
hn = pl.col('addr').str.extract(r'(\d+)').str.strip_chars_start('0')

def typed(split, S, ents):
    s1 = pl.read_parquet(wp('data', f'{split}_s1.parquet'), columns=['id', 'name', 'addr', 'country'])
    s1 = s1.select(pl.col('id').alias('id1'), pl.col('country').alias('ctry'), nk.alias('k1'), hn.alias('h1'))
    s1 = s1.join(s1.group_by('ctry', 'k1').len().rename({'len': 'ntw'}), on=['ctry', 'k1'])
    B = S.sort('p', descending=True).unique(['id2', 'src'], keep='first').join(ents.select('id1'), on='id1', how='semi')
    B = B.filter((pl.col('p') > LO) & (pl.col('p') < HI))
    R = pl.concat([pl.read_parquet(wp('data', f'{split}_s{s}.parquet'), columns=['id', 'name', 'addr']).select(
        pl.col('id').alias('id2'), pl.lit(s, pl.Int8).alias('src'), nk.alias('k2'), hn.alias('h2'), pl.col('addr').is_null().alias('noaddr')) for s in (2, 3)])
    B = B.join(R, on=['id2', 'src']).join(s1, on='id1')
    return B.with_columns((pl.col('k1') == pl.col('k2')).alias('nexact'), (pl.col('ntw') > 1).alias('twin'),
                          ((pl.col('h1') == pl.col('h2')) & pl.col('h1').is_not_null()).alias('hn'))

def report(tag, B, n):
    tot = B.height / n
    g = B.group_by('noaddr', 'nexact', 'twin', 'hn').agg((pl.len() / n).alias('per_ent'), pl.col('p').mean().alias('mp')).sort('per_ent', descending=True)
    print(f'{tag}: borderline best-owner pairs per S1 entity {tot:.4f}  (sum p(1-p) per entity {(B["p"] * (1 - B["p"])).sum() / n:.4f})')
    return g.with_columns(pl.lit(tag).alias('set'))

VAL, TEST = sys.argv[1], sys.argv[2]
ids = pl.read_parquet(wp('data', 'train_s1ids.parquet')).filter(fold_expr('id1') == 0)
tc = pl.read_parquet(wp('data', 'test_s1.parquet'), columns=['id', 'country']).select(pl.col('id').alias('id1'), pl.col('country').alias('ctry'))
out = []
BV = typed('train', pl.read_parquet(VAL, columns=[*KY, 'p']), ids)
for c in ('US', 'India'):
    out.append(report(f'val_{c}', BV.filter(pl.col('ctry') == c), ids.filter(pl.col('ctry') == c).height))
del BV
BT = typed('test', pl.read_parquet(TEST, columns=[*KY, 'p']), tc)
for c in ('US', 'India', 'France'):
    out.append(report(f'test_{c}', BT.filter(pl.col('ctry') == c), tc.filter(pl.col('ctry') == c).height))
T = pl.concat(out).pivot(on='set', index=['noaddr', 'nexact', 'twin', 'hn'], values='per_ent').fill_null(0)
T = T.with_columns((pl.col('test_US') - pl.col('val_US')).alias('dUS'), (pl.col('test_India') - pl.col('val_India')).alias('dIndia')).sort('dUS', descending=True)
pl.Config.set_tbl_rows(20); pl.Config.set_tbl_width_chars(200); pl.Config.set_float_precision(4)
print(T)
