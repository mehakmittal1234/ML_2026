"""Second retrieval pass for address-bearing records missed by the first pass (typo'd city, jittered number, number words, garbled name).
Key = (country, state/last address component, street word) where street word = each alpha token (>= 4 chars, not a street type) of the
address component that holds the first house number; ordinal words are mapped to digits (tenth -> 10th). Pairs are kept when house numbers
are within DH of each other (non-empty on both sides) and the pair is NOT already in the stage-2 pair set.
RESULT: rejected. Recovers 70 of 3,815 address-bearing missed true pairs (1.8%) while adding 5.13 candidates per entity.
usage: python s64_retr2.py <split> <score tag> [DH]   -> cand/r2/<split>.parquet (id1 id2 src + key hits); prints recall on fold 0 (train)"""
import sys, time
from common import *
split, tag = sys.argv[1], sys.argv[2]; DH = int(sys.argv[3]) if len(sys.argv) > 3 else 60
CAP1, CAPP = 60, 6000
t0 = time.time()
ORD = {'first': '1st', 'second': '2nd', 'third': '3rd', 'fourth': '4th', 'fifth': '5th', 'sixth': '6th', 'seventh': '7th', 'eighth': '8th', 'ninth': '9th', 'tenth': '10th',
       'eleventh': '11th', 'twelfth': '12th', 'thirteenth': '13th', 'fourteenth': '14th', 'fifteenth': '15th'}
STY = {'street', 'road', 'avenue', 'drive', 'lane', 'court', 'place', 'circle', 'boulevard', 'parkway', 'highway', 'terrace', 'trail', 'unit', 'suite', 'nagar', 'colony',
       'sector', 'block', 'floor', 'plot', 'flat', 'house', 'near', 'opposite', 'main', 'cross', 'north', 'south', 'east', 'west', 'null', 'none', 'city', 'rue', 'avenue', 'chemin'}
def keys(s):
    d = pl.read_parquet(wp('norm2', f'{split}_s{s}.parquet'), columns=['id', 'ctry', 'at', 'dz']).filter(pl.col('at') != '')
    comps = pl.col('at').str.split(' , ')
    d = d.with_columns(comps.list.last().alias('st'), comps.list.eval(pl.element().filter(pl.element().str.contains(r'\d'))).list.first().fill_null('').alias('sc'),
                       pl.col('dz').str.split(' ').list.first().fill_null('').alias('h'))
    d = d.filter(pl.col('h') != '').with_columns(pl.col('sc').str.split(' ').list.eval(pl.element().replace(ORD)).list.eval(
        pl.element().filter(pl.element().str.contains(r'^[a-z]{4,}$|^\d+(st|nd|rd|th)$') & ~pl.element().is_in(list(STY)))).list.unique().alias('sw'))
    return d.select('id', 'ctry', 'st', 'h', 'sw').explode('sw').drop_nulls('sw')
A = keys(1).rename({'id': 'id1', 'h': 'h1'})
S = pl.read_parquet(wp('scores', tag, f'{split}_s2_c0.parquet'), columns=KY)
out = []
for s in (2, 3):
    B = keys(s).rename({'id': 'id2', 'h': 'h2'})
    k1 = A.group_by('ctry', 'st', 'sw').len().rename({'len': 'n1'}); k2 = B.group_by('ctry', 'st', 'sw').len().rename({'len': 'n2'})
    kk = k1.join(k2, on=['ctry', 'st', 'sw']).filter((pl.col('n1') <= CAP1) & (pl.col('n1') * pl.col('n2') <= CAPP)).with_columns(
        pl.concat_str(['ctry', 'st', 'sw']).hash(seed=1).alias('kh'))
    num = lambda c: pl.col(c).str.slice(0, 9).cast(pl.Int64, strict=False)
    Js = []
    for q in range(8):
        kq = kk.filter(pl.col('kh') % 8 == q).select('ctry', 'st', 'sw')
        J = A.join(kq, on=['ctry', 'st', 'sw']).join(B.join(kq, on=['ctry', 'st', 'sw'], how='semi'), on=['ctry', 'st', 'sw'])
        Js.append(J.filter((num('h1') - num('h2')).abs() <= DH).select('id1', 'id2', pl.lit(s, pl.Int8).alias('src')).unique()); del J
    J = pl.concat(Js).unique().join(S, on=KY, how='anti')
    out.append(J); print(f's{s}: new pairs {J.height} ({time.time()-t0:.0f}s)', flush=True)
N = pl.concat(out)
os.makedirs(wp('cand', 'r2'), exist_ok=True); N.write_parquet(wp('cand', 'r2', f'{split}.parquet'))
if split == 'train':
    gt = load_gt_pairs(); v0 = pl.read_parquet(wp('data', 'train_s1ids.parquet')).filter(fold_expr('id1') == 0)
    miss = gt.join(v0, on='id1', how='semi').join(S, on=KY, how='anti')
    na = pl.concat([pl.read_parquet(wp('norm2', f'train_s{s}.parquet'), columns=['id', 'at']).select(pl.col('id').alias('id2'), pl.lit(s, pl.Int8).alias('src'), (pl.col('at') == '').alias('na')) for s in (2, 3)])
    ma = miss.join(na, on=['id2', 'src']).filter(~pl.col('na')).select(KY)
    got = ma.join(N, on=KY, how='semi').height
    n0 = N.join(v0, on='id1', how='semi')
    print(f'fold 0: address-bearing missed true pairs {ma.height}, recovered by pass 2: {got} ({got / max(1, ma.height):.3f}); '
          f'new candidate pairs for fold-0 entities {n0.height} ({n0.height / v0.height:.2f} per entity), true among them {n0.join(gt, on=KY, how="semi").height}')
print('R2_DONE')
