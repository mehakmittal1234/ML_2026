"""Retrieval pass 4 for pairs outside the stage-2 set, built on two noise types seen in the validation misses:
  KN (number + locality): the record keeps a house number (leading zeros, extra random numbers: '00203 , mumbai , mh' vs
     '203 , fl 2 , ... , mumbai') and its city while the name is garbled. Key = (country, number, city-level word): up to 4 digit groups
     (>= 2 chars, int-normalised) x up to 3 alpha words with S1 document frequency >= CITYDF; S1 key frequency <= CAPN.
  KS (name substrings): concatenated / domain-style names ('tglfoodprivate', 'indiamgmtekdanta', 'sri greatventures com'). Key =
     (country, 6-char substring) of the record name with spaces removed vs the first 6 chars of each S1 name word (>= 6 chars) and of the
     S1 name with spaces removed; S1 key frequency <= CAPS.
Records are processed in chunks. Pair score: 3 x KN hits + 2 x KS hits + IDF of shared address tokens / 5; top K S1 per record,
pairs of the stage-2 set excluded; a pair needs a KN hit or a shared address token.
RESULT (train): recovers 519 of 3,815 fold-0 address-bearing misses (India 438, US 81) inside 2.26M new fold-0 candidates
(0.02% true); even a perfect scorer would add ~+0.0001 on validation: rejected.
usage: python s77_retr4.py <split> [CAPN CAPS K]  -> cand/r4/<split>.parquet"""
import sys, time
from common import *
split = sys.argv[1]
CAPN, CAPS, K = (int(x) for x in sys.argv[2:5]) if len(sys.argv) > 4 else (30, 30, 3)
CITYDF, NCH = 300, 12
t0 = time.time()
GEN = ['private', 'limited', 'pvt', 'ltd', 'llp', 'com', 'www', 'india', 'services', 'service', 'center', 'partners', 'company', 'inc', 'llc', 'corp']
atok = lambda: pl.col('at').str.replace_all(' , ', ' ').str.split(' ')
A = pl.read_parquet(wp('norm2', f'{split}_s1.parquet'), columns=['id', 'ctry', 'nc', 'at']).rename({'id': 'id1'})
AT = A.filter(pl.col('at') != '').select('id1', 'ctry', atok().list.unique().alias('t')).explode('t').filter(pl.col('t').str.len_chars() >= 2)
df = AT.group_by('ctry', 't').len().rename({'len': 'df'}); n1 = A.group_by('ctry').len().rename({'len': 'N1'})
W = df.join(n1, on='ctry').select('ctry', 't', 'df', (pl.col('N1') / pl.col('df')).log().cast(pl.Float32).alias('w'))
CITY = W.filter((pl.col('df') >= CITYDF) & pl.col('t').str.contains(r'^[a-z]{3,}$')).select('ctry', 't', 'df')
def kn_keys(T, idc):
    """T: exploded address tokens (idc..., ctry, t)"""
    num = T.filter(pl.col('t').str.contains(r'^\d{2,}$')).with_columns(pl.col('t').str.strip_chars_start('0').alias('n')).filter(pl.col('n').str.len_chars() >= 2) \
           .select(*idc, 'ctry', 'n').unique().with_columns(pl.int_range(pl.len()).over(idc).alias('r')).filter(pl.col('r') < 4).drop('r')
    cty = T.join(CITY, on=['ctry', 't']).sort('df', descending=True).unique([*idc, 't'], keep='first', maintain_order=True) \
           .with_columns(pl.int_range(pl.len()).over(idc).alias('r')).filter(pl.col('r') < 3).select(*idc, 'ctry', pl.col('t').alias('a'))
    return num.join(cty, on=[*idc, 'ctry']).select(*idc, pl.concat_str(['ctry', 'n', 'a'], separator='|').hash(seed=7).alias('key'))
KN1 = kn_keys(AT, ['id1']); c = KN1.group_by('key').len(); KN1 = KN1.join(c.filter(pl.col('len') <= CAPN).select('key'), on='key')
w = A.select('id1', 'ctry', pl.col('nc').str.split(' ').list.eval(pl.element().filter((pl.element().str.len_chars() >= 6) & ~pl.element().is_in(GEN))).alias('w'),
             pl.col('nc').str.replace_all(' ', '').alias('cat'))
KS1 = pl.concat([w.explode('w').drop_nulls('w').select('id1', 'ctry', pl.col('w').str.slice(0, 6).alias('g')),
                 w.filter(pl.col('cat').str.len_chars() >= 6).select('id1', 'ctry', pl.col('cat').str.slice(0, 6).alias('g'))]).unique() \
        .select('id1', pl.concat_str(['ctry', 'g'], separator='|').hash(seed=8).alias('key'))
c = KS1.group_by('key').len(); KS1 = KS1.join(c.filter(pl.col('len') <= CAPS).select('key'), on='key')
AT = AT.join(W.select('ctry', 't', 'w'), on=['ctry', 't']).select('id1', 't', 'w')
print(f'S1 keys: KN {KN1.height}, KS {KS1.height}, address tokens {AT.height} ({time.time()-t0:.0f}s)', flush=True)
S = pl.read_parquet(wp('scores', 'm4it' if split == 'train' else 'v9scal', f'{split}_s2_c0.parquet'), columns=KY)
R = pl.concat([pl.read_parquet(wp('norm2', f'{split}_s{s}.parquet'), columns=['id', 'ctry', 'nc', 'at']).rename({'id': 'id2'}).with_columns(pl.lit(s, pl.Int8).alias('src')) for s in (2, 3)])
out = []
for k in range(NCH):
    r = R.filter(pl.col('id2').hash(seed=3) % NCH == k)
    RT = r.filter(pl.col('at') != '').select('id2', 'src', 'ctry', atok().list.unique().alias('t')).explode('t').filter(pl.col('t').str.len_chars() >= 2)
    jn = kn_keys(RT, ['id2', 'src']).join(KN1, on='key').group_by(KY).agg(pl.len().cast(pl.Int16).alias('kn'))
    cat = r.select('id2', 'src', 'ctry', pl.col('nc').str.replace_all(' ', '').alias('cat')).filter(pl.col('cat').str.len_chars() >= 6)
    sub = cat.with_columns(pl.int_ranges(0, pl.col('cat').str.len_chars() - 5).alias('i')).explode('i') \
             .select('id2', 'src', pl.concat_str(['ctry', pl.col('cat').str.slice(pl.col('i'), 6)], separator='|').hash(seed=8).alias('key')).unique()
    js = sub.join(KS1, on='key').group_by(KY).agg(pl.len().cast(pl.Int16).alias('ks'))
    N = jn.join(js, on=KY, how='full', coalesce=True).with_columns(pl.col('kn', 'ks').fill_null(0)).join(S, on=KY, how='anti')
    sh = N.select(KY).join(RT.select('id2', 'src', 't'), on=['id2', 'src']).join(AT, on=['id1', 't']).group_by(KY).agg(pl.col('w').sum().alias('aw'), pl.len().cast(pl.Int16).alias('an'))
    N = N.join(sh, on=KY, how='left').with_columns(pl.col('aw').fill_null(0.0), pl.col('an').fill_null(0)).filter((pl.col('kn') > 0) | (pl.col('an') > 0))
    N = N.with_columns((pl.col('kn') * 3 + pl.col('ks') * 2 + pl.col('aw') / 5).alias('r4s')).with_columns(
        pl.col('r4s').rank('ordinal', descending=True).over(['id2', 'src']).cast(pl.Int16).alias('r4rank')).filter(pl.col('r4rank') <= K)
    out.append(N); del r, RT, jn, js, sub, sh
    print(f'  chunk {k + 1}/{NCH}: {sum(x.height for x in out)} pairs ({time.time()-t0:.0f}s)', flush=True)
N = pl.concat(out)
os.makedirs(wp('cand', 'r4'), exist_ok=True); N.write_parquet(wp('cand', 'r4', f'{split}.parquet'))
print(f'new candidate pairs {N.height} ({N.height / R.height:.3f} per record) ({time.time()-t0:.0f}s)')
if split == 'train':
    gt = load_gt_pairs(); v0 = pl.read_parquet(wp('data', 'train_s1ids.parquet')).filter(fold_expr('id1') == 0)
    miss = pl.read_parquet(wp('analysis', 'missed_fold0.parquet')).select(*KY, 'ctry', 'noaddr')
    got = miss.join(N, on=KY, how='semi')
    print('fold-0 misses recovered (address-bearing):', got.filter(~pl.col('noaddr')).group_by('ctry').len().sort('ctry').rows(), 'of', miss.filter(~pl.col('noaddr')).group_by('ctry').len().sort('ctry').rows())
    print('fold-0 misses recovered (no address):', got.filter(pl.col('noaddr')).group_by('ctry').len().sort('ctry').rows(), 'of', miss.filter(pl.col('noaddr')).group_by('ctry').len().sort('ctry').rows())
    n0 = N.join(v0, on='id1', how='semi'); tp = n0.join(gt, on=KY, how='semi')
    print(f'fold-0 new pairs {n0.height}, true {tp.height} ({tp.height / max(1, n0.height):.3f}); true by (kn>0, ks>0):',
          tp.group_by((pl.col('kn') > 0).alias('kn'), (pl.col('ks') > 0).alias('ks')).len().sort('kn', 'ks').rows(),
          'all by (kn>0, ks>0):', n0.group_by((pl.col('kn') > 0).alias('kn'), (pl.col('ks') > 0).alias('ks')).len().sort('kn', 'ks').rows())
print('R4_DONE')
