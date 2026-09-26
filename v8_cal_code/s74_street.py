"""Street-name agreement from the address component that holds the house number (the only street evidence that is not diluted by
city / region / state tokens: in France every address carries 'hauts france', 'pays loire', 'nouvelle aquitaine', so f50's
street_inter is > 0 for 99.9999% of accepted French pairs and cannot see '87 r pologne , lille' vs '87 r colbert , calais').
st = alpha tokens (>= 2 chars) of the first digit-bearing component, minus street types and stop words.
Features per pair: st_both (both sides have street tokens), st_jac (token Jaccard), st_fz (rapidfuzz token_set_ratio of the joined
tokens, robust to typos), st_disjoint (no common token and st_fz < 60).
usage: python s74_street.py <split>   -> feat/f74/<split>.parquet (every stage-2 pair)"""
import sys, glob, time
from rapidfuzz import process, fuzz
from common import *
split = sys.argv[1]; t0 = time.time()
STY = ['st', 'ave', 'rd', 'dr', 'ln', 'ct', 'cir', 'blvd', 'pl', 'pkwy', 'hwy', 'ter', 'trl', 'sq', 'way', 'pt', 'xing', 'plz', 'ctr', 'hts', 'n', 's', 'e', 'w',
       'ne', 'nw', 'se', 'sw', 'r', 'av', 'bd', 'rte', 'ch', 'imp', 'all', 'quai', 'crs', 'cite', 'sq', 'pass', 'res', 'lot', 'zi', 'za', 'zac', 'hameau', 'lieu', 'dit',
       'bis', 'ter', 'b', 'no', 'ste', 'apt', 'unit', 'bldg', 'rm', 'box', 'po', 'fl', 'floor', 'suite', 'de', 'du', 'des', 'la', 'le', 'les', 'l', 'd', 'et', 'the', 'of',
       'road', 'street', 'lane', 'marg', 'main', 'cross', 'nagar', 'extn', 'ph', 'block', 'ngr', 'col', 'sec']
def rec(s):
    d = pl.read_parquet(wp('norm2', f'{split}_s{s}.parquet'), columns=['id', 'at'])
    c = pl.col('at').str.split(' , ')
    comp = c.list.eval(pl.element().filter(pl.element().str.contains(r'\d'))).list.first().fill_null('')
    tok = comp.str.split(' ').list.eval(pl.element().filter(pl.element().str.contains(r'^[a-z]{2,}$') & ~pl.element().is_in(STY))).list.unique().list.sort()
    return d.select('id', tok.alias('st'))
A = rec(1).rename({'id': 'id1', 'st': 'st1'})
R = pl.concat([rec(s).with_columns(pl.lit(s, pl.Int8).alias('src')) for s in (2, 3)]).rename({'id': 'id2', 'st': 'st2'})
S = pl.scan_parquet(sorted(glob.glob(wp('feat', 'f50', f'{split}_c*.parquet')))).select(KY).collect()
out = []
for k in range(4):
    X = S.filter(pl.col('id1') % 4 == k).join(A, on='id1').join(R, on=['id2', 'src'])
    a, b = X['st1'].list.join(' ').to_list(), X['st2'].list.join(' ').to_list()
    fz = process.cpdist(a, b, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
    i = pl.col('st1').list.set_intersection('st2').list.len(); u = pl.col('st1').list.set_union('st2').list.len()
    X = X.with_columns(pl.Series('st_fz', fz)).select(*KY, ((pl.col('st1').list.len() > 0) & (pl.col('st2').list.len() > 0)).alias('st_both'),
                                                     pl.when(u > 0).then(i / u).cast(pl.Float32).alias('st_jac'), 'st_fz')
    X = X.with_columns(pl.when(pl.col('st_both')).then(pl.col('st_fz')).alias('st_fz'),
                       (pl.col('st_both') & (pl.col('st_jac') == 0) & (pl.col('st_fz') < 60)).alias('st_disjoint'))
    out.append(X); print(f'  part {k} ({time.time()-t0:.0f}s)', flush=True)
os.makedirs(wp('feat', 'f74'), exist_ok=True); pl.concat(out).write_parquet(wp('feat', 'f74', f'{split}.parquet'))
print('F74_DONE', split)
