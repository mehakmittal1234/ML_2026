"""Feature factory for the stage-2 pair set (every pair of a score file, i.e. p1 >= 0.005), chunked by id1 % 8.
Pair-intrinsic text features, generator-structure features (source-level versions, near-number clusters, the S1 entity's own copies)
and the existing scores. Label-free except the train label column y.
usage: python s50_feats.py <split> <score tag>   -> feat/f50/<split>_c<k>.parquet"""
import sys, time
from rapidfuzz import process, fuzz
from common import *
from norm import LEGAL_CANON
split, tag = sys.argv[1], sys.argv[2]
OUT = wp('feat', 'f50'); os.makedirs(OUT, exist_ok=True)
LG = pl.Series(sorted(set(LEGAL_CANON.values())))
STY = ['st', 'ave', 'rd', 'dr', 'ln', 'ct', 'cir', 'blvd', 'pl', 'pkwy', 'hwy', 'ter', 'trl', 'sq', 'way', 'pt', 'xing', 'plz', 'ctr', 'hts', 'n', 's', 'e', 'w',
       'ne', 'nw', 'se', 'sw', 'r', 'av', 'bd', 'rte', 'ch', 'imp', 'all', 'quai', 'crs', 'ngr', 'col', 'sec', 'main', 'cross', 'road', 'street', 'lane', 'marg',
       'nagar', 'extn', 'ph', 'block', 'floor', 'fl', 'grd', 'no', 'ste', 'apt', 'unit', 'bldg', 'rm', 'box', 'po']
t0 = time.time()
cp = lambda sc, a, b: process.cpdist(a, b, scorer=sc, workers=-1, dtype=np.float32)

def rec(s):
    d = pl.read_parquet(wp('norm2', f'{split}_s{s}.parquet'), columns=['id', 'ctry', 'nt', 'nc', 'at', 'dz'])
    comps = pl.col('at').str.split(' , ')
    return d.select('id', 'ctry', 'nt', 'nc', pl.col('at').str.replace_all(' , ', ' ').alias('a'),
                    pl.col('dz').str.split(' ').list.first().fill_null('').alias('h'),
                    pl.col('dz').str.split(' ').list.eval(pl.element().filter(pl.element() != '')).list.unique().alias('dl'),
                    pl.col('nc').str.split(' ').list.eval(pl.element().filter(pl.element() != '')).list.unique().alias('tk'),
                    pl.col('nt').str.split(' ').list.eval(pl.element().filter(pl.element().is_in(LG))).list.unique().alias('lg'),
                    pl.col('at').str.split(' ').list.eval(pl.element().filter(pl.element().str.contains(r'^[a-z]{3,}$') & ~pl.element().is_in(STY))).list.unique().alias('w'),
                    comps.list.slice(0, comps.list.len() - 1).list.eval(pl.element().filter(~pl.element().str.contains(r'\d') & (pl.element() != ''))).list.last().fill_null('').alias('city'),
                    (pl.col('at') == '').alias('na'))

A = rec(1).rename({'id': 'id1'})
A = A.join(A.group_by('ctry', 'nc').len().rename({'len': 'k_s1'}), on=['ctry', 'nc'])
R = pl.concat([rec(s).rename({'id': 'id2'}).with_columns(pl.lit(s, pl.Int8).alias('src')) for s in (2, 3)])
# record-level structure (label-free): repeats of (name, house number) per source / other source; name popularity among records
R = R.with_columns(pl.when(pl.col('h') != '').then(pl.len().over('ctry', 'nc', 'h', 'src') - 1).otherwise(0).alias('g_src'),
                   pl.when(pl.col('h') != '').then(pl.len().over('ctry', 'nc', 'h') - pl.len().over('ctry', 'nc', 'h', 'src')).otherwise(0).alias('g_oth'),
                   (pl.len().over('ctry', 'nc') - 1).alias('g_name'))
print(f'records loaded ({time.time()-t0:.0f}s)', flush=True)
S = pl.read_parquet(wp('scores', tag, f'{split}_s2_c0.parquet'), columns=[*KY, 'p1', 'p'])
# competition context over the whole pair set
S = S.with_columns(pl.col('p').rank('ordinal', descending=True).over(['id2', 'src']).cast(pl.Int16).alias('x_rank'),
                   pl.len().over(['id2', 'src']).cast(pl.Int16).alias('x_ncand'),
                   pl.col('p').rank('ordinal', descending=True).over('id1').cast(pl.Int16).alias('a_rank'),
                   pl.len().over('id1').cast(pl.Int16).alias('a_ncand'))
top = S.group_by('id2', 'src').agg(pl.col('p').top_k(2).alias('_t')).with_columns(pl.col('_t').list.get(0).alias('_a'), pl.col('_t').list.get(1, null_on_oob=True).fill_null(0).alias('_b')).drop('_t')
S = S.join(top, on=['id2', 'src']).with_columns(pl.when(pl.col('x_rank') == 1).then(pl.col('p') - pl.col('_b')).otherwise(pl.col('p') - pl.col('_a')).alias('x_gap')).drop('_a', '_b')
# the S1 entity's confident copies (record's best owner, p >= 0.9) with their house numbers
conf = S.filter((pl.col('x_rank') == 1) & (pl.col('p') >= 0.9)).select('id1', pl.col('id2').alias('cid'), pl.col('src').alias('csrc'))
conf = conf.join(R.select(pl.col('id2').alias('cid'), pl.col('src').alias('csrc'), pl.col('h').alias('ch'), pl.col('w').alias('cw')), on=['cid', 'csrc'])
print(f'pairs {S.height}, confident copies {conf.height} ({time.time()-t0:.0f}s)', flush=True)
lab = load_gt_pairs().with_columns(pl.lit(1, pl.Int8).alias('y')) if split == 'train' else None
num = lambda c: pl.col(c).str.slice(0, 9).cast(pl.Int64, strict=False)
for k in range(8):
    fn = os.path.join(OUT, f'{split}_c{k}.parquet')
    if os.path.exists(fn): continue
    P = S.filter(pl.col('id1') % 8 == k)
    X = P.join(A.rename({c: c + '1' for c in A.columns if c not in ('id1', 'ctry', 'k_s1')}), on='id1') \
         .join(R.drop('ctry').rename({c: c + '2' for c in R.columns if c not in ('id2', 'src', 'ctry', 'g_src', 'g_oth', 'g_name')}), on=['id2', 'src'])
    X = X.with_columns(pl.Series('n_tset', cp(fuzz.token_set_ratio, X['nc1'].to_list(), X['nc2'].to_list())),
                                 pl.Series('n_ratio', cp(fuzz.ratio, X['nc1'].to_list(), X['nc2'].to_list())),
                                 pl.Series('n_part', cp(fuzz.partial_ratio, X['nc1'].to_list(), X['nc2'].to_list())),
                                 pl.Series('nt_ratio', cp(fuzz.ratio, X['nt1'].to_list(), X['nt2'].to_list())),
                                 pl.Series('a_tset', cp(fuzz.token_set_ratio, X['a1'].to_list(), X['a2'].to_list())),
                                 pl.Series('a_ratio', cp(fuzz.ratio, X['a1'].to_list(), X['a2'].to_list())))
    i = pl.col('tk1').list.set_intersection('tk2').list.len(); n1 = pl.col('tk1').list.len(); n2 = pl.col('tk2').list.len()
    h1, h2 = pl.col('h1'), pl.col('h2')
    X = X.with_columns(
        i.alias('n_inter'), (n2 - i).alias('n_extra2'), (n1 - i).alias('n_extra1'),
        pl.when((i == n1) & (i == n2)).then(0).when(i == n1).then(1).when(i == n2).then(2).when((n1 == n2) & (i == n1 - 1)).then(3).when(i > 0).then(4).otherwise(5).alias('name_rel'),
        pl.when((pl.col('lg1').list.len() == 0) & (pl.col('lg2').list.len() == 0)).then(0).when(pl.col('lg2').list.set_difference('lg1').list.len() > 0).then(1)
          .when(pl.col('lg2').list.len() < pl.col('lg1').list.len()).then(2).otherwise(3).alias('legal_rel'),
        pl.when(pl.col('na2')).then(0).when((h1 == '') | (h2 == '')).then(1).when(h1 == h2).then(2)
          .when(h1.str.starts_with(h2) | h2.str.starts_with(h1) | h1.str.ends_with(h2) | h2.str.ends_with(h1)).then(3)
          .when((num('h1') - num('h2')).abs() <= 50).then(4).otherwise(5).alias('hn_rel'),
        (num('h1') - num('h2')).abs().cast(pl.Float32).log1p().alias('hn_ldiff'),
        pl.col('w1').list.set_intersection('w2').list.len().alias('street_inter'), pl.col('w2').list.set_difference('w1').list.len().alias('street_extra2'),
        ((pl.col('city1') == pl.col('city2')) & (pl.col('city1') != '')).cast(pl.Int8).alias('city_eq'),
        pl.when((pl.col('dl1').list.len() + pl.col('dl2').list.len()) > 0).then(pl.col('dl1').list.set_intersection('dl2').list.len() / pl.col('dl1').list.set_union('dl2').list.len()).cast(pl.Float32).alias('dg_jac'),
        pl.col('dl2').list.set_difference('dl1').list.len().alias('dg_extra2'), pl.col('na2').cast(pl.Int8).alias('x_noaddr'))
    # the S1 entity's confident copies, excluding x itself
    Cc = conf.join(P.select('id1').unique(), on='id1', how='semi')
    Q = X.select(*KY, 'h1', 'h2').join(Cc, on='id1').filter(~((pl.col('cid') == pl.col('id2')) & (pl.col('csrc') == pl.col('src'))))
    Q = Q.with_columns((pl.col('csrc') == pl.col('src')).alias('same'))
    agg = Q.group_by(KY).agg(pl.col('same').sum().alias('c_src'), (~pl.col('same')).sum().alias('c_oth'),
                             (pl.col('same') & (pl.col('ch') == pl.col('h1')) & (pl.col('h1') != '')).any().cast(pl.Int8).alias('anchor'),
                             (~pl.col('same') & (pl.col('ch') == pl.col('h1')) & (pl.col('h1') != '')).any().cast(pl.Int8).alias('anchor_oth'),
                             ((pl.col('ch') == pl.col('h2')) & (pl.col('h2') != '')).sum().alias('c_at_xh'),
                             ((pl.col('ch') == pl.col('h1')) & (pl.col('h1') != '')).sum().alias('c_at_h1'),
                             (pl.col('same') & (pl.col('ch') == pl.col('h2')) & (pl.col('h2') != '')).sum().alias('c_src_at_xh'))
    X = X.join(agg, on=KY, how='left').with_columns(pl.col('c_src', 'c_oth', 'anchor', 'anchor_oth', 'c_at_xh', 'c_at_h1', 'c_src_at_xh').fill_null(0))
    keep = [*KY, 'ctry', 'p1', 'p', 'x_rank', 'x_ncand', 'a_rank', 'a_ncand', 'x_gap', 'k_s1', 'g_src', 'g_oth', 'g_name',
            'n_tset', 'n_ratio', 'n_part', 'nt_ratio', 'a_tset', 'a_ratio', 'n_inter', 'n_extra2', 'n_extra1', 'name_rel', 'legal_rel', 'hn_rel', 'hn_ldiff',
            'street_inter', 'street_extra2', 'city_eq', 'dg_jac', 'dg_extra2', 'x_noaddr', 'c_src', 'c_oth', 'anchor', 'anchor_oth', 'c_at_xh', 'c_at_h1', 'c_src_at_xh']
    X = X.select(keep)
    if lab is not None:
        X = X.join(lab, on=KY, how='left').with_columns(pl.col('y').fill_null(0), fold_expr('id1').alias('fold'))
    X = X.with_columns(pl.col(pl.Int64).exclude('id1', 'id2').cast(pl.Int32), pl.col(pl.UInt32).cast(pl.Int32))
    X.write_parquet(fn); print(f'chunk {k}: {X.height} pairs ({time.time()-t0:.0f}s)', flush=True)
    del P, X, Q, agg
print('F50_DONE', f'{time.time()-t0:.0f}s')
