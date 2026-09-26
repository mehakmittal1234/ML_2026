"""Premise check for the France fix (label-free for France, labelled for US/India train).
Copy-noise tokens per country are learned from each split's own records (tokens >=4x more frequent in S2/S3 names than in S1 names);
French department components are mapped to the region their city has in S1 (learned by co-occurrence). On borderline top pairs
(0.2 < p1 < 0.75, record's best candidate) we test pattern CLEAN = cleaned names equal-or-contained AND same house number AND
admin-mapped address token-set >= 95: its true rate in train says whether the model is under-confident on it."""
from common import *
from rapidfuzz import process, fuzz
from norm import LEGAL_CANON
LEG = set(LEGAL_CANON.values())

def noise_tokens(split):
    out = {}
    s1 = pl.read_parquet(wp('norm2', f'{split}_s1.parquet'), columns=['ctry', 'nc'])
    cp = pl.concat([pl.read_parquet(wp('norm2', f'{split}_s{s}.parquet'), columns=['ctry', 'nc']) for s in (2, 3)])
    for c in s1['ctry'].unique().to_list():
        a, b = s1.filter(pl.col('ctry') == c), cp.filter(pl.col('ctry') == c)
        f1 = a.select(pl.col('nc').str.split(' ').list.unique()).explode('nc').group_by('nc').len().with_columns((pl.col('len') / a.height).alias('r1')).drop('len')
        f2 = b.select(pl.col('nc').str.split(' ').list.unique()).explode('nc').group_by('nc').len().with_columns((pl.col('len') / b.height).alias('r2')).drop('len')
        j = f2.join(f1, on='nc', how='left').with_columns(pl.col('r1').fill_null(0)).filter((pl.col('r2') >= 0.002) & (pl.col('r2') / (pl.col('r1') + 1e-4) >= 4))
        out[c] = set(j['nc'].to_list()) - {''}
    return out

def admin_map(split):
    """France: copy-only admin components -> S1 region, by city co-occurrence"""
    s1 = pl.read_parquet(wp('norm2', f'{split}_s1.parquet'), columns=['ctry', 'at']).filter(pl.col('ctry') == 'France')
    if s1.height == 0: return {}
    comps = lambda d: d.select(pl.col('at').str.split(' , ').alias('c')).with_row_index('r').explode('c').filter(~pl.col('c').str.contains(r'\d') & (pl.col('c') != ''))
    c1 = comps(s1); vc = c1.group_by('c').len()
    regions = set(vc.filter(pl.col('len') > 0.1 * s1.height)['c'].to_list())
    cities = set(vc.filter((pl.col('len') > 0.001 * s1.height) & ~pl.col('c').is_in(list(regions)))['c'].to_list())
    reg_of_city = c1.join(c1.filter(pl.col('c').is_in(list(regions))).rename({'c': 'reg'}), on='r').filter(pl.col('c').is_in(list(cities))) \
                    .group_by('c', 'reg').len().sort('len', descending=True).unique('c', keep='first')
    cp = pl.concat([pl.read_parquet(wp('norm2', f'{split}_s{s}.parquet'), columns=['ctry', 'at']) for s in (2, 3)]).filter(pl.col('ctry') == 'France')
    c2 = comps(cp); v2 = c2.group_by('c').len()
    cand = set(v2.filter(pl.col('len') > 0.02 * cp.height)['c'].to_list()) - regions - cities
    m = c2.filter(pl.col('c').is_in(list(cand))).rename({'c': 'adm'}).join(c2.filter(pl.col('c').is_in(list(cities))), on='r') \
          .join(reg_of_city, on='c').group_by('adm', 'reg').len().sort('len', descending=True).unique('adm', keep='first')
    return dict(zip(m['adm'].to_list(), m['reg'].to_list()))

def clean_name(nc, noise):
    return ' '.join(t for t in nc.split(' ') if t and t not in noise and t not in LEG)

def analyse(split, lab):
    NZ = noise_tokens(split); AM = admin_map(split)
    print(split, 'noise tokens:', {c: sorted(v)[:40] for c, v in NZ.items()}); print(split, 'admin map:', AM)
    S = pl.read_parquet(wp('feat', 'm1_s2', f'{split}.parquet'), columns=[*KY, 'p1', 'rec_rank'])
    B = S.filter((pl.col('p1') > 0.2) & (pl.col('p1') < 0.75) & (pl.col('rec_rank') == 1))
    A = pl.read_parquet(wp('norm2', f'{split}_s1.parquet'), columns=['id', 'ctry', 'nc', 'at', 'dz']).join(B.select(pl.col('id1').alias('id')).unique(), on='id', how='semi')
    R = pl.concat([pl.read_parquet(wp('norm2', f'{split}_s{s}.parquet'), columns=['id', 'nc', 'at', 'dz']).with_columns(pl.lit(s, pl.Int8).alias('src')) for s in (2, 3)]) \
          .join(B.select(pl.col('id2').alias('id'), 'src').unique(), on=['id', 'src'], how='semi')
    X = B.join(A.rename({'id': 'id1', 'nc': 'n1', 'at': 'a1', 'dz': 'd1'}), on='id1').join(R.rename({'id': 'id2', 'nc': 'n2', 'at': 'a2', 'dz': 'd2'}), on=['id2', 'src'])
    ctry = X['ctry'].to_list()
    c1 = [clean_name(n, NZ.get(c, set())) for n, c in zip(X['n1'].to_list(), ctry)]; c2 = [clean_name(n, NZ.get(c, set())) for n, c in zip(X['n2'].to_list(), ctry)]
    amap = lambda a: ' '.join(AM.get(p.strip(), p.strip()) for p in a.split(' , '))
    a1 = [amap(a) for a in X['a1'].to_list()]; a2 = [amap(a) for a in X['a2'].to_list()]
    X = X.with_columns(pl.Series('cn_ts', process.cpdist(c1, c2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)),
                       pl.Series('am_ts', process.cpdist(a1, a2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)),
                       (pl.col('d1').str.split(' ').list.first() == pl.col('d2').str.split(' ').list.first()).fill_null(False).alias('hn_eq'))
    X = X.with_columns(((pl.col('cn_ts') >= 99.9) & pl.col('hn_eq') & (pl.col('am_ts') >= 95) & (pl.col('a2') != '')).alias('CLEAN'))
    if lab:
        X = X.join(load_gt_pairs().with_columns(pl.lit(1, pl.Int8).alias('y')), on=KY, how='left').with_columns(pl.col('y').fill_null(0))
        print(X.group_by('ctry', 'CLEAN').agg(pl.len(), pl.col('y').mean().round(3).alias('true_rate'), pl.col('p1').mean().round(3).alias('mean_p1')).sort('ctry', 'CLEAN'))
    else:
        print(X.group_by('ctry', 'CLEAN').agg(pl.len(), pl.col('p1').mean().round(3).alias('mean_p1')).sort('ctry', 'CLEAN'))

if __name__ == '__main__':
    analyse('train', True); analyse('test', False)
