"""Stage 2: entity-level evidence on top of stage-1 out-of-fold scores p1.
  record competition : rank of this S1 among the record's candidates, best rival score, gap
  S1 context         : # confident copies already claimed by the S1 (same / other source), best/sum of other copies
  sibling agreement  : does the record agree with the S1's *confidently matched copies* (full name incl. legal form, address,
                       house number, city) -- copies are generated from the hidden true entity, so siblings carry details S1 lacks
usage: python s11_stage2.py build <split> <feat tag> <score tag>   -> feat/<score tag>_s2/<split>.parquet
       python s11_stage2.py fit <score tag> <out tag>               -> scores/<out tag>/train|test.parquet with p2 (cross-fitted on train)"""
import sys, glob, time
import lightgbm as lgb
from rapidfuzz import process, fuzz
from common import *
from norm import LEGAL_CANON
LEGAL = sorted(set(LEGAL_CANON.values()))
PMIN, PSIB = 0.005, 0.7
DROP = int(os.environ.get('S2_DROP', '0'))          # per-mille of S1 entities removed (orphan-copy simulation)
SUF = f'_d{DROP}' if DROP else ''
DROPPED = lambda: (pl.col('id1').hash(seed=777) % 1000) < DROP
PAIRF = ['n_tset', 'n_ratio', 'a_tset', 'a2_empty', 'hn_eq', 'n_exact', 'nm_cnt_s1_of1', 'nm_cnt_s1_of2', 'amb_a', 'amb_b', 'city_in_b', 'st_tset',
         'is_india', 'is_france', 'dg_jac', 'ni_idf_jac', 'ai_idf_jac', 'legal_eq', 'legal_conflict']

def p1_expr(split):
    if split == 'train':
        fo = fold_expr('id1')
        return pl.when(fo == 1).then(pl.col('pB')).when(fo == 2).then(pl.col('pA')).otherwise((pl.col('pA') + pl.col('pB')) / 2).alias('p1')
    return ((pl.col('pA') + pl.col('pB')) / 2).alias('p1')

def rec_fields(split, s, ids):
    d = pl.scan_parquet(wp('norm2', f'{split}_s{s}.parquet')).select('id', 'nt', 'at', 'dz').join(ids.lazy(), on='id', how='semi').collect()
    c = pl.col('at').str.split(' , ')
    lg = pl.Series(LEGAL)
    return d.select('id', 'nt', pl.col('at').str.replace_all(' , ', ' ').alias('a'), pl.col('dz').str.split(' ').list.first().fill_null('').alias('hn'),
                    c.list.slice(0, c.list.len() - 1).list.eval(pl.element().filter(~pl.element().str.contains(r'\d') & (pl.element() != ''))).list.last().fill_null('').alias('city'),
                    pl.col('nt').str.split(' ').list.eval(pl.element().filter(pl.element().is_in(lg))).list.unique().list.sort().list.join(' ').alias('lg'))

def build(split, ftag, mtag):
    t0 = time.time()
    S = pl.scan_parquet(wp('scores', mtag, f'{split}_s[23]_c*.parquet')).select(*KY, 'r', 'rr', 'pA', 'pB').with_columns(p1_expr(split)) \
          .filter(pl.col('p1') >= PMIN).filter(~DROPPED()).drop('pA', 'pB').collect(engine='streaming')
    PSRC = os.environ.get('S2_PSRC')                  # iteration: recompute all context features from better scores (file with KY + p)
    if PSRC:
        S = S.join(pl.read_parquet(wp('scores', PSRC, f'{split}_full_p.parquet')).select(*KY, pl.col('p').alias('_pn')), on=KY, how='left') \
             .with_columns(pl.coalesce('_pn', 'p1').alias('p1')).drop('_pn')
    print(f'scored pairs with p1 >= {PMIN}: {S.height} ({time.time()-t0:.0f}s)', flush=True)
    # record competition
    S = S.with_columns(pl.col('p1').rank('ordinal', descending=True).over(['id2', 'src']).cast(pl.Float32).alias('rec_rank'),
                       pl.len().over(['id2', 'src']).cast(pl.Float32).alias('rec_n'),
                       (pl.col('p1') >= 0.5).sum().over(['id2', 'src']).cast(pl.Float32).alias('rec_n05'))
    top2 = S.group_by(['id2', 'src']).agg(pl.col('p1').top_k(2).alias('_t'))
    top2 = top2.with_columns(pl.col('_t').list.get(0).alias('_b1'), pl.col('_t').list.get(1, null_on_oob=True).fill_null(0).alias('_b2')).drop('_t')
    S = S.join(top2, on=['id2', 'src'], how='left').with_columns(
        pl.when(pl.col('rec_rank') == 1).then(pl.col('_b2')).otherwise(pl.col('_b1')).alias('rec_best_other')).drop('_b1', '_b2')
    S = S.with_columns((pl.col('p1') - pl.col('rec_best_other')).alias('rec_gap'))
    # S1 context: confident copies claimed (record's best S1 = this S1)
    conf = S.filter((pl.col('rec_rank') == 1) & (pl.col('p1') >= 0.5))
    cc = conf.group_by(['id1', 'src']).agg(pl.len().cast(pl.Float32).alias('_n'))
    S = S.join(cc, on=['id1', 'src'], how='left').join(cc.with_columns((5 - pl.col('src')).cast(pl.Int8).alias('src')).rename({'_n': '_no'}), on=['id1', 'src'], how='left')
    isconf = ((pl.col('rec_rank') == 1) & (pl.col('p1') >= 0.5)).cast(pl.Float32)
    S = S.with_columns((pl.col('_n').fill_null(0) - isconf).alias('s1_nconf_src'), pl.col('_no').fill_null(0).alias('s1_nconf_oth')).drop('_n', '_no')
    S = S.with_columns((pl.col('p1').sum().over('id1') - pl.col('p1')).alias('s1_psum_other'),
                       pl.col('p1').rank('ordinal', descending=True).over(['id1', 'src']).cast(pl.Float32).alias('s1_rank'))
    t1 = S.group_by('id1').agg(pl.col('p1').top_k(2).alias('_t')).with_columns(pl.col('_t').list.get(0).alias('_a1'), pl.col('_t').list.get(1, null_on_oob=True).fill_null(0).alias('_a2')).drop('_t')
    S = S.join(t1, on='id1', how='left').with_columns(pl.when(pl.col('p1') >= pl.col('_a1')).then(pl.col('_a2')).otherwise(pl.col('_a1')).alias('s1_pmax_other')).drop('_a1', '_a2')
    print(f'context done, {S.height} pairs ({time.time()-t0:.0f}s)', flush=True)
    # siblings: records whose best S1 is A with p1 >= PSIB
    sib = S.filter((pl.col('rec_rank') == 1) & (pl.col('p1') >= PSIB)).select(pl.col('id1'), pl.col('id2').alias('sid'), pl.col('src').alias('ssrc'))
    Q = S.select(*KY).join(sib, on='id1', how='inner').filter(~((pl.col('sid') == pl.col('id2')) & (pl.col('ssrc') == pl.col('src'))))
    print(f'sibling comparisons {Q.height} ({time.time()-t0:.0f}s)', flush=True)
    R = {}
    for s in (2, 3):
        need = pl.concat([Q.filter(pl.col('src') == s).select(pl.col('id2').alias('id')), Q.filter(pl.col('ssrc') == s).select(pl.col('sid').alias('id'))]).unique()
        R[s] = rec_fields(split, s, need)
    out = []
    for s in (2, 3):
        for s2 in (2, 3):
            q = Q.filter((pl.col('src') == s) & (pl.col('ssrc') == s2))
            q = q.join(R[s], left_on='id2', right_on='id', how='left').join(R[s2].rename({c: c + '_s' for c in R[s2].columns if c != 'id'}), left_on='sid', right_on='id', how='left')
            q = q.with_columns(pl.Series('nt_ts', process.cpdist(q['nt'].to_list(), q['nt_s'].to_list(), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)),
                               pl.Series('nt_r', process.cpdist(q['nt'].to_list(), q['nt_s'].to_list(), scorer=fuzz.ratio, workers=-1, dtype=np.float32)),
                               pl.Series('a_ts', process.cpdist(q['a'].to_list(), q['a_s'].to_list(), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)))
            out.append(q.select(*KY, 'nt_ts', 'nt_r', (pl.col('nt') == pl.col('nt_s')).alias('nt_eq'),
                                pl.when((pl.col('a') != '') & (pl.col('a_s') != '')).then(pl.col('a_ts')).alias('a_ts'),
                                ((pl.col('hn') == pl.col('hn_s')) & (pl.col('hn') != '')).alias('hn_eq_s'),
                                ((pl.col('city') == pl.col('city_s')) & (pl.col('city') != '')).alias('city_eq_s'),
                                ((pl.col('lg') == pl.col('lg_s')) & (pl.col('lg') != '')).alias('lg_eq_s')))
    Q = pl.concat(out)
    A = Q.group_by(KY).agg(pl.len().cast(pl.Float32).alias('sib_n'), pl.col('nt_ts').max().alias('sib_nt_ts'), pl.col('nt_r').max().alias('sib_nt_r'),
                           pl.col('nt_eq').any().cast(pl.Float32).alias('sib_nt_eq'), pl.col('a_ts').max().alias('sib_a_ts'),
                           pl.col('hn_eq_s').any().cast(pl.Float32).alias('sib_hn_eq'), pl.col('city_eq_s').any().cast(pl.Float32).alias('sib_city_eq'),
                           pl.col('lg_eq_s').any().cast(pl.Float32).alias('sib_lg_eq'))
    S = S.join(A, on=KY, how='left').with_columns(pl.col('sib_n').fill_null(0))
    # a few strong pair features
    PF = pl.scan_parquet(wp('feat', ftag, f'{split}_s[23]_c*.parquet')).select(*KY, *PAIRF).join(S.select(*KY).lazy(), on=KY, how='semi').collect()
    S = S.join(PF, on=KY, how='left')
    mtag = os.environ.get('S2_OUT', mtag)
    os.makedirs(wp('feat', f'{mtag}_s2'), exist_ok=True)
    S.write_parquet(wp('feat', f'{mtag}_s2', f'{split}{SUF}.parquet'))
    print(f'BUILD_DONE {S.height} rows ({time.time()-t0:.0f}s)', flush=True)

import unicodedata, re
LEET = str.maketrans({'0': 'o', '1': 'l', '5': 's', '8': 'b', '6': 'g'})
LEET_TOK = re.compile(r'^(?=(?:.*[a-z]){2})(?!\d+(?:st|nd|rd|th)$)[a-z01568]+$')
def leet(s):   # same digit-for-letter fix as s02b_norm2
    if not s or not any(ch.isdigit() for ch in s): return s
    return ' '.join(t.translate(LEET) if (any(ch.isdigit() for ch in t) and LEET_TOK.match(t)) else t for t in s.split(' '))
def _raw(s):
    if s is None: return ''
    s = s.lower()
    if not s.isascii(): s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
    return s
def raw_names(split, s, ids):
    """raw-form name (lowercase, accents/punctuation removed, digit-for-letter typos fixed, legal words NOT canonicalised) + raw house number"""
    d = pl.scan_parquet(wp('data', f'{split}_s{s}.parquet')).select('id', 'name').join(ids.lazy(), on='id', how='semi').collect()
    el = pl.element()
    d = d.with_columns(pl.col('name').str.to_lowercase().str.normalize('NFKD').str.replace_all(r'[^\x00-\x7f]', '').str.replace_all('&', ' and ')
                       .str.replace_all(r'[^a-z0-9 ]', ' ').str.replace_all(r'\s+', ' ').str.strip_chars().str.split(' ').list.eval(
                           pl.when(el.str.contains('^[a-z01568]+$') & (el.str.count_matches('[a-z]') >= 2) & el.str.contains('[01568]') & ~el.str.contains(r'^[0-9]+(st|nd|rd|th)$'))
                             .then(el.str.replace_many(['0', '1', '5', '8', '6'], ['o', 'l', 's', 'b', 'g'])).otherwise(el)).list.join(' ').alias('rn'))
    h = pl.scan_parquet(wp('norm2', f'{split}_s{s}.parquet')).select('id', pl.col('dg').str.split(' ').list.first().fill_null('').alias('hr')).join(ids.lazy(), on='id', how='semi').collect()
    return d.select('id', 'rn').join(h, on='id', how='left')

def build_x(split, mtag):
    """extra stage-2 features: raw-form name similarity (+ its rank/gap inside the record) and raw house-number suffix/prefix relations"""
    t0 = time.time()
    S = pl.read_parquet(wp('feat', f'{mtag}_s2', f'{split}{SUF}.parquet'), columns=KY)
    A = raw_names(split, 1, S.select(pl.col('id1').unique().alias('id'))); print(f'  S1 raw names {A.height} ({time.time()-t0:.0f}s)', flush=True)
    out = []
    for s in (2, 3):
        q = S.filter(pl.col('src') == s)
        B = raw_names(split, s, q.select(pl.col('id2').unique().alias('id'))); print(f'  src {s} raw names {B.height} ({time.time()-t0:.0f}s)', flush=True)
        q = q.join(A.rename({'rn': 'rn1', 'hr': 'hr1'}), left_on='id1', right_on='id', how='left').join(B.rename({'rn': 'rn2', 'hr': 'hr2'}), left_on='id2', right_on='id', how='left')
        a, b = q['rn1'].fill_null('').to_list(), q['rn2'].fill_null('').to_list()
        q = q.with_columns(pl.Series('rn_ratio', process.cpdist(a, b, scorer=fuzz.ratio, workers=-1, dtype=np.float32)),
                           pl.Series('rn_tset', process.cpdist(a, b, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)),
                           pl.Series('rn_tsort', process.cpdist(a, b, scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.float32)),
                           (pl.col('rn1') == pl.col('rn2')).cast(pl.Float32).alias('rn_eq'))
        h1, h2 = pl.col('hr1').fill_null(''), pl.col('hr2').fill_null('')
        both = (h1 != '') & (h2 != '') & (h1 != h2)
        q = q.with_columns(pl.when(both).then((h1.str.ends_with(h2) | h2.str.ends_with(h1)).cast(pl.Float32)).alias('hr_sfx'),
                           pl.when(both).then((h1.str.starts_with(h2) | h2.str.starts_with(h1)).cast(pl.Float32)).alias('hr_pfx'),
                           pl.when(both).then((h1.str.slice(-2) == h2.str.slice(-2)).cast(pl.Float32)).alias('hr_last2'),
                           pl.when(both).then(h2.str.starts_with('0').cast(pl.Float32)).alias('hr2_lead0'))
        out.append(q.select(*KY, 'rn_ratio', 'rn_tset', 'rn_tsort', 'rn_eq', 'hr_sfx', 'hr_pfx', 'hr_last2', 'hr2_lead0')); print(f'  src {s} pair feats ({time.time()-t0:.0f}s)', flush=True)
    X = pl.concat(out)
    g = X.group_by(['id2', 'src']).agg(pl.col('rn_ratio').max().alias('_mx'))
    X = X.join(g, on=['id2', 'src'], how='left')
    t = X.filter(pl.col('rn_ratio') == pl.col('_mx')).group_by(['id2', 'src']).agg(pl.len().cast(pl.Float32).alias('rn_nties'))
    X = X.join(t, on=['id2', 'src'], how='left').with_columns((pl.col('rn_ratio') - pl.col('_mx')).alias('rn_gap_top'),
                                                             pl.col('rn_ratio').rank('min', descending=True).over(['id2', 'src']).cast(pl.Float32).alias('rn_rank_rec')).drop('_mx')
    P = pl.read_parquet(wp('feat', f'{mtag}_s2', f'{split}{SUF}.parquet'), columns=[*KY, 'p1'])
    P = P.with_columns(pl.col('p1').sum().over(['id2', 'src']).alias('rec_psum')).with_columns((pl.col('p1') / pl.col('rec_psum')).alias('rec_share')).drop('p1')
    X = X.join(P, on=KY, how='left')
    X.write_parquet(wp('feat', f'{mtag}_s2', f'{split}{SUF}_x.parquet'))
    print(f'BUILDX_DONE {X.height} rows ({time.time()-t0:.0f}s)', flush=True)

PAR = dict(objective='binary', learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
           lambda_l2=1.0, num_threads=6, verbose=-1)
def fit(mtag, otag, R=500, extra=False):
    t0 = time.time()
    T = pl.read_parquet(wp('feat', f'{mtag}_s2', 'train.parquet'))
    if extra: T = T.join(pl.read_parquet(wp('feat', f'{mtag}_s2', 'train_x.parquet')), on=KY, how='left')
    T = T.join(load_gt_pairs().with_columns(pl.lit(1, pl.Int8).alias('y')), on=KY, how='left') \
          .with_columns(pl.col('y').fill_null(0), fold_expr('id1').alias('fold'))
    C = [c for c in T.columns if c not in ('id1', 'id2', 'y', 'fold')]
    fo = T['fold'].to_numpy(); X = T.select(C).to_numpy().astype(np.float32); y = T['y'].to_numpy()
    p2 = np.zeros(T.height, dtype=np.float32); ms = []
    for tr_f, te_f in (((3, 4), (0, 1, 2, 5, 6, 7)), ((5, 6), (3, 4))):
        tr = np.isin(fo, tr_f); te = np.isin(fo, te_f)
        m = lgb.train(PAR, lgb.Dataset(X[tr], y[tr], feature_name=C), R); ms.append(m)
        if tr_f == (3, 4): p2[te] = m.predict(X[te], num_threads=6)
        else: p2[te] = m.predict(X[te], num_threads=6)
        print(f'  stage-2 model trained on folds {tr_f}: {tr.sum()} rows ({time.time()-t0:.0f}s)', flush=True)
    print('   top gain:', sorted(zip(C, ms[0].feature_importance('gain').round()), key=lambda x: -x[1])[:20], flush=True)
    os.makedirs(wp('scores', otag), exist_ok=True)
    T.select(*KY, 'p1', pl.Series('p', p2)).write_parquet(wp('scores', otag, 'train_s2_c0.parquet'))
    for i, m in enumerate(ms): m.save_model(wp('models', f'{otag}_{i}.txt'))
    tf = wp('feat', f'{mtag}_s2', 'test.parquet')
    if os.path.exists(tf):
        Te = pl.read_parquet(tf)
        if extra: Te = Te.join(pl.read_parquet(wp('feat', f'{mtag}_s2', 'test_x.parquet')), on=KY, how='left')
        pt = np.mean([m.predict(Te.select(C).to_numpy().astype(np.float32), num_threads=6) for m in ms], axis=0)
        Te.select(*KY, 'p1', pl.Series('p', pt).cast(pl.Float32)).write_parquet(wp('scores', otag, 'test_s2_c0.parquet'))
    print('FIT_DONE', f'{time.time()-t0:.0f}s')

if __name__ == '__main__':
    if sys.argv[1] == 'build': build(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == 'buildx': build_x(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'fit': fit(sys.argv[2], sys.argv[3], extra=len(sys.argv) > 4 and sys.argv[4] == 'x')
