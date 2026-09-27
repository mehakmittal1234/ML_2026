"""Vectorised pair features (same feature definitions as feats.py / v4 featlib+s3lib, ~8x faster) + v6 additions.
Record-level fields are precomputed once per record (Side), pair features use polars list/set ops and rapidfuzz cpdist."""
import math, time
import numpy as np
import polars as pl
from rapidfuzz import process, fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from common import *
from norm import LEGAL_CANON

LEGAL = sorted(set(LEGAL_CANON.values()))
STYPES = ['st', 'ave', 'rd', 'dr', 'ln', 'ct', 'cir', 'blvd', 'pl', 'pkwy', 'hwy', 'ter', 'trl', 'sq', 'way', 'pt', 'xing', 'plz', 'ctr', 'hts', 'expy', 'fwy',
          'tpke', 'jct', 'mt', 'mtn', 'ft', 'n', 's', 'e', 'w', 'ne', 'nw', 'se', 'sw', 'r', 'av', 'bd', 'rte', 'ch', 'imp', 'all', 'quai', 'crs', 'fg', 'res',
          'lot', 'pass', 'prom', 'ham', 'bis', 'q', 'ngr', 'col', 'sec', 'main', 'cross', 'road', 'street', 'lane', 'marg', 'nagar', 'extn', 'ph', 'block',
          'floor', 'fl', 'grd', 'no', 'ste', 'apt', 'unit', 'bldg', 'rm']
CW = ['france', 'india', 'usa']
HK = lambda c, n: pl.concat_str([pl.col(c), pl.lit('|'), pl.col(n)]).hash(seed=3)


def city_expr(col='at'):
    c = pl.col(col).str.split(' , ')
    nd = c.list.slice(0, c.list.len() - 1).list.eval(pl.element().filter(~pl.element().str.contains(r'\d') & (pl.element() != '')))
    return nd.list.last().fill_null('').alias('city')


class Ctx:
    def __init__(self, split, nd='norm2'):
        self.split, self.nd = split, nd
        dfn = pl.read_parquet(wp(nd, f'{split}_df_name.parquet')); dfa = pl.read_parquet(wp(nd, f'{split}_df_addr.parquet'))
        self.NTOT = float(sum(pl.scan_parquet(wp(nd, f'{split}_s{s}.parquet')).select(pl.len()).collect().item() for s in (1, 2, 3)))
        self.WMAX = math.log(self.NTOT)
        self.dfn = dfn.select('ctry', 'tok', (self.NTOT / pl.col('df').cast(pl.Float64)).log().cast(pl.Float32).alias('w'))
        self.dfa = dfa.select('ctry', 'tok', (self.NTOT / pl.col('df').cast(pl.Float64)).log().cast(pl.Float32).alias('w'))
        self.NC = {s: pl.scan_parquet(wp(nd, f'{split}_s{s}.parquet')).select(HK('ctry', 'nc').alias('h')).group_by('h')
                      .agg(pl.len().cast(pl.Float32).alias('cnt')).collect() for s in (1, 2, 3)}
        s1 = pl.read_parquet(wp(nd, f'{split}_s1.parquet'), columns=['ctry', 'nc', 'at']).with_columns(city_expr())
        self.amb = s1.select(pl.concat_str([pl.col('ctry'), pl.lit('|'), pl.col('nc'), pl.lit('|'), pl.col('city')]).hash(seed=4).alias('hc')) \
                     .group_by('hc').agg(pl.len().cast(pl.Float32).alias('amb'))
        del s1

    def wsum(self, rec, col, dfx, alpha, name):
        """per record: IDF total of unique tokens (>=2 chars, alpha-only for addresses)"""
        e = pl.col(col).str.split(' ').list.unique().list.eval(pl.element().filter((pl.element().str.len_chars() >= 2) & (pl.element() != ',')))
        if alpha: e = e.list.eval(pl.element().filter(~pl.element().str.contains(r'\d')))
        x = rec.select('id', 'ctry', e.alias('tok')).explode('tok').drop_nulls('tok').join(dfx, on=['ctry', 'tok'], how='left') \
               .with_columns(pl.col('w').fill_null(self.WMAX))
        return x.group_by('id').agg(pl.col('w').sum().alias(name + 'W'), pl.col('w').max().alias(name + 'M'), pl.len().cast(pl.Float32).alias(name + 'N'))


def side(ctx, s, ids=None):
    """precomputed record fields for source s (optionally restricted to ids)"""
    lf = pl.scan_parquet(wp(ctx.nd, f'{ctx.split}_s{s}.parquet'))
    if ids is not None: lf = lf.join(ids.lazy(), on='id', how='semi')
    d = lf.collect()
    lg = pl.Series(LEGAL)
    d = d.with_columns(pl.col('at').str.split(' , ').alias('_c'), pl.col('dz').str.split(' ').list.eval(pl.element().filter(pl.element() != '')).alias('dl'))
    d = d.with_columns(pl.col('_c').list.eval(pl.element().str.contains(r'\d')).list.arg_max().alias('_si'), pl.col('dl').list.first().fill_null('').alias('hn'))
    d = d.with_columns(pl.col('_c').list.get(pl.col('_si'), null_on_oob=True).fill_null('').alias('_stc'))
    d = d.with_columns(
        pl.col('_stc').str.split(' ').list.eval(pl.element().filter(pl.element().str.contains(r'^[a-z]{2,}$') & ~pl.element().is_in(STYPES))).list.join(' ').alias('st'),
        city_expr(), pl.col('nc').str.split(' ').list.eval(pl.element().filter(~pl.element().is_in(CW)).str.slice(0, 1)).list.join('').alias('ini'),
        pl.col('at').str.replace_all(' , ', ' ').alias('a'), pl.col('nc').str.replace_all(' ', '').alias('cat'),
        pl.col('nc').str.split(' ').list.first().fill_null('').alias('nf'),
        pl.col('nt').str.split(' ').list.eval(pl.element().filter(pl.element().is_in(lg))).list.unique().alias('lg'),
        pl.col('nc').str.split(' ').list.eval(pl.element().filter(pl.element().str.len_chars() >= 2)).list.unique().alias('ntk'),
        pl.col('at').str.split(' ').list.eval(pl.element().filter((pl.element().str.len_chars() >= 2) & ~pl.element().str.contains(r'\d'))).list.unique().alias('atk'),
        HK('ctry', 'nc').alias('hnc'))
    d = d.join(ctx.wsum(d, 'nc', ctx.dfn, False, 'wn'), on='id', how='left').join(ctx.wsum(d, 'at', ctx.dfa, True, 'wa'), on='id', how='left')
    d = d.with_columns(pl.col('wnW', 'wnM', 'wnN', 'waW', 'waM', 'waN').fill_null(0))
    return d.drop('_c', '_si', '_stc', 'dg')


def cp(scorer, a, b):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def inter_w(pr, c1, c2, dfx, WMAX):
    """IDF weight of the token-set intersection per pair"""
    x = pr.select('i', 'ctry_1', pl.col(c1).list.set_intersection(pl.col(c2)).alias('t')).explode('t').drop_nulls('t')
    x = x.join(dfx, left_on=['ctry_1', 't'], right_on=['ctry', 'tok'], how='left').with_columns(pl.col('w').fill_null(WMAX))
    return x.group_by('i').agg(pl.col('w').sum().alias('Wc'), pl.col('w').max().alias('Mc'), pl.len().cast(pl.Float32).alias('Nc'))


def compute(pr, ctx, s):
    """pr: pair frame with record fields of side 1 (suffix _1) and side 2 (suffix _2)"""
    pr = pr.with_row_index('i')
    a, b = pr['nc_1'].to_list(), pr['nc_2'].to_list()
    na, ta, tb = pr['na_2'].to_list(), pr['nt_1'].to_list(), pr['nt_2'].to_list()
    aa, ab = pr['a_1'].to_list(), pr['a_2'].to_list()
    f = {'n_ratio': cp(fuzz.ratio, a, b), 'n_tsort': cp(fuzz.token_sort_ratio, a, b), 'n_tset': cp(fuzz.token_set_ratio, a, b),
         'n_partial': cp(fuzz.partial_ratio, a, b), 'n_jw': cp(JaroWinkler.normalized_similarity, a, b),
         'n_alt_tset': cp(fuzz.token_set_ratio, a, na), 'n_alt_ratio': cp(fuzz.ratio, a, na),
         'nt_ratio': cp(fuzz.ratio, ta, tb), 'nt_tset': cp(fuzz.token_set_ratio, ta, tb),
         'a_ratio': cp(fuzz.ratio, aa, ab), 'a_tset': cp(fuzz.token_set_ratio, aa, ab), 'a_tsort': cp(fuzz.token_sort_ratio, aa, ab),
         'a_partial': cp(fuzz.partial_ratio, aa, ab),
         'hn_lev': cp(Levenshtein.distance, pr['hn_1'].to_list(), pr['hn_2'].to_list()),
         'st_tset': cp(fuzz.token_set_ratio, pr['st_1'].to_list(), pr['st_2'].to_list()),
         'st_ratio': cp(fuzz.ratio, pr['st_1'].to_list(), pr['st_2'].to_list())}
    dom, cat = pr['dom_2'].to_list(), pr['cat_1'].to_list()
    pre = [c[:len(d)] if d else '' for c, d in zip(cat, dom)]
    f['dom_prefix'] = cp(fuzz.ratio, dom, pre); f['dom_partial'] = cp(fuzz.partial_ratio, dom, cat)
    F = pl.DataFrame(f).with_row_index('i')
    e1, e2 = pl.col('hn_1') != '', pl.col('hn_2') != ''
    both = e1 & e2
    nan = pl.lit(None, pl.Float32)
    X = pr.select(
        'i', 'id1', 'id2', 'src',
        ((pl.col('nc_1') == pl.col('nc_2')) & (pl.col('nc_1') != '')).cast(pl.Float32).alias('n_exact'),
        ((pl.col('nf_1') == pl.col('nf_2')) & (pl.col('nf_1') != '')).cast(pl.Float32).alias('n_first_eq'),
        (pl.col('na_2') != '').alias('_hasalt'), (pl.col('dom_2') != '').alias('_hasdom'), (pl.col('a_2') != '').alias('_hasa2'),
        (pl.col('a_2') == '').cast(pl.Float32).alias('a2_empty'),
        pl.when(both).then((pl.col('hn_1') == pl.col('hn_2')).cast(pl.Float32)).otherwise(nan).alias('hn_eq'),
        pl.when(both).then(pl.col('dl_2').list.contains(pl.col('hn_1')).cast(pl.Float32)).otherwise(nan).alias('hn1_in2'),
        pl.when(both).then(pl.col('dl_1').list.contains(pl.col('hn_2')).cast(pl.Float32)).otherwise(nan).alias('hn2_in1'),
        both.alias('_both'),
        pl.when(both).then((pl.col('hn_1').str.slice(0, 12).cast(pl.Float64, strict=False) - pl.col('hn_2').str.slice(0, 12).cast(pl.Float64, strict=False)).abs().log1p().cast(pl.Float32)).otherwise(nan).alias('hn_absdiff'),
        pl.col('hn_1').str.len_chars().cast(pl.Float32).alias('hn1_len'), pl.col('hn_2').str.len_chars().cast(pl.Float32).alias('hn2_len'),
        pl.when((pl.col('dl_1').list.len() + pl.col('dl_2').list.len()) > 0)
          .then(pl.col('dl_1').list.set_intersection('dl_2').list.len() / pl.col('dl_1').list.set_union('dl_2').list.len()).otherwise(nan).cast(pl.Float32).alias('dg_jac'),
        pl.col('dl_1').list.len().cast(pl.Float32).alias('dg_n1'), pl.col('dl_2').list.len().cast(pl.Float32).alias('dg_n2'),
        pl.when(pl.col('dl_2').list.len() > 0).then((pl.col('dl_2').list.set_difference('dl_1').list.len() == 0).cast(pl.Float32)).otherwise(nan).alias('dg_sub'),
        ((pl.col('lg_1').list.len() > 0) & (pl.col('lg_2').list.len() > 0)).cast(pl.Float32).alias('legal_both'),
        ((pl.col('lg_1').list.len() > 0) & (pl.col('lg_2').list.len() > 0) & (pl.col('lg_1').list.set_intersection('lg_2').list.len() == 0)).cast(pl.Float32).alias('legal_conflict'),
        ((pl.col('lg_1').list.len() == pl.col('lg_2').list.len()) & (pl.col('lg_1').list.set_intersection('lg_2').list.len() == pl.col('lg_1').list.len())).cast(pl.Float32).alias('legal_eq'),
        ((pl.col('st_1') != '') & (pl.col('st_2') != '')).alias('_bst'),
        (pl.col('st_1') == '').cast(pl.Float32).alias('st_a_empty'), (pl.col('st_2') == '').cast(pl.Float32).alias('st_b_empty'),
        pl.when(pl.col('city_1') == '').then(nan).otherwise(pl.col('city_1').str.split(' ').list.set_intersection(pl.col('a_2').str.split(' ')).list.len()
                                                             / pl.col('city_1').str.split(' ').list.len()).cast(pl.Float32).alias('city_in_b'),
        ((pl.col('cat_2') == pl.col('ini_1')) & pl.col('ini_1').str.len_chars().is_between(2, 6)).cast(pl.Float32).alias('acro'),
        (pl.col('hn_2') == '').cast(pl.Float32).alias('hn_b_empty'), (pl.col('hn_1') == '').cast(pl.Float32).alias('hn_a_empty'),
        (pl.col('ctry_1') == 'India').cast(pl.Float32).alias('is_india'), (pl.col('ctry_1') == 'France').cast(pl.Float32).alias('is_france'),
        'wnW_1', 'wnW_2', 'wnM_1', 'wnN_1', 'wnN_2', 'waW_1', 'waW_2', 'waN_1', 'waN_2', 'hnc_1', 'hnc_2',
        pl.concat_str([pl.col('ctry_1'), pl.lit('|'), pl.col('nc_1'), pl.lit('|'), pl.col('city_1')]).hash(seed=4).alias('hc1'),
        pl.concat_str([pl.col('ctry_1'), pl.lit('|'), pl.col('nc_2'), pl.lit('|'), pl.col('city_1')]).hash(seed=4).alias('hc2'),
        pl.concat_str([pl.col('ctry_1'), pl.lit('|'), pl.col('nc_2')]).hash(seed=3).alias('hnc_21'))
    X = X.join(inter_w(pr, 'ntk_1', 'ntk_2', ctx.dfn, ctx.WMAX).rename({'Wc': 'nWc', 'Mc': 'nMc', 'Nc': 'nNc'}), on='i', how='left') \
         .join(inter_w(pr, 'atk_1', 'atk_2', ctx.dfa, ctx.WMAX).rename({'Wc': 'aWc', 'Mc': 'aMc', 'Nc': 'aNc'}), on='i', how='left')
    X = X.with_columns(pl.col('nWc', 'nNc', 'aWc', 'aNc').fill_null(0))
    X = X.join(ctx.NC[1].rename({'h': 'hnc_1', 'cnt': 'nm_cnt_s1_of1'}), on='hnc_1', how='left') \
         .join(ctx.NC[1].rename({'h': 'hnc_21', 'cnt': 'nm_cnt_s1_of2'}), on='hnc_21', how='left') \
         .join(ctx.NC[s].rename({'h': 'hnc_2', 'cnt': 'nm_cnt_src_of2'}), on='hnc_2', how='left') \
         .join(ctx.amb.rename({'hc': 'hc1', 'amb': 'amb_a'}), on='hc1', how='left').join(ctx.amb.rename({'hc': 'hc2', 'amb': 'amb_b'}), on='hc2', how='left')
    W1, W2, Wc = pl.col('wnW_1'), pl.col('wnW_2'), pl.col('nWc'); A1, A2, Ac = pl.col('waW_1'), pl.col('waW_2'), pl.col('aWc')
    X = X.with_columns(
        (Wc / (W1 + W2 - Wc)).fill_nan(None).alias('ni_idf_jac'), (Wc / W1).fill_nan(None).alias('ni_idf_c1'), (Wc / W2).fill_nan(None).alias('ni_idf_c2'),
        (W1 - Wc).alias('ni_idf_miss1'), (W2 - Wc).alias('ni_idf_miss2'), pl.col('wnN_1').alias('ni_n1'), pl.col('wnN_2').alias('ni_n2'), pl.col('nNc').alias('ni_nc'),
        (Ac / (A1 + A2 - Ac)).fill_nan(None).alias('ai_idf_jac'), (Ac / A1).fill_nan(None).alias('ai_idf_c1'), (Ac / A2).fill_nan(None).alias('ai_idf_c2'),
        (A1 - Ac).alias('ai_idf_miss1'), (A2 - Ac).alias('ai_idf_miss2'), pl.col('waN_1').alias('ai_n1'), pl.col('waN_2').alias('ai_n2'), pl.col('aNc').alias('ai_nc'),
        pl.col('nm_cnt_s1_of2').fill_null(0), pl.col('nm_cnt_src_of2').fill_null(1), pl.col('nm_cnt_s1_of1').fill_null(1),
        pl.col('amb_a').fill_null(1), pl.col('amb_b').fill_null(0))
    X = X.join(F, on='i', how='left')
    X = X.with_columns(
        pl.when(pl.col('_hasalt')).then(pl.col('n_alt_tset')).otherwise(nan).alias('n_alt_tset'),
        pl.when(pl.col('_hasalt')).then(pl.col('n_alt_ratio')).otherwise(nan).alias('n_alt_ratio'),
        pl.when(pl.col('_hasdom')).then(pl.col('dom_prefix')).otherwise(nan).alias('dom_prefix'),
        pl.when(pl.col('_hasdom')).then(pl.col('dom_partial')).otherwise(nan).alias('dom_partial'),
        *[pl.when(pl.col('_hasa2')).then(pl.col(c)).otherwise(nan).alias(c) for c in ('a_ratio', 'a_tset', 'a_tsort', 'a_partial')],
        pl.when(pl.col('_both')).then(pl.col('hn_lev')).otherwise(nan).alias('hn_lev'),
        pl.when(pl.col('_bst')).then(pl.col('st_tset')).otherwise(nan).alias('st_tset'),
        pl.when(pl.col('_bst')).then(pl.col('st_ratio')).otherwise(nan).alias('st_ratio'))
    drop = ['i', '_hasalt', '_hasdom', '_hasa2', '_both', '_bst', 'wnW_1', 'wnW_2', 'wnM_1', 'wnN_1', 'wnN_2', 'waW_1', 'waW_2', 'waN_1', 'waN_2',
            'hnc_1', 'hnc_2', 'hc1', 'hc2', 'hnc_21', 'nWc', 'nMc', 'nNc', 'aWc', 'aMc', 'aNc']
    return X.drop(drop).with_columns(pl.col(pl.Float64).cast(pl.Float32), pl.col('src').cast(pl.Int8))


def pair_features(cand, ctx, sides=None, CH=500_000, log=None):
    """cand: (id1, id2, src, ...extra columns kept). sides: optional preloaded {1: S1side, 2: .., 3: ..}"""
    out = []
    S1 = sides[1] if sides else side(ctx, 1, cand.select(pl.col('id1').unique().alias('id')))
    keep = [c for c in cand.columns if c not in KY]
    for s in (2, 3):
        cs = cand.filter(pl.col('src') == s)
        if cs.height == 0: continue
        S = sides[s] if sides else side(ctx, s, cs.select(pl.col('id2').unique().alias('id')))
        for off in range(0, cs.height, CH):
            ch = cs.slice(off, CH)
            pr = ch.join(S1.rename({c: c + '_1' for c in S1.columns if c != 'id'}), left_on='id1', right_on='id', how='left') \
                   .join(S.rename({c: c + '_2' for c in S.columns if c != 'id'}), left_on='id2', right_on='id', how='left')
            r = compute(pr, ctx, s)
            if keep: r = r.join(ch.select(*KY, *keep), on=KY, how='left')
            out.append(r); del pr
            if log: log(f'src {s} {off + ch.height}/{cs.height}')
        if not sides: del S
    return pl.concat(out)
