"""Distractor-type census: a label-free decomposition of test false merges by pair type.
The copy generator is the same in train and test (US/India copy statistics are identical; India test matches validation on every
count-invariance check), so TRUE best-owner pairs of a given type per S1 entity are a generator constant. For each cell
(country, name relation, address relation, score band) we compare validation true pairs per S1 entity with test pairs per S1 entity;
on test the excess (test - validation true) estimates distractor pairs of that cell. Cells above the decision threshold give the
estimated excess false merges on test that validation never sees.
  name relation (core names, legal forms dropped): eq | order (same token set) | add (one side = other + tokens) | sub (same size,
                 one token differs) | partial | none
  addr relation : noaddr (record has none) | hn_eq (same first house number) | hn_near (non-truncation difference <= 50) |
                  hn_trunc (prefix/suffix) | hn_miss (one side lacks a number) | hn_far
usage: python s34_census.py <val score tag> <test score tag>   -> analysis/census.parquet + printed top cells"""
import sys
from common import *
BANDS = [0.0, 0.3, 0.5, 0.7, 0.9, 0.99, 1.01]


def typed(split, S):
    B = S.sort('p', descending=True).unique(['id2', 'src'], keep='first').filter(pl.col('p') >= 0.05)
    f = lambda s: pl.read_parquet(wp('norm2', f'{split}_s{s}.parquet'), columns=['id', 'nc', 'at', 'dz']).select(
        'id', pl.col('nc').str.split(' ').list.eval(pl.element().filter(pl.element() != '')).list.unique().alias('t'),
        (pl.col('at') == '').alias('na'), pl.col('dz').str.split(' ').list.first().fill_null('').alias('h'))
    A = f(1).rename({'id': 'id1', 't': 't1', 'na': 'na1', 'h': 'h1'})
    R = pl.concat([f(s).rename({'id': 'id2'}).with_columns(pl.lit(s, pl.Int8).alias('src')) for s in (2, 3)]).join(B.select('id2', 'src'), on=['id2', 'src'], how='semi')
    X = B.join(A.join(B.select('id1').unique(), on='id1', how='semi'), on='id1').join(R, on=['id2', 'src'])
    i = pl.col('t1').list.set_intersection('t').list.len(); n1 = pl.col('t1').list.len(); n2 = pl.col('t').list.len()
    name = (pl.when((i == n1) & (i == n2)).then(pl.lit('eq_or_order')).when((i == n1) | (i == n2)).then(pl.lit('add'))
            .when((n1 == n2) & (i == n1 - 1)).then(pl.lit('sub')).when(i > 0).then(pl.lit('partial')).otherwise(pl.lit('none')))
    h, a = pl.col('h'), pl.col('h1')
    num = lambda c: pl.col(c).str.slice(0, 9).cast(pl.Int64, strict=False)
    addr = (pl.when(pl.col('na')).then(pl.lit('noaddr')).when((h == '') | (a == '')).then(pl.lit('hn_miss')).when(h == a).then(pl.lit('hn_eq'))
            .when(h.str.starts_with(a) | a.str.starts_with(h) | h.str.ends_with(a) | a.str.ends_with(h)).then(pl.lit('hn_trunc'))
            .when((num('h') - num('h1')).abs() <= 50).then(pl.lit('hn_near')).otherwise(pl.lit('hn_far')))
    band = pl.col('p').cut(BANDS[1:-1], labels=[f'{BANDS[j]:.2f}-{BANDS[j+1]:.2f}' for j in range(len(BANDS) - 1)], left_closed=True)
    return X.select(*KY, 'p', name.alias('name'), addr.alias('addr'), band.cast(pl.Utf8).alias('band'))


if __name__ == '__main__':
    vtag, ttag = sys.argv[1], sys.argv[2]
    gt = load_gt_pairs(); v0 = pl.read_parquet(wp('data', 'train_s1ids.parquet')).filter(fold_expr('id1') == 0)
    XV = typed('train', pl.read_parquet(wp('scores', vtag, 'train_s2_c0.parquet'), columns=[*KY, 'p'])).join(v0, on='id1')
    XV = XV.join(gt.with_columns(pl.lit(1).alias('y')), on=KY, how='left').with_columns(pl.col('y').fill_null(0))
    nv = dict(v0.group_by('ctry').len().rows())
    CV = XV.group_by('ctry', 'name', 'addr', 'band').agg(pl.len().alias('v_n'), pl.col('y').sum().alias('v_true'), pl.col('p').mean().alias('v_mp'))
    CV = CV.with_columns((pl.col('v_n') / pl.col('ctry').replace_strict(nv, return_dtype=pl.Float64)).alias('v_per'),
                         (pl.col('v_true') / pl.col('ctry').replace_strict(nv, return_dtype=pl.Float64)).alias('v_true_per'))
    del XV
    tc = pl.read_parquet(wp('data', 'test_s1.parquet'), columns=['id', 'country']).select(pl.col('id').alias('id1'), pl.col('country').alias('ctry'))
    XT = typed('test', pl.read_parquet(wp('scores', ttag, 'test_s2_c0.parquet'), columns=[*KY, 'p'])).join(tc, on='id1')
    nt = dict(tc.group_by('ctry').len().rows())
    CT = XT.group_by('ctry', 'name', 'addr', 'band').agg(pl.len().alias('t_n'), pl.col('p').mean().alias('t_mp'))
    CT = CT.with_columns((pl.col('t_n') / pl.col('ctry').replace_strict(nt, return_dtype=pl.Float64)).alias('t_per'))
    # France has no validation: use the US cells as its generator reference (copy generator is country-independent)
    CVf = CV.filter(pl.col('ctry') == 'US').with_columns(pl.lit('France').alias('ctry'))
    C = CT.join(pl.concat([CV, CVf]), on=['ctry', 'name', 'addr', 'band'], how='left').with_columns(pl.col('v_per', 'v_true_per').fill_null(0.0))
    C = C.with_columns((pl.col('t_per') - pl.col('v_true_per')).alias('excess_per'),
                       (pl.col('v_true_per') / pl.col('t_per')).clip(0, 1).alias('est_true_share'))
    C.write_parquet(wp('analysis', 'census.parquet'))
    pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(220); pl.Config.set_float_precision(4)
    for c in ('US', 'India', 'France'):
        d = C.filter(pl.col('ctry') == c)
        acc = d.filter(pl.col('band').is_in(['0.70-0.90', '0.90-0.99', '0.99-1.01']))
        print(f'\n== {c}: test best-owner pairs per S1 (p>=0.05) {d["t_per"].sum():.3f} vs validation {d["v_per"].sum():.3f} (true {d["v_true_per"].sum():.3f});'
              f' accepted-band excess over validation-true {acc["excess_per"].sum():.4f} per S1')
        print(acc.sort('excess_per', descending=True).head(12).select('name', 'addr', 'band', 'v_per', 'v_true_per', 't_per', 'excess_per', 'est_true_share', 't_mp'))
