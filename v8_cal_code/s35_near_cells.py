"""Label-free true share inside the 'same core name, nearby house number' type, by generator-derived cells.
Count invariance: true pairs of a cell per S1 entity are a generator constant (checked on India in s33/s34), so on test
  est_true_share(cell) = validation true pairs per S1 (cell) / test pairs per S1 (cell).
Cells (for pair A-x, x = best-owner record of type eq_or_order + hn_near, raw p >= PMIN):
  anchor : A has a confident (p >= 0.9) best-owner copy in x's SOURCE whose house number equals A's S1 number
  rep_src: x's (core name, house number) is repeated by another record in x's source
  rep_oth: ... by a record in the other source
usage: python s35_near_cells.py <val tag> <test tag> [out tag]
  out tag: writes scores/<out>/test_s2_c0.parquet = test scores with p of every typed test pair in a cell whose estimated true share
  is below SHARE_MIN replaced by that share (other pairs untouched), for the usual s13/s22 writers."""
import sys
from common import *
from s34_census import typed
PMIN, SHARE_MIN = 0.3, 0.6


def cells(split, S, X):
    """X: typed best-owner pairs (s34.typed) -> eq_or_order/hn_near rows with the three cell flags"""
    N = X.filter((pl.col('name') == 'eq_or_order') & (pl.col('addr') == 'hn_near') & (pl.col('p') >= PMIN)).select(*KY, 'p')
    R = pl.concat([pl.read_parquet(wp('norm2', f'{split}_s{s}.parquet'), columns=['id', 'ctry', 'nc', 'dz']).select(
        pl.col('id').alias('id2'), pl.lit(s, pl.Int8).alias('src'), 'ctry', 'nc', pl.col('dz').str.split(' ').list.first().fill_null('').alias('h')) for s in (2, 3)])
    R = R.filter(pl.col('h') != '').with_columns(pl.len().over('ctry', 'nc', 'h', 'src').alias('_ns'), pl.len().over('ctry', 'nc', 'h').alias('_na'))
    N = N.join(R.select('id2', 'src', 'nc', 'h', (pl.col('_ns') > 1).alias('rep_src'), ((pl.col('_na') - pl.col('_ns')) > 0).alias('rep_oth')), on=['id2', 'src'], how='left')
    H1 = pl.read_parquet(wp('norm2', f'{split}_s1.parquet'), columns=['id', 'dz']).select(pl.col('id').alias('id1'), pl.col('dz').str.split(' ').list.first().fill_null('').alias('h1'))
    B = S.sort('p', descending=True).unique(['id2', 'src'], keep='first').filter(pl.col('p') >= 0.9).join(N.select('id1').unique(), on='id1', how='semi')
    B = B.join(R.select('id2', 'src', 'h'), on=['id2', 'src']).join(H1, on='id1').filter(pl.col('h') == pl.col('h1')).select('id1', 'src', pl.col('id2').alias('yid'))
    anc = N.select(KY).join(B, on=['id1', 'src']).filter(pl.col('yid') != pl.col('id2')).select(KY).unique().with_columns(pl.lit(True).alias('anchor'))
    return N.join(anc, on=KY, how='left').with_columns(pl.col('anchor', 'rep_src', 'rep_oth').fill_null(False))


if __name__ == '__main__':
    vtag, ttag = sys.argv[1], sys.argv[2]; out = sys.argv[3] if len(sys.argv) > 3 else None
    gt = load_gt_pairs(); v0 = pl.read_parquet(wp('data', 'train_s1ids.parquet')).filter(fold_expr('id1') == 0)
    SV = pl.read_parquet(wp('scores', vtag, 'train_s2_c0.parquet'), columns=[*KY, 'p'])
    NV = cells('train', SV, typed('train', SV)).join(v0, on='id1').join(gt.with_columns(pl.lit(1).alias('y')), on=KY, how='left').with_columns(pl.col('y').fill_null(0))
    del SV
    nv = dict(v0.group_by('ctry').len().rows())
    CV = NV.group_by('ctry', 'anchor', 'rep_src', 'rep_oth').agg(pl.len().alias('v_n'), pl.col('y').sum().alias('v_true'))
    CV = CV.with_columns((pl.col('v_true') / pl.col('ctry').replace_strict(nv, return_dtype=pl.Float64)).alias('v_true_per'), (pl.col('v_true') / pl.col('v_n')).alias('v_share'))
    ST = pl.read_parquet(wp('scores', ttag, 'test_s2_c0.parquet'), columns=[*KY, 'p'])
    tc = pl.read_parquet(wp('data', 'test_s1.parquet'), columns=['id', 'country']).select(pl.col('id').alias('id1'), pl.col('country').alias('ctry'))
    NT = cells('test', ST, typed('test', ST)).join(tc, on='id1')
    nt = dict(tc.group_by('ctry').len().rows())
    CT = NT.group_by('ctry', 'anchor', 'rep_src', 'rep_oth').agg(pl.len().alias('t_n'), pl.col('p').mean().alias('t_mp'), (pl.col('p') >= 0.7).mean().alias('t_acc'))
    CT = CT.with_columns((pl.col('t_n') / pl.col('ctry').replace_strict(nt, return_dtype=pl.Float64)).alias('t_per'))
    ref = pl.concat([CV, CV.filter(pl.col('ctry') == 'US').with_columns(pl.lit('France').alias('ctry'))])
    C = CT.join(ref, on=['ctry', 'anchor', 'rep_src', 'rep_oth'], how='left').with_columns(pl.col('v_true_per').fill_null(0.0))
    C = C.with_columns((pl.col('v_true_per') / pl.col('t_per')).clip(0, 1).alias('est_share')).sort('ctry', 'anchor', 'rep_src', 'rep_oth')
    pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(220); pl.Config.set_float_precision(4)
    print(C.select('ctry', 'anchor', 'rep_src', 'rep_oth', 'v_n', 'v_share', 'v_true_per', 't_n', 't_per', 't_mp', 't_acc', 'est_share'))
    C.write_parquet(wp('analysis', 'near_cells.parquet'))
    if out:
        low = C.filter((pl.col('est_share') < SHARE_MIN) & (pl.col('ctry') != 'India')).select('ctry', 'anchor', 'rep_src', 'rep_oth', 'est_share')
        fix = NT.join(low, on=['ctry', 'anchor', 'rep_src', 'rep_oth']).select(*KY, pl.col('est_share').cast(pl.Float32).alias('_q'))
        full = pl.read_parquet(wp('scores', ttag, 'test_s2_c0.parquet')).join(fix, on=KY, how='left')
        full = full.with_columns(pl.when(pl.col('_q').is_not_null()).then(pl.min_horizontal('p', '_q')).otherwise(pl.col('p')).alias('p')).drop('_q')
        os.makedirs(wp('scores', out), exist_ok=True); full.write_parquet(wp('scores', out, 'test_s2_c0.parquet'))
        print(f'written scores/{out}: {fix.height} test pairs capped at their cell share (cells with share < {SHARE_MIN}, US/France)')
