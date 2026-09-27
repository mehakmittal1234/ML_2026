"""Hybrid shift correction: count invariance sets HOW MANY true pairs a cell holds, the density-ratio factor (s53) orders WHICH pairs.
Cells: fine = (country, name_rel, hn_rel, anchor, g_src>0, g_oth>0) over the stage-2 pair set (f50), falling back to (country, name_rel,
hn_rel) when the fine cell has < MINTRUE fold-0 true pairs. Expected true pairs of a test cell = fold-0 true pairs per S1 x test S1
(all candidate pairs, so S1 twin composition does not enter). A cell is corrected only if its share (expected true / sum of test p)
is below SHARE_MAX; there q_i = min(p_i, raw_i * lam_c * min(1, fac_i)) with lam_c solving sum q = expected true. Others keep p.
usage: python s54_hybrid.py <raw test tag> <calibrated test tag> <factor tag> <thr> <out tag> [SHARE_MAX]"""
import sys, glob
from common import *
from s32_neighbor_rule import expected_f
rtag, ctag, ftag, thr, out = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5]
SHARE_MAX = float(sys.argv[6]) if len(sys.argv) > 6 else 0.9
MINTRUE = 200
FINE = ['ctry', 'name_rel', 'hn_rel', 'anchor']; COARSE = ['ctry', 'name_rel', 'hn_rel']
cols = [*KY, 'ctry', 'p', 'name_rel', 'hn_rel', 'anchor', 'g_src', 'g_oth']
cell = lambda d: d.with_columns((pl.col('g_src') > 0).alias('gs'), (pl.col('g_oth') > 0).alias('go'))
V = cell(pl.scan_parquet(sorted(glob.glob(wp('feat', 'f50', 'train_c*.parquet')))).filter(pl.col('fold') == 0).select(*cols, 'y').collect())
T = cell(pl.scan_parquet(sorted(glob.glob(wp('feat', 'f50', 'test_c*.parquet')))).select(cols).collect()).filter(pl.col('ctry') != 'France')
nv = dict(pl.read_parquet(wp('data', 'train_s1ids.parquet')).filter(fold_expr('id1') == 0).group_by('ctry').len().rows())
nt = dict(pl.read_parquet(wp('data', 'test_s1.parquet'), columns=['country']).group_by('country').len().rows())
def expected(keys):
    t = V.group_by(keys).agg(pl.col('y').sum().alias('vt'))
    return t.with_columns((pl.col('vt') / pl.col('ctry').replace_strict(nv, return_dtype=pl.Float64) * pl.col('ctry').replace_strict(nt, return_dtype=pl.Float64)).alias('exp_true'))
EF, EC = expected(FINE).rename({'vt': 'vt_f', 'exp_true': 'exp_f'}), expected(COARSE).rename({'vt': 'vt_c', 'exp_true': 'exp_c'})
T = T.join(EF, on=FINE, how='left').join(EC, on=COARSE, how='left').with_columns(pl.col('vt_f', 'vt_c', 'exp_c').fill_null(0))
T = T.with_columns(pl.when(pl.col('vt_f') >= MINTRUE).then(pl.concat_str([pl.col(c).cast(pl.Utf8) for c in FINE], separator='|'))
                     .otherwise(pl.concat_str([pl.col(c).cast(pl.Utf8) for c in COARSE], separator='|')).alias('cell'),
                   pl.when(pl.col('vt_f') >= MINTRUE).then(pl.col('exp_f')).otherwise(pl.col('exp_c')).alias('exp_cell'))
# a coarse cell's expected count must be shared by its non-fine members: subtract the fine cells that stand on their own
fine_in_coarse = T.filter(pl.col('vt_f') >= MINTRUE).select(*COARSE, 'cell', 'exp_f').unique().group_by(COARSE).agg(pl.col('exp_f').sum().alias('_fin'))
T = T.join(fine_in_coarse, on=COARSE, how='left').with_columns(pl.when(pl.col('vt_f') >= MINTRUE).then(pl.col('exp_cell')).otherwise(pl.col('exp_c') - pl.col('_fin').fill_null(0)).clip(0).alias('exp_cell')).drop('_fin')
F = pl.read_parquet(wp('scores', ftag, 'test_factor.parquet'))
C = pl.read_parquet(wp('scores', ctag, 'test_s2_c0.parquet'), columns=[*KY, 'p']).rename({'p': 'pc'})
T = T.join(F, on=KY, how='left').with_columns(pl.col('fac').fill_null(1.0)).join(C, on=KY, how='left')
T = T.with_columns(pl.when(pl.col('vt_f') >= MINTRUE).then(pl.col('vt_f')).otherwise(pl.col('vt_c')).alias('vt_cell'))
G = T.group_by('cell').agg(pl.col('exp_cell').first(), pl.col('vt_cell').first(), pl.col('p').sum().alias('sum_p'), pl.len().alias('n'), pl.col('ctry').first())
# significance: excess over the count-invariant expectation, in Poisson standard errors of the fold-0 true count behind it
G = G.with_columns((pl.col('exp_cell') / pl.col('sum_p')).alias('share'),
                   ((pl.col('sum_p') - pl.col('exp_cell')) / (pl.col('exp_cell') / pl.col('vt_cell').clip(1).sqrt()).clip(1e-9)).alias('z'))
UP = '--up' in sys.argv
fix = G.filter(((pl.col('share') < SHARE_MAX) & (pl.col('sum_p') >= 200) & (pl.col('z') >= 4)) |
               (pl.lit(UP) & (pl.col('share') > 1.0) & (pl.col('sum_p') >= 200) & (pl.col('z') <= -4)))
print(f'cells {G.height}; corrected {fix.height} (share < {SHARE_MAX}, sum p >= 50): pairs {fix["n"].sum()}, expected true {fix["exp_cell"].sum():.0f} vs sum p {fix["sum_p"].sum():.0f}')
pl.Config.set_tbl_rows(25); pl.Config.set_tbl_width_chars(200); pl.Config.set_float_precision(3)
print(fix.sort('sum_p', descending=True).with_columns((pl.col('sum_p') - pl.col('exp_cell')).alias('excess')).select('cell', 'n', 'sum_p', 'exp_cell', 'share', 'z', 'excess').head(25))
Q = []
for r in fix.iter_rows(named=True):
    d = T.filter(pl.col('cell') == r['cell']); pr = d['p'].to_numpy()
    up = r['share'] > 1.0
    base = pr if up else (d['p'] * d['fac'].clip(0, 1)).to_numpy()
    f = (lambda lam: np.minimum(1.0, pr * lam)) if up else (lambda lam: np.minimum(pr, base * lam))
    lo, hi = 0.0, 50.0
    for _ in range(60):
        lam = (lo + hi) / 2
        if f(lam).sum() > r['exp_cell']: hi = lam
        else: lo = lam
    qq = f(lam)
    Q.append(d.select(KY).with_columns(pl.Series('q', qq).cast(pl.Float32), pl.Series('qr', qq / np.maximum(pr, 1e-9)).cast(pl.Float32)))
Q = pl.concat(Q)
full = pl.read_parquet(wp('scores', ctag, 'test_s2_c0.parquet')).join(Q, on=KY, how='left')
# corrected pairs: calibrated score scaled by the same within-cell ratio q/raw (keeps the calibrated scale for the threshold)
full = full.with_columns(pl.when(pl.col('qr').is_not_null()).then((pl.col('p') * pl.col('qr')).clip(0, 1)).otherwise(pl.col('p')).cast(pl.Float32).alias('p')).drop('q', 'qr')
os.makedirs(wp('scores', out), exist_ok=True); full.write_parquet(wp('scores', out, 'test_s2_c0.parquet'))
C0 = pl.read_parquet(wp('scores', ctag, 'test_s2_c0.parquet'), columns=[*KY, 'p'])
M0, M1 = decide(C0, thr), decide(full, thr)
tc = pl.read_parquet(wp('data', 'test_s1.parquet'), columns=['id', 'country']).rename({'id': 'id1', 'country': 'ctry'})
print(f'upward cells: {fix.filter(pl.col("share") > 1).height}', fix.filter(pl.col('share') > 1).select('cell', 'sum_p', 'exp_cell', 'share', 'z').rows()[:12])
print(f'matches {M0.height} -> {M1.height}; removed {M0.join(M1, on=KY, how="anti").height}, added {M1.join(M0, on=KY, how="anti").height};',
      'removed by country', M0.join(M1, on=KY, how='anti').join(tc, on='id1').group_by('ctry').len().sort('ctry').rows())
# expected gain on affected entities, truth ~ corrected scores (self-consistent estimate)
B = full.select(*KY, 'p').filter(pl.col('p') >= 0.01).sort('p', descending=True).unique(['id2', 'src'], keep='first')
B = B.join(C0.rename({'p': 'p0'}), on=KY, how='left').with_columns((pl.col('p0') >= thr).alias('sel0'), (pl.col('p') >= thr).alias('sel1'))
ents = pl.concat([M0.join(M1, on=KY, how='anti'), M1.join(M0, on=KY, how='anti')]).select('id1').unique()
Bc = B.join(ents, on='id1', how='semi')
g = float((expected_f(Bc, 'sel1', 'p', ents) - expected_f(Bc, 'sel0', 'p', ents)).sum()) / 1732544
print(f'expected macro gain vs base (truth ~ corrected scores): {g:+.6f}')
print('written scores/' + out)
