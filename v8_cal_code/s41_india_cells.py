"""India nearby-number types (census excess: eq/hn_near +0.0043, add/hn_near +0.0034, add/hn_far +0.0024, add/hn_trunc +0.0013 per S1).
Cells = (name type, address type, anchor, rep_src, rep_oth) as in s35. Steps:
 (1) estimator noise: fold-0 India entities split in halves A/B; B's true share per cell predicted from A's true pairs per entity
     (count invariance) vs B's actual share -> error of the estimator from sampling alone;
 (2) test: estimated true share per cell = fold-0 true pairs per entity / test pairs per entity; model mean p and acceptance;
 (3) for cells whose share is below the model's belief: expected macro gain of removing their accepted pairs from the calibrated
     test decisions (q = cell share, other pairs keep p) and the ACTUAL fold-0 macro change of the same removal (held-out cost).
usage: python s41_india_cells.py <val tag> <raw test tag> <calibrated test tag> <thr>"""
import sys
from common import *
from s34_census import typed
from s35_near_cells import cells
from s32_neighbor_rule import expected_f
NAMES, ADDRS = ('eq_or_order', 'add'), ('hn_near', 'hn_far', 'hn_trunc')
CK = ['name', 'addr', 'anchor', 'rep_src', 'rep_oth']
vtag, rtag, ctag, thr = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
gt = load_gt_pairs(); v0 = pl.read_parquet(wp('data', 'train_s1ids.parquet')).filter(fold_expr('id1') == 0)
vi = v0.filter(pl.col('ctry') == 'India').with_columns((pl.col('id1').hash(seed=5) % 2).alias('half'))
SV = pl.read_parquet(wp('scores', vtag, 'train_s2_c0.parquet'), columns=[*KY, 'p'])
NV = cells('train', SV, typed('train', SV), NAMES, ADDRS).join(vi, on='id1').join(gt.with_columns(pl.lit(1).alias('y')), on=KY, how='left').with_columns(pl.col('y').fill_null(0))
nh = dict(vi.group_by('half').len().rows())
# (1) estimator noise between halves
H = NV.group_by(*CK, 'half').agg(pl.len().alias('n'), pl.col('y').sum().alias('t'))
A = H.filter(pl.col('half') == 0).select(*CK, (pl.col('t') / nh[0]).alias('tA_per')); B = H.filter(pl.col('half') == 1).select(*CK, (pl.col('n') / nh[1]).alias('nB_per'), (pl.col('t') / pl.col('n')).alias('shareB'), 'n')
E = B.join(A, on=CK, how='left').with_columns(pl.col('tA_per').fill_null(0)).with_columns((pl.col('tA_per') / pl.col('nB_per')).clip(0, 1).alias('estB'))
E = E.filter(pl.col('n') >= 30)
print(f'(1) estimator noise on held-out halves (cells with >= 30 pairs): weighted mean |est - actual| = {((E["estB"] - E["shareB"]).abs() * E["n"]).sum() / E["n"].sum():.4f}, '
      f'max {(E["estB"] - E["shareB"]).abs().max():.3f}; actual shares in these cells: min {E["shareB"].min():.3f} median {E["shareB"].median():.3f}')
# (2) test shares
nI = vi.height
CV = NV.group_by(CK).agg(pl.len().alias('v_n'), pl.col('y').sum().alias('v_true'), pl.col('p').mean().alias('v_mp')).with_columns((pl.col('v_true') / nI).alias('v_true_per'), (pl.col('v_true') / pl.col('v_n')).alias('v_share'))
del SV
tc = pl.read_parquet(wp('data', 'test_s1.parquet'), columns=['id', 'country']).select(pl.col('id').alias('id1'), pl.col('country').alias('ctry')).filter(pl.col('ctry') == 'India')
SR = pl.read_parquet(wp('scores', rtag, 'test_s2_c0.parquet'), columns=[*KY, 'p'])
NT = cells('test', SR, typed('test', SR), NAMES, ADDRS).join(tc, on='id1')
del SR
nT = tc.height
CT = NT.group_by(CK).agg(pl.len().alias('t_n'), pl.col('p').mean().alias('t_mp')).with_columns((pl.col('t_n') / nT).alias('t_per'))
C = CT.join(CV, on=CK, how='left').with_columns(pl.col('v_true_per', 'v_n').fill_null(0)).with_columns((pl.col('v_true_per') / pl.col('t_per')).clip(0, 1).alias('est_share'))
C = C.with_columns((pl.col('t_per') - pl.col('v_true_per')).alias('excess_per')).sort('excess_per', descending=True)
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(230); pl.Config.set_float_precision(4)
print('(2) India cells, largest test excess first:')
print(C.select(*CK, 'v_n', 'v_share', 'v_mp', 't_n', 'excess_per', 'est_share', 't_mp').head(16))
C.write_parquet(wp('analysis', 'india_cells.parquet'))
# (3) candidate cells: estimated share well below the model's belief and enough test mass
cand = C.filter((pl.col('est_share') < pl.col('t_mp') - 0.1) & (pl.col('t_n') >= 300))
SC = pl.read_parquet(wp('scores', ctag, 'test_s2_c0.parquet'), columns=[*KY, 'p'])
Bt = SC.filter(pl.col('p') >= 0.01).sort('p', descending=True).unique(['id2', 'src'], keep='first').join(tc.select('id1'), on='id1', how='semi') \
       .with_columns((pl.col('p') >= thr).alias('sel0')).join(NT.select(*KY, *CK), on=KY, how='left')
SV = pl.read_parquet(wp('scores', vtag, 'train_s2_c0.parquet'), columns=[*KY, 'p']); DV = decide(SV, 0.7); base_v = macro_f05(DV, gt, v0)['macro']
NTOT = 1732544
print(f'(3) {cand.height} candidate cells (est. share < model mean p - 0.1, >= 300 test pairs):')
for r in cand.iter_rows(named=True):
    k = {c: r[c] for c in CK}
    incell = pl.all_horizontal([pl.col(c) == v for c, v in k.items()]).fill_null(False)
    g = Bt.filter(pl.col('sel0') & incell)
    if g.height == 0: continue
    ents = g.select('id1').unique(); Bc = Bt.join(ents, on='id1', how='semi')
    Bc = Bc.with_columns(pl.when(incell).then(pl.lit(r['est_share'])).otherwise(pl.col('p')).alias('q'), (pl.col('sel0') & ~incell).alias('sel1'))
    gain = float((expected_f(Bc, 'sel1', 'q', ents) - expected_f(Bc, 'sel0', 'q', ents)).sum()) / NTOT
    dv = NV.filter(pl.all_horizontal([pl.col(c) == v for c, v in k.items()])).select(KY).join(DV, on=KY, how='semi')
    cost = macro_f05(DV.join(dv, on=KY, how='anti'), gt, v0)['macro'] - base_v
    print(f'  {k}: est share {r["est_share"]:.3f} (model {r["t_mp"]:.3f}), test accepted {g.height:5d} -> expected test gain {gain:+.6f} | fold-0 removal of {dv.height} pairs: {cost:+.6f}', flush=True)
