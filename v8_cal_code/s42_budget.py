"""Budget for the test-shifted nearby-number types: what do the rules capture, and what is the most any rule or retrained specialist
could capture from these types? Monte Carlo over the affected S1 entities: every best-owner pair is true with probability q (cell pairs:
count-invariant cell share; other pairs: calibrated p), each entity also misses Poisson(LAM) true copies. Decisions scored with the
official per-entity F0.5: base = calibrated threshold decisions; rules = base minus the rule cells; oracle = base minus exactly the FALSE
accepted pairs of every shifted cell (cells whose share is below the model's belief). Gains are macro F over all test S1 entities.
usage: python s42_budget.py <val tag> <raw test tag> <calibrated test tag> <thr>"""
import sys
from common import *
from s34_census import typed
from s35_near_cells import cells
NAMES, ADDRS = ('eq_or_order', 'add'), ('hn_near', 'hn_far', 'hn_trunc')
CK = ['name', 'addr', 'anchor', 'rep_src', 'rep_oth']
LAM, DRAWS, NTOT = 0.05, 40, 1732544
vtag, rtag, ctag, thr = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
gt = load_gt_pairs(); v0 = pl.read_parquet(wp('data', 'train_s1ids.parquet')).filter(fold_expr('id1') == 0)
SV = pl.read_parquet(wp('scores', vtag, 'train_s2_c0.parquet'), columns=[*KY, 'p'])
NV = cells('train', SV, typed('train', SV), NAMES, ADDRS).join(v0, on='id1').join(gt.with_columns(pl.lit(1).alias('y')), on=KY, how='left').with_columns(pl.col('y').fill_null(0))
del SV
nv = dict(v0.group_by('ctry').len().rows())
CV = NV.group_by('ctry', *CK).agg(pl.col('y').sum().alias('v_true')).with_columns((pl.col('v_true') / pl.col('ctry').replace_strict(nv, return_dtype=pl.Float64)).alias('v_true_per'))
CV = pl.concat([CV, CV.filter(pl.col('ctry') == 'US').with_columns(pl.lit('France').alias('ctry'))])
tc = pl.read_parquet(wp('data', 'test_s1.parquet'), columns=['id', 'country']).select(pl.col('id').alias('id1'), pl.col('country').alias('ctry'))
nt = dict(tc.group_by('ctry').len().rows())
SR = pl.read_parquet(wp('scores', rtag, 'test_s2_c0.parquet'), columns=[*KY, 'p'])
NT = cells('test', SR, typed('test', SR), NAMES, ADDRS).join(tc, on='id1')
del SR
C = NT.group_by('ctry', *CK).agg(pl.len().alias('t_n'), pl.col('p').mean().alias('t_mp')).with_columns((pl.col('t_n') / pl.col('ctry').replace_strict(nt, return_dtype=pl.Float64)).alias('t_per'))
C = C.join(CV, on=['ctry', *CK], how='left').with_columns(pl.col('v_true_per').fill_null(0.0)).with_columns((pl.col('v_true_per') / pl.col('t_per')).clip(0, 1).alias('share'))
C = C.with_columns((pl.col('share') < pl.col('t_mp') - 0.05).alias('shifted'))
# rule cells = what v11_nb (US/France eq/hn_near, share <= 0.45) plus the India anchor cells (s41) remove
rule = ((pl.col('name') == 'eq_or_order') & (pl.col('addr') == 'hn_near') & pl.col('ctry').is_in(['US', 'France']) & (pl.col('share') <= 0.45)) | \
       ((pl.col('ctry') == 'India') & (pl.col('name') == 'add') & pl.col('anchor') & ~pl.col('rep_src') & ~pl.col('rep_oth'))
C = C.with_columns(rule.alias('rule'))
SC = pl.read_parquet(wp('scores', ctag, 'test_s2_c0.parquet'), columns=[*KY, 'p'])
B = SC.filter(pl.col('p') >= 0.01).sort('p', descending=True).unique(['id2', 'src'], keep='first').with_columns((pl.col('p') >= thr).alias('acc'))
B = B.join(NT.select(*KY, 'ctry', *CK), on=KY, how='left').join(C.select('ctry', *CK, 'share', 'shifted', 'rule'), on=['ctry', *CK], how='left') \
     .with_columns(pl.col('shifted', 'rule').fill_null(False))
aff = B.filter(pl.col('acc') & pl.col('shifted')).select('id1').unique()
B = B.join(aff, on='id1', how='semi').join(tc, on='id1', how='left', suffix='_1').with_columns(pl.coalesce('ctry', 'ctry_1').alias('ctry')).drop('ctry_1')
B = B.with_columns(pl.when(pl.col('shifted')).then(pl.col('share')).otherwise(pl.col('p')).alias('q'))
print(f'affected entities {aff.height}; accepted pairs in shifted cells {B.filter(pl.col("acc") & pl.col("shifted")).height}, of which rule cells {B.filter(pl.col("acc") & pl.col("rule")).height}')
e_idx = B.select('id1').unique().with_row_index('e'); B = B.join(e_idx, on='id1')
ent_c = B.group_by('e').agg(pl.col('ctry').first()).sort('e')['ctry'].to_numpy()
E = e_idx.height; e = B['e'].to_numpy(); q = B['q'].to_numpy(); acc = B['acc'].to_numpy(); shf = B['shifted'].to_numpy(); rl = B['rule'].to_numpy()
rng = np.random.default_rng(0)
res = {k: np.zeros(E) for k in ('base', 'rules', 'oracle')}
for d in range(DRAWS):
    y = rng.random(len(q)) < q
    nt_ = np.bincount(e, weights=y, minlength=E) + rng.poisson(LAM, E)
    for k, sel in (('base', acc), ('rules', acc & ~rl), ('oracle', acc & ~(shf & ~y))):
        npred = np.bincount(e, weights=sel, minlength=E); tp = np.bincount(e, weights=sel & y, minlength=E)
        P = np.where(npred > 0, tp / np.maximum(1, npred), 0); R = np.where(nt_ > 0, tp / np.maximum(1, nt_), 0)
        F = np.where(P + R > 0, 1.25 * P * R / np.maximum(1e-12, 0.25 * P + R), 0); F = np.where((nt_ == 0) & (npred == 0), 1.0, F)
        res[k] += F / DRAWS
for c in ('US', 'India', 'France', None):
    m = np.ones(E, bool) if c is None else ent_c == c
    b = res['base'][m].sum()
    print(f'{c or "ALL":6s}: rules {(res["rules"][m].sum() - b) / NTOT:+.6f} | oracle (all false pairs of shifted cells removed) {(res["oracle"][m].sum() - b) / NTOT:+.6f}  (macro over all test S1)')
C.write_parquet(wp('analysis', 'budget_cells.parquet'))
