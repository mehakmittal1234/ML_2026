"""Apply the near-number cell shares (s35) to the calibrated decisions, cell by cell, keeping only cells whose removal raises
expected F0.5. Expected F is computed on the affected S1 entities only: pairs of the tested cell get q = the cell's estimated true
share, every other best-owner pair keeps its calibrated p (truth ~ Bernoulli(q)); gain is reported as macro F over ALL test S1.
usage: python s36_apply_cells.py <raw test tag> <calibrated test tag> <thr> <out tag> [countries, default US,France]
  -> scores/<out>/test_s2_c0.parquet (calibrated scores, pairs of the chosen cells set to p = 0) and analysis/cells_applied.parquet"""
import sys
from common import *
from s34_census import typed
from s35_near_cells import cells
from s32_neighbor_rule import expected_f
raw, cal, thr, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
CTRY = sys.argv[5].split(',') if len(sys.argv) > 5 else ['US', 'France']
CK = ['ctry', 'anchor', 'rep_src', 'rep_oth']
shares = pl.read_parquet(wp('analysis', 'near_cells.parquet')).select(*CK, 'est_share')
tc = pl.read_parquet(wp('data', 'test_s1.parquet'), columns=['id', 'country']).select(pl.col('id').alias('id1'), pl.col('country').alias('ctry'))
SR = pl.read_parquet(wp('scores', raw, 'test_s2_c0.parquet'), columns=[*KY, 'p'])
NT = cells('test', SR, typed('test', SR)).join(tc, on='id1').join(shares, on=CK).filter(pl.col('ctry').is_in(CTRY)).select(*KY, *CK, 'est_share')
del SR
SC = pl.read_parquet(wp('scores', cal, 'test_s2_c0.parquet'), columns=[*KY, 'p'])
B = SC.filter(pl.col('p') >= 0.01).sort('p', descending=True).unique(['id2', 'src'], keep='first')
B = B.with_columns((pl.col('p') >= thr).alias('sel0')).join(NT, on=KY, how='left')
N = tc.height
cand = B.filter(pl.col('sel0') & pl.col('est_share').is_not_null())
print(f'accepted pairs in near-number cells ({",".join(CTRY)}): {cand.height}')
chosen, total = [], 0.0
for key, g in cand.group_by(CK, maintain_order=True):
    k = dict(zip(CK, key)); share = g['est_share'][0]
    ents = g.select('id1').unique()
    Bc = B.join(ents, on='id1', how='semi')
    incell = pl.all_horizontal([pl.col(c) == v for c, v in k.items()]).fill_null(False)
    Bc = Bc.with_columns(pl.when(incell).then(pl.lit(share)).otherwise(pl.col('p')).alias('q'), (pl.col('sel0') & ~incell).alias('sel1'))
    e0 = expected_f(Bc, 'sel0', 'q', ents); e1 = expected_f(Bc, 'sel1', 'q', ents)
    gain = float((e1 - e0).sum()) / N
    ok = gain > 0
    print(f'  {k}: accepted {g.height:6d}, est. true share {share:.3f}, entities {ents.height:6d}, expected macro gain {gain:+.6f} {"<- apply" if ok else ""}', flush=True)
    if ok: chosen.append(k); total += gain
print(f'total expected macro gain (full test mix, before transfer discount): {total:+.6f}')
if chosen:
    ch = pl.DataFrame(chosen)
    drop = cand.join(ch, on=CK, how='semi').select(KY)
    full = pl.read_parquet(wp('scores', cal, 'test_s2_c0.parquet')).join(drop.with_columns(pl.lit(True).alias('_d')), on=KY, how='left')
    full = full.with_columns(pl.when(pl.col('_d')).then(pl.lit(0.0, pl.Float32)).otherwise(pl.col('p')).alias('p')).drop('_d')
    os.makedirs(wp('scores', out), exist_ok=True); full.write_parquet(wp('scores', out, 'test_s2_c0.parquet'))
    cand.join(ch, on=CK, how='semi').write_parquet(wp('analysis', 'cells_applied.parquet'))
    print(f'written scores/{out}: {drop.height} accepted pairs removed; matches {decide(full, thr).height} (base {decide(SC, thr).height})')
