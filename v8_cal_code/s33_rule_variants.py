"""Sharpen the same-source neighbour rule (s32). Per-copy house-number jitter touches ONE record; a neighbour business's source version
is shared by all its copies in that source. Variants on top of the s32 flag:
  B: x's (core name, house number) is repeated by another record in x's source
  C: ... repeated by another record in any source
  D: B and the repeat is itself accepted for the same S1 entity (the entity would hold two same-source versions)
Validation: flagged accepted pairs and their true share (fold 0). Test: counts; count-invariant true share = val TP per S1 x n_test / flagged.
usage: python s33_rule_variants.py <val tag> <test tag> <thr>"""
import sys
from common import *
from s32_neighbor_rule import flag

def support_cols(split, D):
    R = pl.concat([pl.read_parquet(wp('norm2', f'{split}_s{s}.parquet'), columns=['id', 'ctry', 'nc', 'dz']).select(
        pl.col('id').alias('id2'), pl.lit(s, pl.Int8).alias('src'), 'ctry', 'nc', pl.col('dz').str.split(' ').list.first().fill_null('').alias('h')) for s in (2, 3)])
    R = R.filter(pl.col('h') != '').with_columns((pl.len().over('ctry', 'nc', 'h', 'src') - 1).alias('sup_src'), (pl.len().over('ctry', 'nc', 'h') - 1).alias('sup_any'))
    X = D.join(R.select('id2', 'src', 'ctry', 'nc', 'h', 'sup_src', 'sup_any'), on=['id2', 'src'], how='left').with_columns(pl.col('sup_src', 'sup_any').fill_null(0))
    # D: another ACCEPTED record of the same S1 entity in the same source with the same (nc, h)
    acc = X.select('id1', 'src', 'nc', 'h', 'id2')
    twin = X.filter(pl.col('flag')).select(*KY, 'nc', 'h').join(acc.rename({'id2': 'o'}), on=['id1', 'src', 'nc', 'h']).filter(pl.col('o') != pl.col('id2')) \
            .select(KY).unique().with_columns(pl.lit(True).alias('acc_rep'))
    return X.join(twin, on=KY, how='left').with_columns(pl.col('acc_rep').fill_null(False))

vtag, ttag, thr = sys.argv[1], sys.argv[2], float(sys.argv[3])
gt = load_gt_pairs(); v0 = pl.read_parquet(wp('data', 'train_s1ids.parquet')).filter(fold_expr('id1') == 0)
SV = pl.read_parquet(wp('scores', vtag, 'train_s2_c0.parquet'), columns=[*KY, 'p'])
DV = support_cols('train', flag('train', decide(SV, thr).join(SV, on=KY)))
DV = DV.join(v0.select('id1', pl.col('ctry').alias('c1')), on='id1').join(gt.with_columns(pl.lit(1).alias('y')), on=KY, how='left').with_columns(pl.col('y').fill_null(0))
V = {'A': pl.col('flag'), 'B': pl.col('flag') & (pl.col('sup_src') >= 1), 'C': pl.col('flag') & (pl.col('sup_any') >= 1), 'D': pl.col('flag') & pl.col('acc_rep')}
nv = dict(v0.group_by('ctry').len().rows())
rate = {}
for k, e in V.items():
    d = DV.filter(e)
    rate[k] = {c: d.filter((pl.col('c1') == c) & (pl.col('y') == 1)).height / nv[c] for c in ('US', 'India')}
    print(f'VAL {k}: flagged {d.height:5d}, true share {d["y"].mean():.3f}, false {int((1 - d["y"]).sum())}  | TP per S1 US {rate[k]["US"]:.5f} India {rate[k]["India"]:.5f}')
del SV, DV
ST = pl.read_parquet(wp('scores', ttag, 'test_s2_c0.parquet'), columns=[*KY, 'p'])
tc = pl.read_parquet(wp('data', 'test_s1.parquet'), columns=['id', 'country']).select(pl.col('id').alias('id1'), pl.col('country').alias('c1'))
DT = support_cols('test', flag('test', decide(ST, thr).join(ST, on=KY))).join(tc, on='id1')
n1 = dict(tc.group_by('c1').len().rows())
for k, e in V.items():
    d = DT.filter(e)
    parts = []
    for c in ('US', 'India', 'France'):
        m = d.filter(pl.col('c1') == c).height; r = rate[k]['US' if c == 'France' else c]
        parts.append(f'{c} {m:6d} (est. true share {min(1, r * n1[c] / max(1, m)):.2f}, est. false {max(0, m - r * n1[c]):7.0f})')
    print(f'TEST {k}: ' + ' | '.join(parts))
DT.filter(pl.col('flag')).select(*KY, 'p', 'c1', 'sup_src', 'sup_any', 'acc_rep').write_parquet(wp('analysis', 'rule_flags_test.parquet'))
