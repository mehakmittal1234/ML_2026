"""Label-free LB estimate: expected macro-F0.5 change of each candidate vs v9_cal (decisions), under several truth models
(uncorrected v9scal = pessimistic, hyb4, ns1, ens1), on all entities whose decisions differ; France rule pairs included via pairs.parquet.
RESULT (vs v9_cal, anchored at LB(v9_cal) ~ 0.98225): v13_ens_efd pessimistic -0.00048, corrected truths +0.00135..+0.00163,
estimate 0.9837 (range 0.9818..0.9839); v12c_hyb4_efd 0.9836 (0.9816..0.9841); v12b_hyb4 0.9833 (0.9813..0.9837).
usage: python s65_estimate.py"""
from common import *
from s32_neighbor_rule import expected_f
N = 1732544
base = decide(pl.read_parquet(wp('scores', 'v9scal', 'test_s2_c0.parquet'), columns=[*KY, 'p']), 0.7)
cands = {n: pl.read_parquet(f'../outputs/{n}/pairs.parquet').select(KY) for n in ('v12b_hyb4', 'v12c_hyb4_efd', 'v13_ens_efd')}
truths = {t: pl.read_parquet(wp('scores', t, 'test_s2_c0.parquet'), columns=[*KY, 'p']) for t in ('v9scal', 'hyb4', 'ns1', 'ens1')}
ents = pl.concat([pl.concat([base.join(c, on=KY, how='anti'), c.join(base, on=KY, how='anti')]) for c in cands.values()]).select('id1').unique()
print('entities with any change:', ents.height)
res = {}
for t, S in truths.items():
    B = S.filter(pl.col('p') >= 0.01).sort('p', descending=True).unique(['id2', 'src'], keep='first').join(ents, on='id1', how='semi').select(KY)
    def ef(sel):
        sel = sel.join(ents, on='id1', how='semi').with_columns(pl.lit(True).alias('s'))
        X = pl.concat([B, sel.select(KY)]).unique().join(S.rename({'p': 'q'}), on=KY, how='left').with_columns(pl.col('q').fill_null(0.0)).join(sel, on=KY, how='left').with_columns(pl.col('s').fill_null(False))
        out = 0.0
        for k in range(4):
            e = ents.filter(pl.col('id1') % 4 == k); out += float(expected_f(X.join(e, on='id1', how='semi'), 's', 'q', e).sum())
        return out
    b = ef(base); res[t] = {n: (ef(c) - b) / N for n, c in cands.items()}
    print(f'truth {t:7s}: ' + ' | '.join(f'{n} {v:+.6f}' for n, v in res[t].items()), flush=True)
for n in cands:
    v = [res[t][n] for t in truths]
    print(f'{n}: pessimistic {res["v9scal"][n]:+.5f}, corrected-truth range {min(res[t][n] for t in ("hyb4","ns1","ens1")):+.5f}..{max(res[t][n] for t in ("hyb4","ns1","ens1")):+.5f}'
          f' -> LB estimate {0.98225 + res["ens1"][n]:.4f} (range {0.98225 + res["v9scal"][n]:.4f}..{0.98225 + max(res[t][n] for t in ("hyb4","ns1","ens1")):.4f})')
