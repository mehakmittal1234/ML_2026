"""Which test-probability belief reproduces the observed leaderboard differences?
For belief B (per-pair probability of truth on test) and a submission D, model-implied macro F0.5 via Monte Carlo.
Observed LB: v8 0.981971, v8_cal 0.982143, v8_strict 0.979477, v10_expf 0.981635.
usage: python a35_lbcheck.py <belief tag> [<belief tag> ...]"""
import sys
from common import *
OBS = {'v8': 0.981971, 'v8_cal': 0.982143, 'v8_strict': 0.979477, 'v10_expf': 0.981635}
s1 = pl.read_parquet(wp('data', 'test_s1.parquet'), columns=['entity_id', 'id', 'country'])
def load_sub(name):
    M = pl.read_csv(f'../outputs/{name}/matching_results.tsv', separator='\t', quote_char=None, infer_schema=False).filter(pl.col('matched_entity_ids') != '')
    M = M.with_columns(pl.col('matched_entity_ids').str.split(',')).explode('matched_entity_ids')
    M = M.join(s1.select(pl.col('entity_id').alias('source1_entity_id'), pl.col('id').alias('id1')), on='source1_entity_id')
    return M.select('id1', pl.col('matched_entity_ids').str.slice(1, 1).cast(pl.Int8).alias('src'), pl.col('matched_entity_ids').str.slice(3).cast(pl.Int64).alias('id2'))
if __name__ == '__main__': SUBS = {k: load_sub(k) for k in OBS}
extra = sys.argv[2:] if len(sys.argv) > 2 else []
ent = s1.select(pl.col('id').alias('id1'), 'country').with_row_index('e')
rng = np.random.default_rng(0)
def implied(B, D, nsim=24):
    b = B.join(ent.select('id1', 'e'), on='id1').join(D.with_columns(pl.lit(True).alias('acc')), on=KY, how='left').with_columns(pl.col('acc').fill_null(False))
    # submitted pairs absent from the belief table count as certain false
    miss = D.join(B, on=KY, how='anti').join(ent.select('id1', 'e'), on='id1')
    e = b['e'].to_numpy(); q = b['q'].to_numpy().astype(np.float64); acc = b['acc'].to_numpy(); n = ent.height
    fp0 = np.bincount(miss['e'].to_numpy(), minlength=n).astype(float)
    tot = 0.0
    for s in range(nsim):
        t = rng.random(len(q)) < q
        tp = np.bincount(e, weights=(t & acc), minlength=n); fp = np.bincount(e, weights=(~t & acc), minlength=n) + fp0
        tr = np.bincount(e, weights=t, minlength=n); tr = tr + rng.poisson(0.0209 * tr)
        pr = tp + fp; P = np.where(pr > 0, tp / np.maximum(pr, 1), 0); R = np.where(tr > 0, tp / np.maximum(tr, 1), 0)
        F = np.where((P + R) > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0); F = np.where(tr == 0, (pr == 0).astype(float), F)
        tot += F.mean()
    return tot / nsim
for btag in (sys.argv[1].split(',') if __name__ == '__main__' else []):
    S = pl.read_parquet(wp('scores', btag, 'test_s2_c0.parquet'), columns=[*KY, 'p'])
    B = S.sort('p', descending=True).unique(['id2', 'src'], keep='first').filter(pl.col('p') > 0.002).rename({'p': 'q'})
    r = {k: implied(B, D) for k, D in SUBS.items()}
    for x in extra:
        r[x] = implied(B, load_sub(x))
    base = r['v8']
    print(f'belief {btag}: ' + '  '.join(f'{k} {v:.5f} (d {v-base:+.5f} | obs {OBS[k]-OBS["v8"]:+.5f})' if k in OBS else f'{k} {v:.5f} (d {v-base:+.5f})' for k, v in r.items()), flush=True)
