"""Label-free prior-shift recalibration for test (per country).
Identity used: the generator gives ~3.46 true matches per S1 entity (train US and India are identical: 3.459 / 3.465), retrieval keeps
~98.6% of them, and on train the model's expected matches (sum of p over each record's best candidate) equal 0.992-0.995 of that.
On test the same sum is 1.001 (US/India) to 1.011 (France) -> the model over-assigns probability on test.
For each country find b with sum sigmoid(logit(p) + b) = KAPPA * S1 * 3.46 * 0.986, then decide with the usual threshold.
usage: python s18_priorshift.py <score tag> <thr> <out tag>   -> scores/<out tag>/test_s2_c0.parquet (p recalibrated)"""
import sys
from scipy.special import logit, expit
from scipy.optimize import brentq
from common import *
mtag, thr, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
KAPPA = {k: v * float(os.environ.get('KAPPA_MULT', '1')) for k, v in {'US': 0.9949, 'India': 0.9921, 'France': 0.9921}.items()}
S = pl.read_parquet(wp('scores', mtag, 'test_s2_c0.parquet')).select(*KY, 'p1', 'p')
c = pl.read_parquet(wp('data', 'test_s1.parquet'), columns=['id', 'country']).rename({'id': 'id1'})
S = S.join(c, on='id1')
best = S.sort('p', descending=True).unique(['id2', 'src'], keep='first')
n1 = dict(c.group_by('country').len().iter_rows())
B = {}
for ctry in sorted(n1):
    p = np.clip(best.filter(pl.col('country') == ctry)['p'].to_numpy().astype(np.float64), 1e-7, 1 - 1e-7)
    target = KAPPA[ctry] * n1[ctry] * 3.46 * 0.986
    b = brentq(lambda b: expit(logit(p) + b).sum() - target, -5, 5)
    B[ctry] = b
    q = expit(logit(p) + b)
    t_eq = expit(logit(thr) - b)
    print(f'{ctry:7s} sum_p {p.sum():.0f} target {target:.0f} ratio {p.sum()/target:.4f} -> bias {b:+.3f}; accepted at {thr}: {(p >= thr).sum()} -> {(q >= thr).sum()} '
          f'(equivalent raw threshold {t_eq:.3f})', flush=True)
S = S.with_columns(pl.struct('p', 'country').map_batches(lambda s: pl.Series(expit(logit(np.clip(s.struct.field('p').to_numpy().astype(np.float64), 1e-7, 1 - 1e-7))
                   + np.array([B[x] for x in s.struct.field('country').to_list()]))), return_dtype=pl.Float64).cast(pl.Float32).alias('p'))
os.makedirs(wp('scores', out), exist_ok=True)
S.select(*KY, 'p1', 'p').write_parquet(wp('scores', out, 'test_s2_c0.parquet'))
print('PRIOR_DONE', B)
