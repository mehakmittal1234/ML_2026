"""Blend of the two test-adapted score sets and the expected-F0.5 decision (v13 step).
ens1 = mean of hyb4 and ns1 probabilities (every other column taken from ns1); <out>_efd encodes the expected-F0.5 decoder's choice
(efd_decide, lam 0.08) as scores 1 (chosen) / 0 (not chosen), so s55_build with threshold 0.5 applies it.
usage: python s58_blend.py <tag A> <tag B> <out tag>   -> scores/<out>/test_s2_c0.parquet and scores/<out>_efd/test_s2_c0.parquet"""
import sys
from common import *
from efd import efd_decide
a, b, out = sys.argv[1], sys.argv[2], sys.argv[3]
A = pl.read_parquet(wp('scores', a, 'test_s2_c0.parquet'), columns=[*KY, 'p'])
B = pl.read_parquet(wp('scores', b, 'test_s2_c0.parquet'), columns=[*KY, 'p']).rename({'p': 'p2'})
ens = A.join(B, on=KY).with_columns(((pl.col('p') + pl.col('p2')) / 2).cast(pl.Float32).alias('p')).drop('p2')
full = pl.read_parquet(wp('scores', b, 'test_s2_c0.parquet')).drop('p').join(ens, on=KY)
os.makedirs(wp('scores', out), exist_ok=True); full.write_parquet(wp('scores', out, 'test_s2_c0.parquet'))
S = full.select(*KY, 'p'); P = efd_decide(S, lam=0.08)
print(f'{out}: expected-F decoder keeps {P.height} pairs (threshold 0.7 would keep {decide(S, 0.7).height})')
os.makedirs(wp('scores', out + '_efd'), exist_ok=True)
S.join(P.with_columns(pl.lit(1.0).alias('_c')), on=KY, how='left').with_columns(
    pl.when(pl.col('_c').is_not_null()).then(pl.lit(1.0)).otherwise(pl.lit(0.0)).cast(pl.Float32).alias('p')).drop('_c') \
 .write_parquet(wp('scores', out + '_efd', 'test_s2_c0.parquet'))
