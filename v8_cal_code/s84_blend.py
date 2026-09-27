"""Logit blend of a base score set with the shift-robust re-scorer (s81 / s83), for test.
z = sigmoid(w * logit(q) + (1 - w) * logit(p)). Fold 0: m4it 0.98941; blend with round-2 q, w 0.3 -> 0.98978, w 0.5 -> 0.98975.
Pairs missing from the robust score file keep p. The base's p1 column is carried along (s18 / s55 / s70 expect it).
usage: python s84_blend.py <base tag> <robust tag> <w> <out tag>   -> scores/<out>/test_s2_c0.parquet"""
import sys
from common import *
base, rob, w, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
P = pl.read_parquet(wp('scores', base, 'test_s2_c0.parquet'), columns=[*KY, 'p1', 'p'])
Q = pl.read_parquet(wp('scores', rob, 'test_s2_c0.parquet'), columns=[*KY, 'p']).rename({'p': 'q'})
lg = lambda c: (pl.col(c).clip(1e-6, 1 - 1e-6) / (1 - pl.col(c).clip(1e-6, 1 - 1e-6))).log()
X = P.join(Q, on=KY, how='left').with_columns(pl.coalesce('q', 'p').alias('q'))
X = X.with_columns((1 / (1 + (-(w * lg('q') + (1 - w) * lg('p'))).exp())).cast(pl.Float32).alias('p')).drop('q')
os.makedirs(wp('scores', out), exist_ok=True); X.write_parquet(wp('scores', out, 'test_s2_c0.parquet'))
print(f'{out}: {X.height} pairs, w {w}; threshold 0.7 keeps {decide(X, 0.7).height} (base {decide(P, 0.7).height})')
