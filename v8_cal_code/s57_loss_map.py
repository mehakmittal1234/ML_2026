"""Where is the remaining expected loss on test? Model-implied expected F0.5 of the threshold decisions (s24 machinery) under a score file,
loss (1 - EF) per S1 entity attributed to the type of the entity's most uncertain best-owner candidate (largest p(1-p)).
usage: python s57_loss_map.py <score tag> <thr>"""
import sys, glob
from common import *
from s24_diag_country import implied
tag, thr = sys.argv[1], float(sys.argv[2])
import s24_diag_country as S24; S24.THR = thr
S = pl.read_parquet(wp('scores', tag, 'test_s2_c0.parquet'), columns=[*KY, 'p'])
tc = pl.read_parquet(wp('data', 'test_s1.parquet'), columns=['id', 'country']).select(pl.col('id').alias('id1'), pl.col('country').alias('ctry'))
E = implied(S, tc).with_columns((1 - pl.col('EF')).alias('loss'))
F = pl.scan_parquet(sorted(glob.glob(wp('feat', 'f50', 'test_c*.parquet')))).select(*KY, 'name_rel', 'hn_rel', 'k_s1', 'x_noaddr').collect()
B = S.filter(pl.col('p') >= 0.01).sort('p', descending=True).unique(['id2', 'src'], keep='first').join(F, on=KY, how='left')
B = B.with_columns((pl.col('p') * (1 - pl.col('p'))).alias('u')).sort('u', descending=True).unique('id1', keep='first')
lab = pl.when(pl.col('u').is_null() | (pl.col('u') < 0.01)).then(pl.lit('no uncertain candidate'))
lab = lab.when(pl.col('x_noaddr') == 1).then(pl.when(pl.col('k_s1') > 1).then(pl.lit('no-address, S1 twin (class A)')).otherwise(pl.lit('no-address, unique name')))
lab = lab.when(pl.col('hn_rel').is_in([3, 4, 5])).then(pl.lit('different house number (near/far/trunc)'))
lab = lab.when(pl.col('hn_rel') == 2).then(pl.lit('same house number, name differs')).when(pl.col('hn_rel') == 1).then(pl.lit('house number missing one side')).otherwise(pl.lit('other'))
E = E.join(B.select('id1', 'u', 'x_noaddr', 'k_s1', 'hn_rel'), on='id1', how='left').with_columns(lab.alias('type'))
N = tc.height
print(f'{tag} @ {thr}: implied test F {E["EF"].mean():.5f}  (loss {1 - E["EF"].mean():.5f})')
T = E.group_by('ctry', 'type').agg((pl.col('loss').sum() / N).alias('loss_share')).sort('loss_share', descending=True)
pl.Config.set_tbl_rows(30); pl.Config.set_float_precision(5)
print(T.head(20)); print(E.group_by('ctry').agg((pl.col('loss').sum() / N).alias('loss')).sort('ctry'))
