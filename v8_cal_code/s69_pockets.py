"""Where does the p chain reject pairs the structural model (s68) calls true, much more often on test than on validation?
Cells: country x name relation x house-number relation x no-address. For each cell: rate of 'struct says true, p says no'
(q >= 0.9, p < 0.5) per pair on validation fold 0 (with the true share among them) and on test.
Count-invariance check on test for the flagged pairs: the S1 entity's number of OTHER confident records (best owner, p >= 0.9).
True copies are size-biased (train: 3.20 for p >= 0.9 copies); extra records sit on random S1 entities (higher, ~3.35).
usage: python s69_pockets.py <struct test tag> <final score tag for the accepted set>"""
import sys, glob
from common import *
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250)
stag, ftag = sys.argv[1], sys.argv[2]
CELL = ['ctry', 'name_rel', 'hn_rel', 'x_noaddr']
f50 = lambda split: pl.scan_parquet(sorted(glob.glob(wp('feat', 'f50', f'{split}_c*.parquet')))).select(*KY, 'name_rel', 'hn_rel', 'x_noaddr').collect()
V = pl.read_parquet(wp('analysis', 'struct_fold0.parquet')).join(f50('train'), on=KY)
T = pl.read_parquet(wp('scores', stag, 'test_struct.parquet')).join(f50('test'), on=KY)
flag = (pl.col('q') >= 0.9) & (pl.col('p') < 0.5)
v = V.group_by(CELL).agg(pl.len().alias('nv'), flag.mean().alias('rate_v'), (flag & (pl.col('y') == 1)).sum().alias('flag_true_v'), flag.sum().alias('flag_v'))
t = T.group_by(CELL).agg(pl.len().alias('nt'), flag.mean().alias('rate_t'), flag.sum().alias('flag_t'))
c = t.join(v, on=CELL, how='left').with_columns((pl.col('flag_true_v') / pl.col('flag_v')).alias('true_share_v'),
                                                (pl.col('flag_t') - pl.col('rate_v').fill_null(0) * pl.col('nt')).alias('excess_t'))
print(c.sort('excess_t', descending=True).head(30))
# count invariance for flagged test pairs vs accepted test pairs, per country
S = pl.read_parquet(wp('scores', ftag, 'test_s2_c0.parquet'), columns=[*KY, 'p'])
nconf = S.sort('p', descending=True).unique(['id2', 'src'], keep='first').filter(pl.col('p') >= 0.9).group_by('id1').len().rename({'len': 'nconf'})
F = T.filter(flag).join(nconf, on='id1', how='left').with_columns(pl.col('nconf').fill_null(0))
A = T.filter((pl.col('p') >= 0.9) & (pl.col('q') >= 0.9)).join(nconf, on='id1', how='left').with_columns((pl.col('nconf') - 1).alias('nconf'))
print('other confident records per S1: flagged (struct yes, p no) vs agreed-true pairs')
print(F.group_by('ctry').agg(pl.len(), pl.col('nconf').mean().alias('flagged')).join(A.group_by('ctry').agg(pl.col('nconf').mean().alias('agreed_true')), on='ctry'))
T.filter(flag).write_parquet(wp('analysis', 'pockets_test.parquet'))
