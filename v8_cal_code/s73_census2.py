"""Census of accepted pairs per S1 entity by pair type, test vs generator constant (US, India).
The generator's noise is applied per copy, so true copies per S1 entity in each (name edit x address relation) cell is a constant:
train truth rate. Expected accepted on test = truth rate x validation recall of the cell (fold 0, m4it thr 0.7) x #S1.
A deficit is true copies the test pipeline drops (the 'service' decoration was one: 0.0043 vs 0.0350 per US entity, fixed in v14 and
confirmed on the leaderboard: v14e 0.9865). A surplus is accepted distractors.
Cell = country x name_rel x legal_rel x hn_rel x deco (decoration word added: service / services / center / partners / none).
usage: python s73_census2.py <output dir (pairs.parquet)>"""
import sys, glob
from common import *
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250)
OUT = sys.argv[1]
W = ['service', 'services', 'center', 'partners']
CELL = ['ctry', 'name_rel', 'legal_rel', 'hn_rel', 'deco']
def typed(split, P):
    F = pl.scan_parquet(sorted(glob.glob(wp('feat', 'f50', f'{split}_c*.parquet')))).select(*KY, 'ctry', 'name_rel', 'legal_rel', 'hn_rel').collect()
    X = P.select(KY).join(F, on=KY)
    n1 = pl.read_parquet(wp('norm2', f'{split}_s1.parquet'), columns=['id', 'nc']).rename({'id': 'id1', 'nc': 'nc1'})
    n2 = pl.concat([pl.read_parquet(wp('norm2', f'{split}_s{s}.parquet'), columns=['id', 'nc']).with_columns(pl.lit(s, pl.Int8).alias('src')) for s in (2, 3)]).rename({'id': 'id2', 'nc': 'nc2'})
    X = X.join(n1, on='id1').join(n2, on=['id2', 'src'])
    e = pl.lit('none')
    for w in W[::-1]:
        r = rf'\b{w}\b'; e = pl.when(pl.col('nc2').str.contains(r) & ~pl.col('nc1').str.contains(r)).then(pl.lit(w)).otherwise(e)
    return X.with_columns(e.alias('deco')).drop('nc1', 'nc2')
gt = load_gt_pairs()
n_tr = pl.read_parquet(wp('norm2', 'train_s1.parquet'), columns=['ctry']).group_by('ctry').len().rename({'len': 'n_tr'})
n_te = pl.read_parquet(wp('norm2', 'test_s1.parquet'), columns=['ctry']).group_by('ctry').len().rename({'len': 'n_te'})
G = typed('train', gt)                                  # true pairs outside the stage-2 set are not typed (retrieval misses)
truth = G.group_by(CELL).len().rename({'len': 'true_tr'})
v0 = pl.read_parquet(wp('data', 'train_s1ids.parquet')).filter(fold_expr('id1') == 0)
G0 = G.join(v0, on='id1', how='semi')
Pv = decide(pl.read_parquet(wp('scores', 'm4it', 'train_s2_c0.parquet'), columns=[*KY, 'p']), 0.7).join(v0, on='id1', how='semi')
rec = G0.join(Pv, on=KY, how='semi').group_by(CELL).len().rename({'len': 'tp0'}).join(G0.group_by(CELL).len().rename({'len': 'true0'}), on=CELL, how='right')
fp0 = typed('train', Pv.join(gt, on=KY, how='anti')).group_by(CELL).len().rename({'len': 'fp0'})
cur = typed('test', pl.read_parquet(os.path.join(OUT, 'pairs.parquet'))).group_by(CELL).len().rename({'len': 'acc_te'})
C = truth.join(rec, on=CELL, how='left').join(fp0, on=CELL, how='left').join(n_tr, on='ctry').join(n_te, on='ctry').join(cur, on=CELL, how='left') \
      .with_columns(pl.col('tp0', 'fp0', 'acc_te').fill_null(0)).filter(pl.col('ctry') != 'France')
n0 = v0.join(pl.read_parquet(wp('norm2', 'train_s1.parquet'), columns=['id', 'ctry']).rename({'id': 'id1'}), on='id1').group_by('ctry').len().rename({'len': 'n0'})
C = C.join(n0, on='ctry').with_columns((pl.col('tp0') / pl.col('true0')).alias('recall'),
                                       ((pl.col('true_tr') / pl.col('n_tr')) * (pl.col('tp0') / pl.col('true0')).fill_null(0) * pl.col('n_te')).alias('exp_tp'),
                                       (pl.col('fp0') / pl.col('n0') * pl.col('n_te')).alias('exp_fp_valrate'))
C = C.with_columns((pl.col('acc_te') - pl.col('exp_tp') - pl.col('exp_fp_valrate')).alias('diff'),
                   ((pl.col('acc_te') - pl.col('exp_tp') - pl.col('exp_fp_valrate')) / (pl.col('exp_tp') + 1).sqrt()).alias('z'))
print('largest deficits (accepted on test below truth x recall):')
print(C.sort('diff').head(25).select(*CELL, 'true_tr', 'recall', 'exp_tp', 'exp_fp_valrate', 'acc_te', 'diff', 'z'))
print('largest surpluses:')
print(C.sort('diff', descending=True).head(15).select(*CELL, 'true_tr', 'recall', 'exp_tp', 'exp_fp_valrate', 'acc_te', 'diff', 'z'))
print('totals by country:', C.group_by('ctry').agg(pl.col('exp_tp').sum(), pl.col('exp_fp_valrate').sum(), pl.col('acc_te').sum(),
      pl.col('diff').filter(pl.col('diff') < 0).sum().alias('sum_deficit'), pl.col('diff').filter(pl.col('diff') > 0).sum().alias('sum_surplus')).rows())
C.write_parquet(wp('analysis', 'census2.parquet'))
