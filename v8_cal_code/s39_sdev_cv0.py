"""s38 without the fold shift: fold-0 scores (and test) average two models and are sharper than folds 1-7, so the stacker is trained
and evaluated INSIDE fold 0 with a 2-way split by S1 entity (each half scored by the model trained on the other half).
Compares m4it as is / base stacker / base + sdev on fold 0 (official macro F0.5); final test model = trained on all of fold 0.
RESULT (m4it, fold 0): as is 0.98941 | base stacker 0.98934 | base + sdev 0.98936 -> +0.00002 over its control: REJECTED.
The +0.00059 seen in s38 was the stacker correcting the fold-shift, not new information; the stage-2 sibling features
(sib_a_ts / sib_hn_eq / sib_city_eq) already carry the source-level address signal on train-like data.
usage: python s39_sdev_cv0.py <val tag> [test tag]"""
import sys
import lightgbm as lgb
from common import *
from s38_sdev_stack import frame, BASE, SD, PAR
vtag = sys.argv[1]; ttag = sys.argv[2] if len(sys.argv) > 2 else None
gt = load_gt_pairs(); v0 = pl.read_parquet(wp('data', 'train_s1ids.parquet')).filter(fold_expr('id1') == 0)
T = frame('train', vtag).join(v0.select('id1'), on='id1', how='semi').join(gt.with_columns(pl.lit(1).alias('y')), on=KY, how='left') \
        .with_columns(pl.col('y').fill_null(0), (pl.col('id1').hash(seed=99) % 2).alias('half'))
ALL = pl.read_parquet(wp('scores', vtag, 'train_s2_c0.parquet'), columns=[*KY, 'p'])
models = {}
for name, F in (('m4it as is', None), ('base stacker', BASE), ('base + sdev', BASE + SD)):
    q = T['p'].to_numpy().copy()
    if F is not None:
        for h in (0, 1):
            tr = T.filter(pl.col('half') != h); te = T['half'].to_numpy() == h
            m = lgb.train(PAR, lgb.Dataset(tr.select(F).to_numpy().astype(np.float32), tr['y'].to_numpy()), 300)
            q[te] = m.predict(T.filter(pl.col('half') == h).select(F).to_numpy().astype(np.float32), num_threads=4)
        models[name] = lgb.train(PAR, lgb.Dataset(T.select(F).to_numpy().astype(np.float32), T['y'].to_numpy()), 300)
    V = ALL.join(T.select(KY), on=KY, how='anti').vstack(T.select(KY).with_columns(pl.Series('p', q).cast(pl.Float32)))
    sc = {t: macro_f05(decide(V, t), gt, v0)['macro'] for t in (0.6, 0.65, 0.7, 0.75, 0.8)}
    bt = max(sc, key=sc.get)
    print(f'{name:14s}: ' + ' '.join(f'{t}:{v:.5f}' for t, v in sc.items()) + f'  | best {bt} {sc[bt]:.5f}', flush=True)
if ttag:
    Te = frame('test', ttag)
    for nm, F, tg in (('base stacker', BASE, f'{ttag}_stk0'), ('base + sdev', BASE + SD, f'{ttag}_sdev0')):
        q = models[nm].predict(Te.select(F).to_numpy().astype(np.float32), num_threads=4)
        os.makedirs(wp('scores', tg), exist_ok=True); Te.select(*KY, 'p1').with_columns(pl.Series('p', q).cast(pl.Float32)).write_parquet(wp('scores', tg, 'test_s2_c0.parquet'))
    print('test scored ->', f'scores/{ttag}_stk0', f'scores/{ttag}_sdev0')
