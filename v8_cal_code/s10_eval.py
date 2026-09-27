"""Official-metric evaluation on the validation fold (S1 fold 0) with full competition (every train pair scored out-of-fold).
Decision: each S2/S3 record -> its highest-scoring S1 if score >= thr. Macro F0.5 per S1 entity (singletons included).
Also: error decomposition = macro gain if one error class were fixed (retrieval miss / model reject / lost competition / false merge).
usage: python s10_eval.py <score tag> [score column p] [thr list]"""
import sys, glob
from common import *
mtag = sys.argv[1]; SC = sys.argv[2] if len(sys.argv) > 2 else 'p'
THRS = [float(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
S = pl.scan_parquet(wp('scores', mtag, 'train_s[23]_c*.parquet')).select(*KY, pl.col(SC).alias('p')).collect()
ids = pl.read_parquet(wp('data', 'train_s1ids.parquet')).filter(fold_expr('id1') == 0)
gt = load_gt_pairs(); gt0 = gt.join(ids, on='id1', how='semi')
print(f'scored pairs {S.height}; validation S1 {ids.height}, true pairs {gt0.height}; candidate recall {gt0.join(S, on=KY, how="semi").height/gt0.height:.4f}')
res = []
for thr in THRS:
    pred = decide(S, thr)
    m = macro_f05(pred, gt, ids, by='ctry')
    p0 = pred.join(ids, on='id1', how='semi'); tp = p0.join(gt0, on=KY, how='semi').height
    res.append((thr, m['macro'], tp / max(1, p0.height), tp / gt0.height))
    print(f'thr {thr}: macro {m["macro"]:.5f} | pair P {tp/max(1,p0.height):.4f} R {tp/gt0.height:.4f} | ' +
          ' '.join(f'{r[0]} {r[1]:.5f}' for r in m['by'].iter_rows()), flush=True)
best = max(res, key=lambda x: x[1]); thr = best[0]
print(f'best thr {thr}: macro {best[1]:.5f}')
# ---- error decomposition at best thr
pred = decide(S, thr)
base = macro_f05(pred, gt, ids)['macro']
best_of_rec = S.sort('p', descending=True).unique(['id2', 'src'], keep='first').select(*KY, pl.col('p').alias('pbest'))
G = gt0.join(S.select(*KY, 'p'), on=KY, how='left').join(best_of_rec.rename({'id1': 'best_id1'}), on=['id2', 'src'], how='left') \
       .join(pred.with_columns(pl.lit(True).alias('acc')), on=KY, how='left').with_columns(pl.col('acc').fill_null(False))
G = G.with_columns(pl.when(pl.col('acc')).then(pl.lit('ok')).when(pl.col('p').is_null()).then(pl.lit('fn_retrieval'))
                     .when(pl.col('best_id1') != pl.col('id1')).then(pl.lit('fn_lost_competition')).otherwise(pl.lit('fn_model_reject')).alias('err'))
FP = pred.join(ids, on='id1', how='semi').join(gt, on=KY, how='anti')
FP = FP.join(gt.select('id2', 'src', pl.col('id1').alias('true_id1')), on=['id2', 'src'], how='left') \
       .with_columns(pl.when(pl.col('true_id1').is_null()).then(pl.lit('fp_distractor_record')).otherwise(pl.lit('fp_other_owner')).alias('err'))
print(G.group_by('err').len().sort('err')); print(FP.group_by('err').len().sort('err'))
def gain_fix_fn(kind):
    add = G.filter(pl.col('err') == kind).select(*KY)
    return macro_f05(pl.concat([pred, add]), gt, ids)['macro'] - base
def gain_fix_fp(kind):
    rm = FP.filter(pl.col('err') == kind).select(*KY)
    return macro_f05(pred.join(rm, on=KY, how='anti'), gt, ids)['macro'] - base
print('macro gain if fixed (validation):')
for k in ('fn_retrieval', 'fn_model_reject', 'fn_lost_competition'):
    print(f'   {k:22s} +{gain_fix_fn(k):.5f}')
for k in ('fp_distractor_record', 'fp_other_owner'):
    print(f'   {k:22s} +{gain_fix_fp(k):.5f}')
T = macro_f05(pred, gt, ids, return_table=True)
T = T.with_columns(pl.col('nt').clip(0, 6).alias('k'))
print(T.group_by('k').agg(pl.len(), pl.col('F').mean().alias('meanF'), (1 - pl.col('F')).sum().alias('loss')).sort('k')
      .with_columns((pl.col('loss') / T.height).alias('macro_loss')))
G.write_parquet(wp('analysis', f'val_errors_{mtag}.parquet')); FP.write_parquet(wp('analysis', f'val_fp_{mtag}.parquet'))
