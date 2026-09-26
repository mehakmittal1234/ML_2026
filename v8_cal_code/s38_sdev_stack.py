"""Does the source-level sibling signal (s23 sdev features) add information beyond the best existing scores?
Stack on m4it scores: base stacker = logit(p), logit(p1), record competition (rank, gap to best rival); +sdev stacker adds the
s23 features. Both trained on folds 1-7 (out-of-fold scores), evaluated on fold 0 with the official macro F0.5 (one owner/record).
usage: python s38_sdev_stack.py <val tag> [test tag]  (test tag: also scores test -> scores/<test tag>_sdev/test_s2_c0.parquet)"""
import sys
import lightgbm as lgb
from scipy.special import logit
from common import *
SD = ['sd_n_same', 'sd_n_oth', 'sd_ydev_same', 'sd_xdev', 'sd_sh_same', 'sd_sh_oth', 'sd_cov_same', 'sd_aeq_same']
PAR = dict(objective='binary', learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, num_threads=4, verbose=-1)

def frame(split, tag):
    S = pl.read_parquet(wp('scores', tag, f'{split}_s2_c0.parquet'), columns=[*KY, 'p1', 'p'])
    S = S.with_columns(pl.col('p').clip(1e-6, 1 - 1e-6).map_batches(lambda s: pl.Series(logit(s.to_numpy()))).alias('lp'),
                       pl.col('p1').clip(1e-6, 1 - 1e-6).map_batches(lambda s: pl.Series(logit(s.to_numpy()))).alias('lp1'),
                       pl.col('p').rank('ordinal', descending=True).over(['id2', 'src']).cast(pl.Float32).alias('rrank'))
    top = S.group_by('id2', 'src').agg(pl.col('p').top_k(2).alias('_t')).with_columns(pl.col('_t').list.get(0).alias('_a'), pl.col('_t').list.get(1, null_on_oob=True).fill_null(0).alias('_b')).drop('_t')
    S = S.join(top, on=['id2', 'src']).with_columns(pl.when(pl.col('rrank') == 1).then(pl.col('p') - pl.col('_b')).otherwise(pl.col('p') - pl.col('_a')).alias('rgap')).drop('_a', '_b')
    return S.join(pl.read_parquet(wp('feat', f'{tag}_s2', f'{split}_sdev.parquet')), on=KY, how='left')

BASE = ['lp', 'lp1', 'rrank', 'rgap']
if __name__ == '__main__':
    vtag = sys.argv[1]; ttag = sys.argv[2] if len(sys.argv) > 2 else None
    T = frame('train', vtag).with_columns(fold_expr('id1').alias('fold')).join(load_gt_pairs().with_columns(pl.lit(1).alias('y')), on=KY, how='left').with_columns(pl.col('y').fill_null(0))
    tr, va = T.filter(pl.col('fold') != 0), T.filter(pl.col('fold') == 0)
    gt = load_gt_pairs(); v0 = pl.read_parquet(wp('data', 'train_s1ids.parquet')).filter(fold_expr('id1') == 0)
    res = {}
    for name, F in (('m4it as is', None), ('base stacker', BASE), ('base + sdev', BASE + SD)):
        if F is None:
            q = va['p'].to_numpy()
        else:
            m = lgb.train(PAR, lgb.Dataset(tr.select(F).to_numpy().astype(np.float32), tr['y'].to_numpy(), feature_name=F), 300)
            q = m.predict(va.select(F).to_numpy().astype(np.float32), num_threads=4); res[name] = m
        V = va.select(KY).with_columns(pl.Series('p', q))
        sc = {t: macro_f05(decide(V, t), gt, v0)['macro'] for t in (0.6, 0.65, 0.7, 0.75, 0.8)}
        bt = max(sc, key=sc.get)
        print(f'{name:14s}: ' + ' '.join(f'{t}:{v:.5f}' for t, v in sc.items()) + f'  | best {bt} {sc[bt]:.5f}', flush=True)
    m = res['base + sdev']; print('gain importance:', sorted(zip(BASE + SD, m.feature_importance('gain').round()), key=lambda x: -x[1]))
    if ttag:
        Te = frame('test', ttag)
        b = res['base stacker'].predict(Te.select(BASE).to_numpy().astype(np.float32), num_threads=4)
        s = m.predict(Te.select(BASE + SD).to_numpy().astype(np.float32), num_threads=4)
        for tg, q in ((f'{ttag}_stk', b), (f'{ttag}_sdev', s)):
            os.makedirs(wp('scores', tg), exist_ok=True); Te.select(*KY, 'p1').with_columns(pl.Series('p', q).cast(pl.Float32)).write_parquet(wp('scores', tg, 'test_s2_c0.parquet'))
        print('test scored ->', f'scores/{ttag}_stk', f'scores/{ttag}_sdev')
