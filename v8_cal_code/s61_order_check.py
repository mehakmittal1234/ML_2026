"""Held-out check of the hybrid's within-cell ordering. The density-ratio discriminator D (s53 --inv features) is retrained with fold 0
EXCLUDED from the source (source = folds 1-7, target = test), then applied to fold-0 pairs. Among fold-0 pairs with the same core name and
a nearby house number (hn_rel = 4), label-known groups:
   true      : genuine copies with a jittered/different number
   neighbour : copies of ANOTHER S1 entity with the same core name on the same street a few doors away (real generator neighbours)
   other     : remaining false pairs
AUC of the factor (higher = more source-like = more likely true) for true vs neighbour tells whether ordering inside the shifted cells
is informative.   RESULT: India fold 0, factor AUC true vs real same-name neighbour copies 0.924 (held out); c_at_xh 1.37 vs 0.07, sv_s_hn 0.68 vs 0.02.
usage: python s61_order_check.py"""
import sys, glob, time
import lightgbm as lgb
from common import *
sys.argv += ['--inv']
from s53_dr import FEATS, PAR, R, load, comp
t0 = time.time()
def auc(pos, neg):
    x = np.r_[pos, neg]; r = pd_rank(x); n1, n0 = len(pos), len(neg)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def pd_rank(x):
    o = np.argsort(x, kind='mergesort'); r = np.empty(len(x)); r[o] = np.arange(1, len(x) + 1); return r
ks = lambda split: (lambda a: a.join(a.group_by('ctry', 'nc').len(), on=['ctry', 'nc']).select(pl.col('id').alias('id1'), 'ctry', pl.col('len').clip(1, 4).alias('kc')))(
    pl.read_parquet(wp('norm2', f'{split}_s1.parquet'), columns=['id', 'ctry', 'nc']))
eS_all, eT_all = ks('train'), ks('test')
gt = load_gt_pairs()
own = gt.select('id2', 'src', pl.col('id1').alias('owner'))
N1 = pl.read_parquet(wp('norm2', 'train_s1.parquet'), columns=['id', 'nc']).rename({'id': 'id1'})
for c in ('India', 'US'):
    Src = load('train', (pl.col('fold') != 0) & (pl.col('ctry') == c))
    Tgt = load('test', pl.col('ctry') == c)
    cS, cT = comp(eS_all.filter((pl.col('ctry') == c) & (fold_expr('id1') != 0))), comp(eT_all.filter(pl.col('ctry') == c))
    w = (cT / cT.sum()) / (cS / cS.sum()); wS = Src['kc'].replace_strict({k: float(w[k - 1]) for k in range(1, 5)}, return_dtype=pl.Float64).to_numpy()
    X = np.vstack([Src.select(FEATS).to_numpy(), Tgt.select(FEATS).to_numpy()]).astype(np.float32)
    m = lgb.train(PAR, lgb.Dataset(X, np.r_[np.ones(Src.height), np.zeros(Tgt.height)], weight=np.r_[wS, np.ones(Tgt.height)], feature_name=FEATS, free_raw_data=True), R)
    del X
    V = load('train', (pl.col('fold') == 0) & (pl.col('ctry') == c) & (pl.col('hn_rel') == 4) & (pl.col('name_rel') == 0))
    d = m.predict(V.select(FEATS).to_numpy().astype(np.float32), num_threads=4)
    V = V.with_columns(pl.Series('f', d / (1 - d))).join(own, on=['id2', 'src'], how='left').join(N1, on='id1').join(N1.rename({'id1': 'owner', 'nc': 'nc_o'}), on='owner', how='left')
    V = V.with_columns(pl.when(pl.col('y') == 1).then(pl.lit('true')).when((pl.col('nc_o') == pl.col('nc')) & pl.col('owner').is_not_null()).then(pl.lit('neighbour')).otherwise(pl.lit('other')).alias('g'))
    t, nb = V.filter(pl.col('g') == 'true'), V.filter(pl.col('g') == 'neighbour')
    print(f'{c}: fold-0 same-name nearby-number pairs: true {t.height}, same-name neighbour copies {nb.height}, other {V.height - t.height - nb.height} ({time.time()-t0:.0f}s)')
    if t.height and nb.height:
        print(f'   AUC factor (true vs neighbour): {auc(t["f"].to_numpy(), nb["f"].to_numpy()):.3f} | AUC base p: {auc(t["p"].to_numpy(), nb["p"].to_numpy()):.3f}'
              f' | median factor true {t["f"].median():.3f} neighbour {nb["f"].median():.3f}')
        for col in ('anchor', 'c_at_xh', 'sv_s_hn', 'g_src'):
            print(f'   {col}: mean true {t[col].mean():.3f}  neighbour {nb[col].mean():.3f}')
