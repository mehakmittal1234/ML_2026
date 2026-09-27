"""Pair-level, model-light generator comparison. Every S2/S3 record is paired with its top-1 stage-1 candidate (rec_rank==1 in m1_s2).
Edit statistics of those pairs: train split by GT (true copy paired with its owner / distractor), test = mixture -> deconvolve with pt.
Also reports the share of true copies whose top-1 candidate is NOT their owner (competition), which the deconvolution treats as 'true' rows."""
from rapidfuzz.distance import Levenshtein
from common import *
def edits(split):
    P = pl.read_parquet(wp('feat', 'm1_s2', f'{split}.parquet'), columns=[*KY, 'p1', 'rec_rank']).filter(pl.col('rec_rank') == 1).select(*KY, 'p1')
    n1 = pl.read_parquet(wp('norm2', f'{split}_s1.parquet'), columns=['id', 'ctry', 'nc', 'dg']).rename({'id': 'id1', 'nc': 'c1', 'dg': 'g1'})
    n2 = pl.concat([pl.read_parquet(wp('norm2', f'{split}_s{s}.parquet'), columns=['id', 'nc', 'dg']).with_columns(pl.lit(s, pl.Int8).alias('src')) for s in (2, 3)]).rename({'id': 'id2', 'nc': 'c2', 'dg': 'g2'})
    X = P.join(n1, on='id1').join(n2, on=['id2', 'src'])
    c1, c2, g1, g2 = X['c1'].to_list(), X['c2'].to_list(), X['g1'].to_list(), X['g2'].to_list()
    n = X.height
    add = np.zeros(n, np.int8); far = np.zeros(n, np.int8); drop = np.zeros(n, np.int8); hsub = np.zeros(n, np.int8); htr = np.zeros(n, np.int8); heq = np.zeros(n, np.int8)
    for i in range(n):
        s1 = set((c1[i] or '').split()); s2 = set((c2[i] or '').split())
        d = s1 - s2; a = s2 - s1
        drop[i] = min(len(d), 3); add[i] = min(len(a), 3)
        far[i] = min(sum(1 for t in a if max((Levenshtein.normalized_similarity(t, u) for u in s1), default=0.0) < 0.5), 3)
        x = (g1[i] or '').split(' ')[0] if g1[i] else ''; y = (g2[i] or '').split(' ')[0] if g2[i] else ''
        if x and y:
            if x == y: heq[i] = 1
            elif x.startswith(y): htr[i] = 1
            elif len(x) == len(y): hsub[i] = 1
    return X.select(*KY, 'p1', 'ctry').with_columns(pl.Series('add', add), pl.Series('far', far), pl.Series('drop', drop), pl.Series('h_eq', heq), pl.Series('h_trunc', htr), pl.Series('h_sub', hsub))
tr = edits('train')
gt = load_gt_pairs()
tr = tr.join(gt.select('id2', 'src', pl.col('id1').alias('owner')), on=['id2', 'src'], how='left').with_columns(
    pl.when(pl.col('owner').is_null()).then(pl.lit('distr')).when(pl.col('owner') == pl.col('id1')).then(pl.lit('true_own')).otherwise(pl.lit('true_other')).alias('kind'))
te = edits('test')
print('train kinds:', tr.group_by('ctry', 'kind').len().sort('ctry', 'kind').rows())
COLS = ['add', 'far', 'drop', 'h_eq', 'h_trunc', 'h_sub']
rows = []
for c in ('US', 'India'):
    t = tr.filter(pl.col('ctry') == c); x = te.filter(pl.col('ctry') == c)
    pd_tr = t.filter(pl.col('kind') == 'distr').height / t.height
    # test distractor share among records that have a top-1 pair: scale train's distractor-per-S1 by the record surplus
    n1tr = pl.read_parquet(wp('data', 'train_s1.parquet'), columns=['country']).filter(pl.col('country') == c).height
    n1te = pl.read_parquet(wp('data', 'test_s1.parquet'), columns=['country']).filter(pl.col('country') == c).height
    true_te = t.filter(pl.col('kind') != 'distr').height / n1tr * n1te
    pt = true_te / x.height
    print(f'{c}: train distractor share among top-1 pairs {pd_tr:.3f}; test implied true share {pt:.3f}')
    for col in COLS:
        for v in sorted(t[col].unique().to_list()):
            mt = (t.filter(pl.col('kind') != 'distr')[col] == v).mean(); md = (t.filter(pl.col('kind') == 'distr')[col] == v).mean(); mx = (x[col] == v).mean()
            est = (mx - (1 - pt) * md) / pt
            rows.append((c, col, v, round(mt, 4), round(md, 4), round(mx, 4), round(est, 4), round(est - mt, 4)))
T = pl.DataFrame(rows, schema=['ctry', 'stat', 'value', 'train_true', 'train_distr', 'test_all', 'test_true_est', 'shift'], orient='row')
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(200); print(T.filter(~((pl.col('stat').str.starts_with('h_')) & (pl.col('value') == 0))))
tr.write_parquet(wp('analysis', 'a42_train_top1.parquet')); te.write_parquet(wp('analysis', 'a42_test_top1.parquet'))
