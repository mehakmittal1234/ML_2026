"""US no-address excess (census: eq_or_order/noaddr accepted +0.009 per S1 on test). Count invariance by S1 name-group size k:
the copy generator ignores name popularity, so the number of NO-ADDRESS records carrying a group's exact core name, per S1 entity of
the group, is c (true copies) + d_k (records not owned by the group). Train measures c and d_k with labels; on test any rise of the
per-entity count within a stratum is non-owned records (orphans) carrying that exact name.
Also: how the model scores them (accepted share) and whether the S1 owner has other confident copies.
RESULT: no test excess. US no-address exact-name records per S1 entity, train vs test: k=1 0.1138 vs 0.1133, k=2 0.1200 vs 0.1159,
k=3 0.1181 vs 0.1134; no-address distractors are ~0.0006 per entity in train. Accepted per entity and mean p match by k (k=1 precision
0.988, k>=2 ~0.88 on train). The census '+0.009' was a composition effect: 60% of test US S1 names are unique vs 52% in train.
usage: python s40_noaddr_groups.py <val tag> <test tag>"""
import sys
from common import *
vtag, ttag = sys.argv[1], sys.argv[2]

def groups(split):
    A = pl.read_parquet(wp('norm2', f'{split}_s1.parquet'), columns=['id', 'ctry', 'nc']).filter(pl.col('nc') != '')
    A = A.join(A.group_by('ctry', 'nc').len().rename({'len': 'k'}), on=['ctry', 'nc'])
    R = pl.concat([pl.read_parquet(wp('norm2', f'{split}_s{s}.parquet'), columns=['id', 'ctry', 'nc', 'at']).filter(pl.col('at') == '')
                   .select(pl.col('id').alias('id2'), pl.lit(s, pl.Int8).alias('src'), 'ctry', 'nc') for s in (2, 3)])
    return A, R

def summary(split, A, R, tag, own=None):
    kk = pl.col('k').clip(1, 4)
    G = A.group_by('ctry', 'nc', 'k').len().rename({'len': 'nS1'})
    X = R.join(G, on=['ctry', 'nc'])                                   # no-address records whose exact name is an S1 group
    S = pl.read_parquet(wp('scores', tag, f'{split}_s2_c0.parquet'), columns=[*KY, 'p'])
    best = S.sort('p', descending=True).unique(['id2', 'src'], keep='first').rename({'id1': 'bid', 'p': 'bp'})
    X = X.join(best, on=['id2', 'src'], how='left').with_columns(pl.col('bp').fill_null(0.0))
    if own is not None:
        X = X.join(own, on=['id2', 'src'], how='left').join(A.select(pl.col('id').alias('true_id1'), pl.col('nc').alias('nc_t')), on='true_id1', how='left') \
             .with_columns(pl.when(pl.col('true_id1').is_null()).then(pl.lit('distractor')).when(pl.col('nc_t') == pl.col('nc')).then(pl.lit('group'))
                           .otherwise(pl.lit('other_owner')).alias('who'))
    return X.with_columns(kk.alias('kc'))

gt = load_gt_pairs()
out = {}
for split, tag in (('train', vtag), ('test', ttag)):
    A, R = groups(split)
    own = gt.select('id2', 'src', pl.col('id1').alias('true_id1')) if split == 'train' else None
    X = summary(split, A, R, tag, own)
    nS1 = A.with_columns(pl.col('k').clip(1, 4).alias('kc')).group_by('ctry', 'kc').len().rename({'len': 'n_ent'})
    agg = [pl.len().alias('rec'), (pl.col('bp') >= 0.7).sum().alias('acc')]
    if split == 'train':
        agg += [(pl.col('who') == 'group').sum().alias('rec_true'), (pl.col('who') == 'distractor').sum().alias('rec_dist'),
                ((pl.col('bp') >= 0.7) & (pl.col('who') == 'group') & (pl.col('bid') == pl.col('true_id1'))).sum().alias('acc_right'),
                ((pl.col('bp') >= 0.7) & (pl.col('who') == 'distractor')).sum().alias('acc_dist')]
    T = X.group_by('ctry', 'kc').agg(*agg).join(nS1, on=['ctry', 'kc'])
    cols = [c for c in T.columns if c not in ('ctry', 'kc', 'n_ent')]
    out[split] = T.with_columns(*[(pl.col(c) / pl.col('n_ent')).alias(c + '_per') for c in cols]).sort('ctry', 'kc')
pl.Config.set_tbl_rows(30); pl.Config.set_tbl_width_chars(230); pl.Config.set_float_precision(4)
print('TRAIN (all folds; per S1 entity of the stratum)'); print(out['train'].select('ctry', 'kc', 'n_ent', *[c for c in out['train'].columns if c.endswith('_per')]))
print('TEST'); print(out['test'].select('ctry', 'kc', 'n_ent', *[c for c in out['test'].columns if c.endswith('_per')]))
