"""Edit-type 'tells' separating true copies from planted near-duplicate distractors (labelled train, stage-2 pairs).
  legal relation : same / dropped (record's legal set is a strict subset of S1's) / added-or-swapped (record has a legal token S1 lacks) / none
  digit relation : all digit groups of each address concatenated in order (raw, zeros kept):
                   equal / truncated (record's digits are a strict prefix of S1's, or of one group) / altered (same length, a digit differs) / other
  name edit      : record core-name tokens absent from S1's core name -> noise token (copy-noise list) / real vocabulary word (S1 df>=5) / other (typo, OOV)
usage: python a15_tells.py"""
from common import *
from norm import LEGAL_CANON
import a14_france_premise as A14
LG = sorted(set(LEGAL_CANON.values()))

def fields(split, s, ids):
    d = pl.scan_parquet(wp('norm2', f'{split}_s{s}.parquet')).select('id', 'ctry', 'nt', 'nc', 'dg').join(ids.lazy(), on='id', how='semi').collect()
    lg = pl.Series(LG)
    return d.select('id', 'ctry', pl.col('nc').str.split(' ').alias('tk'),
                    pl.col('nt').str.split(' ').list.eval(pl.element().filter(pl.element().is_in(lg))).list.unique().alias('lg'),
                    pl.col('dg').str.replace_all(' ', '|').alias('dgs'))

def tells(split, S):
    NZ = A14.noise_tokens(split)
    voc = pl.read_parquet(wp('norm2', f'{split}_s1.parquet'), columns=['ctry', 'nc']).select('ctry', pl.col('nc').str.split(' ')).explode('nc') \
            .group_by('ctry', 'nc').len().filter(pl.col('len') >= 5).rename({'nc': 'tok'}).drop('len')
    A = fields(split, 1, S.select(pl.col('id1').unique().alias('id')))
    R = pl.concat([fields(split, s, S.filter(pl.col('src') == s).select(pl.col('id2').unique().alias('id'))).with_columns(pl.lit(s, pl.Int8).alias('src')) for s in (2, 3)])
    X = S.join(A.rename({'id': 'id1', 'tk': 'tk1', 'lg': 'lg1', 'dgs': 'd1'}), on='id1').join(R.rename({'id': 'id2', 'tk': 'tk2', 'lg': 'lg2', 'dgs': 'd2'}).drop('ctry'), on=['id2', 'src'])
    n1, n2 = pl.col('lg1').list.len(), pl.col('lg2').list.len()
    added = pl.col('lg2').list.set_difference('lg1').list.len() > 0
    X = X.with_columns(pl.when((n1 == 0) & (n2 == 0)).then(pl.lit('none')).when(added).then(pl.lit('added/swapped'))
                         .when(n2 < n1).then(pl.lit('dropped')).otherwise(pl.lit('same')).alias('legal_rel'))
    d1, d2 = pl.col('d1').str.replace_all(r'\|', ''), pl.col('d2').str.replace_all(r'\|', '')
    X = X.with_columns(pl.when((pl.col('d1') == '') | (pl.col('d2') == '')).then(pl.lit('missing'))
                         .when(d1 == d2).then(pl.lit('equal'))
                         .when(d1.str.starts_with(d2) | pl.col('d1').str.starts_with(pl.col('d2'))).then(pl.lit('truncated'))
                         .when(d1.str.len_chars() == d2.str.len_chars()).then(pl.lit('altered'))
                         .otherwise(pl.lit('other')).alias('dig_rel'))
    ex = X.select('id1', 'id2', 'src', 'ctry', pl.col('tk2').list.set_difference('tk1').alias('t')).explode('t').drop_nulls('t').filter(pl.col('t') != '')
    ex = ex.join(voc.with_columns(pl.lit(True).alias('invoc')), left_on=['ctry', 't'], right_on=['ctry', 'tok'], how='left')
    nz = pl.DataFrame([(c, t) for c, ts in NZ.items() for t in ts], schema=['ctry', 't'], orient='row').with_columns(pl.lit(True).alias('noise'))
    ex = ex.join(nz, on=['ctry', 't'], how='left').with_columns(pl.col('noise').fill_null(False))
    ek = ex.group_by(KY).agg(pl.col('noise').sum().alias('n_noise'), (pl.col('invoc').fill_null(False) & ~pl.col('noise')).sum().alias('n_realword'),
                             (~pl.col('invoc').fill_null(False) & ~pl.col('noise')).sum().alias('n_other'))
    X = X.join(ek, on=KY, how='left').with_columns(pl.col('n_noise', 'n_realword', 'n_other').fill_null(0))
    X = X.with_columns(pl.when(pl.col('n_realword') > 0).then(pl.lit('real word added')).when(pl.col('n_other') > 0).then(pl.lit('typo/OOV added'))
                         .when(pl.col('n_noise') > 0).then(pl.lit('noise only')).otherwise(pl.lit('nothing added')).alias('name_edit'))
    return X

if __name__ == '__main__':
    S = pl.read_parquet(wp('feat', 'm1_s2', 'train.parquet'), columns=[*KY, 'p1', 'rec_rank'])
    S = S.filter(fold_expr('id1').is_in([0, 1]))            # 2 folds are plenty for rates
    X = tells('train', S).join(load_gt_pairs().with_columns(pl.lit(1, pl.Int8).alias('y')), on=KY, how='left').with_columns(pl.col('y').fill_null(0))
    B = X.filter((pl.col('p1') > 0.2) & (pl.col('p1') < 0.75) & (pl.col('rec_rank') == 1))
    pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
    for col in ('legal_rel', 'dig_rel', 'name_edit'):
        print(X.group_by(col).agg(pl.len().alias('all_pairs'), pl.col('y').mean().round(3).alias('true_rate_all'),
                                   pl.col('p1').mean().round(3).alias('mean_p1')).join(
              B.group_by(col).agg(pl.len().alias('borderline'), pl.col('y').mean().round(3).alias('true_rate_borderline')), on=col, how='left').sort(col))
    print(B.group_by('legal_rel', 'dig_rel').agg(pl.len(), pl.col('y').mean().round(3)).sort('len', descending=True).head(16))
    X.select(*KY, 'legal_rel', 'dig_rel', 'name_edit', 'n_noise', 'n_realword', 'n_other').write_parquet(wp('analysis', 'tells_train_f01.parquet'))
