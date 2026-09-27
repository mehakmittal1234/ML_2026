"""Label-free generator comparison (record level). Train records are split into true copies / distractors via GT. Test records are a mixture:
test_mean = pt * mu_true_test + (1 - pt) * mu_distr_test, with pt = expected true copies / records (3.46 per S1 * N_S1 per country).
Assuming distractors are generated as in train (mu_distr_test = mu_distr_train), mu_true_test = (test_mean - (1 - pt) * mu_distr_train) / pt.
Out-of-range estimates (<0 or >1 for rates) would falsify the 'distractors unchanged' assumption.
Record statistics are record-only (no pairing needed)."""
from common import *
LEG = r'\b(llc|l\.l\.c\.?|inc\.?|incorporated|corp\.?|corporation|co\.?|company|ltd\.?|limited|pvt|private|llp|lp|pc|p\.c\.|pllc|sarl|sas|sasu|eurl|sa|sci)\b'
def stats(split):
    R = pl.concat([pl.read_parquet(wp('data', f'{split}_s{s}.parquet'), columns=['id', 'name', 'addr', 'country']).with_columns(pl.lit(s, pl.Int8).alias('src')) for s in (2, 3)]).rename({'id': 'id2'})
    n = pl.col('name').fill_null(''); a = pl.col('addr').fill_null('')
    tok = n.str.split(' ').list.eval(pl.element().filter(pl.element() != '')).list.len()
    return R.with_columns(
        (a.str.strip_chars() == '').alias('no_addr'),
        tok.clip(0, 12).alias('n_tok'),
        n.str.to_lowercase().str.contains(LEG).alias('legal'),
        n.str.contains(r'[A-Za-z][0156][A-Za-z]|[A-Za-z][0156]\b').alias('leet'),
        ((n == n.str.to_uppercase()) & n.str.contains('[A-Z]')).alias('all_caps'),
        n.str.contains(r'[#@\|\[\]\*<>]|--|\(id:').alias('deco'),
        n.str.to_lowercase().str.contains(r'www\.|\.com').alias('web'),
        n.str.contains(r'[^\x00-\x7F]').alias('non_ascii'),
        a.str.split(',').list.len().clip(0, 10).alias('a_ncomp'),
        a.str.count_matches(r'\d+').clip(0, 10).alias('a_ndig'),
        n.str.len_chars().clip(0, 80).alias('n_len'),
        n.str.contains(r'  ').alias('dbl_space'))
COLS = ['no_addr', 'n_tok', 'legal', 'leet', 'all_caps', 'deco', 'web', 'non_ascii', 'a_ncomp', 'a_ndig', 'n_len', 'dbl_space']
tr = stats('train').join(load_gt_pairs().select('id2', 'src').with_columns(pl.lit(True).alias('owned')), on=['id2', 'src'], how='left').with_columns(pl.col('owned').fill_null(False))
te = stats('test')
n1tr = pl.read_parquet(wp('data', 'train_s1.parquet'), columns=['country']).group_by('country').len()
n1te = pl.read_parquet(wp('data', 'test_s1.parquet'), columns=['country']).group_by('country').len()
rows = []
for c in ('US', 'India'):
    t = tr.filter(pl.col('country') == c); x = te.filter(pl.col('country') == c)
    k_tr = t['owned'].sum() / n1tr.filter(pl.col('country') == c)['len'][0]
    pt = k_tr * n1te.filter(pl.col('country') == c)['len'][0] / x.height
    print(f'{c}: train true copies per S1 {k_tr:.3f}; train true share {t["owned"].mean():.3f}; test records {x.height}, implied test true share pt={pt:.3f}')
    for col in COLS:
        mt = t.filter(pl.col('owned'))[col].cast(pl.Float64).mean(); md = t.filter(~pl.col('owned'))[col].cast(pl.Float64).mean()
        mx = x[col].cast(pl.Float64).mean(); est = (mx - (1 - pt) * md) / pt
        rows.append((c, col, round(mt, 4), round(md, 4), round(mx, 4), round(est, 4), round(est - mt, 4)))
T = pl.DataFrame(rows, schema=['ctry', 'stat', 'train_true', 'train_distr', 'test_all', 'test_true_est', 'shift'], orient='row')
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200); print(T)
# distribution (not only mean) for name token count: implied test-true histogram vs train-true
for c in ('US', 'India'):
    t = tr.filter(pl.col('country') == c); x = te.filter(pl.col('country') == c)
    k_tr = t['owned'].sum() / n1tr.filter(pl.col('country') == c)['len'][0]; pt = k_tr * n1te.filter(pl.col('country') == c)['len'][0] / x.height
    ht = t.filter(pl.col('owned'))['n_tok'].value_counts(normalize=True).rename({'proportion': 'train_true'})
    hd = t.filter(~pl.col('owned'))['n_tok'].value_counts(normalize=True).rename({'proportion': 'train_distr'})
    hx = x['n_tok'].value_counts(normalize=True).rename({'proportion': 'test_all'})
    H = ht.join(hd, on='n_tok', how='full', coalesce=True).join(hx, on='n_tok', how='full', coalesce=True).fill_null(0).sort('n_tok')
    H = H.with_columns(((pl.col('test_all') - (1 - pt) * pl.col('train_distr')) / pt).alias('test_true_est')).with_columns(pl.col(pl.Float64).round(4))
    print(c, 'name token count distribution'); print(H)
