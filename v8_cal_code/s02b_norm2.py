"""norm2 = norm + targeted fixes found in the missed-pair analysis (each measurable separately):
  (1) digit-for-letter name typos: 0->o 1->l 5->s 8->b 6->g inside alphabetic tokens (84% of mixed name tokens are exactly this)
  (2) alias markers followed by punctuation ('dba:', 'formerly:') are now split (v4 regex required whitespace)
  (3) '#handle' / '@handle' / single concatenated-token names are treated like domains (get the name-prefix keys)
  (4) dz = digit groups with leading zeros stripped
writes norm2/{split}_s{s}.parquet : id ctry nt nc na dom at dg dz"""
import re, time
from common import *
import norm as N0

os.makedirs(wp('norm2'), exist_ok=True)
LEET = str.maketrans({'0': 'o', '1': 'l', '5': 's', '8': 'b', '6': 'g'})
LEET_TOK = re.compile(r'^(?=(?:.*[a-z]){2})(?!\d+(?:st|nd|rd|th)$)[a-z01568]+$')
ALIAS2 = re.compile(r'\s(?:a/k/a|aka|d/b/a|dba|doing business as|trading as|t/a|f/k/a|fka|formerly known as|formerly|also known as)\W*\s', re.I)
ALIAS_RAW = re.compile(r'(?i)(?:a/k/a|\baka|d/b/a|\bdba|doing business as|trading as|\bt/a|f/k/a|\bfka|formerly|also known as)\W*\s')

def leet(s):
    if not s or not any(ch.isdigit() for ch in s):
        return s
    return ' '.join(t.translate(LEET) if (any(ch.isdigit() for ch in t) and LEET_TOK.match(t)) else t for t in s.split(' '))

t0 = time.time()
N0.ALIAS_RE = ALIAS2          # patched alias regex (module-level, used inside norm_name)
for sp in ('train', 'test'):
    for s in (1, 2, 3):
        d = pl.read_parquet(wp('norm', f'{sp}_s{s}.parquet'))
        raw = pl.read_parquet(wp('data', f'{sp}_s{s}.parquet'), columns=['id', 'name'])
        d = d.join(raw, on='id', how='left')
        # (2) re-normalise alias-looking names with the fixed regex
        am = d['name'].str.contains(ALIAS_RAW.pattern).fill_null(False)
        sub = d.filter(am)
        if sub.height:
            res = [N0.norm_name(n) for n in sub['name'].to_list()]
            fix = pl.DataFrame({'id': sub['id'], 'nt_f': [' '.join(r[0]) for r in res], 'nc_f': [' '.join(r[1]) for r in res],
                                'na_f': [' '.join(r[2]) for r in res]})
            d = d.join(fix, on='id', how='left').with_columns(pl.coalesce('nt_f', 'nt').alias('nt'), pl.coalesce('nc_f', 'nc').alias('nc'),
                                                              pl.coalesce('na_f', 'na').alias('na')).drop('nt_f', 'nc_f', 'na_f')
        # (1) leet fix on name token strings
        d = d.with_columns(*[pl.col(c).map_elements(leet, return_dtype=pl.Utf8) for c in ('nt', 'nc', 'na')])
        # (3) handles / concatenated single-token names -> domain-like
        handle = pl.col('name').str.contains(r'^\s*[#@]') | ((pl.col('nc').str.count_matches(' ') == 0) & (pl.col('nc').str.len_chars() >= 10))
        d = d.with_columns(pl.when((pl.col('dom') == '') & handle & (s != 1)).then(pl.col('nc').str.replace_all(r'[^a-z]', '')).otherwise(pl.col('dom')).alias('dom'))
        d = d.with_columns(pl.when(pl.col('dom') != '').then(pl.col('dom').map_elements(leet, return_dtype=pl.Utf8)).otherwise(pl.col('dom')).alias('dom'))
        # (4) zero-stripped digit groups
        d = d.with_columns(pl.col('dg').str.split(' ').list.eval(pl.element().str.strip_chars_start('0').replace('', '0')).list.join(' ').alias('dz'))
        d = d.with_columns(pl.when(pl.col('dg') == '').then(pl.lit('')).otherwise(pl.col('dz')).alias('dz'))
        d.drop('name').write_parquet(wp('norm2', f'{sp}_s{s}.parquet'))
        print(sp, s, d.height, 'alias re-normalised', sub.height, f'{time.time()-t0:.0f}s', flush=True)
print('NORM2_DONE')
