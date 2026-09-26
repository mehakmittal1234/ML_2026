"""Blocking keys v2 (from norm2). v4 key types (0-13, identical recipes but zero-stripped digit groups) + targeted families:
  14 R1: each of the 3 rarest name tokens alone         (typo / replaced word in the other name tokens; name-only records)
  15 A1: each of the 3 rarest address alpha tokens alone (invented brand name + distinctive address word)
  16 HS: house number x street word                     (same street address, different/alternate city)
  17 ND: rarest 2 name tokens x each of first 3 digit groups (junk numbers inserted before the real house number)
  18 HZ: each of first 3 digit groups x rarest 2 address tokens (same, address side)
All keys are subject to the same frequency caps at join time (common tokens never join).
usage: python s03b_keys2.py <split>  -> keys2/<split>_s{s}/part_*.parquet (key u64, id i32, kt u8)"""
import polars as pl, sys, time, os
from common import wp
N = wp('norm2') + '/'; K = wp('keys2') + '/'
split = sys.argv[1]
tables = [f'{split}_s1', f'{split}_s2', f'{split}_s3']
KT = {'N1': 0, 'N2': 1, 'NA': 2, 'AA': 3, 'HA': 4, 'NH': 5, 'DA': 6, 'D1': 7, 'NX': 8, 'DG': 9, 'D3': 10, 'DH': 11, 'NC': 12, 'NP': 13,
      'R1': 14, 'A1': 15, 'HS': 16, 'ND': 17, 'HZ': 18}
STY = ['st','ave','rd','dr','ln','ct','cir','blvd','pl','pkwy','hwy','ter','trl','sq','way','pt','xing','plz','ctr','hts','expy','fwy','tpke','jct','mt','mtn','ft',
       'n','s','e','w','ne','nw','se','sw','r','av','bd','rte','ch','imp','all','quai','crs','fg','res','lot','pass','prom','ham','bis','q','ngr','col','sec','main',
       'cross','road','street','lane','marg','nagar','extn','ph','block','floor','fl','grd','no','ste','apt','unit','bldg','rm','box','po']
t0 = time.time()

def tokcol(col, alpha):
    e = pl.col(col).str.split(' ').list.unique()
    e = e.list.eval(pl.element().filter((pl.element().str.len_chars() >= 2) & (pl.element() != ',')))
    if alpha:
        e = e.list.eval(pl.element().filter(~pl.element().str.contains(r'\d')))
    return e

def doc_freq(col, alpha):
    parts = []
    for t in tables:
        lf = pl.scan_parquet(N + t + '.parquet').select(pl.col('ctry'), tokcol(col, alpha).alias('tok')).explode('tok')
        parts.append(lf.drop_nulls('tok').group_by(['ctry', 'tok']).agg(pl.len().alias('df')))
    return pl.concat(parts).group_by(['ctry', 'tok']).agg(pl.col('df').sum().cast(pl.UInt32)).collect(engine='streaming')

os.makedirs(K, exist_ok=True)
if os.path.exists(N + f'{split}_df_addr.parquet'):
    dfn = pl.read_parquet(N + f'{split}_df_name.parquet'); dfa = pl.read_parquet(N + f'{split}_df_addr.parquet')
else:
    dfn = doc_freq('nc', False); dfn.write_parquet(N + f'{split}_df_name.parquet')
    dfa = doc_freq('at', True); dfa.write_parquet(N + f'{split}_df_addr.parquet')
print(f'df done: name {dfn.height} addr {dfa.height} ({time.time()-t0:.0f}s)', flush=True)
dfn = dfn.filter(pl.col('df') >= 2); dfa = dfa.filter(pl.col('df') >= 2)

def top3(ch, col, dfx, alpha, name):
    x = ch.select('id', 'ctry', tokcol(col, alpha).alias('tok')).explode('tok').drop_nulls('tok')
    x = x.join(dfx, on=['ctry', 'tok'], how='inner').sort(['id', 'df', 'tok'])
    return x.group_by('id', maintain_order=True).agg(pl.col('tok').head(3).alias(name))

def P(*parts):
    return pl.concat_str([p if isinstance(p, pl.Expr) else pl.lit(p) for p in parts])

CH = 400_000
for t in tables:
    src = int(t[-1]); lf = pl.scan_parquet(N + t + '.parquet')
    n = lf.select(pl.len()).collect().item(); out = 0
    if os.path.exists(K + t + '/DONE'):
        print(t, 'exists, skipping', flush=True); continue
    os.makedirs(K + t, exist_ok=True)
    for off in range(0, n, CH):
        ch = lf.slice(off, CH).collect()
        ch = ch.join(top3(ch, 'nc', dfn, False, 'nr'), on='id', how='left') \
               .join(top3(ch, 'at', dfa, True, 'ar'), on='id', how='left') \
               .join(top3(ch, 'na', dfn, False, 'mr'), on='id', how='left')
        g = lambda c, i: pl.col(c).list.get(i, null_on_oob=True)
        ch = ch.with_columns(
            pl.col('nr').list.sort().alias('ns'), pl.col('ar').list.sort().alias('as_'), pl.col('mr').list.sort().alias('ms'),
            pl.col('dz').str.split(' ').list.eval(pl.element().filter(pl.element() != '')).list.unique(maintain_order=True).alias('dl'),
            pl.when(pl.col('dom') != '').then(pl.col('dom').str.slice(0, 8))
              .otherwise(pl.when(pl.lit(src == 1)).then(pl.col('nc').str.replace_all(' ', '').str.slice(0, 8))).alias('dp'))
        comps = pl.col('at').str.split(' , ')
        cityc = comps.list.slice(0, comps.list.len() - 1).list.eval(pl.element().filter(~pl.element().str.contains(r'\d') & (pl.element() != ''))).list.last()
        stc = comps.list.eval(pl.element().filter(pl.element().str.contains(r'\d'))).list.first().fill_null('')
        ch = ch.with_columns(
            pl.col('ns').list.eval(pl.element().filter(pl.element().str.len_chars() >= 5).str.slice(0, 4)).alias('npf'),
            cityc.fill_null('').str.split(' ').list.eval(pl.element().filter(pl.element().str.len_chars() >= 3)).list.eval(pl.element().sort_by(pl.element().str.len_chars(), descending=True)).list.first().alias('ctk'),
            pl.col('at').str.replace_all(' , ', ' ').str.split(' ').list.eval(pl.element().filter((pl.element().str.len_chars() >= 3) & ~pl.element().str.contains(r'\d'))).list.unique(maintain_order=True).alias('atk'),
            stc.str.split(' ').list.eval(pl.element().filter(pl.element().str.contains(r'^[a-z]{3,}$') & ~pl.element().is_in(STY))).list.unique(maintain_order=True).alias('stw'))
        c = pl.col('ctry')
        n0, n1, n2 = g('ns', 0), g('ns', 1), g('ns', 2); r0, r1, r2 = g('nr', 0), g('nr', 1), g('nr', 2)
        a0, a1, a2 = g('as_', 0), g('as_', 1), g('as_', 2); q0, q1, q2 = g('ar', 0), g('ar', 1), g('ar', 2)
        m0, m1, m2 = g('ms', 0), g('ms', 1), g('ms', 2); w0, w1 = g('mr', 0), g('mr', 1)
        d0, d1, d2 = g('dl', 0), g('dl', 1), g('dl', 2); dp = pl.col('dp')
        nlen = pl.col('ns').list.len()
        K_ = {
            'N1_0': pl.when(nlen == 1).then(P(c, '|N1|', n0)),
            'N2_0': P(c, '|N2|', n0, '|', n1), 'N2_1': P(c, '|N2|', n0, '|', n2), 'N2_2': P(c, '|N2|', n1, '|', n2),
            'NA_0': P(c, '|NA|', r0, '|', q0), 'NA_1': P(c, '|NA|', r0, '|', q1),
            'NA_2': P(c, '|NA|', r1, '|', q0), 'NA_3': P(c, '|NA|', r1, '|', q1),
            'AA_0': P(c, '|AA|', a0, '|', a1), 'AA_1': P(c, '|AA|', a0, '|', a2), 'AA_2': P(c, '|AA|', a1, '|', a2),
            'HA_0': P(c, '|HA|', d0, '|', q0), 'HA_1': P(c, '|HA|', d0, '|', q1),
            'HA_2': P(c, '|HA|', d1, '|', q0), 'HA_3': P(c, '|HA|', d1, '|', q1),
            'NH_0': P(c, '|NH|', r0, '|', d0), 'NH_1': P(c, '|NH|', r1, '|', d0),
            'DA_0': P(c, '|DA|', dp, '|', q0), 'DA_1': P(c, '|DA|', dp, '|', q1),
            'D1_0': pl.when((src == 1) | (pl.col('at') == '') | (pl.col('dom') != '')).then(P(c, '|D1|', dp)),
            'NX_0': pl.when(pl.col('nc') != '').then(P(c, '|NX|', pl.col('nc'))),
            'DG_0': pl.when(pl.col('dl').list.len() >= 2).then(P(c, '|DG|', d0, ' ', d1)),
            'D3_0': pl.when(pl.col('dl').list.len() >= 3).then(P(c, '|D3|', d0, ' ', d1, ' ', d2)),
            'DH_0': P(c, '|DH|', dp, '|', d0),
            'NP_0': P(c, '|NP|', g('npf', 0), '|', g('npf', 1)), 'NP_1': P(c, '|NP|', g('npf', 0), '|', g('npf', 2)), 'NP_2': P(c, '|NP|', g('npf', 1), '|', g('npf', 2)),
            # ---- new targeted families
            'R1_0': P(c, '|R1|', r0), 'R1_1': P(c, '|R1|', r1), 'R1_2': P(c, '|R1|', r2),
            'A1_0': P(c, '|A1|', q0), 'A1_1': P(c, '|A1|', q1), 'A1_2': P(c, '|A1|', q2),
            'HS_0': P(c, '|HS|', d0, '|', g('stw', 0)), 'HS_1': P(c, '|HS|', d0, '|', g('stw', 1)), 'HS_2': P(c, '|HS|', d0, '|', g('stw', 2)),
        }
        for i, dd in enumerate((d0, d1, d2)):
            K_[f'ND_{2*i}'] = P(c, '|ND|', r0, '|', dd); K_[f'ND_{2*i+1}'] = P(c, '|ND|', r1, '|', dd)
            K_[f'HZ_{2*i}'] = P(c, '|HZ|', dd, '|', q0); K_[f'HZ_{2*i+1}'] = P(c, '|HZ|', dd, '|', q1)
        nce = pl.when(pl.col('nc') != '')
        if src == 1:
            K_['NC_0'] = nce.then(P(c, '|NC|', pl.col('nc'), '|', pl.col('ctk')))
        else:
            for i in range(6):
                K_[f'NC_{i}'] = nce.then(P(c, '|NC|', pl.col('nc'), '|', g('atk', i)))
            K_.update({'N2_3': P(c, '|N2|', m0, '|', m1), 'N2_4': P(c, '|N2|', m0, '|', m2), 'N2_5': P(c, '|N2|', m1, '|', m2),
                       'NA_4': P(c, '|NA|', w0, '|', q0), 'NA_5': P(c, '|NA|', w1, '|', q0),
                       'NX_1': pl.when(pl.col('na') != '').then(P(c, '|NX|', pl.col('na'))),
                       'R1_3': P(c, '|R1|', w0), 'R1_4': P(c, '|R1|', w1)})
        kk = ch.select('id', *[e.alias(k) for k, e in K_.items()]).unpivot(index='id', variable_name='v', value_name='ks').drop_nulls('ks')
        kk = kk.select(pl.col('ks').hash(seed=7).alias('key'), pl.col('id').cast(pl.Int32),
                       pl.col('v').str.slice(0, 2).replace_strict(KT, return_dtype=pl.UInt8).alias('kt')).unique(['key', 'id'])
        kk.write_parquet(K + t + f'/part_{off//CH:03d}.parquet', compression='zstd'); out += kk.height
        del kk, ch
    open(K + t + '/DONE', 'w').write('ok')
    print(f'{t}: {n} recs, {out} keys ({out/n:.1f}/rec) ({time.time()-t0:.0f}s)', flush=True)
print('KEYS_DONE', flush=True)
