"""Signed house-number offset: a generator tell no existing feature carries (every number feature is |h1 - h2|).
Train fold 0, same street/nearby number (hn_rel 4): the record's number is ABOVE the S1 number for 99.0% of US and 85% of India
DISTRACTORS (planted neighbours are made by adding a positive offset), but for only ~50% of true copies (jitter is symmetric).
Test is label-free checkable: true pairs are symmetric, so the excess of accepted d > 0 over d < 0 counts accepted distractors.
Features (S1 first number h1, record first number h2): signed difference, sign, relative offset, length change, digit Hamming distance,
common prefix/suffix, single-digit substitution, transposition, anagram.
modes:  feats <split>                   -> feat/f67/<split>.parquet
        cv                              : stage-3 LightGBM on folds 1-7 (base p is out-of-fold there), scored on fold 0; control = same
                                          model without the f67 features. Official macro F0.5 on fold 0 (fold-0 pairs with both numbers
                                          get the new score, all other pairs keep m4it).
        test <out tag>                  : fit on all folds, write the log-odds shift delta = logit(q) - logit(p) for every test pair
usage: python s67_sign.py feats train|test  |  python s67_sign.py cv  |  python s67_sign.py test <out tag>"""
import sys, glob, time
import lightgbm as lgb
from common import *
mode = sys.argv[1]; t0 = time.time()
SF = ['sd', 'sg', 'srel', 'dlen', 'ham', 'cpre', 'csuf', 'sub1', 'trans', 'anag', 'sub_last']
BF = ['lp', 'lp1', 'is_india', 'is_france', 'name_rel', 'legal_rel', 'hn_rel', 'hn_ldiff', 'k_s1', 'g_src', 'g_oth', 'n_tset', 'a_tset', 'street_inter',
      'city_eq', 'dg_jac', 'anchor', 'c_at_xh', 'c_at_h1', 'x_rank', 'x_gap']
PAR = dict(objective='binary', learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
           lambda_l2=10.0, num_threads=4, verbose=-1)
R = 400

def digit_feats(h1, h2):
    """h1, h2: digit strings (may be ''), lists. Returns dict of numpy arrays."""
    n = len(h1); o = {k: np.full(n, np.nan, dtype=np.float32) for k in ('ham', 'cpre', 'csuf', 'sub1', 'trans', 'anag', 'sub_last')}
    for i, (a, b) in enumerate(zip(h1, h2)):
        if not a or not b: continue
        k = 0
        while k < min(len(a), len(b)) and a[k] == b[k]: k += 1
        s = 0
        while s < min(len(a), len(b)) - k and a[-1 - s] == b[-1 - s]: s += 1
        o['cpre'][i], o['csuf'][i] = k, s
        an = sorted(a) == sorted(b); o['anag'][i] = an
        if len(a) == len(b):
            diff = [j for j in range(len(a)) if a[j] != b[j]]
            o['ham'][i] = len(diff); o['sub1'][i] = len(diff) == 1; o['sub_last'][i] = diff == [len(a) - 1]
            o['trans'][i] = len(diff) == 2 and diff[1] == diff[0] + 1 and an
        else:
            o['ham'][i] = -1; o['sub1'][i] = 0; o['sub_last'][i] = 0; o['trans'][i] = 0
    return o

if mode == 'feats':
    split = sys.argv[2]
    S = pl.concat([pl.scan_parquet(sorted(glob.glob(wp('feat', 'f50', f'{split}_c*.parquet')))).select(KY).collect()])
    h = lambda s: pl.read_parquet(wp('norm2', f'{split}_s{s}.parquet'), columns=['id', 'dz']).select(
        pl.col('id'), pl.col('dz').str.split(' ').list.first().fill_null('').str.replace_all(r'\D', '').str.slice(0, 9).alias('h'))
    X = S.join(h(1).rename({'id': 'id1', 'h': 'h1'}), on='id1').join(pl.concat([h(s).with_columns(pl.lit(s, pl.Int8).alias('src')) for s in (2, 3)]).rename({'id': 'id2', 'h': 'h2'}), on=['id2', 'src'])
    i1, i2 = pl.col('h1').cast(pl.Int64, strict=False), pl.col('h2').cast(pl.Int64, strict=False)
    X = X.with_columns((i2 - i1).clip(-100000, 100000).cast(pl.Float32).alias('sd')).with_columns(
        pl.col('sd').sign().alias('sg'), (pl.col('sd') / i1.clip(1, None)).clip(-10, 10).cast(pl.Float32).alias('srel'),
        pl.when((pl.col('h1') != '') & (pl.col('h2') != '')).then(pl.col('h2').str.len_chars().cast(pl.Int32) - pl.col('h1').str.len_chars().cast(pl.Int32)).cast(pl.Float32).alias('dlen'))
    out = []
    for c in range(0, X.height, 2_000_000):
        x = X.slice(c, 2_000_000); o = digit_feats(x['h1'].to_list(), x['h2'].to_list())
        out.append(x.select(*KY, 'sd', 'sg', 'srel', 'dlen').with_columns(*[pl.Series(k, v) for k, v in o.items()]))
        print(f'  {c + x.height}/{X.height} ({time.time()-t0:.0f}s)', flush=True)
    os.makedirs(wp('feat', 'f67'), exist_ok=True); pl.concat(out).write_parquet(wp('feat', 'f67', f'{split}.parquet'))
    print('F67_DONE', split, X.height)
    sys.exit()

def load(split, filt):
    F = pl.scan_parquet(sorted(glob.glob(wp('feat', 'f50', f'{split}_c*.parquet')))).filter(filt).collect()
    F = F.with_columns(pl.col('p').clip(1e-6, 1 - 1e-6).alias('pc'), pl.col('p1').clip(1e-6, 1 - 1e-6).alias('p1c')).with_columns(
        (pl.col('pc') / (1 - pl.col('pc'))).log().alias('lp'), (pl.col('p1c') / (1 - pl.col('p1c'))).log().alias('lp1'),
        (pl.col('ctry') == 'India').cast(pl.Int8).alias('is_india'), (pl.col('ctry') == 'France').cast(pl.Int8).alias('is_france'))
    return F.join(pl.read_parquet(wp('feat', 'f67', f'{split}.parquet')), on=KY, how='left')

def fit(T, feats):
    return lgb.train(PAR, lgb.Dataset(T.select(feats).to_numpy().astype(np.float32), T['y'].to_numpy(), feature_name=feats, free_raw_data=True), R)

NUM = pl.col('hn_rel') >= 2                         # both sides carry a house number

if mode == 'cv':
    gt = load_gt_pairs(); v0 = pl.read_parquet(wp('data', 'train_s1ids.parquet')).filter(fold_expr('id1') == 0)
    BASE = pl.read_parquet(wp('scores', 'm4it', 'train_s2_c0.parquet'), columns=[*KY, 'p'])
    Tr = load('train', (pl.col('fold') != 0) & NUM); Va = load('train', (pl.col('fold') == 0) & NUM)
    print(f'train pairs {Tr.height}, fold-0 pairs {Va.height} ({time.time()-t0:.0f}s)', flush=True)
    res = {}
    for name, feats in (('control', BF), ('+sign', BF + SF)):
        m = fit(Tr, feats); q = m.predict(Va.select(feats).to_numpy().astype(np.float32), num_threads=4)
        V = BASE.join(Va.select(KY), on=KY, how='anti').vstack(Va.select(*KY, pl.Series('p', q).cast(pl.Float32)))
        y = Va['y'].to_numpy(); ll = float(-(y * np.log(np.clip(q, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - q, 1e-6, 1))).mean())
        sc = {t: macro_f05(decide(V, t), gt, v0, by='ctry') for t in (0.6, 0.7, 0.8)}
        print(f'{name:8s} logloss {ll:.5f} | ' + ' '.join(f'{t}:{v["macro"]:.5f}' for t, v in sc.items()) + ' ' + str([(c, round(v, 5)) for c, v, _ in sc[0.7]['by'].rows()]) + f' ({time.time()-t0:.0f}s)', flush=True)
        if name == '+sign': print('   top gain:', sorted(zip(feats, m.feature_importance('gain').round()), key=lambda x: -x[1])[:14])
        res[name] = V
    yb = Va['y'].to_numpy(); pb = Va['p'].to_numpy()
    print(f'm4it as is logloss {float(-(yb * np.log(np.clip(pb, 1e-6, 1)) + (1 - yb) * np.log(np.clip(1 - pb, 1e-6, 1))).mean()):.5f} | '
          + ' '.join(f'{t}:{macro_f05(decide(BASE, t), gt, v0)["macro"]:.5f}' for t in (0.6, 0.7, 0.8)))
    os.makedirs(wp('scores', 's67cv'), exist_ok=True); res['+sign'].write_parquet(wp('scores', 's67cv', 'train_s2_c0.parquet'))
    res['control'].write_parquet(wp('scores', 's67cv', 'train_control.parquet'))

if mode == 'test':
    out = sys.argv[2]
    Tr = load('train', NUM); Te = load('test', NUM)
    d = {}
    for name, feats in (('control', BF), ('sign', BF + SF)):
        m = fit(Tr, feats); q = np.clip(m.predict(Te.select(feats).to_numpy().astype(np.float32), num_threads=4), 1e-6, 1 - 1e-6)
        d[name] = np.log(q / (1 - q)) - Te['lp'].to_numpy()
        print(f'{name}: fitted ({time.time()-t0:.0f}s)', flush=True)
    os.makedirs(wp('scores', out), exist_ok=True)
    Te.select(*KY, 'ctry', 'p', 'sd').with_columns(pl.Series('delta', d['sign']).cast(pl.Float32), pl.Series('delta_ctl', d['control']).cast(pl.Float32)) \
      .write_parquet(wp('scores', out, 'test_delta.parquet'))
    print('deltas written ->', out)
