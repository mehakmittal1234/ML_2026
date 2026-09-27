"""Learn Indic-script -> Latin token dictionaries from TRAINING ground-truth pairs (port of v4 learn_indic.py + learn_indic_final.py).
Writes work/data/indic_dict_raw.pkl ({'name','addr'}: tok -> (latin, dice, count)) and work/data/indic_name_dict.pkl (tok -> latin)."""
import re, unicodedata, collections, pickle, random, time
import numpy as np
from rapidfuzz import fuzz, process
from common import *
from translit import translit, skeleton

t0 = time.time()
INDIC = re.compile(r'[ऀ-෿]')
def lat_toks(s):
    s = unicodedata.normalize('NFKD', s.lower()); s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    return re.findall(r'[a-z0-9]+', s)
def ind_toks(s): return [t.strip('()[]{},;:"\'') for t in s.split() if INDIC.search(t)]

pairs = load_gt_pairs()
s1 = pl.read_parquet(wp('data', 'train_s1.parquet'), columns=['id', 'name', 'addr']).rename({'id': 'id1', 'name': 'n1', 'addr': 'a1'})
# ---------------- (1) raw dice dictionaries for name and addr (v4 learn_indic.py)
res = {}
for field in ['name', 'addr']:
    cij = collections.Counter(); ci = collections.Counter(); cl = collections.Counter(); npair = 0
    for s in (2, 3):
        b = pl.read_parquet(wp('data', f'train_s{s}.parquet'), columns=['id', field, 'country']).filter(pl.col('country') == 'India')
        b = b.filter(pl.col(field).str.contains(r'[\x{0900}-\x{0DFF}]')).rename({'id': 'id2', field: 'x'}).drop('country')
        j = b.join(pairs.filter(pl.col('src') == s), on='id2').join(s1, on='id1')
        col = 'n1' if field == 'name' else 'a1'
        for x, y in zip(j['x'].to_list(), j[col].to_list()):
            I = set(ind_toks(x)); L = set(lat_toks(y)); npair += 1
            for i in I:
                ci[i] += 1
                for l in L: cij[(i, l)] += 1
            for l in L: cl[l] += 1
        del b, j
    best = {}
    for (i, l), c in cij.items():
        d = 2 * c / (ci[i] + cl[l])
        if i not in best or d > best[i][1]: best[i] = (l, d, c)
    res[field] = best
    print(field, 'pairs:', npair, 'indic tokens:', len(best), f'({time.time()-t0:.0f}s)', flush=True)
pickle.dump(res, open(wp('data', 'indic_dict_raw.pkl'), 'wb'))

# ---------------- (2) name dictionary (v4 learn_indic_final.py), vectorised vocabulary match
def phon(tr, l): return 2 * fuzz.ratio(skeleton(tr), skeleton(l)) + fuzz.ratio(tr, l)
cij = collections.Counter(); ci = collections.Counter(); all_indic = collections.Counter(); latin = collections.Counter()
for sp in ('train', 'test'):
    for s in (1, 2, 3):
        b = pl.read_parquet(wp('data', f'{sp}_s{s}.parquet'), columns=['id', 'name', 'country']).filter(pl.col('country') == 'India')
        isind = b['name'].str.contains(r'[\x{0900}-\x{0DFF}]')
        lat = b.filter(~isind)['name'].str.to_lowercase().str.extract_all(r'[a-z]{3,}').explode().drop_nulls().value_counts()
        for t, c in zip(lat[lat.columns[0]].to_list(), lat['count'].to_list()): latin[t] += c
        b = b.filter(isind)
        for n in b['name'].to_list():
            for t in ind_toks(n): all_indic[t] += 1
        if sp == 'train' and s != 1:
            j = b.rename({'id': 'id2'}).join(pairs.filter(pl.col('src') == s), on='id2').join(s1, on='id1')
            for x, y in zip(j['name'].to_list(), j['n1'].to_list()):
                I = set(ind_toks(x)); L = set(lat_toks(y))
                for i in I:
                    ci[i] += 1
                    for l in L: cij[(i, l)] += 1
        del b
print('counts done', len(all_indic), 'indic name tokens', f'({time.time()-t0:.0f}s)', flush=True)
cand = collections.defaultdict(list)
for (i, l), c in cij.items(): cand[i].append((l, c / ci[i]))
learned = {}
for i, lst in cand.items():
    if ci[i] < 3: continue
    mx = max(p for _, p in lst); short = [l for l, p in lst if p >= 0.8 * mx]; tr = translit(i)
    learned[i] = max(short, key=lambda l: phon(tr, l))
vocab = [t for t, c in latin.items() if c >= 200]; vsk = [skeleton(t) for t in vocab]

def big_match_many(us):
    trs = [translit(u) for u in us]; sks = [skeleton(t) for t in trs]
    out = []
    for a in range(0, len(us), 1000):
        m1 = process.cdist(sks[a:a+1000], vsk, scorer=fuzz.ratio, dtype=np.float32, workers=-1)
        m2 = process.cdist(trs[a:a+1000], vocab, scorer=fuzz.ratio, dtype=np.float32, workers=-1)
        sc = (2 * m1 + m2) / 3; bi = sc.argmax(1)
        out += [(vocab[k], float(sc[r, k]), trs[a + r]) for r, k in enumerate(bi)]
    return out

random.seed(1); samp = random.sample(sorted(learned), 250)
bm = big_match_many(samp)
for th in (60, 70, 75, 80, 85):
    acc = [p[0] == learned[k] for p, k in zip(bm, samp) if p[1] >= th]
    print(f'thr {th}: coverage {len(acc)/len(samp):.2f}, precision {sum(acc)/max(1,len(acc)):.3f}', flush=True)
final = dict(learned); unl = [u for u in all_indic if u not in learned]
TH = 75
for u, (best, sc, tr) in zip(unl, big_match_many(unl)):
    final[u] = best if sc >= TH else tr
print('vocab:', len(vocab), 'learned:', len(learned), 'unlearned:', len(unl), f'({time.time()-t0:.0f}s)', flush=True)
pickle.dump(final, open(wp('data', 'indic_name_dict.pkl'), 'wb'))
print('INDIC_DONE')
