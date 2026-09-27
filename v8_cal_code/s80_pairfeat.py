"""Pair-intrinsic features for every stage-2 pair (train 8.7M, test 7.7M), for a shift-robust re-scorer.
The base score chain uses S1-context / count features over all retrieved candidates, and split-dependent vocabularies (tells noise words
differ between train and test: e.g. 'holdings', 'downtown' become noise words on test). Those inputs shift on test (40% distractors vs
26%). Here only pair-level evidence is computed:
  feats2.compute  : stage-1 pair features (name / address similarities, IDF overlaps, house-number, legal-form, city relations)
  raw-name        : raw-form name similarities and raw house-number suffix / prefix relations (s11 build_x), plus the record-level
                    rank / gap of the raw-name ratio among the record's candidates
--trainidf (test only): IDF weights from the TRAIN corpus for US / India tokens (France keeps test IDF, it has no train data), written to
feat/f80t/. Split-specific IDF is a test-only failure: 'service' has IDF 5.02 on train and 5.46 on test (its document frequency share
drops with the test's distractor mix), which pushes it over the model's learned noise-word / real-word boundary (SHAP -6.3 vs -0.9).
usage: python s80_pairfeat.py <split> [--trainidf]   -> feat/f80[t]/<split>_c<k>.parquet"""
import sys, glob, time
from rapidfuzz import process, fuzz
from common import *
import feats2
from s11_stage2 import raw_names
split = sys.argv[1]; t0 = time.time()
TRIDF = '--trainidf' in sys.argv
OUT = wp('feat', 'f80t' if TRIDF else 'f80'); os.makedirs(OUT, exist_ok=True)
S = pl.scan_parquet(sorted(glob.glob(wp('feat', 'f50', f'{split}_c*.parquet')))).select(KY).collect()
ctx = feats2.Ctx(split)
if TRIDF:
    tr = feats2.Ctx('train')
    ctx.dfn = pl.concat([tr.dfn, ctx.dfn.filter(~pl.col('ctry').is_in(tr.dfn['ctry'].unique().implode()))])
    ctx.dfa = pl.concat([tr.dfa, ctx.dfa.filter(~pl.col('ctry').is_in(tr.dfa['ctry'].unique().implode()))])
    ctx.WMAX = tr.WMAX; del tr
    print('IDF: train tables for', sorted(ctx.dfn['ctry'].unique().to_list()), flush=True)
print(f'{split}: {S.height} pairs, ctx loaded ({time.time()-t0:.0f}s)', flush=True)
DROPC = ['nm_cnt_s1_of1', 'nm_cnt_s1_of2', 'nm_cnt_src_of2', 'amb_a', 'amb_b']      # corpus counts: shift with the distractor share
parts = []
for k in range(8):
    fn = os.path.join(OUT, f'{split}_c{k}.parquet')
    P = S.filter(pl.col('id1') % 8 == k)
    if not os.path.exists(fn):
        F = feats2.pair_features(P, ctx).drop([c for c in DROPC])
        A = raw_names(split, 1, P.select(pl.col('id1').unique().alias('id')))
        out = []
        for s in (2, 3):
            q = P.filter(pl.col('src') == s)
            B = raw_names(split, s, q.select(pl.col('id2').unique().alias('id')))
            q = q.join(A.rename({'rn': 'rn1', 'hr': 'hr1'}), left_on='id1', right_on='id', how='left').join(B.rename({'rn': 'rn2', 'hr': 'hr2'}), left_on='id2', right_on='id', how='left')
            a, b = q['rn1'].fill_null('').to_list(), q['rn2'].fill_null('').to_list()
            q = q.with_columns(pl.Series('rn_ratio', process.cpdist(a, b, scorer=fuzz.ratio, workers=-1, dtype=np.float32)),
                               pl.Series('rn_tset', process.cpdist(a, b, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)),
                               pl.Series('rn_tsort', process.cpdist(a, b, scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.float32)),
                               (pl.col('rn1') == pl.col('rn2')).cast(pl.Float32).alias('rn_eq'))
            h1, h2 = pl.col('hr1').fill_null(''), pl.col('hr2').fill_null('')
            both = (h1 != '') & (h2 != '') & (h1 != h2)
            q = q.with_columns(pl.when(both).then((h1.str.ends_with(h2) | h2.str.ends_with(h1)).cast(pl.Float32)).alias('hr_sfx'),
                               pl.when(both).then((h1.str.starts_with(h2) | h2.str.starts_with(h1)).cast(pl.Float32)).alias('hr_pfx'),
                               pl.when(both).then((h1.str.slice(-2) == h2.str.slice(-2)).cast(pl.Float32)).alias('hr_last2'),
                               pl.when(both).then(h2.str.starts_with('0').cast(pl.Float32)).alias('hr2_lead0'))
            out.append(q.select(*KY, 'rn_ratio', 'rn_tset', 'rn_tsort', 'rn_eq', 'hr_sfx', 'hr_pfx', 'hr_last2', 'hr2_lead0'))
        F = F.join(pl.concat(out), on=KY, how='left')
        F.write_parquet(fn); del F, A, B, out
    print(f'  chunk {k}: {P.height} pairs ({time.time()-t0:.0f}s)', flush=True)
# record-level context of the raw-name ratio (within the record's own candidate list: not a count over other records)
X = pl.scan_parquet(sorted(glob.glob(os.path.join(OUT, f'{split}_c*.parquet')))).select(*KY, 'rn_ratio', 'n_tset', 'a_tset').collect()
X = X.with_columns(pl.col('rn_ratio').max().over(['id2', 'src']).alias('_mx'), pl.col('n_tset').max().over(['id2', 'src']).alias('_mn'),
                   pl.col('a_tset').max().over(['id2', 'src']).alias('_ma'))
X = X.select(*KY, (pl.col('rn_ratio') - pl.col('_mx')).alias('rn_gap_top'), (pl.col('n_tset') - pl.col('_mn')).alias('nt_gap_top'),
             (pl.col('a_tset') - pl.col('_ma')).alias('at_gap_top'), pl.col('rn_ratio').rank('min', descending=True).over(['id2', 'src']).cast(pl.Float32).alias('rn_rank_rec'),
             pl.len().over(['id2', 'src']).cast(pl.Float32).alias('rec_ncand'))
X.write_parquet(os.path.join(OUT, f'{split}_rec.parquet'))
print('F80_DONE', split, f'{time.time()-t0:.0f}s')
