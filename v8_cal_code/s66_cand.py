"""candidate_pairs.tsv from the STAGE-2 pair set (the set fed to the final matching model: stage-2 LightGBM, specialist, hybrid, decoder).
Organiser answer (participant FAQ): candidate_pairs.tsv holds the final candidate set actually fed to the matching model, a pruned later
stage is fine, every matched ID must appear in it, and smaller candidate sets per Source 1 entity rank higher. v8_cal wrote the full
retrieval set (top-8 per S2/S3 record, ~44 pairs per S1); the stage-2 set is ~4.4 per S1.
Checks that every pair of <outdir>/matching_results.tsv is in the set, refuses to overwrite an existing candidate file.
usage: python s66_cand.py <stage-2 score tag> <outdir> [<outdir> ...]"""
import sys, time
from common import *
tag, outs = sys.argv[1], sys.argv[2:]
t0 = time.time()
S = pl.read_parquet(wp('scores', tag, 'test_s2_c0.parquet'), columns=KY).unique()
s1 = pl.read_parquet(wp('data', 'test_s1.parquet'), columns=['entity_id', 'id'])
e = pl.concat_str([pl.lit('S'), pl.col('src').cast(pl.Utf8), pl.lit('-'), pl.col('id2').cast(pl.Utf8)])
C = S.sort(['id1', 'src', 'id2']).group_by('id1', maintain_order=True).agg(e.str.join(',').alias('candidate_entity_ids'))
C = s1.join(C, left_on='id', right_on='id1', how='left', maintain_order='left').with_columns(pl.col('candidate_entity_ids').fill_null(''))
Cs = S.select(pl.col('id1'), e.alias('e2'))
print(f'stage-2 set {S.height} pairs ({S.height / s1.height:.2f} per S1), {C.filter(pl.col("candidate_entity_ids") == "").height} S1 without candidates ({time.time()-t0:.0f}s)', flush=True)
for out in outs:
    fn = os.path.join(out, 'candidate_pairs.tsv')
    if os.path.exists(fn):
        print(f'{fn} exists, skipped'); continue
    M = pl.read_csv(os.path.join(out, 'matching_results.tsv'), separator='\t', infer_schema=False).join(s1, left_on='source1_entity_id', right_on='entity_id') \
        .select(pl.col('id').alias('id1'), pl.col('matched_entity_ids').str.split(',').alias('e2')).explode('e2').filter(pl.col('e2').is_not_null() & (pl.col('e2') != ''))
    miss = M.join(Cs, on=['id1', 'e2'], how='anti').height
    if miss:
        print(f'{out}: {miss} of {M.height} matched pairs NOT in the stage-2 set -> not written'); continue
    C.select(pl.col('entity_id').alias('source1_entity_id'), 'candidate_entity_ids').write_csv(fn, separator='\t', quote_style='never')
    print(f'{out}: all {M.height} matched pairs covered -> {fn} ({time.time()-t0:.0f}s)', flush=True)
