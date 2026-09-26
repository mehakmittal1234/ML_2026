"""Write the v6 submission files (does NOT submit anything).
matching_results.tsv : each S2/S3 record -> its highest-scoring S1 if score >= thr (stage-2 scores; stage-1 for pairs below stage-2 floor never pass)
candidate_pairs.tsv  : every pair the matching model scored (retrieval top-K per record), one row per S1, comma list (README format)
usage: python s13_submit.py <score tag> <thr> <cand feat tag> <outdir>"""
import sys, time, duckdb
from common import *
mtag, thr, ftag, OUT = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
os.makedirs(OUT, exist_ok=True); t0 = time.time()
S = pl.read_parquet(wp('scores', mtag, 'test_s2_c0.parquet'), columns=[*KY, 'p'])
M = decide(S, thr, 'p')
s1 = pl.read_parquet(wp('data', 'test_s1.parquet'), columns=['entity_id', 'id'])        # original file order
agg = M.sort(['id1', 'src', 'id2']).with_columns(pl.concat_str([pl.lit('S'), pl.col('src').cast(pl.Utf8), pl.lit('-'), pl.col('id2').cast(pl.Utf8)]).alias('e2')) \
       .group_by('id1', maintain_order=True).agg(pl.col('e2').str.join(',').alias('matched_entity_ids'))
res = s1.join(agg, left_on='id', right_on='id1', how='left', maintain_order='left').with_columns(pl.col('matched_entity_ids').fill_null(''))
res.select(pl.col('entity_id').alias('source1_entity_id'), 'matched_entity_ids').write_csv(os.path.join(OUT, 'matching_results.tsv'), separator='\t', quote_style='never')
print(f'matching_results: {res.height} S1 rows, {agg.height} non-empty, {M.height} matched pairs ({time.time()-t0:.0f}s)', flush=True)
# candidate file: chunked by id1 hash (a sorted string_agg over ~76M pairs exceeds memory); row order is irrelevant to the validator
cand = pl.concat([pl.scan_parquet(wp('feat', ftag, f'test_s{s}_c*.parquet')).select('id1', 'id2', pl.lit(s, pl.Int8).alias('src')) for s in (2, 3)])
n, seen = 0, []
with open(os.path.join(OUT, 'candidate_pairs.tsv'), 'w', encoding='utf-8') as fh:
    fh.write('source1_entity_id\tcandidate_entity_ids\n')
    for k in range(8):
        c = cand.filter(pl.col('id1') % 8 == k).collect().sort(['id1', 'src', 'id2'])
        n += c.height
        g = c.with_columns(pl.concat_str([pl.lit('S'), pl.col('src').cast(pl.Utf8), pl.lit('-'), pl.col('id2').cast(pl.Utf8)]).alias('e')) \
             .group_by('id1', maintain_order=True).agg(pl.col('e').str.join(','))
        g = g.join(s1.select('id', 'entity_id'), left_on='id1', right_on='id', how='inner')
        fh.write(''.join(f'{a}\t{b}\n' for a, b in zip(g['entity_id'].to_list(), g['e'].to_list())))
        seen.append(g.select('id1')); del c, g
    miss = s1.join(pl.concat(seen), left_on='id', right_on='id1', how='anti')
    fh.write(''.join(f'{a}\t\n' for a in miss['entity_id'].to_list()))
print(f'candidate_pairs: {n} pairs ({n/s1.height:.1f}/S1), {miss.height} S1 without candidates ({time.time()-t0:.0f}s)', flush=True)
