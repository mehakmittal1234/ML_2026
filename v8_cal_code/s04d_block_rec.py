"""Record-side (id2) blocking: for S2/S3 records, rank ALL S1 candidates sharing an uncapped key and keep the top M per record.
Each S2/S3 record has at most one owner, so ranking from the record's side is not crowded out by same-name S1 entities.
usage: python s04d_block_rec.py <split> <keys dir> <id2 filter: all | gt:<folds>> <tag> <CAP2> <CAPP> <M> <NP>
writes cand/<tag>/<split>_s{s}_q{k}.parquet : id1 id2 nk ntyp kmask wsum wmax rrank"""
import duckdb, sys, time, glob
from common import *
split, KD, filt, tag, CAP2, CAPP, M, NP = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8])
KEEPTRUE = len(sys.argv) > 9 and sys.argv[9] == 'keeptrue'
K = wp(KD) + '/'; OUT = wp('cand', tag); os.makedirs(OUT, exist_ok=True)
for f in glob.glob(OUT + f'/{split}_s[23]_q*.parquet'): os.remove(f)
con = duckdb.connect()
con.execute(f"SET memory_limit='3500MB'; SET threads=6; SET temp_directory='{wp('tmp')}'; SET preserve_insertion_order=false; SET enable_progress_bar=false")
t0 = time.time()
con.execute(f"CREATE TABLE k1 AS SELECT key, id::BIGINT id, kt FROM '{K}{split}_s1/*.parquet'")
con.execute("CREATE TABLE c1 AS SELECT key, count(*)::INTEGER c1 FROM k1 GROUP BY key")
for s in (2, 3):
    src_f = f"'{K}{split}_s{s}/*.parquet'"
    if filt.startswith('gt:') or filt.startswith('gtplus:'):
        # gt:<folds> = records owned by S1 of those folds; gtplus:<folds>:<mod> = those + a 1/mod hash sample of ALL records
        parts = filt.split(':'); fo = [int(x) for x in parts[1].split(',')]
        ids = load_gt_pairs().filter(pl.col('src') == s).join(pl.read_parquet(wp('data', 'train_s1ids.parquet')).filter(fold_expr('id1').is_in(fo)), on='id1', how='semi').select(pl.col('id2').alias('id'))
        if filt.startswith('gtplus:'):
            allr = pl.read_parquet(wp('data', f'{split}_s{s}.parquet'), columns=['id'])
            ids = pl.concat([ids, allr.filter(pl.col('id').hash(seed=17) % int(parts[2]) == 0)]).unique()
        con.register('f2', ids.to_arrow()); flt = "WHERE b.id IN (SELECT id FROM f2)"
    else:
        flt = ''
    con.execute(f"""CREATE OR REPLACE TABLE kk AS SELECT key, (1.0/(c1*c2))::FLOAT w FROM
                    (SELECT b.key, count(*) c2 FROM {src_f} b SEMI JOIN c1 USING (key) GROUP BY b.key) JOIN c1 USING (key)
                    WHERE c2 <= {CAP2} AND c1*c2 <= {CAPP}""")
    con.execute(f"CREATE OR REPLACE TABLE ks2 AS SELECT b.key, b.id::BIGINT id, kk.w FROM {src_f} b JOIN kk USING (key) {flt}")
    con.execute(f"CREATE OR REPLACE TABLE k1s AS SELECT k1.* FROM k1 SEMI JOIN (SELECT DISTINCT key FROM ks2) USING (key)")
    if KEEPTRUE:
        con.register('gtp_', load_gt_pairs().filter(pl.col('src') == s).select('id1', 'id2').to_arrow()); con.execute("CREATE OR REPLACE TABLE gtp AS SELECT * FROM gtp_")
    tot = 0
    for p in range(NP):
        fn = f'{OUT}/{split}_s{s}_q{p}.parquet'
        con.execute(f"""COPY (
            SELECT id1, id2, nk, bit_count(km)::TINYINT ntyp, km::INTEGER kmask, wsum, wmax,
                   row_number() OVER (PARTITION BY id2 ORDER BY bit_count(km) DESC, wsum DESC, id1)::SMALLINT rrank
            FROM (SELECT a.id id1, b.id id2, count(*)::SMALLINT nk, bit_or((1::INTEGER) << a.kt) km, sum(b.w)::FLOAT wsum, max(b.w)::FLOAT wmax
                  FROM k1s a JOIN (SELECT * FROM ks2 WHERE id % {NP} = {p}) b USING (key) GROUP BY 1, 2)
            QUALIFY rrank <= {M} {"OR (id1, id2) IN (SELECT (id1, id2) FROM gtp)" if KEEPTRUE else ""}) TO '{fn}' (FORMAT PARQUET)""")
        n = con.execute(f"SELECT count(*) FROM '{fn}'").fetchone()[0]; tot += n
        print(f'  s{s} part {p}: {n} pairs ({time.time()-t0:.0f}s)', flush=True)
    print(f's{s}: pairs kept {tot} ({time.time()-t0:.0f}s)', flush=True)
print('BLOCK2_DONE', f'{time.time()-t0:.0f}s', flush=True)
