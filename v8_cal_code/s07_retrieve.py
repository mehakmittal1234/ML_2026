"""v6 candidate generation (fused, per record partition; the big intermediate pool is never stored):
  record-side key join over keys3 (v4 key families on norm2 + R1/HS/ND) with frequency caps -> top-MC per record by crude order
  -> cheap fuzzy features -> stage-0 ranker -> keep the top-K S1 candidates per S2/S3 record.
usage: python s07_retrieve.py <split> <tag> <ranker tag> <CAP2> <CAPP> <MC> <K> <NP>
writes cand/<tag>/<split>_s{s}_c{p}.parquet : id1 id2 nk ntyp kmask wsum wmax rrank c_* r rr"""
import duckdb, sys, time, glob
import lightgbm as lgb
from rapidfuzz import process, fuzz
from common import *
import s06_ranker as RK

split, tag, rtag, CAP2, CAPP, MC, KEEP, NP = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8])
K = wp('keys3') + '/'; OUT = wp('cand', tag); os.makedirs(OUT, exist_ok=True)
m = lgb.Booster(model_file=wp('models', f'ranker_{rtag}.txt'))
con = duckdb.connect()
con.execute(f"SET memory_limit='3000MB'; SET threads=6; SET temp_directory='{wp('tmp')}'; SET preserve_insertion_order=false; SET enable_progress_bar=false")
t0 = time.time()
con.execute(f"CREATE TABLE k1 AS SELECT key, id::BIGINT id, kt FROM '{K}{split}_s1/*.parquet'")
con.execute("CREATE TABLE c1 AS SELECT key, count(*)::INTEGER c1 FROM k1 GROUP BY key")
def side(s):
    d = pl.read_parquet(wp('norm2', f'{split}_s{s}.parquet'), columns=['id', 'nc', 'na', 'at', 'dz'])
    return d.select('id', 'nc', 'na', pl.col('at').str.replace_all(' , ', ' ').alias('a'), pl.col('dz').str.split(' ').list.first().fill_null('').alias('h'))
S1 = side(1)
print(f'setup {time.time()-t0:.0f}s', flush=True)
for s in (2, 3):
    src_f = f"'{K}{split}_s{s}/*.parquet'"
    con.execute(f"""CREATE OR REPLACE TABLE kk AS SELECT key, (1.0/(c1*c2))::FLOAT w FROM
                    (SELECT b.key, count(*) c2 FROM {src_f} b SEMI JOIN c1 USING (key) GROUP BY b.key) JOIN c1 USING (key)
                    WHERE c2 <= {CAP2} AND c1*c2 <= {CAPP}""")
    con.execute(f"CREATE OR REPLACE TABLE k1s AS SELECT k1.* FROM k1 SEMI JOIN kk USING (key)")
    S = side(s); tot = 0
    NQ = NP // 4                                   # one key-join query serves 4 output partitions (S1 keys scanned NQ times, not NP)
    for q in range(NQ):
        parts = [p for p in range(q, NP, NQ) if not os.path.exists(f'{OUT}/{split}_s{s}_c{p}.parquet')]
        if not parts: continue
        con.execute(f"CREATE OR REPLACE TABLE ks2 AS SELECT b.key, b.id::BIGINT id, kk.w FROM {src_f} b JOIN kk USING (key) WHERE b.id % {NQ} = {q}")
        cq = con.execute(f"""
            SELECT id1, id2, nk, bit_count(km)::TINYINT ntyp, km::INTEGER kmask, wsum, wmax,
                   row_number() OVER (PARTITION BY id2 ORDER BY bit_count(km) DESC, wsum DESC, id1)::SMALLINT rrank
            FROM (SELECT a.id id1, b.id id2, count(*)::SMALLINT nk, bit_or((1::INTEGER) << a.kt) km, sum(b.w)::FLOAT wsum, max(b.w)::FLOAT wmax
                  FROM k1s a JOIN ks2 b USING (key) GROUP BY 1, 2)
            QUALIFY rrank <= {MC}""").pl()
        t1 = time.time()
        for p in parts:
          fn = f'{OUT}/{split}_s{s}_c{p}.parquet'
          c = cq.filter(pl.col('id2') % NP == p)
          x = c.select('id1', 'id2').join(S1, left_on='id1', right_on='id', how='left').join(S, left_on='id2', right_on='id', how='left', suffix='_2')
          n1, n2, a1, a2 = x['nc'].to_list(), x['nc_2'].to_list(), x['a'].to_list(), x['a_2'].to_list()
          cf = pl.DataFrame({
              'c_ntset': process.cpdist(n1, n2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32),
              'c_nratio': process.cpdist(n1, n2, scorer=fuzz.ratio, workers=-1, dtype=np.float32),
              'c_natset': process.cpdist(n1, x['na_2'].to_list(), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32),
              'c_atset': process.cpdist(a1, a2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32),
              'c_apart': process.cpdist(a1, a2, scorer=fuzz.partial_ratio, workers=-1, dtype=np.float32)})
          cf = cf.with_columns((x['h'] == x['h_2']).cast(pl.Int8).alias('c_hneq'), (x['h_2'] == '').cast(pl.Int8).alias('c_hn2e'), (x['a_2'] == '').cast(pl.Int8).alias('c_a2e'))
          d = RK.addf(pl.concat([c, cf], how='horizontal')); del x, cf, n1, n2, a1, a2
          d = d.with_columns(pl.Series('r', m.predict(d.select(RK.F).to_numpy().astype(np.float32), num_threads=6)).cast(pl.Float32))
          d = d.with_columns(pl.col('r').rank('ordinal', descending=True).over('id2').cast(pl.Int16).alias('rr'))
          d = d.filter(pl.col('rr') <= KEEP).drop([f'k{i}' for i in range(RK.NB)])
          d.write_parquet(fn); tot += d.height
          print(f'  s{s} part {p}: pool {c.height} -> kept {d.height} (block {t1-t0:.0f}s, total {time.time()-t0:.0f}s)', flush=True)
          del c, d
        del cq
    print(f's{s}: kept {tot} ({time.time()-t0:.0f}s)', flush=True)
    del S
print('RETRIEVE_DONE', f'{time.time()-t0:.0f}s', flush=True)
