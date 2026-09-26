"""Stage-1 matching model (LightGBM), cross-fitted over S1 folds so every train pair gets an out-of-fold score.
  model A: trained on pairs of S1 fold 1, model B: on fold 2 (both exclude validation fold 0).
  S1-side context (from ranker scores over ALL candidates of the S1) is joined in.
usage: python s09_model.py ctx   <split> <tag>              -> feat/<tag>/<split>_s1ctx.parquet (+ pair-level id1 ranks)
       python s09_model.py train <tag> <mtag> <rounds>        -> models/<mtag>_{A,B}.txt
       python s09_model.py score <split> <tag> <mtag>         -> scores/<mtag>/<split>_s{s}_c{p}.parquet (p = OOF / averaged score)"""
import sys, glob, time
import duckdb
import lightgbm as lgb
from common import *

NB = 19
PAR = dict(objective='binary', learning_rate=0.06, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
           lambda_l2=1.0, num_threads=6, verbose=-1, max_bin=127)
NONF = {'id1', 'id2', 'y', 'kmask', 'fold'}

def ffiles(split, tag):
    return sorted(glob.glob(wp('feat', tag, f'{split}_s[23]_c*.parquet')))

def build_ctx(split, tag):
    """per (id1, src) and per id1 aggregates of the ranker score over all retrieved candidates + pair rank within id1"""
    con = duckdb.connect(); con.execute(f"SET memory_limit='2500MB'; SET threads=6; SET temp_directory='{wp('tmp')}'; SET enable_progress_bar=false")
    src = ' UNION ALL '.join(f"SELECT id1, id2, {s}::TINYINT src, r, rr FROM '{wp('feat', tag, f'{split}_s{s}_c*.parquet')}'" for s in (2, 3))
    con.execute(f"""COPY (
        SELECT id1, id2, src,
               row_number() OVER (PARTITION BY id1, src ORDER BY r DESC)::SMALLINT s1_rank_r,
               (sum(CASE WHEN rr = 1 AND r >= 0.5 THEN 1 ELSE 0 END) OVER (PARTITION BY id1, src))::FLOAT s1_ntop_src,
               (sum(CASE WHEN rr = 1 AND r >= 0.5 THEN 1 ELSE 0 END) OVER (PARTITION BY id1))::FLOAT s1_ntop_all,
               (count(*) OVER (PARTITION BY id1))::FLOAT s1_ncand,
               (sum(r) OVER (PARTITION BY id1, src))::FLOAT s1_rsum_src
        FROM ({src})) TO '{wp('feat', tag, f'{split}_pairctx.parquet')}' (FORMAT PARQUET)""")
    print('ctx rows', con.execute(f"SELECT count(*) FROM '{wp('feat', tag, f'{split}_pairctx.parquet')}'").fetchone())

def load(split, tag, filt=None, label=True):
    lf = pl.scan_parquet(ffiles(split, tag))
    if filt is not None: lf = lf.filter(filt)
    d = lf.collect()
    ctx = pl.scan_parquet(wp('feat', tag, f'{split}_pairctx.parquet')).join(d.select(*KY).lazy(), on=KY, how='semi').collect()
    d = d.join(ctx, on=KY, how='left')
    d = d.with_columns(*[((pl.col('kmask') // (1 << i)) % 2).cast(pl.Float32).alias(f'k{i}') for i in range(NB)],
                       pl.col('wsum').log().alias('lwsum'), pl.col('wmax').log().alias('lwmax'),
                       (pl.col('s1_rsum_src') - pl.col('r')).alias('s1_rsum_other'))
    if label and split == 'train':
        d = d.join(load_gt_pairs().with_columns(pl.lit(1, pl.Int8).alias('y')), on=KY, how='left').with_columns(pl.col('y').fill_null(0))
    return d

def cols_of(d):
    return [c for c in d.columns if c not in NONF and d[c].dtype != pl.Utf8]

if __name__ == '__main__':
    mode = sys.argv[1]; t0 = time.time()
    if mode == 'ctx':
        build_ctx(sys.argv[2], sys.argv[3])
    elif mode == 'train':
        tag, mtag, R = sys.argv[2], sys.argv[3], int(sys.argv[4])
        for nm, fo in (('A', 1), ('B', 2)):
            # training rows: fold fo, top-5 per record; half of the trivially easy negatives (ranker r < 0.001) dropped, the rest weighted x2
            d = load('train', tag, (fold_expr('id1') == fo) & (pl.col('rr') <= 5) & ((pl.col('r') >= 0.001) | (pl.col('id2').hash(seed=9) % 2 == 0)))
            C = cols_of(d)
            X = d.select(C).to_numpy().astype(np.float32); y = d['y'].to_numpy(); n = d.height
            w = np.where(d['r'].to_numpy() < 0.001, 2.0, 1.0); del d
            m = lgb.train(PAR, lgb.Dataset(X, y, weight=w, feature_name=C, free_raw_data=True), R); del X
            m.save_model(wp('models', f'{mtag}_{nm}.txt'))
            print(f'model {nm}: fold {fo}, {n} rows, pos {int(y.sum())} ({time.time()-t0:.0f}s)', flush=True)
            print('   top gain:', sorted(zip(C, m.feature_importance('gain').round()), key=lambda x: -x[1])[:15], flush=True)
    elif mode == 'score':
        split, tag, mtag = sys.argv[2], sys.argv[3], sys.argv[4]
        os.makedirs(wp('scores', mtag), exist_ok=True)
        mA = lgb.Booster(model_file=wp('models', f'{mtag}_A.txt')); mB = lgb.Booster(model_file=wp('models', f'{mtag}_B.txt'))
        C = mA.feature_name()
        for f in ffiles(split, tag):
            out = wp('scores', mtag, os.path.basename(f))
            if os.path.exists(out): continue
            d = pl.read_parquet(f)
            ctx = pl.scan_parquet(wp('feat', tag, f'{split}_pairctx.parquet')).join(d.select(*KY).lazy(), on=KY, how='semi').collect()
            d = d.join(ctx, on=KY, how='left').with_columns(*[((pl.col('kmask') // (1 << i)) % 2).cast(pl.Float32).alias(f'k{i}') for i in range(NB)],
                        pl.col('wsum').log().alias('lwsum'), pl.col('wmax').log().alias('lwmax'), (pl.col('s1_rsum_src') - pl.col('r')).alias('s1_rsum_other'))
            X = d.select(C).to_numpy().astype(np.float32)
            pA = mA.predict(X, num_threads=6); pB = mB.predict(X, num_threads=6)
            fo = d.select(fold_expr('id1')).to_series().to_numpy() if split == 'train' else np.full(d.height, -1)
            p = np.where(fo == 1, pB, np.where(fo == 2, pA, (pA + pB) / 2))
            d.select(*KY, 'r', 'rr', 'a2_empty', 'is_india', 'is_france').with_columns(pl.Series('p', p).cast(pl.Float32), pl.Series('pA', pA).cast(pl.Float32),
                                                                         pl.Series('pB', pB).cast(pl.Float32)).write_parquet(out)
            print(os.path.basename(f), d.height, f'{time.time()-t0:.0f}s', flush=True)
    print('DONE', mode, f'{time.time()-t0:.0f}s')
