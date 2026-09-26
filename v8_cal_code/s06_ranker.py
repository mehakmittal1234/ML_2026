"""Stage-0 record-side ranker: scores every pool pair (key-family bits + cheap fuzzy features) and ranks the S1 candidates of
each S2/S3 record. Trained on a 1% record sample (records owned by validation-fold S1 excluded); evaluated on validation owners.
usage: python s06_ranker.py train <pool tag>                 -> models/ranker_<tag>.txt + recall@M on validation owners
       python s06_ranker.py apply <split> <pool tag> <model tag>  -> adds 'r' (score) and 'rr' (rank within record) -> ..._rk.parquet"""
import sys, glob, time
import lightgbm as lgb
from common import *
NB = 19
def addf(d):
    return d.with_columns(*[((pl.col('kmask') // (1 << i)) % 2).cast(pl.UInt8).alias(f'k{i}') for i in range(NB)],
                          pl.col('wsum').log().alias('lw'), pl.col('wmax').log().alias('lm'),
                          pl.len().over('id2').cast(pl.Float32).alias('npool'),
                          (pl.col('wsum') / pl.col('wsum').max().over('id2')).alias('wrel'),
                          (pl.col('c_ntset') - pl.col('c_ntset').max().over('id2')).alias('ntset_gap'),
                          (pl.col('c_atset') - pl.col('c_atset').max().over('id2')).alias('atset_gap'),
                          (pl.col('c_ntset') + pl.col('c_atset')).alias('c_sum'),
                          ((pl.col('c_ntset') + pl.col('c_atset')) - (pl.col('c_ntset') + pl.col('c_atset')).max().over('id2')).alias('sum_gap'))
F = [f'k{i}' for i in range(NB)] + ['nk', 'ntyp', 'lw', 'lm', 'rrank', 'npool', 'wrel', 'c_ntset', 'c_nratio', 'c_natset', 'c_atset', 'c_apart',
                                     'c_hneq', 'c_hn2e', 'c_a2e', 'ntset_gap', 'atset_gap', 'c_sum', 'sum_gap']
PAR = dict(objective='binary', learning_rate=0.1, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1,
           num_threads=6, verbose=-1)
files = lambda split, tag, s: sorted(glob.glob(wp('cand', tag, f'{split}_s{s}_q*_cf.parquet')))

if __name__ == '__main__' and sys.argv[1] == 'train':
    tag = sys.argv[2]; t0 = time.time()
    gt = load_gt_pairs(); f0 = pl.read_parquet(wp('data', 'train_s1ids.parquet')).filter(fold_expr('id1') == 0).select('id1')
    lab = gt.with_columns(pl.lit(1, pl.Int8).alias('y'))
    val_rec = gt.join(f0, on='id1', how='semi').select('id2', 'src')
    TR = []
    for s in (2, 3):
        vr = val_rec.filter(pl.col('src') == s).select('id2')
        for f in files('train', tag, s):
            d = pl.scan_parquet(f).join(vr.lazy(), on='id2', how='anti').filter(pl.col('id2').hash(seed=17) % 100 == 0).collect()
            TR.append(addf(d).with_columns(pl.lit(s, pl.Int8).alias('src')).join(lab, on=KY, how='left').with_columns(pl.col('y').fill_null(0)))
    TR = pl.concat(TR)
    print(f'train rows {TR.height} (pos {TR["y"].sum()}, records {TR.select(pl.struct("id2", "src").n_unique()).item()}) ({time.time()-t0:.0f}s)', flush=True)
    m = lgb.train(PAR, lgb.Dataset(TR.select(F).to_numpy().astype(np.float32), TR['y'].to_numpy()), 400)
    m.save_model(wp('models', f'ranker_{tag}.txt')); del TR
    G = []
    for s in (2, 3):
        vr = val_rec.filter(pl.col('src') == s).select('id2')
        for f in files('train', tag, s):
            d = addf(pl.scan_parquet(f).join(vr.lazy(), on='id2', how='semi').collect())
            d = d.with_columns(pl.Series('r', m.predict(d.select(F).to_numpy().astype(np.float32), num_threads=6)).cast(pl.Float32))
            d = d.with_columns(pl.col('r').rank('ordinal', descending=True).over('id2').cast(pl.Int16).alias('rr'), pl.lit(s, pl.Int8).alias('src'))
            G.append(d.join(gt.lazy().filter(pl.col('src') == s).collect(), on=KY, how='semi').select(*KY, 'rr', 'rrank', 'r',
                     (pl.col('r').max().over('id2')).alias('rmax'))); del d
    G = gt.join(f0, on='id1', how='semi').join(pl.concat(G), on=KY, how='left').with_columns(pl.col('rr').fill_null(9999), pl.col('rrank').fill_null(9999))
    print('validation owners recall@M  ranker: ' + ' '.join(f'{M}:{(G["rr"] <= M).mean():.4f}' for M in (1, 2, 3, 5, 8, 10, 20, 60)))
    print('                            crude : ' + ' '.join(f'{M}:{(G["rrank"] <= M).mean():.4f}' for M in (1, 2, 3, 5, 8, 10, 20, 60)))
    print('top gain:', sorted(zip(F, m.feature_importance('gain').round()), key=lambda x: -x[1])[:12])
    G.write_parquet(wp('analysis', f'ranker_val_{tag}.parquet'))
    print('RANKER_DONE', f'{time.time()-t0:.0f}s')

if __name__ == '__main__' and sys.argv[1] == 'apply':
    split, tag, mtag = sys.argv[2], sys.argv[3], sys.argv[4]; t0 = time.time()
    m = lgb.Booster(model_file=wp('models', f'ranker_{mtag}.txt'))
    for s in (2, 3):
        for f in files(split, tag, s):
            out = f.replace('_cf.parquet', '_rk.parquet')
            if os.path.exists(out): continue
            d = addf(pl.read_parquet(f))
            d = d.with_columns(pl.Series('r', m.predict(d.select(F).to_numpy().astype(np.float32), num_threads=6)).cast(pl.Float32))
            d = d.with_columns(pl.col('r').rank('ordinal', descending=True).over('id2').cast(pl.Int16).alias('rr'))
            d.drop([f'k{i}' for i in range(NB)]).write_parquet(out)
            print(os.path.basename(out), d.height, f'{time.time()-t0:.0f}s', flush=True); del d
    print('APPLY_DONE')
