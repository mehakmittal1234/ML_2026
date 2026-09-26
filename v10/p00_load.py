"""Raw TSV -> parquet with file row order kept. /home/user/w/data/{split}_s{s}.parquet: row id name addr country ;
train_pairs.parquet: id1 id2 src gpos (position of the id inside the ground-truth list)."""
import os, time, polars as pl
DS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset'); W = '/home/user/w/data'
t0 = time.time()
rd = lambda p: pl.read_csv(p, separator='\t', quote_char=None, infer_schema=False)
for sp in ('train', 'test'):
    for s in (1, 2, 3):
        d = rd(f'{DS}/{sp}/{sp}_source{s}.tsv').with_row_index('row')
        d = d.select('row', pl.col('entity_id').str.slice(3).cast(pl.Int64).alias('id'), pl.col('business_name').alias('name'),
                     pl.col('business_address').alias('addr'), 'country')
        assert d['id'].n_unique() == d.height
        d.write_parquet(f'{W}/{sp}_s{s}.parquet'); print(sp, s, d.height, f'{time.time()-t0:.0f}s', flush=True)
gt = rd(f'{DS}/train/train_ground_truth.tsv').with_row_index('grow')
p = gt.with_columns(pl.col('matched_entity_ids').fill_null('').str.split(',')).explode('matched_entity_ids').filter(pl.col('matched_entity_ids') != '')
p = p.with_columns(pl.int_range(pl.len()).over('grow').alias('gpos'))
p = p.select(pl.col('source1_entity_id').str.slice(3).cast(pl.Int64).alias('id1'), pl.col('matched_entity_ids').str.slice(3).cast(pl.Int64).alias('id2'),
             pl.col('matched_entity_ids').str.slice(1, 1).cast(pl.Int8).alias('src'), 'gpos', 'grow')
p.write_parquet(f'{W}/train_pairs.parquet'); print('pairs', p.height, f'{time.time()-t0:.0f}s')
