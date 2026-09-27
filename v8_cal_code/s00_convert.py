"""TSV -> parquet. data/{split}_s{1,2,3}.parquet (entity_id,name,addr,country,id), data/train_pairs.parquet (id1,id2,src),
data/train_s1ids.parquet (id1, ctry)."""
import time
from common import *

t0 = time.time()
for sp in ('train', 'test'):
    for s in (1, 2, 3):
        d = read_tsv(os.path.join(DS, sp, f'{sp}_source{s}.tsv')).rename({'business_name': 'name', 'business_address': 'addr'})
        d = d.with_columns(pl.col('entity_id').str.slice(3).cast(pl.Int64).alias('id'))
        assert d['id'].n_unique() == d.height
        d.write_parquet(wp('data', f'{sp}_s{s}.parquet'))
        print(sp, s, d.height, f'{time.time()-t0:.0f}s', flush=True)
gt = read_tsv(os.path.join(DS, 'train', 'train_ground_truth.tsv'))
pairs = gt.with_columns(pl.col('matched_entity_ids').fill_null('').str.split(',')).explode('matched_entity_ids') \
          .filter(pl.col('matched_entity_ids') != '') \
          .select(pl.col('source1_entity_id').str.slice(3).cast(pl.Int64).alias('id1'),
                  pl.col('matched_entity_ids').str.slice(3).cast(pl.Int64).alias('id2'),
                  pl.col('matched_entity_ids').str.slice(1, 1).cast(pl.Int8).alias('src'))
pairs.write_parquet(wp('data', 'train_pairs.parquet'))
s1 = pl.read_parquet(wp('data', 'train_s1.parquet'), columns=['id', 'country']).rename({'id': 'id1', 'country': 'ctry'})
assert gt.height == s1.height
s1.write_parquet(wp('data', 'train_s1ids.parquet'))
print('pairs', pairs.height, 'unique id2', pairs.select(pl.struct('id2', 'src').n_unique()).item(), f'{time.time()-t0:.0f}s')
