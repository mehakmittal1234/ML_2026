"""India add-on drop list (s41): cells 'one name word added' x (nearby | far | truncated house number) x anchor only
(the entity has a same-source copy at its S1 number; x is not repeated in either source). Expected test gain +0.0003 (s41/s42),
fold-0 cost of the same removal -0.00016. Writes <outdir>/drop_pairs_india.tsv (every test pair of those cells, raw p >= 0.3) and
<outdir>/matching_results.tsv = <base matching> minus both lists.
usage: python s43_india_addon.py <raw test tag> <base matching_results.tsv> <outdir>"""
import sys, subprocess
from common import *
from s34_census import typed
from s35_near_cells import cells
rtag, base, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
tc = pl.read_parquet(wp('data', 'test_s1.parquet'), columns=['id', 'entity_id', 'country']).rename({'id': 'id1', 'country': 'ctry'})
SR = pl.read_parquet(wp('scores', rtag, 'test_s2_c0.parquet'), columns=[*KY, 'p'])
N = cells('test', SR, typed('test', SR), ('add',), ('hn_near', 'hn_far', 'hn_trunc')).join(tc, on='id1')
L = N.filter((pl.col('ctry') == 'India') & pl.col('anchor') & ~pl.col('rep_src') & ~pl.col('rep_oth'))
L = L.with_columns(pl.concat_str([pl.lit('S'), pl.col('src').cast(pl.Utf8), pl.lit('-'), pl.col('id2').cast(pl.Utf8)]).alias('e2'),
                   pl.concat_str([pl.col('name'), pl.col('addr'), pl.lit('anchor')], separator='+').alias('cell'))
L.select(pl.col('entity_id').alias('source1_entity_id'), pl.col('e2').alias('entity_id'), pl.col('ctry').alias('country'), 'cell') \
 .write_csv(os.path.join(OUT, 'drop_pairs_india.tsv'), separator='\t')
print('India add-on list:', L.height, 'pairs;', L.group_by('addr').len().rows())
subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'apply_drop_list.py'), base, os.path.join(OUT, 'drop_pairs_india.tsv'),
                os.path.join(OUT, 'matching_results.tsv')], check=True)
