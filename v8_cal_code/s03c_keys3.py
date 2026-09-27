"""keys3 = keys2 without the two rejected key families (15 A1: rare address token alone, 18 HZ: digit group x address token).
This step was run inline in the original session; this script reproduces it (verified identical on train_s2 part_000).
usage: python s03c_keys3.py <split>  -> keys3/<split>_s{s}/part_*.parquet"""
import sys, glob
from common import *
split = sys.argv[1]
for s in (1, 2, 3):
    src, dst = wp('keys2', f'{split}_s{s}'), wp('keys3', f'{split}_s{s}')
    os.makedirs(dst, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(src, 'part_*.parquet'))):
        pl.read_parquet(f).filter(~pl.col('kt').is_in([15, 18])).write_parquet(os.path.join(dst, os.path.basename(f)))
    open(os.path.join(dst, 'DONE'), 'w').write('ok')
    print(f'{split}_s{s} done', flush=True)
