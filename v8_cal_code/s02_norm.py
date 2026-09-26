"""Normalise every record with v4's norm.py (parallel). norm/{split}_s{s}.parquet: id ctry nt nc na dom at dg"""
import time, multiprocessing as mp
from common import *

def work(args):
    from norm import norm_name, norm_addr
    ids, N, A, C = args
    nt, nc, na, dm, at, dg = [], [], [], [], [], []
    for nm, ad, c in zip(N, A, C):
        t_all, core, alt, dom = norm_name(nm or ''); atoks, digs = norm_addr(ad, c)
        nt.append(' '.join(t_all)); nc.append(' '.join(core)); na.append(' '.join(alt)); dm.append(dom)
        at.append(' '.join(atoks)); dg.append(' '.join(digs))
    return pl.DataFrame({'id': ids, 'ctry': C, 'nt': nt, 'nc': nc, 'na': na, 'dom': dm, 'at': at, 'dg': dg},
                        schema_overrides={'id': pl.Int64})

if __name__ == '__main__':
    t0 = time.time()
    with mp.Pool(6) as pool:
        for sp in ('train', 'test'):
            for s in (1, 2, 3):
                out = wp('norm', f'{sp}_s{s}.parquet')
                if os.path.exists(out): continue
                d = pl.read_parquet(wp('data', f'{sp}_s{s}.parquet'), columns=['id', 'name', 'addr', 'country'])
                B = 100_000
                jobs = [(d['id'][i:i+B].to_list(), d['name'][i:i+B].to_list(), d['addr'][i:i+B].to_list(), d['country'][i:i+B].to_list())
                        for i in range(0, d.height, B)]
                del d
                res = pl.concat(pool.map(work, jobs, chunksize=1)); del jobs
                res.write_parquet(out)
                print(sp, s, res.height, f'{time.time()-t0:.0f}s', flush=True); del res
    print('NORM_DONE')
