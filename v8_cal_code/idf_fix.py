"""Train-scale IDF for test (root-cause fix for test-only rejections).
feats2.Ctx computes w = log(NTOT_split / df_split(ctry, tok)) with ONE global NTOT but per-country df. Test's country mix differs from train's
(US 38% of test S1 vs 60% of train; France new), so every token's weight shifts by a country constant (US +0.44, India -0.18, France ~+1.0..1.4),
and the models (trained on train-scale weights) misread common decoration words ('service', 'groupe', 'cie', ...) as rare distinctive words.
CtxFixed maps each test token's relative in-country frequency onto train's scale:
    w_adj(ctry, tok) = log(NTOT_train) - log(df_test(ctry, tok) * Nc_train(ref(ctry)) / Nc_test(ctry))
with Nc = records of that country over S1+S2+S3 and ref(France) = US (the models treat France like US: is_india = 0). WMAX = log(NTOT_train)."""
import math
from common import *
import feats2

def country_sizes(split, nd='norm2'):
    return dict(pl.concat([pl.scan_parquet(wp(nd, f'{split}_s{s}.parquet')).select('ctry') for s in (1, 2, 3)]).group_by('ctry').len().collect().iter_rows())

class CtxFixed(feats2.Ctx):
    def __init__(self, split='test', nd='norm2', ref=None):
        super().__init__(split, nd)
        NTOT_tr = float(sum(pl.scan_parquet(wp(nd, f'train_s{s}.parquet')).select(pl.len()).collect().item() for s in (1, 2, 3)))
        ntr, nte = country_sizes('train', nd), country_sizes(split, nd)
        ref = ref or {'US': 'US', 'India': 'India', 'France': 'US'}
        fac = {c: math.log(ntr[ref.get(c, 'US')] / nte[c]) for c in nte}          # log(Nc_train(ref) / Nc_test(c))
        self.fac = fac
        f = pl.DataFrame({'ctry': list(fac), 'fac': list(fac.values())})
        for attr, kind in (('dfn', 'name'), ('dfa', 'addr')):
            df = pl.read_parquet(wp(nd, f'{split}_df_{kind}.parquet')).join(f, on='ctry', how='left').with_columns(pl.col('fac').fill_null(0.0))
            setattr(self, attr, df.select('ctry', 'tok', (math.log(NTOT_tr) - pl.col('df').cast(pl.Float64).log() - pl.col('fac')).cast(pl.Float32).alias('w')))
        self.WMAX = math.log(NTOT_tr)
