"""Root cause of test-only rejection of decorated true copies: compare stage-1 inputs for US 'S1 name + service' same-address pairs, train vs test."""
from common import *
import glob
a = pl.read_parquet(wp('analysis','deco_service_train.parquet')).select(KY); b = pl.read_parquet(wp('analysis','deco_service_test.parquet')).select(KY)
def feats(sp, K):
    parts = []
    for s in (2, 3):
        for f in sorted(glob.glob(wp('feat','v6c',f'{sp}_s{s}_c*.parquet'))):
            parts.append(pl.scan_parquet(f).join(K.lazy().filter(pl.col('src')==s), on=KY, how='semi'))
    F = pl.concat(parts, how='diagonal_relaxed').collect()
    ctx = pl.scan_parquet(wp('feat','v6c',f'{sp}_pairctx.parquet')).join(K.lazy(), on=KY, how='semi').collect()
    return F.join(ctx, on=KY, how='left')
A = feats('train', a); B = feats('test', b)
print('rows', A.height, B.height)
print('model files:', sorted(os.path.basename(m) for m in glob.glob(wp('models','*'))))
cols = [c for c in A.columns if c not in KY and A[c].dtype not in (pl.Utf8, pl.List) and c in B.columns]
rows = []
for c in cols:
    x = A[c].cast(pl.Float64, strict=False); y = B[c].cast(pl.Float64, strict=False)
    sd = x.drop_nulls().drop_nans().std() or 1e-9
    mx, my = x.drop_nans().mean() or 0, y.drop_nans().mean() or 0
    rows.append((c, round(mx, 4), round(my, 4), round((my-mx)/sd, 2), round(x.is_null().mean(),3), round(y.is_null().mean(),3)))
T = pl.DataFrame(rows, schema=['feature','train_mean','test_mean','std_shift','train_null','test_null'], orient='row').sort(pl.col('std_shift').abs(), descending=True)
pl.Config.set_tbl_rows(30); print(T.head(25))
A.write_parquet(wp('analysis','deco_A.parquet')); B.write_parquet(wp('analysis','deco_B.parquet'))
