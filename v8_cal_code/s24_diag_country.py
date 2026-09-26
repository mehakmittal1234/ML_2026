"""Label-free per-country diagnostics of a score file (no decoder change; decisions = common.decide at THR).
 (1) predicted copies per S1 entity, per source: compared with the generator's true distribution (train) and with the
     validation relation predicted-vs-true, so a country whose predicted shape departs from the validation shape is flagged;
 (2) model-implied expected F0.5 of the threshold decisions (Poisson-binomial over each entity's best-owner candidates):
     checked against the actual F on validation fold 0, then reported for every test country.
usage: python s24_diag_country.py <val score parquet> <test score parquet> [thr]"""
import sys
from common import *
from efd import _expected_f, _gain_table
THR = 0.7
K, C, PMIN, LAM = 12, 18, 0.01, 0.05

def implied(S, ctry):
    """expected F of decide(S, THR) per entity, from the scores themselves"""
    B = S.filter(pl.col('p') >= PMIN).sort('p', descending=True).unique(['id2', 'src'], keep='first')
    B = B.sort(['id1', 'p'], descending=[False, True]).with_columns(pl.int_range(pl.len()).over('id1').alias('r'))
    tail = B.filter(pl.col('r') >= K).group_by('id1').agg(pl.col('p').sum().alias('tq'))
    H = B.filter(pl.col('r') < K)
    ent = ctry.join(H.group_by('id1').agg((pl.col('p') >= THR).sum().alias('k')), on='id1', how='left').join(tail, on='id1', how='left') \
              .with_columns(pl.col('k').fill_null(0), pl.col('tq').fill_null(0.0)).with_row_index('e')
    H = H.join(ent.select('id1', 'e'), on='id1')
    Q = np.zeros((ent.height, K)); Q[H['e'].to_numpy(), H['r'].to_numpy()] = H['p'].to_numpy()
    G = _gain_table(K, C); ef = np.zeros(ent.height); lam = LAM + ent['tq'].to_numpy(); kk = ent['k'].to_numpy()
    for s in range(0, ent.height, 100_000):
        EF = _expected_f(Q[s:s + 100_000], lam[s:s + 100_000], G, C); ef[s:s + 100_000] = EF[np.arange(EF.shape[0]), kk[s:s + 100_000]]
    return ent.with_columns(pl.Series('EF', ef))

def counts(pred, ctry):
    c = pred.group_by('id1').agg((pl.col('src') == 2).sum().alias('n2'), (pl.col('src') == 3).sum().alias('n3'))
    return ctry.join(c, on='id1', how='left').with_columns(pl.col('n2', 'n3').fill_null(0).cast(pl.Int32)).with_columns((pl.col('n2') + pl.col('n3')).alias('n'))

def dist(d, col, top=7):
    v = d.group_by(pl.col(col).clip(0, top)).len().sort(col).with_columns(pl.col('len') / d.height)
    return ' '.join(f'{k}:{x:.4f}' for k, x in v.rows())

if __name__ == '__main__':
    VAL, TEST = sys.argv[1], sys.argv[2]; THR = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
    gt = load_gt_pairs()
    ids = pl.read_parquet(wp('data', 'train_s1ids.parquet')); v0 = ids.filter(fold_expr('id1') == 0).select('id1', 'ctry')
    SV = pl.read_parquet(VAL, columns=[*KY, 'p']); ST = pl.read_parquet(TEST, columns=[*KY, 'p'])
    tc = pl.read_parquet(wp('data', 'test_s1.parquet'), columns=['id', 'country']).select(pl.col('id').alias('id1'), pl.col('country').alias('ctry'))
    print('== (1) copies per entity (total S2+S3), clipped at 7')
    tv = counts(gt.join(v0, on='id1', how='semi'), v0); pv = counts(decide(SV, THR).join(v0, on='id1', how='semi'), v0)
    for c in ('US', 'India'):
        print(f'val {c:6s} true: {dist(tv.filter(pl.col("ctry") == c), "n")}')
        print(f'val {c:6s} pred: {dist(pv.filter(pl.col("ctry") == c), "n")}')
    pt = counts(decide(ST, THR), tc)
    for c in ('US', 'India', 'France'):
        d = pt.filter(pl.col('ctry') == c)
        print(f'test {c:6s} pred: {dist(d, "n")}   | S2>5: {(d["n2"] > 5).mean():.5f}  S3>6: {(d["n3"] > 6).mean():.5f}  mean {d["n"].mean():.3f}')
    print('   validation over-cap (impossible counts): S2>5', f'{(pv["n2"] > 5).mean():.5f}', 'S3>6', f'{(pv["n3"] > 6).mean():.5f}')
    print('\n== (2) model-implied expected F0.5 of decide(thr) vs actual')
    ev = implied(SV, v0).join(macro_f05(decide(SV, THR), gt, v0, return_table=True).select('id1', 'F'), on='id1')
    for c in ('US', 'India'):
        d = ev.filter(pl.col('ctry') == c); print(f'val  {c:6s} implied {d["EF"].mean():.5f}  actual {d["F"].mean():.5f}')
    print(f'val  all    implied {ev["EF"].mean():.5f}  actual {ev["F"].mean():.5f}')
    et = implied(ST, tc)
    for c in ('US', 'India', 'France'):
        d = et.filter(pl.col('ctry') == c); print(f'test {c:6s} implied {d["EF"].mean():.5f}   (n={d.height})')
    w = et.group_by('ctry').agg(pl.col('EF').mean(), pl.len()); print(f'test all    implied {et["EF"].mean():.5f}')
