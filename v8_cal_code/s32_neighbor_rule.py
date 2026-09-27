"""Same-source neighbour rule (test-only distractor type found in s29-s31).
The generator gives each source ONE address version per entity. Test US/France contain same-name businesses a few doors away on the
same street, each with several copies; the model accepts them (train has almost none, so no supervised feature can learn them).
Rule: drop accepted pair (A, x) when
  - x's first house number is a NEAR difference from A's S1 house number (not a truncation/prefix/suffix, |diff| <= DMAX), and
  - A has another accepted copy y in x's SOURCE with p >= PCONF whose house number EQUALS A's S1 number
    (that source already has A's version, and x disagrees with it).
Reports: (1) validation cost (fold 0, official macro F0.5); (2) test counts by country; (3) label-free expected test gain:
flagged pairs get their count-invariance true share (validation flagged-TP rate per S1 entity x test S1 count / flagged count),
all other pairs keep p, and expected F0.5 of the decisions with / without the flagged pairs is compared.
usage: python s32_neighbor_rule.py <val score tag> <test score tag> <thr> [out tag]  (out tag: writes scores/<out>/test_s2_c0.parquet
       with flagged test pairs set to p = 0, so s13/s22 build the submission unchanged)"""
import sys
from common import *
from scipy.stats import poisson
DMAX, PCONF = 50, 0.9


def hn_table(split, s):
    return pl.read_parquet(wp('norm2', f'{split}_s{s}.parquet'), columns=['id', 'dz']).select('id', pl.col('dz').str.split(' ').list.first().fill_null('').alias('h'))


def flag(split, D):
    """D: accepted pairs (id1,id2,src,p). returns D with bool 'flag'"""
    H1 = hn_table(split, 1).rename({'id': 'id1', 'h': 'h1'})
    HR = pl.concat([hn_table(split, s).rename({'id': 'id2'}).with_columns(pl.lit(s, pl.Int8).alias('src')) for s in (2, 3)])
    X = D.join(H1, on='id1', how='left').join(HR, on=['id2', 'src'], how='left').with_columns(pl.col('h1', 'h').fill_null(''))
    num = lambda c: pl.col(c).str.slice(0, 9).cast(pl.Int64, strict=False)
    h, a = pl.col('h'), pl.col('h1')
    near = ((h != '') & (a != '') & (h != a) & ~h.str.starts_with(a) & ~a.str.starts_with(h) & ~h.str.ends_with(a) & ~a.str.ends_with(h)
            & ((num('h') - num('h1')).abs() <= DMAX))
    X = X.with_columns(near.fill_null(False).alias('near'))
    anchor = X.filter((pl.col('p') >= PCONF) & (h == a) & (a != '')).select('id1', 'src', pl.col('id2').alias('yid'))
    fl = X.filter(pl.col('near')).select(KY).join(anchor, on=['id1', 'src']).filter(pl.col('yid') != pl.col('id2')).select(KY).unique().with_columns(pl.lit(True).alias('flag'))
    return D.join(fl, on=KY, how='left').with_columns(pl.col('flag').fill_null(False))


def expected_f(D_all, sel_col, q_col, ents, K=14, lam=0.05):
    """expected F0.5 per entity of an arbitrary selected set: selected best-owner pairs vs the rest, truth ~ Bernoulli(q)"""
    D = D_all.join(ents.select('id1'), on='id1', how='semi')
    D = D.sort(['id1', q_col], descending=[False, True]).with_columns(pl.int_range(pl.len()).over(['id1', sel_col]).alias('r'))
    ent = ents.select('id1').with_row_index('e'); D = D.join(ent, on='id1')
    E = ent.height; C = K + 8
    def mat(sub):
        Q = np.zeros((E, K)); s2 = sub.filter(pl.col('r') < K); Q[s2['e'].to_numpy(), s2['r'].to_numpy()] = s2[q_col].to_numpy()
        extra = np.zeros(E); s3 = sub.filter(pl.col('r') >= K).group_by('e').agg(pl.col(q_col).sum()); extra[s3['e'].to_numpy()] = s3[q_col].to_numpy()
        return Q, extra
    Qs, xs = mat(D.filter(pl.col(sel_col))); Qr, xr = mat(D.filter(~pl.col(sel_col)))
    k = np.asarray(D.filter(pl.col(sel_col)).group_by('e').len().join(ent.select('e'), on='e', how='right').fill_null(0).sort('e')['len'])
    def dist(Q, extra_mean):
        P = np.zeros((E, C)); P[:, 0] = 1
        for i in range(Q.shape[1]):
            q = Q[:, i:i + 1]; n = P * (1 - q); n[:, 1:] += P[:, :-1] * q; n[:, -1] += P[:, -1] * q[:, 0]; P = n
        if extra_mean is not None:
            pm = poisson.pmf(np.arange(C)[None, :], extra_mean[:, None])
            out = np.zeros_like(P)
            for m in range(C): out[:, m:] += P[:, :C - m] * pm[:, m:m + 1]
            P = out
        return P
    pa = dist(Qs, None); pr = dist(Qr, lam + xr + xs)                # (selected tail beyond K is negligible; counted as missed)
    a = np.arange(C)[None, :, None]; b = np.arange(C)[None, None, :]; kk = k[:, None, None].astype(float)
    g = np.where(kk > 0, 1.25 * a / np.maximum(1e-12, 0.25 * (a + b) + kk), ((a == 0) & (b == 0)).astype(float))
    return (pa[:, :, None] * pr[:, None, :] * g).sum((1, 2))


if __name__ == '__main__':
    vtag, ttag, thr = sys.argv[1], sys.argv[2], float(sys.argv[3]); out = sys.argv[4] if len(sys.argv) > 4 else None
    gt = load_gt_pairs(); ids = pl.read_parquet(wp('data', 'train_s1ids.parquet')); v0 = ids.filter(fold_expr('id1') == 0)
    SV = pl.read_parquet(wp('scores', vtag, 'train_s2_c0.parquet'), columns=[*KY, 'p'])
    DV = flag('train', decide(SV, thr).join(SV, on=KY))
    base = macro_f05(DV.select(KY), gt, v0, by='ctry'); new = macro_f05(DV.filter(~pl.col('flag')).select(KY), gt, v0, by='ctry')
    FV = DV.filter(pl.col('flag')).join(v0, on='id1').join(gt.with_columns(pl.lit(1).alias('y')), on=KY, how='left').with_columns(pl.col('y').fill_null(0))
    print(f'VALIDATION fold 0: flagged {FV.height} accepted pairs, true {FV["y"].sum()} ({FV["y"].mean():.3f}); macro {base["macro"]:.5f} -> {new["macro"]:.5f} ({new["macro"]-base["macro"]:+.5f})')
    rate = {c: FV.filter((pl.col('ctry') == c) & (pl.col('y') == 1)).height / v0.filter(pl.col('ctry') == c).height for c in ('US', 'India')}
    rate['France'] = rate['US']
    print('   flagged TRUE pairs per S1 entity (count-invariant generator constant):', {c: round(v, 5) for c, v in rate.items()})
    ST = pl.read_parquet(wp('scores', ttag, 'test_s2_c0.parquet'), columns=[*KY, 'p'])
    tc = pl.read_parquet(wp('data', 'test_s1.parquet'), columns=['id', 'country']).select(pl.col('id').alias('id1'), pl.col('country').alias('ctry'))
    DT = flag('test', decide(ST, thr).join(ST, on=KY)).join(tc, on='id1')
    n1 = dict(tc.group_by('ctry').len().rows())
    B = ST.filter(pl.col('p') >= 0.01).sort('p', descending=True).unique(['id2', 'src'], keep='first').join(DT.select(*KY, 'flag'), on=KY, how='left') \
          .with_columns(pl.col('flag').fill_null(False), pl.col('p').is_between(thr, 1.0).alias('sel0')).join(tc, on='id1')
    B = B.with_columns((pl.col('sel0') & ~pl.col('flag')).alias('sel1'))
    tot0 = tot1 = 0.0
    print('TEST:')
    for c in ('US', 'India', 'France'):
        nf = DT.filter((pl.col('ctry') == c) & pl.col('flag')).height
        t = min(1.0, rate[c] * n1[c] / max(1, nf))
        Bc = B.filter(pl.col('ctry') == c).with_columns(pl.when(pl.col('flag')).then(pl.lit(t)).otherwise(pl.col('p')).alias('q'))
        ents = tc.filter(pl.col('ctry') == c)
        e0 = expected_f(Bc, 'sel0', 'q', ents).mean(); e1 = expected_f(Bc, 'sel1', 'q', ents).mean()
        tot0 += e0 * n1[c]; tot1 += e1 * n1[c]
        print(f'  {c:6s}: accepted {DT.filter(pl.col("ctry") == c).height}, flagged {nf} (mean p {DT.filter((pl.col("ctry") == c) & pl.col("flag"))["p"].mean():.3f});'
              f' est. true share of flagged {t:.3f}; expected F {e0:.5f} -> {e1:.5f} ({e1 - e0:+.5f})', flush=True)
    N = sum(n1.values()); print(f'  ALL   : expected F {tot0 / N:.5f} -> {tot1 / N:.5f} ({(tot1 - tot0) / N:+.5f}) on the full test mix')
    if out:
        os.makedirs(wp('scores', out), exist_ok=True)
        full = pl.read_parquet(wp('scores', ttag, 'test_s2_c0.parquet'))
        full = full.join(DT.filter(pl.col('flag')).select(*KY, pl.lit(True).alias('_f')), on=KY, how='left') \
                   .with_columns(pl.when(pl.col('_f')).then(pl.lit(0.0, pl.Float32)).otherwise(pl.col('p')).alias('p')).drop('_f')
        full.write_parquet(wp('scores', out, 'test_s2_c0.parquet')); print('written scores/' + out)
