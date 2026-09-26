"""Source-level sibling features for stage 2 (label-free).
The generator is hierarchical: hidden entity -> one version per source -> per-copy noise. Copies of one entity in the same source
therefore share that source's address corruptions (alternate city, typos, 'null' parts, state renaming). On train, two
same-source copies of one owner share an address token absent from the owner's S1 address 59.8% of the time; a same-source copy of
a same-name twin does so 5.5% of the time (cross-source copies of one owner: 18.3%). Names carry no such layer (1.2-1.4% for all).
For pair (A, x), siblings y = A's confident copies (record's best S1 = A, p1 >= PSIB), x excluded; dev(r) = canonical address
tokens of r absent from A's address.
  sd_n_same / sd_n_oth   : # siblings with an address in x's source / the other source
  sd_sh_same / sd_sh_oth : max # dev tokens x shares with one sibling (same source / other source)
  sd_cov_same            : share of x's dev tokens found in some same-source sibling's dev
  sd_ydev_same           : mean # dev tokens of the same-source siblings (is there a source-level deviation to share?)
  sd_xdev                : # dev tokens of x
  sd_aeq_same            : x's address token set equals a same-source sibling's
usage: python s23_sibdev.py <split> [stage-2 dir, default m1]  -> feat/<dir>_s2/<split>_sdev.parquet"""
import sys, time
from common import *
PSIB = 0.7
split = sys.argv[1]; SRC = sys.argv[2] if len(sys.argv) > 2 else 'm1'
t0 = time.time()
S = pl.read_parquet(wp('feat', f'{SRC}_s2', f'{split}.parquet'), columns=[*KY, 'p1', 'rec_rank'])
sib = S.filter((pl.col('rec_rank') == 1) & (pl.col('p1') >= PSIB)).select('id1', pl.col('id2').alias('sid'), pl.col('src').alias('ssrc'))
atok = lambda: pl.col('at').str.split(' ').list.eval(pl.element().filter((pl.element() != ',') & (pl.element() != ''))).list.unique()
A1 = pl.read_parquet(wp('norm2', f'{split}_s1.parquet'), columns=['id', 'at']).select(pl.col('id').alias('id1'), atok().alias('ta'))
REC = pl.concat([pl.read_parquet(wp('norm2', f'{split}_s{s}.parquet'), columns=['id', 'at']).select(pl.col('id').alias('id2'), pl.lit(s, pl.Int8).alias('src'), atok().alias('tx'))
                 for s in (2, 3)])
print(f'loaded: pairs {S.height}, siblings {sib.height} ({time.time()-t0:.0f}s)', flush=True)
out = []
NCH = 8
for k in range(NCH):
    P = S.filter(pl.col('id1') % NCH == k).select(KY)
    Y = sib.filter(pl.col('id1') % NCH == k)
    # dev tokens of every candidate record x w.r.t. its S1, and of every sibling y
    ax = A1.join(P.select('id1').unique(), on='id1', how='semi').explode('ta').drop_nulls('ta').rename({'ta': 't'})
    DX = P.join(REC, on=['id2', 'src']).explode('tx').drop_nulls('tx').rename({'tx': 't'}).join(ax, on=['id1', 't'], how='anti')
    DY = DX.join(Y.rename({'sid': 'id2', 'ssrc': 'src'}), on=KY, how='semi').rename({'id2': 'sid', 'src': 'ssrc'})
    nx = DX.group_by(KY).agg(pl.len().cast(pl.Float32).alias('sd_xdev'))
    ny = DY.group_by('id1', 'sid', 'ssrc').agg(pl.len().cast(pl.Float32).alias('_ny'))
    # siblings (with an address) of each pair, by source relation
    Yr = Y.join(REC.filter(pl.col('tx').list.len() > 0).select(pl.col('id2').alias('sid'), pl.col('src').alias('ssrc'), pl.col('tx').alias('ty')), on=['sid', 'ssrc'])
    PY = P.join(Yr.select('id1', 'sid', 'ssrc'), on='id1').filter(~((pl.col('sid') == pl.col('id2')) & (pl.col('ssrc') == pl.col('src'))))
    PY = PY.with_columns((pl.col('ssrc') == pl.col('src')).alias('same')).join(ny, on=['id1', 'sid', 'ssrc'], how='left').with_columns(pl.col('_ny').fill_null(0))
    cnt = PY.group_by(KY).agg(pl.col('same').sum().cast(pl.Float32).alias('sd_n_same'), (~pl.col('same')).sum().cast(pl.Float32).alias('sd_n_oth'),
                              pl.col('_ny').filter(pl.col('same')).mean().cast(pl.Float32).alias('sd_ydev_same'))
    # shared dev tokens per (pair, sibling)
    SH = DX.join(DY, on=['id1', 't']).filter(~((pl.col('sid') == pl.col('id2')) & (pl.col('ssrc') == pl.col('src')))).with_columns((pl.col('ssrc') == pl.col('src')).alias('same'))
    per = SH.group_by(*KY, 'sid', 'same').agg(pl.len().alias('n'))
    sh = per.group_by(KY).agg(pl.col('n').filter(pl.col('same')).max().cast(pl.Float32).alias('sd_sh_same'), pl.col('n').filter(~pl.col('same')).max().cast(pl.Float32).alias('sd_sh_oth'))
    cov = SH.filter(pl.col('same')).group_by(KY).agg(pl.col('t').n_unique().cast(pl.Float32).alias('_cov'))
    # exact address token-set equality with a same-source sibling
    eq = P.join(REC.rename({'tx': 'tx_x'}), on=['id2', 'src']).join(Yr, on='id1').filter((pl.col('ssrc') == pl.col('src')) & (pl.col('sid') != pl.col('id2')))
    eq = eq.filter(pl.col('tx_x').list.len() > 0).with_columns((pl.col('tx_x').list.sort() == pl.col('ty').list.sort()).alias('e')).group_by(KY).agg(pl.col('e').any().cast(pl.Float32).alias('sd_aeq_same'))
    X = P.join(cnt, on=KY, how='left').join(nx, on=KY, how='left').join(sh, on=KY, how='left').join(cov, on=KY, how='left').join(eq, on=KY, how='left')
    X = X.with_columns(pl.col('sd_n_same', 'sd_n_oth', 'sd_xdev').fill_null(0),
                       pl.when(pl.col('sd_n_same') > 0).then(pl.col('sd_sh_same').fill_null(0)).alias('sd_sh_same'),
                       pl.when(pl.col('sd_n_oth') > 0).then(pl.col('sd_sh_oth').fill_null(0)).alias('sd_sh_oth'),
                       pl.when((pl.col('sd_n_same') > 0) & (pl.col('sd_xdev') > 0)).then(pl.col('_cov').fill_null(0) / pl.col('sd_xdev')).alias('sd_cov_same'),
                       pl.when(pl.col('sd_n_same') > 0).then(pl.col('sd_aeq_same').fill_null(0)).alias('sd_aeq_same')).drop('_cov')
    out.append(X); print(f'chunk {k}: {X.height} pairs ({time.time()-t0:.0f}s)', flush=True)
    del P, Y, ax, DX, DY, PY, SH, per, eq, X
pl.concat(out).write_parquet(wp('feat', f'{SRC}_s2', f'{split}_sdev.parquet'))
print('SDEV_DONE', f'{time.time()-t0:.0f}s')
