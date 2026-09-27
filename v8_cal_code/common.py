"""Shared paths, IO and the official metric for the v6 experiments.

Official metric (README): F0.5 per Source-1 entity, macro-averaged over ALL Source-1 entities.
An entity with no true matches scores 1 if nothing is predicted, else 0; an entity with true matches
and an empty prediction scores 0.
"""
import os, sys
import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = os.path.join(ROOT, 'dataset')
V6 = os.path.dirname(os.path.abspath(__file__))    # this code folder (was ROOT/v6; changed so a copy never writes into v6/work)
W = os.path.join(V6, 'work')            # all generated artifacts
for d in ('data', 'norm', 'keys', 'cand', 'feat', 'models', 'scores', 'tmp', 'out', 'analysis'):
    os.makedirs(os.path.join(W, d), exist_ok=True)
sys.path.insert(0, os.path.join(V6))

KY = ['id1', 'id2', 'src']


def wp(*parts):
    return os.path.join(W, *parts)


def read_tsv(path):
    return pl.read_csv(path, separator='\t', quote_char=None, infer_schema=False)


def load_gt_pairs():
    """training ground truth as (id1, id2, src) int pairs"""
    return pl.read_parquet(wp('data', 'train_pairs.parquet'))


def val_ids():
    """validation S1 entities (fixed hash split, 1/8 of train S1 = ~276K entities)"""
    return pl.read_parquet(wp('data', 'train_s1ids.parquet')).filter(fold_expr('id1') == 0).select('id1')


def fold_expr(col='id1', k=8):
    return (pl.col(col).hash(seed=20260925) % k).cast(pl.Int8)


def f05(P, R):
    return np.where((P + R) > 0, 1.25 * P * R / np.maximum(1e-12, 0.25 * P + R), 0.0)


def macro_f05(pred, gt, ids, by=None, return_table=False):
    """pred, gt: DataFrames with (id1,id2,src). ids: DataFrame(id1 [,by]) of evaluated S1 entities.
    returns overall macro F0.5 (and optional per-group table)."""
    t = gt.join(ids.select('id1'), on='id1', how='semi').group_by('id1').agg(pl.len().alias('nt'))
    p = pred.join(ids.select('id1'), on='id1', how='semi').group_by('id1').agg(pl.len().alias('np_'))
    c = pred.join(gt, on=KY, how='semi').join(ids.select('id1'), on='id1', how='semi').group_by('id1').agg(pl.len().alias('tp'))
    e = ids.join(t, on='id1', how='left').join(p, on='id1', how='left').join(c, on='id1', how='left') \
           .with_columns(pl.col('nt').fill_null(0), pl.col('np_').fill_null(0), pl.col('tp').fill_null(0))
    nt, npp, tp = e['nt'].to_numpy(), e['np_'].to_numpy(), e['tp'].to_numpy()
    P = np.where(npp > 0, tp / np.maximum(1, npp), 0.0)
    R = np.where(nt > 0, tp / np.maximum(1, nt), 0.0)
    F = f05(P, R)
    F = np.where((nt == 0) & (npp == 0), 1.0, F)
    e = e.with_columns(pl.Series('F', F))
    if return_table:
        return e
    out = {'macro': float(F.mean()), 'n': len(F)}
    if by is not None:
        out['by'] = e.group_by(by).agg(pl.col('F').mean().alias('macro'), pl.len().alias('n')).sort(by)
    return out


def decide(D, thr, score='p'):
    """threshold + each S2/S3 record goes to at most one S1 (highest score)"""
    return D.filter(pl.col(score) >= thr).sort(score, descending=True).unique(['id2', 'src'], keep='first').select(KY)
