# v8_cal — code that produced `outputs/v8_cal`

A self-contained copy of the scripts from `v6/` that produce `outputs/v8_cal/matching_results.tsv` (5,802,692 matched pairs) and
`candidate_pairs.tsv`. `v6/` itself was not changed.

## How to run
Keep this folder inside `student_resource/` (next to `dataset/`, `utils/`, `.venv/`), then:

```bash
./run_v8_cal.sh
```

Output goes to `outputs/v8_cal_repro/` (a new folder, so the original `outputs/v8_cal` is not overwritten). All intermediate files go
to `v8_cal_code/work/`. Environment: `../.venv` (python 3.14, polars 1.44, duckdb 1.5, lightgbm 4.7, rapidfuzz 3.14, scipy).
Runtime on an 8 GB M2: retrieval alone takes about 3.5 h (train 83 min + test 125 min). The full run takes several hours.

## Pipeline
| step | script(s) | what |
|---|---|---|
| 1 | `s00_convert`, `s01_indic`, `s02_norm`, `s02b_norm2` | TSV to parquet, Indic-script dictionaries learned from train pairs, normalisation (+ digit-for-letter, alias, handle, zero-strip fixes) |
| 2 | `s03b_keys2`, `s03c_keys3` | blocking keys: v4 families + R1 / HS / ND (A1 and HZ dropped) |
| 3 | `s04d_block_rec`, `s05_cheap`, `s06_ranker` | stage-0 record-side ranker, trained on the `f3` pool |
| 4 | `s07_retrieve` | key join, then top-60 per S2/S3 record, then the ranker keeps the top 8 per record |
| 5 | `s08_feat` (+ `feats2`), `s09_model` | pair features and the stage-1 LightGBM `m1`, cross-fitted so train scores are out-of-fold |
| 6 | `s11_stage2 build/buildx`, `s16_support`, `s17_tells` (+ `a15_tells`, `a14_france_premise`) | stage-2 features: record competition, S1 context, sibling agreement, raw-name and house-number relations, name support, edit-type tells |
| 7 | `s14b_final v7t sup,tells` | stage-2 LightGBM on all train folds (700 rounds). This gives the base score for every test pair (the v7 model). |
| 8 | `s19_specialist final v8s` | borderline specialist (1500 rounds, 161 features). It re-scores pairs with 0.01 < p1 < 0.99; all other pairs keep their v7t score. |
| 9 | `s18_priorshift v8s 0.7 v8scal` | per-country prior-shift recalibration of test probabilities to the generator-implied match count |
| 10 | `s13_submit v8scal 0.7 v6c` | threshold 0.7, each S2/S3 record goes to its single best S1, then the files are written in README format |

`s10_eval.py` (official macro F0.5 on validation fold 0) is included for evaluation only; the run script does not use it.

## Checked against the saved artifacts
- Threshold 0.7 on the saved `v8scal` scores gives exactly the 5,802,692 matches in `outputs/v8_cal`.
- Recalibration biases recomputed from `v8s` match the saved `v8scal` scores: France -0.229, India -0.435, US -0.654. These are
  equivalent to raw thresholds of 0.746, 0.783 and 0.818.
- In `v8s`, every pair outside the 0.01–0.99 band has exactly its v7t score. 1,983,300 pairs were re-scored by the specialist.
- The saved models have 700 trees (v7t, including the sup and tells features) and 1500 trees (v8s specialist).

## Differences from the original run
- `common.py`: the artifact directory is now `v8_cal_code/work` instead of `v6/work`. This is the only code edit.
- `s03c_keys3.py` is new. The keys2 → keys3 filter was originally run inline; its output was checked to be identical to the saved
  `keys3` on `train_s2`.
- The `s04d_block_rec` arguments for the ranker pool were not logged. They were reconstructed from the saved `f3` pool: filter
  `gtplus:0:100` matches the pool's owner mix, M=60, 4 partitions, and the keys contain no A1/HZ bits. The caps `1000 100000` are
  assumed to be the same as in retrieval.
- LightGBM uses multiple threads and bagging, so a rerun can differ from the original in the last decimal places of the scores.
  A handful of borderline matches may flip.
