# Handoff: root cause of the "true copies rejected on test only" finding, and its fix

From the local session that built v8_cal (the base of this repo). Everything here uses only the competition files.

## 1. The bug
`feats2.Ctx` computes the name and address IDF as `w = log(NTOT / df(ctry, tok))`.
- `NTOT` is ONE global record count over all countries (S1 + S2 + S3).
- `df` is counted per country.

Test's country mix differs from train's: US is 38% of test S1 against 60% of train S1, and France is new. So on test every token's weight shifts by a
country constant (frequent tokens, test − train):

| Country | Shift (median) | IQR |
|---|---|---|
| US | +0.44 | 0.40–0.48 |
| India | −0.18 | −0.22 to −0.14 |
| France | about +1.5 vs the US train scale | – |

Examples (US): service 5.02 → 5.46, services 4.34 → 4.78, center 3.65 → 4.10.

The models learned common decoration words ("service", "services", …) by their exact IDF value, for example in `ni_idf_miss2` (the IDF of the record's
tokens missing from S1). For the US "S1 name + service" pairs that value is **exactly 5.0207 on train** and 5.4566 on test. On test these words
therefore look like rare, distinctive added words, so true copies are downgraded.

Stage-1 p1, US pairs where the record adds exactly one word to the S1 name, same address:

| Added word | Train true rate | Train p1 | Test p1 |
|---|---|---|---|
| service | 1.000 | 0.979 | 0.782 |
| services | 0.999 | 0.991 | 0.761 |
| partners | 0.708 | 0.708 | 0.490 |
| llc | 0.817 | 0.832 | 0.709 |
| inc | 0.899 | 0.903 | 0.824 |

France is the worst case: common French decoration words ("groupe", "cie", "associés") look about 1.5 IDF units rarer than anything seen in training.

## 2. The fix (no retraining)
`code/idf_fix.py` → `CtxFixed(split)`: the same as `feats2.Ctx`, except the test IDF is mapped onto the train scale per country:

    w_adj(ctry, tok) = log(NTOT_train) - log(df_test(ctry, tok) * Nc_train(ref(ctry)) / Nc_test(ctry))
    WMAX = log(NTOT_train)

- `Nc` = records of that country over S1+S2+S3.
- `ref(France) = US`, because the models treat France like the US (`is_india = 0`).
- Measured factors, log(Nc_train(ref) / Nc_test): US 0.517, India −0.097, France 1.489.

**Proof** (`code/a45_idf_probe.py`): recompute the features of 1,989 test "S1 + service" pairs with `CtxFixed` and re-score them with the unchanged
stage-1 models (m1_A / m1_B):

| Pairs | Stage-1 p before | After fix | Share ≥ 0.5 |
|---|---|---|---|
| 1,989 test "S1 + service" pairs | 0.782 | **0.982** (train equivalent 0.979) | 89% → 99.3% |
| 20,000 random test pairs | 0.0749 | 0.0752 (average change 0.001) | unchanged |

`ni_idf_miss2` for the service pairs goes 5.457 → 5.008 (train 5.021).

## 3. Second bug of the same kind (smaller)
`a15_tells.tells()` defines "real vocabulary word" as S1 df ≥ 5. Test's US S1 table is half the size of train's, so fewer words qualify on test.
`code/a15_tells.py` adds `TELLS_SCALE=1`, which scales the threshold per country by the S1 size (France on the US train scale). The default is unchanged.

The blocking-key weights (`w = 1/(c1*c2)`, used as wsum/wmax) are also split-dependent, but barely shifted for the affected pairs (+0.08 SD). They are
not fixed here.

## 4. Re-scoring test with the fix (no retraining): `code/run_v20.sh`
| Step | Script | What |
|---|---|---|
| 1 | `s08F_feat.py test v6c 8 40` | s08_feat with `CtxFixed` → `feat/v6cF` (candidates reused from `cand/v6c`); about 10 min |
| 2 | `s09F_score.py` | stage-1 scoring with `models/m1_A.txt` + `m1_B.txt` → `scores/m1F` (pair context reused from `feat/v6c/test_pairctx.parquet`) |
| 3 | `s11_stage2.py build test v6cF m1F`; `buildx test m1F`; `S16_SRC=m1F s16_support.py test`; `TELLS_SRC=m1F TELLS_SCALE=1 s17_tells.py test` | stage-2 context rebuilt from the new p1 |
| 4 | `s14F_score.py m1F v7t v7tF` | saved stage-2 model `models/v7t_stage2_all.txt` (extras sup, tells) |
| 5 | `S19_SRC=m1F S19_TELLS=m1F S19_FTAG=v6cF S19_NOP1S=1 s19F_score.py v8s v7tF v8sF` | saved specialist `models/v8s_specialist.txt` |
| 6 | `s18_priorshift.py v8sF 0.7 v8sFcal`, then `s22_country_thr.py v8sFcal <out> 0.7 0.7 0.7` | recalibration and decisions |

- `code/s19_specialist.py` and `code/s17_tells.py` carry small environment switches (S19_TELLS, S19_FTAG, S19_NOP1S, TELLS_SRC). With the defaults
  they behave as before.
- Paths go through `common.wp()` (work/ subfolders). Adjust to your layout.

## 5. Models included (`models/`)
| File | What |
|---|---|
| `m1_A.txt`, `m1_B.txt` | Stage-1 LightGBM (cross-fitted); test p1 = mean(A, B) |
| `v7t_stage2_all.txt` | Stage-2 model trained on all folds, extras sup + tells (v7/v8 lineage) |
| `v8s_specialist.txt` | Borderline specialist, all folds, S19_SRC=m1 (behind v8_cal, LB 0.982143) |
| `v9s_specialist.txt` | Round-2 specialist (S19_SRC=it1), behind v9s / m4it |
| `ranker_f3.txt` | Stage-0 record-side ranker |

## 6. Other findings worth using
- **Leaderboard-consistency check** (`code/a35_lbcheck.py`). It scores any set of test probabilities against real leaderboard differences:
  - Known points: v8 0.981971, v8_cal 0.982143, v8_strict 0.979477, v10_expf (v9scal + expected-F decoder) 0.981635, v14e 0.984861.
  - v8scal fits the first three best (total error 0.00035). v9scal failed badly: it predicted v10 at +0.0019 and v10 actually scored −0.0003. So
    **build on the v8 lineage, not on v9s / m4it**.
  - Every old probability set predicts v14e at −0.004 to −0.005 vs v8_cal, while it scored +0.0027. That is the IDF bug in action.
- **Label-free generator deconvolution** (`a41_generator.py`, `a42_pairdeconv.py`). Test true copies are generated like train's: identical in the US,
  slightly noisier names in India (one-word additions +5 pts). Test has about 3× more plausible look-alike distractors per S1.
- **France:** true copies keep the house number 91% of the time against 73% in US/India. In France the address is the reliable signal and the name
  the weak one.

The local v20 result (the full v8-lineage test re-score with the fix) is still running; the fixed test scores and the matching file will follow.
