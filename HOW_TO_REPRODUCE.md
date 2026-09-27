# How to reproduce the 0.984861 submission (v14e)

This guide takes you from the raw challenge data to the exact `matching_results.tsv` that scored **0.984861** on the public
leaderboard. It uses one script, `v8_cal_code/run_v14e.sh`. The same run also produces the optional follow-up file `v15_same`.

The last part of the chain (steps 5.4 to 8) was re-run from scratch and matched the submitted file byte for byte.

---

## 1. What you need

**Hardware.** 4 CPU cores and 16 GB RAM are enough (that is what was used). Keep about 10 GB of free disk for `v8_cal_code/work/`.
No GPU is needed.

**Software.** Python 3.11 with the packages in `v8_cal_code/requirements.txt`:

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -r v8_cal_code/requirements.txt
```

**Code.** Clone the repository and switch to the branch:

```bash
git clone https://github.com/mehakmittal1234/ML_2026.git
cd ML_2026
git checkout claude/awesome-ride-rnl6x4
```

**Inputs.**

| Input | Where to get it | Put it here |
|---|---|---|
| Challenge dataset (7 TSV files) | Unstop portal, "Download Dataset" (team leader account) | `dataset/train/train_source{1,2,3}.tsv`, `dataset/train/train_ground_truth.tsv`, `dataset/test/test_source{1,2,3}.tsv` |
| Official validator | the `utils/` folder of the student resource | `utils/validate_submission.py` |
| Base train scores | shared Drive file, `train_s2_c0.parquet` (117,324,783 bytes, md5 `ef7d028dcaff276cddeff1b09e571751`) | `v8_cal_code/work/scores/m4it/train_s2_c0.parquet` |
| Base test scores | shared Drive file, `test_s2_c0.parquet` (107,786,962 bytes, md5 `83fec309501645bed1326999bdee593d`) | `v8_cal_code/work/scores/v9s/test_s2_c0.parquet` |

The two base score files are on Drive:
https://drive.google.com/file/d/1exBjzanZUUQhJ_NFTaf5vI8cqyuaOgPp/view and
https://drive.google.com/file/d/1HUzCq9inBs9O6k-Bj8pmrA5T79VbgfaA/view.
Use the md5 values above to tell which is which.

The two files hold the stage-2 model's probabilities for every candidate pair: out-of-fold on train, and on test. Validation score
of that model: 0.98941. Section 6 explains how to rebuild them if you do not have them.

## 2. Folder layout before running

```
ML_2026/
├── dataset/
│   ├── train/  train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
│   └── test/   test_source1.tsv   test_source2.tsv   test_source3.tsv
├── utils/validate_submission.py
└── v8_cal_code/
    ├── run_v14e.sh
    ├── requirements.txt
    ├── s*.py …
    └── work/scores/
        ├── m4it/train_s2_c0.parquet
        └── v9s/test_s2_c0.parquet
```

## 3. Run it

```bash
cd v8_cal_code
PY=python ./run_v14e.sh
```

- Output goes to `outputs/` (set `OUT=/some/dir` to change it).
- Progress is printed with timestamps. Each step's log is in `v8_cal_code/work/tmp/<step>.log`.
- If a step fails, fix the cause and run the script again. Finished steps are skipped (marker files `work/tmp/done_*`). To redo a
  step, delete its marker.
- The builders refuse to overwrite an existing output folder. Use a fresh `OUT` folder, or delete old outputs first.
- Expect roughly 2–3 hours on 4 cores. Most of the time goes into normalisation (step 1) and the feature and model steps
  (steps 4–6).

## 4. Check the results at each step

| Step | Script | You should see |
|---|---|---|
| 1 | `s00_convert`, `s01_indic`, `s02_norm`, `s02b_norm2` | `work/data/*.parquet`, `work/norm2/*.parquet` |
| 2 | `s18_priorshift v9s 0.7 v9scal` | threshold 0.7 on `v9scal` gives 5,802,685 matches |
| 3 | `s35_near_cells`, `s37_droplist` | `outputs/v11_nb/drop_pairs.tsv` with 30,001 rows |
| 4 | `s50_feats`, `s56_srcver` (train and test) | `work/feat/f50/`, `work/feat/f56/` |
| 5.1 | `s53_dr test v9s dr3 --inv` | `work/scores/dr3/test_factor.parquet` |
| 5.2 | `s54_hybrid … hyb4 0.9` | `work/scores/hyb4/` (removes 27,519 accepted pairs) |
| 5.3 | `s62_ns test … ns1` | `work/scores/ns1/` |
| 5.4 | `s58_blend hyb4 ns1 ens1` | "expected-F decoder keeps 5,768,150 pairs" |
| 5.5 | `s55_build ens1_efd …` | `outputs/v13_ens_efd`: 5,765,507 matches (France 852,568, India 2,705,435, US 2,207,504) |
| 6 | `s67_sign feats`, `s68_struct test st1` (600 rounds) | `work/feat/f67/`, `work/scores/st1/test_struct.parquet` |
| 7.1 | `s70_add` | `outputs/v14_add`: +32,801, total 5,798,308 |
| 7.2 | `s71_deco` | `outputs/v14c_deco`: +11,788, total 5,810,096 |
| 7.3 | `s72_frdeco … full` | `outputs/v14e_frfull`: +24,931, total **5,835,027** (France 877,499, India 2,719,394, US 2,238,134) |
| 7.4 (optional) | `s75_samename` | `outputs/v15_same`: +6,055, total 5,841,082 |
| 8 | `s66_cand`, official validator | `candidate_pairs.tsv` with 7,674,729 pairs (4.43 per Source-1 entity); "PASS" for both outputs |

**Final check:** the md5 of the file to submit should match.

| File | md5 |
|---|---|
| `outputs/v14e_frfull/matching_results.tsv` | `1f7052312d78b1cd5942ca5fc3a5d35a` |
| `outputs/v15_same/matching_results.tsv` | `091798c08669e9067a404c48a34fdc5c` |

Steps 5.4 to 8 were re-run and reproduced both files byte for byte. The LightGBM steps (5.1, 5.2, 5.3 and 6) are deterministic with
the same package versions and the same number of threads (4). On a different machine a handful of pairs can move, so the md5 can
differ while the match counts stay within a few pairs of the table.

## 5. Submit

- **Leaderboard:** upload `outputs/v14e_frfull/matching_results.tsv` as it is (0.984861). `outputs/v15_same/matching_results.tsv`
  is the optional follow-up (not leaderboard-tested; estimated about +0.0002).
- **Final package:** each output folder also has `candidate_pairs.tsv`. The organisers review it and rank smaller candidate sets
  higher. Per the organiser FAQ, the zip's top level holds only `output/` (with `matching_results.tsv` and `candidate_pairs.tsv`),
  `code/` and the methodology document. `requirements.txt` and `README.md` go in `code/business_entity_resolution/`.

Ready-made copies of the leaderboard files are in `submissions/` on the branch (zipped `matching_results.tsv`).

## 6. If you do not have the two base score files

The base scores come from the stage-2 cascade (validation 0.98941). Rebuilding them is the long part:

1. `v8_cal_code/run_upto_stage2.sh`: normalisation, blocking keys, retrieval (top 8 Source-1 candidates per record), pair features,
   stage-1 LightGBM, and the stage-2 feature files. This took about 3.5 hours for retrieval alone on an 8 GB laptop, and needs about
   40 GB of disk.
2. `s14b_final.py v7t sup,tells`, then `s19_specialist.py final v8s`. This gives the v8 scores, the ones behind v8_cal (LB 0.982143).
3. The "iteration round 2" (v9) that produced `m4it` / `v9s`: recompute the stage-2 context features from the v8 scores
   (`S2_PSRC`, `S2_OUT` environment variables of `s11_stage2.py`), then refit. The exact commands for this round were run in an
   earlier session and are not recorded in this repository. If you cannot recover them, use the two Drive files.

Using the v8 scores in place of `v9s` / `m4it` runs through the same chain, but it starts from a weaker base and has not been
measured on the leaderboard.

## 7. What each stage does (one line each)

- **Steps 1–2:** load the data, normalise names and addresses, and recalibrate the base probabilities per country.
- **Step 3:** list pairs in "same name, nearby house number" groups where test has more pairs than the data generator can explain.
  These are dropped for France.
- **Steps 4–5:** features and two test-adapted corrections of the base scores: a count-based density-ratio correction, and a
  specialist trained on real same-name neighbours. Their blend is decided with an expected-F0.5 decoder, which gives v13.
- **Step 6:** a model that uses pair structure only, never the base scores. It finds true copies the base scores reject on test.
- **Step 7:** add those copies back:
  - pairs the structural model is sure about (US and India)
  - decorated names ("Service", "Partners", "& Associés", "Et Fils", …), up to the per-entity rate seen in train
  - (optional) same-name copies with a lower house number, a truncated number, or a unique name
- **Step 8:** candidate file and the official validator.

## 8. Troubleshooting

- **"Killed" / out of memory.** Close other programs. Every step fits in 16 GB when run alone.
- **"exists; refusing to overwrite".** An output folder is already there. Delete it or set a new `OUT`.
- **A step fails halfway.** Read `work/tmp/<step>.log`, fix the cause, re-run `./run_v14e.sh`. Finished steps are skipped.
- **Different md5, same counts.** Expected on a different CPU or thread count (see section 4). The file is still valid; the
  validator must print PASS.
