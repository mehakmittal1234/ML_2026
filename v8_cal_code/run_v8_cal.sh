#!/bin/zsh
# Full pipeline that produced outputs/v8_cal (dataset/ -> matching_results.tsv + candidate_pairs.tsv).
# Keep this folder inside student_resource/ (it reads ../dataset and writes its own ./work). Stops at the first failing step.
set -e
PY=${PY:-../.venv/bin/python}
cd "$(dirname "$0")"
mkdir -p work/tmp
run() { echo "== $*"; $PY "$@" > work/tmp/$(echo "$*" | tr ' /' '__').log 2>&1; }

# 1. data, Indic transliteration dictionaries, normalisation
run s00_convert.py
run s01_indic.py
run s02_norm.py
run s02b_norm2.py
# 2. blocking keys (v4 families + R1/HS/ND; A1/HZ dropped)
for sp in train test; do run s03b_keys2.py $sp; run s03c_keys3.py $sp; done
# 3. stage-0 record-side ranker: pool = fold-0 owned records + 1% sample, cheap features, LightGBM ranker
run s04d_block_rec.py train keys3 gtplus:0:100 f3 1000 100000 60 4
run s05_cheap.py train f3 q norm2
run s06_ranker.py train f3
# 4. retrieval: key join -> top-60 per record -> ranker -> keep top-8 per S2/S3 record
for sp in train test; do run s07_retrieve.py $sp v6c f3 1000 100000 60 8 40; done
# 5. pair features + stage-1 LightGBM (cross-fitted, out-of-fold on train)
for sp in train test; do run s08_feat.py $sp v6c 8 40; run s09_model.py ctx $sp v6c; done
run s09_model.py train v6c m1 700
for sp in train test; do run s09_model.py score $sp v6c m1; done
# 6. stage-2 features: competition / S1 context / siblings, raw-name + house-number relations, name support, edit-type tells
for sp in train test; do
  run s11_stage2.py build $sp v6c m1
  run s11_stage2.py buildx $sp m1
  run s16_support.py $sp
  run s17_tells.py $sp
done
# 7. v7t: stage-2 LightGBM on all train folds (700 rounds) -> base scores for every test pair
run s14b_final.py v7t sup,tells
# 8. v8s: borderline specialist (1500 rounds, all features) re-scores pairs with 0.01 < p1 < 0.99; others keep v7t
run s19_specialist.py final v8s
# 9. v8scal: per-country prior-shift recalibration (equivalent raw thresholds France 0.746, India 0.783, US 0.818)
run s18_priorshift.py v8s 0.7 v8scal
# 10. submission files at threshold 0.7 (one owner per S2/S3 record) + official validator
run s13_submit.py v8scal 0.7 v6c ../outputs/v8_cal_repro
cd .. && python3 utils/validate_submission.py --matching outputs/v8_cal_repro/matching_results.tsv \
  --candidate outputs/v8_cal_repro/candidate_pairs.tsv --test-dir dataset/test
echo V8_CAL_DONE
