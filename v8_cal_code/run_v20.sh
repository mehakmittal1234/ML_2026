#!/bin/bash
# v20 = v8_cal lineage re-scored on test with TRAIN-SCALE IDF (root-cause fix of test-only rejections). No model is retrained.
PY=${PY:-python3}
cd "$(dirname "$0")"
set -e
echo "$(date +%T) s08F features"; $PY s08F_feat.py test v6c 8 40 > work/tmp/v20_s08F.log 2>&1
echo "$(date +%T) s09F stage-1"; $PY s09F_score.py > work/tmp/v20_s09F.log 2>&1
echo "$(date +%T) s11 build";   $PY s11_stage2.py build test v6cF m1F > work/tmp/v20_s11b.log 2>&1
echo "$(date +%T) s11 buildx";  $PY s11_stage2.py buildx test m1F > work/tmp/v20_s11x.log 2>&1
echo "$(date +%T) s16 support"; S16_SRC=m1F $PY s16_support.py test > work/tmp/v20_s16.log 2>&1
echo "$(date +%T) s17 tells";   TELLS_SRC=m1F TELLS_SCALE=1 $PY s17_tells.py test > work/tmp/v20_s17.log 2>&1
echo "$(date +%T) stage-2 score"; $PY s14F_score.py m1F v7t v7tF > work/tmp/v20_s14F.log 2>&1
echo "$(date +%T) specialist";  S19_SRC=m1F S19_TELLS=m1F S19_FTAG=v6cF S19_NOP1S=1 $PY s19F_score.py v8s v7tF v8sF > work/tmp/v20_s19F.log 2>&1
echo "$(date +%T) priorshift";  $PY s18_priorshift.py v8sF 0.7 v8sFcal > work/tmp/v20_s18.log 2>&1
$PY s22_country_thr.py v8sFcal ../outputs/v20_idffix 0.7 0.7 0.7 > work/tmp/v20_s22.log 2>&1
cd .. && python3 utils/validate_submission.py --matching outputs/v20_idffix/matching_results.tsv --candidate /dev/null/none | tail -1
cp outputs/v8/candidate_pairs.tsv outputs/v20_idffix/
echo "$(date +%T) V20_DONE"
