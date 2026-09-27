#!/bin/bash
# Reproduce the leaderboard submission v14e_frfull (public LB 0.984861) and the optional v15_same from:
#   - the raw dataset in ../dataset/{train,test}/ and the official validator in ../utils/
#   - the two base score files (stage-2 model "m4it / v9s", validation 0.98941):
#       work/scores/m4it/train_s2_c0.parquet   (md5 ef7d028dcaff276cddeff1b09e571751)
#       work/scores/v9s/test_s2_c0.parquet     (md5 83fec309501645bed1326999bdee593d)
# See ../HOW_TO_REPRODUCE.md. Safe to re-run: finished steps are skipped (markers in work/tmp/done_*).
set -e
PY=${PY:-python3}
OUT=${OUT:-../outputs}
cd "$(dirname "$0")"
mkdir -p work/tmp "$OUT"
run() { local tag=$(echo "$*" | tr ' /:,' '____')
        if [ -f work/tmp/done_$tag ]; then echo "skip   $*"; return; fi
        echo "$(date +%T) run $*" | tee -a work/tmp/PROGRESS
        $PY "$@" > work/tmp/$tag.log 2>&1 || { echo "FAILED: $*  (see work/tmp/$tag.log)"; exit 1; }
        touch work/tmp/done_$tag; }

# 0. inputs
[ -f ../dataset/test/test_source1.tsv ] || { echo "missing ../dataset/test/test_source1.tsv"; exit 1; }
[ -f ../dataset/train/train_ground_truth.tsv ] || { echo "missing ../dataset/train/train_ground_truth.tsv"; exit 1; }
[ -f work/scores/v9s/test_s2_c0.parquet ] || { echo "missing work/scores/v9s/test_s2_c0.parquet (base test scores)"; exit 1; }
[ -f work/scores/m4it/train_s2_c0.parquet ] || { echo "missing work/scores/m4it/train_s2_c0.parquet (base train scores)"; exit 1; }

# 1. data conversion, Indic dictionaries, normalisation
run s00_convert.py
run s01_indic.py
run s02_norm.py
run s02b_norm2.py

# 2. per-country recalibration of the base test scores (threshold 0.7 -> 5,802,685 matches = v9_cal)
run s18_priorshift.py v9s 0.7 v9scal

# 3. neighbour-business drop list (used for France in step 5)
run s35_near_cells.py m4it v9s
run s37_droplist.py v9s v9scal 0.7 "$OUT/v11_nb"

# 4. pair features for the stage-2 pair sets
run s50_feats.py train m4it
run s50_feats.py test v9s
run s56_srcver.py train m4it
run s56_srcver.py test v9s

# 5. test adaptation -> v13_ens_efd (5,765,507 matches)
run s53_dr.py test v9s dr3 --inv
run s54_hybrid.py v9s v9scal dr3 0.7 hyb4 0.9
run s62_ns.py test v9s v9scal 0.7 ns1
run s58_blend.py hyb4 ns1 ens1
run s55_build.py ens1_efd 0.5 "$OUT/v13_ens_efd" "$OUT/v11_nb/drop_pairs.tsv" France

# 6. score-free structural detector
run s67_sign.py feats train
run s67_sign.py feats test
R=600 run s68_struct.py test st1

# 7. restore true copies the score chain rejects on test -> v14e_frfull (5,835,027 matches, LB 0.984861)
run s70_add.py "$OUT/v13_ens_efd" st1 ens1 0.97 US,India "$OUT/v14_add"
run s71_deco.py "$OUT/v14_add" st1 ens1 US:0.8,India:0.97 0.9 "$OUT/v14c_deco"
run s72_frdeco.py "$OUT/v14c_deco" st1 ens1 full 0 1 "$OUT/v14e_frfull"
# optional: v15_same = v14e + 6,055 same-name copies (not leaderboard-tested, estimated +0.0002)
run s75_samename.py "$OUT/v14e_frfull" "$OUT/v15_same"

# 8. candidate files (stage-2 set, 4.43 per S1 entity) and the official validator
run s66_cand.py v9scal "$OUT/v14e_frfull" "$OUT/v15_same"
for v in v14e_frfull v15_same; do
  echo "== $v"; $PY ../utils/validate_submission.py --matching "$OUT/$v/matching_results.tsv" --candidate "$OUT/$v/candidate_pairs.tsv" --test-dir ../dataset/test | tail -1
  md5sum "$OUT/$v/matching_results.tsv"
done
echo "expected md5: v14e_frfull 1f7052312d78b1cd5942ca5fc3a5d35a | v15_same 091798c08669e9067a404c48a34fdc5c"
echo V14E_DONE
