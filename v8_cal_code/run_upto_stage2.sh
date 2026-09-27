#!/bin/bash
# Cloud rerun of the v8_cal pipeline through the stage-2 feature files (steps 1-6 of run_v8_cal.sh), bash instead of zsh.
# Logs: work/tmp/*.log ; progress markers in work/tmp/PROGRESS
set -e
PY=${PY:-/home/user/venv/bin/python}
cd "$(dirname "$0")"
mkdir -p work/tmp
run() { local tag=$(echo "$*" | tr ' /' '__')
        if [ -f work/tmp/done_$tag ]; then echo "skip $*"; return; fi
        echo "$(date +%T) == $*" | tee -a work/tmp/PROGRESS; $PY "$@" > work/tmp/$tag.log 2>&1; touch work/tmp/done_$tag; }
run s00_convert.py
run s01_indic.py
run s02_norm.py
run s02b_norm2.py
for sp in train test; do run s03b_keys2.py $sp; run s03c_keys3.py $sp; rm -rf work/keys2/${sp}_s*; done
run s04d_block_rec.py train keys3 gtplus:0:100 f3 1000 100000 60 4
run s05_cheap.py train f3 q norm2
run s06_ranker.py train f3
for sp in train test; do run s07_retrieve.py $sp v6c f3 1000 100000 60 8 40; done
rm -rf work/cand/f3
for sp in train test; do run s08_feat.py $sp v6c 8 40; run s09_model.py ctx $sp v6c; done
run s09_model.py train v6c m1 700
for sp in train test; do run s09_model.py score $sp v6c m1; done
for sp in train test; do
  run s11_stage2.py build $sp v6c m1
  run s11_stage2.py buildx $sp m1
  run s16_support.py $sp
  run s17_tells.py $sp
done
echo "$(date +%T) STAGE2_FEATURES_DONE" | tee -a work/tmp/PROGRESS
