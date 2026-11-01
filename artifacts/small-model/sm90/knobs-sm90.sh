#!/bin/bash
# Codebook state compression sweep (SM90): secondary pipeline knobs, proxy only (CPU). WKV component 0 refitted with
# another codebook size (8-bit x 10 stages, 12-bit x 7 stages vs the main fit's 10-bit x 8), 3x the fitting points
# (600k) and 50 Lloyd iterations (vs 20); then knobs_sm90.py measures the held-out coding error per bit.
# Every process is killed at 15:50 (the 16:00 break; CPU quiet too).
WT=C:/Users/jonat/workspace/wt/sm90
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
CK=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/scale-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon/expertium/E89-F65e-F77a/master.pth
export OMP_NUM_THREADS=4 VQ_FIT_THREADS=4
cd $WT
B=artifacts/small-model/sm90/books
L=artifacts/small-model/sm90/logs
until grep -q '\[fit\] done' $L/fit.nohup 2>/dev/null; do sleep 20; done
LEFT=$(( $(date -d 15:50 +%s) - $(date +%s) ))
[ $LEFT -gt 600 ] || { echo "[knobs] no time before the break"; exit 0; }
A="--student $CK --out-dir $B"
F="fit --parts wkv --holdout-every 10 --components 0 --norm-bits 10"
timeout $LEFT "$PY" -X utf8 $WT/scripts/state_vq_fit.py $A $F --max-points 200000 --wkv-index-bits 8 --wkv-stages 10 --out knob-b8.npz > $L/knob-b8.log 2>&1 &
timeout $LEFT "$PY" -X utf8 $WT/scripts/state_vq_fit.py $A $F --max-points 200000 --wkv-index-bits 12 --wkv-stages 7 --out knob-b12.npz > $L/knob-b12.log 2>&1 &
timeout $LEFT "$PY" -X utf8 $WT/scripts/state_vq_fit.py $A $F --max-points 600000 --wkv-stages 8 --out knob-p600.npz > $L/knob-p600.log 2>&1 &
timeout $LEFT "$PY" -X utf8 $WT/scripts/state_vq_fit.py $A $F --max-points 200000 --iters 50 --wkv-stages 8 --out knob-i50.npz > $L/knob-i50.log 2>&1 &
wait
LEFT=$(( $(date -d 15:55 +%s) - $(date +%s) ))
[ $LEFT -gt 60 ] && timeout $LEFT "$PY" -X utf8 $WT/artifacts/small-model/sm90/knobs_sm90.py > $L/knobs-eval.log 2>&1
echo "[knobs] done $(date +%H:%M:%S)"
