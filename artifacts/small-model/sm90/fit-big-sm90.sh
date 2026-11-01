#!/bin/bash
# Codebook state compression sweep (SM90): the fitting-data knob. The CPU proof showed the deep WKV stages overfit at
# 200k points (component 0 held-out / in-sample residual 1.15x at stage 1 -> 4.4x at stage 20), so the WKV books are
# refitted on every fit-split head of the corpus (max 900k; 45 users) with the same seeds; the held-out proxy then
# picks the WKV books (DESIGN.md). Killed at 15:35 if not done (break at 16:00).
WT=C:/Users/jonat/workspace/wt/sm90
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
export OMP_NUM_THREADS=4 VQ_FIT_THREADS=4
cd $WT
B=artifacts/small-model/sm90/books
L=artifacts/small-model/sm90/logs
LEFT=$(( $(date -d 15:35 +%s) - $(date +%s) ))
S="--out-dir $B fit --parts wkv --wkv-stages 20 16 12 8 6 4 4 3 --norm-bits 6 8 10 --holdout-every 10 --max-points 900000"
timeout $LEFT "$PY" -X utf8 $WT/scripts/state_vq_fit.py $S --components 0 --out fitbig-c0.npz > $L/fitbig-c0.log 2>&1 &
timeout $LEFT "$PY" -X utf8 $WT/scripts/state_vq_fit.py $S --components 1 --out fitbig-c1.npz > $L/fitbig-c1.log 2>&1 &
timeout $LEFT "$PY" -X utf8 $WT/scripts/state_vq_fit.py $S --components 2 --out fitbig-c2.npz > $L/fitbig-c2.log 2>&1 &
timeout $LEFT "$PY" -X utf8 $WT/scripts/state_vq_fit.py $S --components 3 5 --out fitbig-c35.npz > $L/fitbig-c35.log 2>&1 &
timeout $LEFT "$PY" -X utf8 $WT/scripts/state_vq_fit.py $S --components 4 6 7 --out fitbig-c467.npz > $L/fitbig-c467.log 2>&1 &
wait
echo "[fitbig] done $(date +%H:%M:%S)"
