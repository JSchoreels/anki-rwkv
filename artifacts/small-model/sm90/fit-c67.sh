#!/bin/bash
# Codebook state compression sweep (SM90): WKV rank components 6 and 7 (4 and 3 stages) for the rank-8 knob.
WT=C:/Users/jonat/workspace/wt/sm90
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
CK=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/scale-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon/expertium/E89-F65e-F77a/master.pth
export OMP_NUM_THREADS=4 VQ_FIT_THREADS=4
cd $WT
"$PY" -X utf8 $WT/scripts/state_vq_fit.py --student $CK --out-dir artifacts/small-model/sm90/books fit --parts wkv \
  --wkv-stages 20 16 12 8 6 4 4 3 --norm-bits 6 8 10 --holdout-every 10 --max-points 200000 --components 6 7 \
  --out fit-c67.npz > artifacts/small-model/sm90/logs/fit-c67.log 2>&1
echo "[fit-c67] done $(date +%H:%M:%S)"
