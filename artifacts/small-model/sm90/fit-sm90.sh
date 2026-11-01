#!/bin/bash
# Codebook state compression sweep (SM90): k-means codebooks on the released small model's corpus (45 users fitted,
# every 10th held out for the proxy). WKV components 0-5 with (20, 16, 12, 8, 6, 4) residual 10-bit stages (the
# ladder uses prefixes), norm ranges 6/8/10 bits; shift m4 x 8 and m8 x 4 stages of 12-bit codes. 6 CPU processes.
WT=C:/Users/jonat/workspace/wt/sm90
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
CK=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/scale-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon/expertium/E89-F65e-F77a/master.pth
export OMP_NUM_THREADS=4 VQ_FIT_THREADS=4
cd $WT
B=artifacts/small-model/sm90/books
L=artifacts/small-model/sm90/logs
until [ "$(ls $B/corpus/*.npz 2>/dev/null | wc -l)" -ge 50 ] && grep -q '\[collect\] done' $L/collect.nohup 2>/dev/null; do sleep 20; done
A="--student $CK --out-dir $B"
S="--wkv-stages 20 16 12 8 6 4 --norm-bits 6 8 10 --holdout-every 10 --max-points 200000"
"$PY" -X utf8 $WT/scripts/state_vq_fit.py $A fit --parts wkv $S --components 0 --out fit-c0.npz > $L/fit-c0.log 2>&1 &
"$PY" -X utf8 $WT/scripts/state_vq_fit.py $A fit --parts wkv $S --components 1 5 --out fit-c15.npz > $L/fit-c15.log 2>&1 &
"$PY" -X utf8 $WT/scripts/state_vq_fit.py $A fit --parts wkv $S --components 2 3 --out fit-c23.npz > $L/fit-c23.log 2>&1 &
"$PY" -X utf8 $WT/scripts/state_vq_fit.py $A fit --parts wkv $S --components 4 --out fit-c4.npz > $L/fit-c4.log 2>&1 &
for r in t c; do
  "$PY" -X utf8 $WT/scripts/state_vq_fit.py $A fit --parts shift --holdout-every 10 --max-points 300000 --shift-m 4 8 \
    --shift-stages 8 4 --norm-bits 6 8 10 --roles $r --out fit-shift-$r.npz > $L/fit-shift-$r.log 2>&1 &
done
wait
"$PY" -X utf8 $WT/scripts/state_vq_fit.py $A merge --files fit-c0.npz fit-c15.npz fit-c23.npz fit-c4.npz fit-shift-t.npz \
  fit-shift-c.npz --out codebooks-sm90.npz > $L/merge.log 2>&1
echo "[fit] done $(date +%H:%M:%S) rc $?"
