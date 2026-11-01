#!/bin/bash
# Codebook state compression sweep (SM90): final codebooks (all fits merged, 6/8-bit norm ranges with hi = the largest
# norm), pinned sha256, then the CPU proof (proof-cpu.sh).
WT=C:/Users/jonat/workspace/wt/sm90
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
H=$WT/artifacts/small-model/sm90
L=$H/logs
export OMP_NUM_THREADS=4 VQ_FIT_THREADS=4
cd $WT
until grep -q '\[fit\] done' $L/fit.nohup 2>/dev/null; do sleep 15; done
A="--out-dir artifacts/small-model/sm90/books"
"$PY" -X utf8 $WT/scripts/state_vq_fit.py $A merge --files fit-c0.npz fit-c15.npz fit-c23.npz fit-c4.npz fit-c67.npz \
  fit-shift-t.npz fit-shift-c.npz --out codebooks-merged.npz > $L/finalize.log 2>&1
"$PY" -X utf8 $WT/scripts/state_vq_fit.py $A refit-norms --codebooks codebooks-merged.npz --holdout-every 10 \
  --max-points 200000 --norm-bits 6 8 --norm-top max --out codebooks-sm90.npz >> $L/finalize.log 2>&1
"$PY" -X utf8 -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" $H/books/codebooks-sm90.npz > $H/books/codebooks-sm90.sha256
echo "[finalize] books $(cat $H/books/codebooks-sm90.sha256) $(date +%H:%M:%S)"
bash $H/proof-cpu.sh
