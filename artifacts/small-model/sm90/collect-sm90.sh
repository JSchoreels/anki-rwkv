#!/bin/bash
# Codebook state compression sweep (SM90): corpus of the released small model's stored card/note states
# (50 training users of its run, 3k-60k reviews, none of the 732 set; leak-free inputs), 3 CPU shards.
WT=C:/Users/jonat/workspace/wt/sm90
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
CK=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/scale-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon/expertium/E89-F65e-F77a/master.pth
export OMP_NUM_THREADS=4 VQ_FIT_THREADS=2
cd $WT
for i in 0 1 2; do
  "$PY" -X utf8 $WT/scripts/state_vq_fit.py --student $CK --out-dir artifacts/small-model/sm90/books --exclude-pool 732 \
    --deleted-cards neighbour collect --users 50 --shard $i/3 --per-user 1500 > artifacts/small-model/sm90/logs/collect-$i.log 2>&1 &
done
wait
echo "[collect] done $(date +%H:%M:%S)"
