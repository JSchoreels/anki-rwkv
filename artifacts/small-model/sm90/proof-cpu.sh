#!/bin/bash
# Codebook state compression sweep (SM90) CPU proof, run once codebooks-sm90.npz is final (5 CPU processes at most):
# (a) k-means determinism (two identical small fits), (b) proof_codec.py, (c, d) proof-int8.sh again on the final tree,
# (e) check_state_vq_score_gpu.py cpu for every screen recipe (3 processes, references for the GPU ticket), shift proxy.
WT=C:/Users/jonat/workspace/wt/sm90
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
H=$WT/artifacts/small-model/sm90
L=$H/logs
CK=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/scale-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon/expertium/E89-F65e-F77a/master.pth
export OMP_NUM_THREADS=4 VQ_FIT_THREADS=4
cd $WT
for part in a b c; do
  "$PY" -X utf8 $WT/scripts/check_state_vq_score_gpu.py cpu --cases $H/check-cases-$part.json --users 2 \
    --output $H/check-sm90-cpu-$part.json --ref-dir $H/check-ref > $L/check-cpu-$part.log 2>&1 &
done
for k in 1 2; do
  "$PY" -X utf8 $WT/scripts/state_vq_fit.py --out-dir artifacts/small-model/sm90/books fit --parts wkv --holdout-every 10 \
    --wkv-stages 1 1 1 1 3 --components 4 --norm-bits 8 --max-points 30000 --out det-$k.npz > $L/proof-fit-det-$k.log 2>&1 &
done
wait
"$PY" -X utf8 -c "
import json, numpy as np
a, b = np.load('$H/books/det-1.npz'), np.load('$H/books/det-2.npz')
same = sorted(a.files) == sorted(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)
d = max(float(np.abs(a[k] - b[k]).max()) for k in a.files)
json.dump({'what': 'two k-means fits, same seed and corpus (component 4, 3 stages, 30k points, 4 threads)', 'files': a.files, 'bitwise_equal': same, 'max_abs_diff': d}, open('$H/proof/proof-fit-det.json', 'w'), indent=1)
print('[proof] fit determinism bitwise', same, d)"
"$PY" -X utf8 $H/proof_codec.py > $L/proof-codec.log 2>&1; echo "[proof] codec exit $?"
"$PY" -X utf8 $H/proxy_sm90.py --books codebooks-sm90.npz --parts shift --out $H/proxy-sm90-shift.json > $L/proxy-shift.log 2>&1 &
mv $H/proof/cpu-sm90.jsonl $H/proof/cpu-sm90-pre-recipes.jsonl 2>/dev/null
bash $H/proof-int8.sh
"$PY" -X utf8 $H/proof_int8.py > $L/proof-int8-compare.log 2>&1; echo "[proof] int8 exit $?"
wait
echo "[proof-cpu] done $(date +%H:%M:%S)"
