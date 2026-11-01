#!/bin/bash
# Codebook state compression sweep (SM90) CPU proof: (d) the int8 / int4+int8 paths of the sm90 tree are bitwise the
# released tree's (wt/sm72s) and (e) the CPU scorer with compression off reproduces the recorded released rows
# (gpu/small-lf-neighbour.jsonl, strict 1e-5 rule). 3 smallest users of the 732 set, leak-free inputs.
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
CK=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/scale-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon/expertium/E89-F65e-F77a/master.pth
H=C:/Users/jonat/workspace/wt/sm90/artifacts/small-model/sm90
export OMP_NUM_THREADS=4
"$PY" -X utf8 -c "import json; json.dump({'users': [2388, 4676, 1712]}, open('$H/proof-users.json', 'w'))"
for T in sm90 sm72s; do
  cd C:/Users/jonat/workspace/wt/$T
  env -u RWKV_DELETED_CARDS CUDA_VISIBLE_DEVICES= "$PY" -X utf8 C:/Users/jonat/workspace/wt/$T/scripts/small_model_heldout.py --device cpu --workers 1 \
    --pool 732 --users-file $H/proof-users.json --model rel=expertium:w64:$CK --only rel/w8+s8 --only rel/w8+s4c8n \
    --split-existing --deleted-cards neighbour --out $H/proof/cpu-$T.jsonl > $H/logs/proof-int8-$T.log 2>&1 || echo "[proof] $T scorer failed"
done
echo "[proof-int8] done"
