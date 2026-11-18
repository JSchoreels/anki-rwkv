#!/bin/bash
# Training-free state check (SM116): CPU proof. The unmodified model and the 100% projection on the 8 proof users,
# same CPU scorer, then sm116_rule.py proof -> check-cpu-sm116.json (also vs the show-time retrain (SM92)'s GPU rows).
cd /c/Users/jonat/workspace/wt/sm116
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
M=C:/Users/jonat/workspace/wt/sm92/artifacts/small-model/sm92-e26-dec-1ep32k-muon/expertium/E89-F65e-F77a/master.pth
D=artifacts/small-model/sm116
A="--device cpu --pool 732 --users-file $D/users-proof.json --inputs e2s-query --split-existing --deleted-cards neighbour"
rm -f $D/work/proof-base.jsonl $D/work/proof-k100.jsonl
env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START $PY -X utf8 scripts/small_model_heldout.py $A \
  --model sm92=expertium:w64:$M --only sm92/w8+s8 --out $D/work/proof-base.jsonl > $D/work/proof-base.out 2>&1 &
SM116_BASIS=$D/work/basis.npz SM116_KEEP=1.0 env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START $PY -X utf8 $D/sm116_state.py score $A \
  --model sm116k100=expertium:w64:$M --only sm116k100/w8+s8 --out $D/work/proof-k100.jsonl > $D/work/proof-k100.out 2>&1 &
wait
$PY -X utf8 $D/sm116_rule.py proof
