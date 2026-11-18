#!/bin/bash
# SM116 collection part b, rerun after the first run's part b went above 6 GB RAM on a 158k-review user (killed):
# part b's users with <= 125,000 reviews (work/users-directions-b.json).
cd /c/Users/jonat/workspace/wt/sm116
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
M=C:/Users/jonat/workspace/wt/sm92/artifacts/small-model/sm92-e26-dec-1ep32k-muon/expertium/E89-F65e-F77a/master.pth
D=artifacts/small-model/sm116
SM116_STATS=$D/work/stats-b.npz env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START $PY -X utf8 $D/sm116_state.py collect \
  --device cpu --pool 1500 --users-file $D/work/users-directions-b.json \
  --model sm92=expertium:w64:$M --only sm92/w8+s8 --inputs e2s-query --split-existing --deleted-cards neighbour \
  --time-limit 100000 --out $D/work/collect-b.jsonl > $D/work/collect-b.out 2>&1
echo "part b2 exit $?" >> $D/work/collect.log
