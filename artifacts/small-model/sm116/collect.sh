#!/bin/bash
# The training-free state check (SM116): state directions from the 100 direction users, 2 CPU processes (halves of
# the scorer's stratified order). Stats -> work/stats-{a,b}.npz (never committed), rows -> work/collect-{a,b}.jsonl.
cd /c/Users/jonat/workspace/wt/sm116
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
M=C:/Users/jonat/workspace/wt/sm92/artifacts/small-model/sm92-e26-dec-1ep32k-muon/expertium/E89-F65e-F77a/master.pth
D=artifacts/small-model/sm116
run() {  # $1 part, $2 start
  SM116_STATS=$D/work/stats-$1.npz env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START $PY -X utf8 $D/sm116_state.py collect \
    --device cpu --pool 1500 --users-file $D/users-directions.json --start $2 --count 50 \
    --model sm92=expertium:w64:$M --only sm92/w8+s8 --inputs e2s-query --split-existing --deleted-cards neighbour \
    --time-limit 100000 --out $D/work/collect-$1.jsonl > $D/work/collect-$1.out 2>&1
  echo "part $1 exit $?" >> $D/work/collect.log
}
run a 0 & run b 50 & wait
echo "collect done" >> $D/work/collect.log
