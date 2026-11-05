#!/usr/bin/env bash
# The small model in the review simulator (SM41), part B screen: the 20 users of part A (diag/sample.json) x 1 split,
# 60 days, the lanes of policy/lanes-<tag>.txt -> policy/<tag>/rows/<user>.jsonl. Usage: run-screen.sh <tag>.
# 4 processes x 1 thread, below-normal priority, CPU only; quiet.json holds inside the engine; resumable per split.
set -u
TAG=$1
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/sm41
P=$WT/artifacts/small-model/sm41/policy
O=$P/$TAG
F=$WT/artifacts/small-model/sm41/fullrun/fits
LANES=$(cat "$P/lanes-$TAG.txt" | tr -d '\r\n')
cd "$WT"
mkdir -p "$O/logs" "$O/rows"
echo "start $(date '+%F %T') pid $$" >> "$O/runner.log"
export PY O F LANES
"$PY" -c "
import json
d = json.load(open('$WT/artifacts/small-model/sm41/diag/sample.json'))['users']
for u in sorted(d, key=lambda u: -u['window_reviews']): print(u['user'], u['split_day'])
" | tr -d '\r' | xargs -P 4 -L 1 sh -c '
  "$PY" -X utf8 scripts/sm41_engine.py run --user $0 --split $1 --days 60 --lanes "$LANES" \
    --fsrs7-fits "$F/$0.jsonl" --min-free-gb 3 --out "$O/rows/$0.jsonl" >> "$O/logs/$0.log" 2>&1'
echo "done $(date '+%F %T')" >> "$O/runner.log"
