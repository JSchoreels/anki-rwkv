#!/usr/bin/env bash
# The small model in the review simulator (SM41), part A diagnostic: 20 sampled users x 1 split (diag/sample.json),
# 60 days, lanes tf + ihr-0.90 / fsrs7-0.90 x seeds 0-3 (diag/lanes.txt), per-review logs (diag/logs/<user>-<split>.npz).
# 4 processes x 1 thread, below-normal priority, CPU only; quiet.json holds inside the engine; resumable per split.
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/sm41
O=$WT/artifacts/small-model/sm41/diag
F=$WT/artifacts/small-model/sm41/fullrun/fits
LANES=$(cat "$O/lanes.txt" | tr -d '\r\n')
cd "$WT"
mkdir -p "$O/run-logs" "$O/rows" "$O/logs"
echo "start $(date '+%F %T') pid $$" >> "$O/runner.log"
export PY O F LANES
"$PY" -c "
import json
d = json.load(open('$O/sample.json'))['users']
for u in sorted(d, key=lambda u: -u['window_reviews']): print(u['user'], u['split_day'])
" | tr -d '\r' | xargs -P 4 -L 1 sh -c '
  "$PY" -X utf8 scripts/sm41_engine.py run --user $0 --split $1 --days 60 --lanes "$LANES" \
    --fsrs7-fits "$F/$0.jsonl" --min-free-gb 3 --out "$O/rows/$0.jsonl" --log-dir "$O/logs" >> "$O/run-logs/$0.log" 2>&1'
echo "done $(date '+%F %T')" >> "$O/runner.log"
