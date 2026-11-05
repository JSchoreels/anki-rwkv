#!/usr/bin/env bash
# The small model in the review simulator (SM41), part B: confirm the time-aware target at beta = 3 (best of screen 4
# at matched memorised cards) on every full-run user and split -> policy/confirm3/rows/<user>.jsonl; then the reports.
# 4 processes x 1 thread, below-normal priority, CPU only; quiet.json holds inside the engine; resumable per split.
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/sm41
P=$WT/artifacts/small-model/sm41/policy
F=$WT/artifacts/small-model/sm41/fullrun/fits
O=$P/confirm3
cd "$WT"
mkdir -p "$O/logs" "$O/rows"
echo "start $(date '+%F %T') pid $$" >> "$O/runner.log"
LANES=$(cat "$P/lanes-confirm3.txt" | tr -d '\r\n')
export PY O F LANES
"$PY" -c "
import json
d = json.load(open('$WT/artifacts/small-model/sm41/fullrun/users.json'))
us = sorted(d['users'], key=lambda u: -sum(s['window_reviews'] for s in u['splits']))
for u in us: print(u['user'], ','.join(str(s['split_day']) for s in u['splits']))
" | tr -d '\r' | xargs -P 4 -L 1 sh -c '
  "$PY" -X utf8 scripts/sm41_engine.py run --user $0 --splits $1 --days 60 --lanes "$LANES" \
    --fsrs7-fits "$F/$0.jsonl" --min-free-gb 3 --out "$O/rows/$0.jsonl" >> "$O/logs/$0.log" 2>&1'
"$PY" -X utf8 -W ignore scripts/sm41_frontier.py "$O/rows/*.jsonl" s3 > "$P/frontier-confirm3.out" 2>&1
echo "done $(date '+%F %T')" >> "$O/runner.log"
