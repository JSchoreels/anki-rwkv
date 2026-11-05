#!/usr/bin/env bash
# The small model in the review simulator (SM41), part B chain (CPU only): screens 2 and 3 (20 users), the plan's
# decision rule (sm41_policy.py choose -> lanes-confirm.txt), then the confirmation run on every full-run user and
# split (117 users, 304 splits, 60 days) -> confirm/rows/<user>.jsonl, and the reports. 4 processes x 1 thread,
# below-normal priority; quiet.json holds inside the engine (per user and per split); resumable (finished splits
# are skipped), so a relaunch of this script continues where it stopped.
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/sm41
P=$WT/artifacts/small-model/sm41/policy
F=$WT/artifacts/small-model/sm41/fullrun/fits
cd "$WT"
echo "start $(date '+%F %T') pid $$" >> "$P/chain.log"
bash "$P/run-screen.sh" screen2
bash "$P/run-screen.sh" screen3
"$PY" -X utf8 -W ignore scripts/sm41_policy.py choose --rows "$P/screen*/rows/*.jsonl" --out "$P/lanes-confirm.txt" > "$P/choose.out" 2>&1
"$PY" -X utf8 -W ignore scripts/sm41_policy.py report --rows "$P/screen*/rows/*.jsonl" --md "$P/screen-tables.md" > /dev/null 2>&1
echo "screens done $(date '+%F %T')" >> "$P/chain.log"
O=$P/confirm
mkdir -p "$O/logs" "$O/rows"
LANES=$(cat "$P/lanes-confirm.txt" | tr -d '\r\n')
export PY O F LANES
"$PY" -c "
import json
d = json.load(open('$WT/artifacts/small-model/sm41/fullrun/users.json'))
us = sorted(d['users'], key=lambda u: -sum(s['window_reviews'] for s in u['splits']))
for u in us: print(u['user'], ','.join(str(s['split_day']) for s in u['splits']))
" | tr -d '\r' | xargs -P 4 -L 1 sh -c '
  "$PY" -X utf8 scripts/sm41_engine.py run --user $0 --splits $1 --days 60 --lanes "$LANES" \
    --fsrs7-fits "$F/$0.jsonl" --min-free-gb 3 --out "$O/rows/$0.jsonl" >> "$O/logs/$0.log" 2>&1'
"$PY" -X utf8 -W ignore scripts/sm41_policy.py report --rows "$O/rows/*.jsonl" --md "$P/confirm-tables.md" > /dev/null 2>&1
"$PY" -X utf8 -W ignore scripts/sm41_policy.py report --rows "$O/rows/*.jsonl" --exclude-sample --md "$P/confirm-heldout-tables.md" > /dev/null 2>&1
echo "done $(date '+%F %T')" >> "$P/chain.log"
