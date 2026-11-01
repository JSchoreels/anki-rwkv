#!/usr/bin/env bash
# The small model in the review simulator (SM41) stage 1 prototype: 5 users x 90 days, every lane, CPU, 4 processes
# (below-normal priority, 1 thread each). Rows -> prototype.jsonl, logs -> run-<user>.log, then the report.
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/sm41
O=$WT/artifacts/small-model/sm41
cd "$WT"
"$PY" -c "
import json
for u in json.load(open('$O/users.json'))['users']: print(u['user'], u['split_day'])
" | tr -d '\r' | xargs -P 4 -L 1 sh -c "\"$PY\" -X utf8 scripts/sm41_sim.py run --user \$0 --split-day \$1 --days 90 --seeds 2 --out $O/prototype.jsonl > $O/run-\$0.log 2>&1"
"$PY" -X utf8 scripts/sm41_sim.py report > $O/report.out 2>&1
echo "done $(date +%H:%M:%S)" >> $O/report.out
