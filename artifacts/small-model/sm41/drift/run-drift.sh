#!/usr/bin/env bash
# The small model in the review simulator (SM41) stage 2, the drift study (CPU only): 50 users x 2-3 split points x
# 60 days + 10 users x 1 split x 120 days, every arm (scripts/sm41_drift.py ARMS), 3 seeds, all lanes of a split in
# one batch. 4 processes, 1 thread each, below-normal priority. Resumable: a finished (user, split, days) row in
# rows/<user>-<days>.jsonl is skipped. Each process holds new users / splits during quiet.json windows (from 10 min before).
# Rows -> rows/<user>-<days>.jsonl, logs -> logs/<user>-<days>.log, then the report (drift-tables.md, drift-result.json).
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/sm41
O=$WT/artifacts/small-model/sm41/drift
cd "$WT"
mkdir -p "$O/logs" "$O/rows"
echo "start $(date '+%F %T') pid $$" >> "$O/runner.log"
"$PY" -c "
import json
d = json.load(open('$O/users.json'))
for u in d['users']: print(u['user'], ','.join(str(s['split_day']) for s in u['splits']), d['days'])
for u in d['long']: print(u['user'], u['split']['split_day'], d['long_days'])
" | tr -d '\r' | xargs -P 4 -L 1 sh -c "\"$PY\" -X utf8 scripts/sm41_drift.py run --user \$0 --splits \$1 --days \$2 --out $O/rows/\$0-\$2.jsonl >> $O/logs/\$0-\$2.log 2>&1"
"$PY" -X utf8 scripts/sm41_drift.py report > "$O/report.out" 2>&1
echo "done $(date '+%F %T')" >> "$O/runner.log"
