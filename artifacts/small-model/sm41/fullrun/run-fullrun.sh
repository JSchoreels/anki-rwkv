#!/usr/bin/env bash
# The small model in the review simulator (SM41) stage 2 part 3, the full run (CPU only, no GPU): 117 users x 2-3
# split points (304) x 60 days, 59 lanes per split (DESIGN.md section 10). 4 processes, 1 thread each, below-normal
# priority, largest users first. One job per user: (1) the pre-split FSRS-7 refits of its splits (srs-benchmark venv,
# scripts/sm41_fsrs7_fit.py -> fits/<user>.jsonl), (2) every split's 59 lanes (scripts/sm41_engine.py, one warm-up
# pass -> rows/<user>.jsonl, log logs/<user>.log). Resumable: finished (user, split) fits and rows are skipped.
# Every job holds new users / splits during quiet.json windows (scope all / cpu, from 10 min before) and holds a new
# split while less than 3 GB of RAM is available. At the end: check + report (fullrun-tables.md, fullrun-result.json).
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
PYB=C:/Users/jonat/workspace/srs-benchmark/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/sm41
O=$WT/artifacts/small-model/sm41/fullrun
LANES=$(cat "$O/lanes.txt" | tr -d '\r\n')
cd "$WT"
mkdir -p "$O/logs" "$O/rows" "$O/fits"
echo "start $(date '+%F %T') pid $$" >> "$O/runner.log"
export PY PYB O LANES
"$PY" -c "
import json
d = json.load(open('$O/users.json'))
us = sorted(d['users'], key=lambda u: -sum(s['window_reviews'] for s in u['splits']))
for u in us: print(u['user'], ','.join(str(s['split_day']) for s in u['splits']))
" | tr -d '\r' | xargs -P 4 -L 1 sh -c '
  "$PYB" -X utf8 scripts/sm41_fsrs7_fit.py fit --users "$O/users.json" --only $0 --out "$O/fits/$0.jsonl" >> "$O/logs/$0.log" 2>&1 &&
  "$PY" -X utf8 scripts/sm41_engine.py run --user $0 --splits $1 --days 60 --lanes "$LANES" \
    --fsrs7-fits "$O/fits/$0.jsonl" --min-free-gb 3 --out "$O/rows/$0.jsonl" >> "$O/logs/$0.log" 2>&1'
"$PY" -X utf8 scripts/sm41_fullrun.py check > "$O/check.out" 2>&1
"$PY" -X utf8 scripts/sm41_fullrun.py report > "$O/report.out" 2>&1
echo "done $(date '+%F %T')" >> "$O/runner.log"
