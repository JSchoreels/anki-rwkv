#!/usr/bin/env bash
# The real-data check of the time-aware target (SM41r): the released small model teacher-forced on the REAL 60-day
# windows of every full-run user and split (the engine's tf lane alone + read-only hooks, scripts/sm41_realcheck.py)
# -> realcheck/logs/<user>-<split>.npz; then the analysis -> realcheck/realcheck-tables.md / .json.
# CPU only, 1 thread per process, below-normal priority, quiet.json holds inside each job. 2 processes while the
# beta = 3 confirmation (policy/confirm3) runs, 2 more once its runner.log says "done". Resumable (per split file,
# per-user lock dirs so the two pools never take the same user).
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/sm41
R=$WT/artifacts/small-model/sm41/realcheck
C3=$WT/artifacts/small-model/sm41/policy/confirm3/runner.log
cd "$WT"
mkdir -p "$R/logs" "$R/joblogs" "$R/locks"
echo "start $(date '+%F %T') pid $$" >> "$R/runner.log"
"$PY" -c "
import json
d = json.load(open('$WT/artifacts/small-model/sm41/fullrun/users.json'))
us = sorted(d['users'], key=lambda u: -sum(s['window_reviews'] for s in u['splits']))
for u in us: print(u['user'], ','.join(str(s['split_day']) for s in u['splits']))
" | tr -d '\r' > "$R/jobs.txt"
export PY R
JOB='mkdir "$R/locks/$0" 2>/dev/null || exit 0
  "$PY" -X utf8 scripts/sm41_realcheck.py run --user $0 --splits $1 --out-dir "$R/logs" >> "$R/joblogs/$0.log" 2>&1
  echo "$0 exit $? $(date +%T)" >> "$R/runner.log"'
xargs -P 2 -L 1 sh -c "$JOB" < "$R/jobs.txt" &
A=$!
( while ! grep -q done "$C3" 2>/dev/null; do sleep 60; done
  echo "confirm3 done, 2 more processes $(date +%T)" >> "$R/runner.log"
  xargs -P 2 -L 1 sh -c "$JOB" < "$R/jobs.txt" ) &
B=$!
wait $A $B
"$PY" -X utf8 -W ignore scripts/sm41_realcheck.py analyze --logs "$R/logs" --out-dir "$R" > "$R/analyze.out" 2>&1
echo "done $(date '+%F %T')" >> "$R/runner.log"
