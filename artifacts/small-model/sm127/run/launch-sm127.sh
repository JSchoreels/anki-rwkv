#!/usr/bin/env bash
# Fluency vs desired retention in the simulator with answer-time feedback (SM127): the pre-registered PILOT (design.md
# sections 3, 11). CPU only, no GPU, no queue ticket. 12 users x 365 days x 22 lanes; 4 processes x 1 thread,
# below-normal priority, largest windows first. Every user process checks quiet.json (scope all / cpu) and the daily
# 20:00-01:00 break (Sunday from 16:00) before it starts (no start within 60 min of a window) and pauses at the next
# simulated day inside a window; it holds while less than 6 GB of RAM is available.
#   stop:   touch C:/Users/jonat/workspace/wt/sm127/artifacts/small-model/sm127/run/STOP (checked before every user and
#           every simulated day; a stopped user writes nothing and is redone on relaunch)
#   resume: launch again (users whose rows/<user>.json exists are skipped)
# Log: run/launch-sm127.log ("done" at the end, "STOP" lines when stopped); per user: logs/<user>.log; rows/<user>.json.
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/sm127
S=$WT/artifacts/small-model/sm127
R=$S/run
LOG=$R/launch-sm127.log
mkdir -p "$R" "$S/rows" "$S/logs"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1
export PY S
log() { echo "[sm127 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
cd "$WT" || exit 1
log "started pid $$: $(git -C "$WT" log --oneline -1)"
if [ -n "$(git -C "$WT" status --porcelain -- scripts)" ]; then log "STOP: uncommitted changes in scripts/"; exit 1; fi
[ -e "$R/STOP" ] && { log "STOP file present: not starting"; exit 1; }
"$PY" -c "
import json
d = json.load(open(r'$S/users.json'))
P = set(d['pilot'])
us = sorted([c for c in d['users'] if c['user'] in P], key=lambda c: -c['window_reviews'] * 1.0 - c['pre_reviews'] / 20)
for c in us: print(c['user'])
" | tr -d '\r' | xargs -P 4 -I{} sh -c '
  if [ -e "$S/run/STOP" ]; then echo "STOP before user {}" >> "$S/logs/{}.log"; exit 0; fi
  "$PY" -X utf8 scripts/sm127_fluency.py run --user {} --days 365 --out "$S/rows/{}.json" >> "$S/logs/{}.log" 2>&1
  echo "exit $? $(date "+%m-%d %H:%M:%S")" >> "$S/logs/{}.log"'
n=$(ls "$S"/rows/*.json 2>/dev/null | wc -l)
log "user processes ended: $n of 12 rows"
if [ -e "$R/STOP" ]; then log "STOP file present: no report"; exit 1; fi
if [ "$n" -lt 12 ]; then log "STOP: incomplete ($n of 12 rows); relaunch to finish"; exit 1; fi
"$PY" -X utf8 scripts/sm127_fluency.py report --set pilot > "$S/report-pilot.out" 2>&1
log "report exit $?"
log "done"
