#!/usr/bin/env bash
# Time-drift diagnostic of fluency vs desired retention (SM127), design.md 12.2. CPU only: 12 pilot users x 150 days x
# 8 lanes, 3 processes x 1 thread, below-normal priority, largest windows first. Holds in quiet windows (quiet.json
# all/cpu, daily 20:00-01:00); stop: touch timedrift/STOP; resume: launch again (rows/<user>.json exists = done).
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/sm127
S=$WT/artifacts/small-model/sm127
T=$S/timedrift
LOG=$T/run/launch-td.log
mkdir -p "$T/run" "$T/rows" "$T/logs"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1
export PY S T
log() { echo "[sm127-td $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
cd "$WT" || exit 1
log "started pid $$: $(git -C "$WT" log --oneline -1)"
if [ -n "$(git -C "$WT" status --porcelain -- scripts)" ]; then log "STOP: uncommitted changes in scripts/"; exit 1; fi
[ -e "$T/STOP" ] && { log "STOP file present: not starting"; exit 1; }
"$PY" -c "
import json
d = json.load(open(r'$S/users.json'))
P = set(d['pilot'])
us = sorted([c for c in d['users'] if c['user'] in P], key=lambda c: -c['window_reviews'] * 1.0 - c['pre_reviews'] / 20)
for c in us: print(c['user'])
" | tr -d '\r' | xargs -P 3 -I{} sh -c '
  if [ -e "$T/STOP" ]; then echo "STOP before user {}" >> "$T/logs/{}.log"; exit 0; fi
  "$PY" -X utf8 scripts/sm127_timedrift.py run --user {} >> "$T/logs/{}.log" 2>&1
  echo "exit $? $(date "+%m-%d %H:%M:%S")" >> "$T/logs/{}.log"'
n=$(ls "$T"/rows/*.json 2>/dev/null | wc -l)
log "user processes ended: $n of 12 rows"
if [ -e "$T/STOP" ]; then log "STOP file present: no report"; exit 1; fi
if [ "$n" -lt 12 ]; then log "STOP: incomplete ($n of 12 rows); relaunch to finish"; exit 1; fi
"$PY" -X utf8 scripts/sm127_timedrift.py report > "$T/report-td.out" 2>&1
log "report exit $?"
log "done"
