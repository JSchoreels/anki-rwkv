#!/usr/bin/env bash
# Fluency vs desired retention in the simulator with answer-time feedback (SM127): FULL RUN v3 (design.md section 13).
# 150 users x 365 days x 22 lanes, answer-time level anchor (--fix) and per-user fluency threshold (--per-user-s).
# CPU only, no GPU, no queue ticket. At most 3 processes x 1 thread, below-normal priority, each under 6 GB (a user holds
# while < 6 GB of RAM is free). Largest users first; a user whose rows-v3/<user>.json exists is skipped. No daily break
# (SM127_DAILY_BREAK unset); quiet.json windows (scope all / cpu) and the STOP file hold / stop it.
#   stop:   touch C:/Users/jonat/workspace/wt/sm127/artifacts/small-model/sm127/run-v3/STOP
#           (checked before every user and every simulated day; a stopped user writes nothing and is redone on relaunch)
#   resume: launch again (finished users are skipped)
# Log: run-v3/launch-v3.log ("done" at the end, "STOP:" lines); per user: logs-v3/<user>.log; rows-v3/<user>.json.
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/sm127
S=$WT/artifacts/small-model/sm127
R=$S/run-v3
LOG=$R/launch-v3.log
mkdir -p "$R" "$S/rows-v3" "$S/logs-v3"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1
unset SM127_DAILY_BREAK
export SM127_STOP=$R/STOP
export PY S
log() { echo "[sm127v3 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
cd "$WT" || exit 1
log "started pid $$: $(git -C "$WT" log --oneline -1)"
if [ -n "$(git -C "$WT" status --porcelain -- scripts)" ]; then log "STOP: uncommitted changes in scripts/"; exit 1; fi
[ -e "$R/STOP" ] && { log "STOP file present: not starting"; exit 1; }
"$PY" -c "
import json
d = json.load(open(r'$S/users.json'))
us = sorted(d['users'], key=lambda c: -c['window_reviews'] * 1.0 - c['pre_reviews'] / 20)
assert len(us) == 150, len(us)
for c in us: print(c['user'])
" | tr -d '\r' | xargs -P 3 -I{} sh -c '
  if [ -e "$S/run-v3/STOP" ]; then echo "STOP before user {}" >> "$S/logs-v3/{}.log"; exit 0; fi
  "$PY" -X utf8 scripts/sm127_fluency.py run --user {} --days 365 --fix --per-user-s --out "$S/rows-v3/{}.json" >> "$S/logs-v3/{}.log" 2>&1
  echo "exit $? $(date "+%m-%d %H:%M:%S")" >> "$S/logs-v3/{}.log"'
n=$(ls "$S"/rows-v3/*.json 2>/dev/null | wc -l)
log "user processes ended: $n of 150 rows"
if [ -e "$R/STOP" ]; then log "STOP file present: no report"; exit 1; fi
if grep -l Traceback "$S"/logs-v3/*.log > /dev/null 2>&1; then log "STOP: Traceback in $(grep -l Traceback "$S"/logs-v3/*.log | tr '\n' ' ')"; fi
if [ "$n" -lt 150 ]; then log "STOP: incomplete ($n of 150 rows); read the logs, relaunch to finish"; exit 1; fi
# stop rule (3): budget_violations > 0 on a policy lane, or nofb_rows = 0
bad=$("$PY" -c "
import json, glob
bad = []
for f in sorted(glob.glob(r'$S/rows-v3/*.json')):
    r = json.load(open(f))
    pol = [k for k, nm in enumerate(r['lanes']) if nm != 'tf' and not nm.startswith('fr-')]
    if any(r['budget_violations'][k] > 0 for k in pol) or r['nofb_rows'] <= 0 or not r.get('time_anchor'):
        bad.append(r['user'])
print(' '.join(map(str, bad)))
" | tr -d '\r')
if [ -n "$bad" ]; then log "STOP: sanity check failed for users: $bad (budget violations / nofb rows / no anchor); report not run"; exit 1; fi
"$PY" -X utf8 scripts/sm127_fluency.py report --set full --rows "$S/rows-v3" --md "$S/sm127-tables-v3.md" --json "$S/sm127-result-v3.json" > "$S/report-v3.out" 2>&1
log "report exit $?"
log "done"
