#!/usr/bin/env bash
# Response-time prediction (SM82) stage 2: answer-time curves. The frozen released small model + the answer-time head on
#   richer frozen features (SM82c) queried "what if this card were reviewed after t days" on 150 sampled users of the 732
#   set (test reviews on a later day than the card's previous one): 15 elapsed-time grid points (0.25x..4x the actual
#   interval, 1..365 days), only the review's instant query row changes (scripts/sm82s2_curves.py).
# Steps: CPU check (identity = SM82c's recorded row, query-row independence, shift exactness) -> GPU ticket: curve
#   queries (resumable; pass 2 only if users are missing). The analysis (CPU) follows by hand: HANDOFF.md.
# Breaks: quiet.json (scope all/gpu) plus the daily 20:00-01:00 break and Sunday from 16:00, re-read at every check.
#   launch   WMI hidden (HANDOFF.md); log artifacts/small-model/sm82/stage2/gpu/launch-sm82s2.log
#   stop     touch C:/Users/jonat/workspace/wt/sm82s2/artifacts/small-model/sm82/stage2/gpu/STOP (checked between tickets)
# Do not edit scripts/ expertium/ rwkv/ of the sm82s2 worktree while it runs.
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
QUIET=C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json
WT=C:/Users/jonat/workspace/wt/sm82s2
S=$WT/artifacts/small-model/sm82/stage2
G=$S/gpu
OUT=$S/work/curves
mkdir -p "$G" "$OUT"
LOG=$G/launch-sm82s2.log
H="$PY -X utf8 $WT/scripts/sm82s2_curves.py"
MASTER=C:/Users/jonat/workspace/wt/sm82c/artifacts/small-model/sm82c-head-released/master.pth   # read only
USERS=$S/users.json
NAME="response-time prediction (SM82) stage 2"
DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS   # cap 0.8
log() { echo "[sm82s2 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
stopped() { [ -e "$G/STOP" ] && { log "STOP: STOP file present"; return 0; }; return 1; }
nap() { [ "$DRYRUN" = 1 ] && return 0; local s=$1; [ "$s" -gt 600 ] && s=600; sleep "$s"; }
block() {  # -> "NOW <seconds to its end>" inside a quiet window (quiet.json scope all/gpu, or the daily break), else "NEXT <minutes>"
  Q="$QUIET" "$PY" -X utf8 - <<'PYEOF'
import json, os, re
from datetime import datetime, timedelta
now = datetime.now()
try:
    text = open(os.environ["Q"], encoding="utf-8").read()
except OSError as error:
    print(f"ERR quiet.json unreadable: {error}"); raise SystemExit
try:
    wins = [(w["start"], w["end"]) for w in json.loads(text) if w.get("scope", "all") in ("all", "gpu")]
except (ValueError, KeyError, TypeError):  # malformed JSON: take every start/end pair textually, scope assumed "all"
    wins = re.findall(r'"start"\s*:\s*"([^"]+)"[^{}]*?"end"\s*:\s*"([^"]+)"', text, re.S)
blocks = [(datetime.fromisoformat(s), datetime.fromisoformat(e)) for s, e in wins]
day = now.replace(hour=0, minute=0, second=0, microsecond=0)
blocks += [(day + timedelta(days=d, hours=20), day + timedelta(days=d + 1, hours=1)) for d in range(-1, 9)]  # daily break
blocks += [(day + timedelta(days=d, hours=16), day + timedelta(days=d, hours=20))                          # Sunday from 16:00
           for d in range(-1, 9) if (day + timedelta(days=d)).weekday() == 6]
blocks.sort()
merged = []
for s, e in blocks:
    if merged and s <= merged[-1][1]:
        merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    else:
        merged.append((s, e))
for s, e in merged:
    if s <= now < e:
        print(f"NOW {int((e - now).total_seconds()) + 60}"); raise SystemExit
    if s > now:
        print(f"NEXT {int((s - now).total_seconds() // 60)}"); raise SystemExit
print("NEXT 100000")
PYEOF
}

wait_free() {  # wait_free MINUTES WHAT: returns once MINUTES fit before the next window (sets FREE = minutes to it)
  local n=0
  while true; do
    stopped && return 1
    local b; b=$(block)
    case "$b" in
      NOW*) local s=${b#NOW }
            if [ "$DRYRUN" = 1 ]; then log "DRYRUN: $2 would wait inside a quiet window (${s}s left)"; FREE=100000; return 0; fi
            [ $((n % 6)) = 0 ] && log "$2: quiet window, ${s}s left"; n=$((n + 1)); nap "$s";;
      NEXT*) FREE=${b#NEXT }
             [ "$FREE" -ge $(( $1 + 3 )) ] && return 0
             if [ "$DRYRUN" = 1 ]; then log "DRYRUN: $2 ($1 min) would wait: $FREE min to the next window"; FREE=100000; return 0; fi
             [ $((n % 6)) = 0 ] && log "$2 needs $1 min, $FREE min before the next window; waiting"; n=$((n + 1)); nap $(( FREE * 60 + 60 ));;
      *) log "window check failed: $b"; nap 600;;
    esac
  done
}

q() {  # q MINUTES PURPOSE -- CMD...   one ticket that must end before the next window
  local minutes=$1 purpose=$2 rc; shift 3
  while true; do
    wait_free "$minutes" "$purpose" || return 1
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $NAME: $purpose (expected $minutes min)"; echo "    cmd: $*" >> "$LOG"; return 0; fi
    local maxwait=$(( FREE - minutes - 3 )); [ "$maxwait" -lt 1 ] && maxwait=1
    log "enqueue: $purpose (expected $minutes min, max wait $maxwait min, $FREE min to the next window)"
    "$PY" "$QUEUE" run --owner small-model --purpose "$NAME: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm82/stage2/launch-sm82s2.sh" -- "$@"
    rc=$?; log "exit $rc: $purpose"
    [ "$rc" = 75 ] && { log "queue wait would reach the next window; re-planning"; sleep 60; continue; }
    return $rc
  done
}
cpu_step() { wait_free "$1" "$2"; }   # CPU work also stays out of the windows
queried() { F="$OUT/query.jsonl" "$PY" -X utf8 -c "
import json, os
from pathlib import Path
p = Path(os.environ['F'])
print(len({r['user_id'] for r in map(json.loads, open(p, encoding='utf-8')) if 'error' not in r}) if p.exists() else 0)"; }

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1) (DRYRUN=$DRYRUN)"
rc=0
[ -f "$MASTER" ] && [ -f "$USERS" ] || { log "STOP: missing $MASTER or $USERS"; exit 1; }
if [ "$DRYRUN" != 1 ] && [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then log "STOP: uncommitted source changes"; exit 1; fi
cpu_step 5 "CPU check" || exit 1
if [ ! -e "$G/check-cpu-ok" ]; then
  line=$(CUDA_VISIBLE_DEVICES= $H check --user 2388 --out "$G/check-cpu.json" 2>&1 | tail -1)
  log "$line"
  [[ "$line" == "[sm82s2-check] PASS"* ]] || { log "STOP: CPU check failed (see $G/check-cpu.json)"; exit 1; }
  [ "$DRYRUN" = 1 ] || touch "$G/check-cpu-ok"
fi
n=$(cat "$USERS" | "$PY" -c "import json, sys; print(len(json.load(sys.stdin)['users']))")
for pass in 1 2; do
  got=$(queried)
  [ "$got" -ge "$n" ] && break
  q $([ $pass = 1 ] && echo 50 || echo 15) "answer-time curve queries on the frozen released small model + the answer-time head on richer frozen features (SM82c): $n users of the 732 set, 15 elapsed-time grid points (pass $pass)" -- \
    $H query --master "$MASTER" --users-file "$USERS" --out-dir "$OUT" >> "$G/query.out" 2>&1 || { log "STOP: curve query ticket failed (pass $pass)"; rc=1; break; }
  [ "$DRYRUN" = 1 ] && break
done
log "queried users: $(queried) of $n"
log "done (rc $rc)"
exit $rc
