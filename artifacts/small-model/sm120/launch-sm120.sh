#!/usr/bin/env bash
# Fast Good/Easy prediction (SM120): scoring only, no training (design: artifacts/small-model/sm120/design.md). The answer-time
#   heads of the time heads for the family (SM117) (six models) and of the time heads for the new candidates (SM119) (the
#   longer Small Instant-Only model (SM113), the longer cool-down for Tiny (SM115), and the show-time two-head cool-down for
#   Small (SM118) if its head exists) scored on the 732 set (723 users) for P(Good or Easy), P(Easy), P(set and answered
#   within S s), the speed given the grade and the grade-weighted time; then the CPU report sm120-result.{md,json}.
#   Answer time = front + back, question shown to grade press, capped at the max answer seconds ("within S s" includes
#   reading the back).
# Nothing is enqueued until the launcher of SM119 has ended (a "done" or "STOP" line in its log; the queue is FIFO).
# Breaks: quiet.json (scope all/gpu) plus the daily 20:00-01:00 break and Sunday from 16:00, re-read at every check
#   (<= 10 min); every ticket starts only when its budget fits before the next window. GPU cap 0.8, missing users at 0.92.
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm120/artifacts/small-model/sm120/launch-sm120.sh
#   launch   WMI hidden (artifacts/small-model/sm120/HANDOFF.md); log artifacts/small-model/sm120/gpu/launch-sm120.log
#   resume   the launch command again (scored users are skipped)
#   stop     touch C:/Users/jonat/workspace/wt/sm120/artifacts/small-model/sm120/gpu/STOP (checked between tickets)
# Read only: wt/sm117 (heads), wt/sm119 (heads, log), wt/sm118 (master). Do not edit scripts/ of wt/sm120 while it runs.
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
QUIET=C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json
WT=C:/Users/jonat/workspace/wt/sm120
S=$WT/artifacts/small-model/sm120
G=$S/gpu
mkdir -p "$G"
LOG=$G/launch-sm120.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm120-dryrun.log
H="$PY -X utf8 $WT/scripts/sm117_head.py"
F="$PY -X utf8 $WT/scripts/sm120_fast.py"
SM117HEADS=C:/Users/jonat/workspace/wt/sm117/artifacts/small-model
SM119HEADS=C:/Users/jonat/workspace/wt/sm119/artifacts/small-model
SM119LOG=$SM119HEADS/sm119/gpu/launch-sm119.log   # read only
MODELS="sm106 sm98 sm101 sm99 sm91s2 sm92"        # + the SM119 heads that exist once SM119 has ended
NAME="Fast Good/Easy prediction (SM120)"
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START   # cap 0.8
log() { echo "[sm120 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
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
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm120/launch-sm120.sh" -- "$@"
    rc=$?; log "exit $rc: $purpose"
    [ "$rc" = 75 ] && { log "queue wait would reach the next window; re-planning"; sleep 60; continue; }
    return $rc
  done
}
cpu_step() { wait_free "$1" "$2"; }   # CPU work also stays out of the windows

train_done() { [ -f "$1/head.pth" ] && [ -f "$1/master.pth" ] && R="$1/train.json" "$PY" -X utf8 -c "
import json, os, sys
sys.exit(0 if json.load(open(os.environ['R'])).get('status') == 'complete' else 1)" 2>/dev/null; }
head_dir() { case "$1" in sm113|sm115|sm118) echo "$SM119HEADS/sm119-$1-head";; *) echo "$SM117HEADS/sm117-$1-head";; esac; }
scored_users() { R="$1" "$PY" -X utf8 -c "
import json, os
from pathlib import Path
p = Path(os.environ['R'])
print(len({r['user_id'] for r in map(json.loads, open(p, encoding='utf-8')) if 'error' not in r}) if p.exists() else 0)"; }
budget() {
  case "$1:$2" in
    fork:score) echo 45;;
    *:score) echo 30;;
    *:retry) echo 15;;
  esac
}

after_sm119() {  # nothing is enqueued until the time heads for the new candidates (SM119)'s launcher has ended (its log's last
  # line is "done ..." or a "STOP" line); then the SM119 heads whose training completed are added to MODELS
  local n=0 last key hd
  while true; do
    stopped && return 1
    last=$(tail -1 "$SM119LOG" 2>/dev/null)
    if [[ "$last" =~ ^\[sm119\ [^]]*\]\ (done|STOP) ]]; then log "the time heads for the new candidates (SM119) has ended: ${last:0:220}"; break; fi
    [ "${SKIP_SM119_WAIT:-0}" = 1 ] && { log "SKIP_SM119_WAIT=1: not waiting for the time heads for the new candidates (SM119)"; break; }
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN: would wait for the time heads for the new candidates (SM119) to end (last line: ${last:0:140})"; break; fi
    [ $((n % 12)) = 0 ] && log "waiting for the time heads for the new candidates (SM119) to end (no done / STOP line in its launcher log yet; last: ${last:0:140})"
    n=$((n + 1)); sleep 300
  done
  for key in sm113 sm115 sm118; do
    hd=$SM119HEADS/sm119-$key-head
    if train_done "$hd"; then
      if [ "$key" = sm118 ]; then
        local m118=C:/Users/jonat/workspace/wt/sm118/artifacts/small-model/sm118-e26-dec2ep32k-t91s2-p2/expertium/E89-F65e-F77a/master.pth
        export SM119_SM118_SHA256
        SM119_SM118_SHA256=$(R="$hd/train.json" "$PY" -X utf8 -c "import json, os; print(json.load(open(os.environ['R']))['released_master_sha256'])")
        if [ "$(sha256sum "$m118" | cut -d' ' -f1)" != "$SM119_SM118_SHA256" ]; then
          log "sm118: its master's sha256 differs from the head's train.json; not scored"; continue
        fi
      fi
      MODELS="$MODELS $key"
      [ -e "$SM119HEADS/sm119/gpu/grade-ok-sm119-$key" ] || log "WARNING: $key has no SM119 grade-ok marker (the scorer still checks the trunk equals the release)"
    else
      log "$key: no completed SM119 head at $hd; not scored"
    fi
  done
  log "models: $MODELS"
  return 0
}

preconditions() {
  local f key
  for f in "$S/priors.json" "$S/check-cpu.json" "$WT/artifacts/small-model/sm117/time-baselines.jsonl"; do
    [ -f "$f" ] || { log "STOP: missing $f"; return 1; }
  done
  R="$S/check-cpu.json" "$PY" -X utf8 -c "import json, os, sys; sys.exit(0 if json.load(open(os.environ['R']))['pass'] else 1)" \
    || { log "STOP: the CPU check did not pass ($S/check-cpu.json)"; return 1; }
  for key in $MODELS; do
    train_done "$(head_dir "$key")" || { log "STOP: $key head missing or incomplete at $(head_dir "$key")"; return 1; }
    $H info --model "$key" > /dev/null 2>> "$LOG" || { log "STOP: $key master missing or its sha256 differs"; return 1; }
  done
  if [ "$DRYRUN" != 1 ] && [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    log "STOP: uncommitted source changes in scripts/ expertium/ rwkv/"; return 1; fi
  return 0
}

score_model() {  # score_model KEY: pass 1 at cap 0.8, pass 2 (missing users only) at 0.92
  local key=$1 HD OUT got
  eval "$($H info --model "$key")"
  HD=$(head_dir "$key")
  OUT=$G/sm120-$key-rows.jsonl
  if [ "$DRYRUN" = 1 ] || [ "$(scored_users "$OUT")" -lt 723 ]; then
    q "$(budget "$KIND" score)" "scoring $MNAME: P(Good or Easy), P(Easy), P(set and answered within S s) on the 732 set (723 users, pass 1, cap 0.8)" -- \
      env -u RWKV_DELETED_CARDS $F score --model "$key" --head-dir "$HD" --out "$OUT" >> "$G/score-$key.out" 2>&1 \
      || { log "STOP: $key scoring ticket failed (see $G/score-$key.out)"; return 1; }
  fi
  if [ "$DRYRUN" = 1 ] || [ "$(scored_users "$OUT")" -lt 723 ]; then
    q "$(budget "$KIND" retry)" "scoring $MNAME: P(Good or Easy), P(Easy), P(set and answered within S s) on the 732 set (missing users, pass 2, cap 0.92)" -- \
      env -u RWKV_DELETED_CARDS HELDOUT_GPU_MEMORY_FRACTION=0.92 $F score --model "$key" --head-dir "$HD" --out "$OUT" >> "$G/score-$key.out" 2>&1 \
      || { log "STOP: $key scoring pass 2 failed (see $G/score-$key.out)"; return 1; }
  fi
  if [ "$DRYRUN" != 1 ]; then
    got=$(scored_users "$OUT")
    log "$key scored users: $got of 723"
    [ "$got" -ge 700 ] || { log "STOP: $key scoring has only $got of 723 users after 2 passes (see $G/score-$key.out)"; return 1; }
  fi
  return 0
}

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1) (DRYRUN=$DRYRUN)"
rc=0
cpu_step 5 "preparation" || rc=1
[ $rc -eq 0 ] && { after_sm119 || rc=1; }
[ $rc -eq 0 ] && { preconditions || rc=1; }
for key in $MODELS; do
  [ $rc -eq 0 ] && { score_model "$key" || rc=1; }
done
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then
    log "DRYRUN: then $F report"
  else
    cpu_step 15 "report" || rc=1
    out=$($F report 2>&1); r=$?
    log "report exit $r: $(echo "$out" | tail -1)"; [ "$r" = 0 ] || rc=1
  fi
fi
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; else log "done"; fi
else
  log "STOP: ended with exit $rc"
fi
exit $rc
