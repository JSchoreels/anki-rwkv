#!/usr/bin/env bash
# Scheduling-time vs review-time accuracy (SM81): FSRS-7 / FSRS7-Instant / small model / big model at scheduling time,
# at the start of the day and at review time, on the same reviews of the 732 set (726 users), leak-free inputs, w8+s8.
#   1. CPU (after the break, alongside the data-order re-seed's (SM80) GPU work): FSRS-7 per-user fits (fsrs-rs
#      fit-v7, the published protocol), FSRS-7 default, FSRS7-Instant (shared-5k protocol) per-review predictions
#      (scripts/sm81_fsrs.py, srs-benchmark venv, 6 processes, below-normal priority), ~30-45 min;
#   2. wait for the data-order re-seed (SM80) launcher to end ("] done" / STOP);
#   3. GPU tickets: scorer with --start-of-day --dump (normal pass + start-of-day pass per user), the released small
#      model (the leak-free small model (SM72s), cool-down from epoch 25) first, then the big model (1 leak-free
#      epoch); each on 723 users at the 0.8 cap (2 passes) + the 3 long users at 0.92;
#   4. CPU: scripts/sm81_table.py -> artifacts/small-model/sm81/sm81-result.md / .json.
# Breaks: quiet.json (scope all/gpu) plus the daily 20:00-01:00 break (Sunday from 16:00 is in quiet.json) as a
#   safeguard, re-read at every check; a ticket / CPU step starts only when it fits before the next window.
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm81/artifacts/small-model/sm81/launch-sm81.sh
#   launch   WMI hidden (HANDOFF.md); log artifacts/small-model/sm81/gpu/launch-sm81.log
#   resume   the launch command again (finished users / steps are skipped)
#   stop     touch C:/Users/jonat/workspace/wt/sm81/artifacts/small-model/sm81/gpu/STOP
#   no wait  touch .../sm81/gpu/NO-WAIT (skip the wait for SM80's launcher)
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
SRSPY=C:/Users/jonat/workspace/srs-benchmark/.venv/Scripts/python.exe                                  # read only
FSRSRS=C:/Users/jonat/workspace/fsrs-rs/tools/instant-replay/target/release                              # read only
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
QUIET=C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json
WT=C:/Users/jonat/workspace/wt/sm81
S=$WT/artifacts/small-model/sm81
G=$S/gpu
mkdir -p "$G"
LOG=$G/launch-sm81.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm81-dryrun.log
SC=$WT/scripts/small_model_heldout.py
SM80LOG=C:/Users/jonat/workspace/wt/sm80/artifacts/small-model/sm80/gpu/launch-sm80.log                  # read only
S72=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s                                        # read only
USERS732=$S72/eval-users-732-fit.json; ULONG=$S72/users-732-long.json; USERS726=$S/users-726.json
SMALL=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/scale-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon/expertium/E89-F65e-F77a/master.pth
SLBL=s-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon
SOLD=$S72/gpu/small-lf-neighbour.jsonl
BIGCK=C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72-big-leakfree-1ep/expertium/big-leakfree-1ep/master.pth
BLBL=big-leakfree-1ep
BOLD=C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72/gpu-big/bigleak-neighbour.jsonl
OUTROWS=$G/sm81-neighbour.jsonl
DUMP=$G/dump
FSRS=$G/fsrs
MODE=neighbour
MIN_FSRS=45; MIN_SMALL=30; MIN_BIG=35; MIN_PASS2=10; MIN_LONG=10; MIN_TABLE=5
NAME="scheduling-time vs review-time accuracy (SM81)"
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS   # scorer cap 0.8
log() { echo "[sm81 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
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
for d in range(-1, 9):   # the daily 20:00-01:00 break; Sunday from 16:00
    start = day + timedelta(days=d)
    blocks.append((start + timedelta(hours=16 if start.weekday() == 6 else 20), start + timedelta(days=1, hours=1)))
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
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $purpose (expected $minutes min)"; echo "    cmd: $*" >> "$LOG"; return 0; fi
    local maxwait=$(( FREE - minutes - 3 )); [ "$maxwait" -lt 1 ] && maxwait=1
    log "enqueue: $purpose (expected $minutes min, max wait $maxwait min, $FREE min to the next window)"
    "$PY" "$QUEUE" run --owner small-model --purpose "small-model: $NAME: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm81/launch-sm81.sh" -- "$@"
    rc=$?; log "exit $rc: $purpose"
    [ "$rc" = 75 ] && { log "queue wait would reach the next window; re-planning"; sleep 60; continue; }
    return $rc
  done
}
cpu_step() { wait_free "$1" "$2"; }   # CPU work also stays out of the windows
label_users() { F="$1" LB="$2" "$PY" -X utf8 -c "
import json, os
from pathlib import Path
p = Path(os.environ['F'])
print(len({r['user_id'] for r in map(json.loads, open(p, encoding='utf-8')) if r.get('label') == os.environ['LB'] and 'start_of_day' in r}) if p.exists() else 0)"; }
fsrs_users() { [ -f "$FSRS/users.jsonl" ] && grep -c . "$FSRS/users.jsonl" || echo 0; }

# ---- wait for the data-order re-seed (SM80) launcher: its log's last line is "done" or a STOP / failure line
sm80_last() { tail -1 "$SM80LOG" 2>/dev/null; }
sm80_ended() { local l; l=$(sm80_last)
  [[ "$l" == *"] done" ]] || [[ "$l" == *"] STOP"* ]] || [[ "$l" == *"STOP file present"* ]] || [[ "$l" =~ (failed|stopped|rule\ exit\ [1-9]) ]]; }
wait_sm80() {
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: would wait for the data-order re-seed (SM80) launcher; its last line now: $(sm80_last | cut -c1-200)"; return 0; fi
  local n=0
  until sm80_ended || [ -e "$G/NO-WAIT" ]; do
    stopped && return 1
    [ $(( n % 30 )) = 0 ] && log "waiting for the data-order re-seed (SM80) launcher to end; its last line: $(sm80_last | cut -c1-200)"
    n=$(( n + 1 )); sleep 120
  done
  log "SM80 launcher ended (or NO-WAIT): $(sm80_last | cut -c1-200)"
}

preconditions() {
  for f in "$SMALL" "$BIGCK" "$USERS732" "$ULONG" "$USERS726" "$SOLD" "$BOLD" "$SRSPY" "$FSRSRS/fit-v7.exe" "$FSRSRS/instant-replay.exe" \
           C:/Users/jonat/workspace/srs-benchmark/runs/shared-5k/shared.json C:/Users/jonat/workspace/srs-benchmark/runs/shared.json; do
    [ -f "$f" ] || { log "STOP: missing $f"; return 1; }
  done
  if [ "$DRYRUN" != 1 ] && [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    log "STOP: uncommitted source changes in scripts/ expertium/ rwkv/ (provenance would be dirty)"; return 1; fi
  return 0
}

fsrs_step() {  # FSRS-7 fits + FSRS7-Instant replay on CPU (2 attempts; resumable per user)
  local attempt
  for attempt in 1 2; do
    [ "$(fsrs_users)" -ge 726 ] && break
    cpu_step $MIN_FSRS "FSRS-7 fits and FSRS7-Instant replay (CPU, 6 processes)" || return 1
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN: FSRS step: SM81_OUT=$FSRS SM81_USERS=$USERS726 SM81_PROCESSES=6 $SRSPY -X utf8 $WT/scripts/sm81_fsrs.py (cwd $G/fsrs-cwd)"; return 0; fi
    log "FSRS step attempt $attempt: $(fsrs_users) of 726 users done"
    mkdir -p "$G/fsrs-cwd"
    (cd "$G/fsrs-cwd" && SM81_OUT="$FSRS" SM81_USERS="$USERS726" SM81_PROCESSES=6 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       "$SRSPY" -X utf8 "$WT/scripts/sm81_fsrs.py" >> "$G/fsrs.out" 2>&1)
    log "FSRS step attempt $attempt exit $?: $(fsrs_users) of 726 users"
  done
  [ "$DRYRUN" = 1 ] || [ "$(fsrs_users)" -ge 726 ] || log "FSRS step: only $(fsrs_users) of 726 users (see $G/fsrs/errors.jsonl); the table uses the users both sides have"
  return 0
}

score() {  # score USERS_FILE LABEL MODEL_SPEC MIN1 MIN2 WANT WHAT: leak-free scoring with the start-of-day pass, 2 passes
  local pass
  for pass in 1 2; do
    if [ "$DRYRUN" = 1 ] || [ "$(label_users "$OUTROWS" "$2/w8+s8")" -lt "$6" ]; then
      q $([ $pass = 1 ] && echo "$4" || echo "$5") "scoring $2 with the start-of-day pass on $7 (pass $pass)" -- \
        env -u RWKV_DELETED_CARDS "$PY" -X utf8 "$SC" --pool 732 --users-file "$1" --model "$2=$3" --only "$2/w8+s8" \
        --time-limit 2400 --split-existing --deleted-cards $MODE --start-of-day --dump "$DUMP" --out "$OUTROWS" >> "$G/score.out" 2>&1 \
        || { log "STOP: scoring ticket failed ($2)"; return 1; }
      [ "$DRYRUN" = 1 ] && return 0
    fi
  done
  local got; got=$(label_users "$OUTROWS" "$2/w8+s8")
  [ "$got" -ge "$6" ] || { log "STOP: $2 scored on $got of $6 users after 2 passes"; return 1; }
  return 0
}
score726() {  # score726 LABEL MODEL_SPEC MIN: 723 users at the 0.8 cap, then the 3 long users at 0.92
  score "$USERS732" "$1" "$2" "$3" $MIN_PASS2 723 "the 732 pool minus 9 never-fit (723)" || return 1
  if [ "$DRYRUN" = 1 ] || [ "$(label_users "$OUTROWS" "$1/w8+s8")" -lt 726 ]; then
    q $MIN_LONG "scoring $1 with the start-of-day pass on the 3 long users of the 732 pool (cap 0.92)" -- \
      env -u RWKV_DELETED_CARDS HELDOUT_GPU_MEMORY_FRACTION=0.92 "$PY" -X utf8 "$SC" --pool 732 --users-file "$ULONG" --model "$1=$2" \
      --only "$1/w8+s8" --time-limit 2400 --split-existing --deleted-cards $MODE --start-of-day --dump "$DUMP" --out "$OUTROWS" >> "$G/score.out" 2>&1 \
      || log "long users ticket failed ($1); continuing with $(label_users "$OUTROWS" "$1/w8+s8") users"
  fi
  [ "$DRYRUN" = 1 ] || log "$1: $(label_users "$OUTROWS" "$1/w8+s8") users scored with the start-of-day pass"
  return 0
}

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1) (DRYRUN=$DRYRUN)"
rc=0
preconditions || rc=1
[ $rc -eq 0 ] && { fsrs_step || rc=1; }
[ $rc -eq 0 ] && { wait_sm80 || rc=1; }
[ $rc -eq 0 ] && { score726 "$SLBL" "expertium:w64:$SMALL" $MIN_SMALL || rc=1; }
[ $rc -eq 0 ] && { score726 "$BLBL" "fork::$BIGCK" $MIN_BIG || rc=1; }
TABLE=("$PY" -X utf8 "$WT/scripts/sm81_table.py" --rows "$OUTROWS" --dump "$DUMP" --fsrs "$FSRS" --label-small "$SLBL/w8+s8"
       --label-big "$BLBL/w8+s8" --existing-small "$SOLD" --existing-big "$BOLD" --out-md "$S/sm81-result.md" --out-json "$S/sm81-result.json")
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then
    log "DRYRUN: then ${TABLE[*]}"
  else
    cpu_step $MIN_TABLE "table" || rc=1
    if [ $rc -eq 0 ]; then
      "${TABLE[@]}" > "$G/table.out" 2>&1; r=$?
      log "table exit $r: $(tail -1 "$G/table.out" | cut -c1-300)"; [ "$r" = 0 ] || rc=1
    fi
  fi
fi
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; else log "done"; fi
else
  log "STOP: ended with exit $rc"
fi
exit $rc
