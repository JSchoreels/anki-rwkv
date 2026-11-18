#!/usr/bin/env bash
# Duration-leak scoring (SM31) and robustness to other apps (SM29): the released small model and the big model
# (1 leak-free epoch) scored as they are (no retraining) with changed inputs, grade heads only, on the 732 set,
# leak-free (deleted cards: neighbour), w8+s8 (scripts/small_model_heldout.py --inputs V, scripts/sm31_inputs.py).
#   1. GPU tickets, one scoring pass each (723 users at the 0.8 cap, a pass 2 only for users a time limit left out),
#      in this order: duration-leak scoring (SM31) show-time intervals (e2s-query) small, big; end-to-start on every
#      row (e2s-all) small, big; robustness to other apps (SM29) passfail small, big; notime small, big;
#      passfail-notime small, big;
#   2. CPU: scripts/sm31_table.py -> artifacts/small-model/sm31/sm31-result.{md,json} and
#      artifacts/small-model/sm29/sm29-result.{md,json} on the 722-user rule pool;
#   3. the 3 long users of the 732 set (cap 0.92), same order, then the table again (adds the 732+long pool).
# The normal-input rows are the existing leak-free read-out rows (read only), which today's scorer reproduces exactly.
# Breaks: quiet.json (scope all/gpu) plus the daily 20:00-01:00 break (Sunday from 16:00) as a safeguard, re-read at
#   every check; a ticket starts only when it fits before the next window.
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm31/artifacts/small-model/sm31/launch-sm31.sh
#   launch   WMI hidden (HANDOFF.md); log artifacts/small-model/sm31/gpu/launch-sm31.log
#   resume   the launch command again (finished users / tickets are skipped)
#   stop     touch C:/Users/jonat/workspace/wt/sm31/artifacts/small-model/sm31/gpu/STOP
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
QUIET=C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json
WT=C:/Users/jonat/workspace/wt/sm31
S=$WT/artifacts/small-model/sm31
G=$S/gpu
mkdir -p "$G"
LOG=$G/launch-sm31.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm31-dryrun.log
SC=$WT/scripts/small_model_heldout.py
S72=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s                                        # read only
USERS732=$S72/eval-users-732-fit.json; ULONG=$S72/users-732-long.json
SMALL=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/scale-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon/expertium/E89-F65e-F77a/master.pth
SLBL=s-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon
BIGCK=C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72-big-leakfree-1ep/expertium/big-leakfree-1ep/master.pth
BLBL=big-leakfree-1ep
OUTROWS=$G/sm31-inputs.jsonl
MODE=neighbour
MIN_SMALL=20; MIN_BIG=25; MIN_PASS2=10; MIN_LONG_SMALL=20; MIN_LONG_BIG=10; MIN_TABLE=5
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START   # scorer cap 0.8
log() { echo "[sm31 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
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
    "$PY" "$QUEUE" run --owner small-model --purpose "$purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm31/launch-sm31.sh" -- "$@"
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
print(len({r['user_id'] for r in map(json.loads, open(p, encoding='utf-8')) if r.get('label') == os.environ['LB'] and 'benchmark' in r}) if p.exists() else 0)"; }

preconditions() {
  for f in "$SMALL" "$BIGCK" "$USERS732" "$ULONG"; do
    [ -f "$f" ] || { log "STOP: missing $f"; return 1; }
  done
  if [ "$DRYRUN" != 1 ] && [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    log "STOP: uncommitted source changes in scripts/ expertium/ rwkv/ (provenance would be dirty)"; return 1; fi
  return 0
}

what() {  # what VARIANT -> ticket purpose prefix
  case "$1" in
    e2s-query) echo "duration-leak scoring (SM31): show-time intervals (e2s-query)";;
    e2s-all) echo "duration-leak scoring (SM31): end-to-start intervals on every row (e2s-all)";;
    passfail) echo "robustness to other apps (SM29): pass / fail ratings only (passfail)";;
    notime) echo "robustness to other apps (SM29): no answer times (notime)";;
    passfail-notime) echo "robustness to other apps (SM29): pass / fail ratings and no answer times (passfail-notime)";;
  esac
}
model_name() { if [ "$1" = "$SLBL" ]; then echo "the released small model"; else echo "the big model (1 leak-free epoch)"; fi; }

score_main() {  # score_main VARIANT LABEL MODEL_SPEC MIN: 723 users at the 0.8 cap, pass 2 for leftovers
  local v=$1 lb=$2 spec=$3 pass
  for pass in 1 2; do
    if [ "$DRYRUN" = 1 ] || [ "$(label_users "$OUTROWS" "$lb/w8+s8@$v")" -lt 723 ]; then
      q $([ $pass = 1 ] && echo "$4" || echo "$MIN_PASS2") "$(what "$v"): scoring $(model_name "$lb") on the 723 users of the 732 set (pass $pass)" -- \
        env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START "$PY" -X utf8 "$SC" --pool 732 --users-file "$USERS732" --model "$lb=$spec" \
        --only "$lb/w8+s8" --inputs "$v" --time-limit 1800 --split-existing --deleted-cards $MODE --out "$OUTROWS" >> "$G/score.out" 2>&1 \
        || { log "STOP: scoring ticket failed ($lb $v)"; return 1; }
      [ "$DRYRUN" = 1 ] && return 0
    fi
  done
  local got; got=$(label_users "$OUTROWS" "$lb/w8+s8@$v")
  [ "$got" -ge 723 ] || { log "STOP: $lb $v scored on $got of 723 users after 2 passes"; return 1; }
  log "$lb $v: $got users"
  return 0
}
score_long() {  # score_long VARIANT LABEL MODEL_SPEC MIN: the 3 long users at 0.92 (a failure is logged, not fatal)
  local v=$1 lb=$2 spec=$3
  if [ "$DRYRUN" = 1 ] || [ "$(label_users "$OUTROWS" "$lb/w8+s8@$v")" -lt 726 ]; then
    q "$4" "$(what "$v"): scoring $(model_name "$lb") on the 3 long users of the 732 set (cap 0.92)" -- \
      env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START HELDOUT_GPU_MEMORY_FRACTION=0.92 "$PY" -X utf8 "$SC" --pool 732 --users-file "$ULONG" \
      --model "$lb=$spec" --only "$lb/w8+s8" --inputs "$v" --time-limit 2400 --split-existing --deleted-cards $MODE --out "$OUTROWS" >> "$G/score.out" 2>&1 \
      || { stopped && return 1; log "long users ticket failed ($lb $v); continuing"; }
  fi
  return 0
}
table() {
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: then $PY -X utf8 $WT/scripts/sm31_table.py --rows $OUTROWS"; return 0; fi
  cpu_step $MIN_TABLE "table" || return 1
  "$PY" -X utf8 "$WT/scripts/sm31_table.py" --rows "$OUTROWS" > "$G/table.out" 2>&1; local r=$?
  log "table exit $r: $(tr '\n' ' ' < "$G/table.out" | cut -c1-600)"
  return 0
}

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1) (DRYRUN=$DRYRUN)"
rc=0
preconditions || rc=1
VARIANTS="e2s-query e2s-all passfail notime passfail-notime"
for v in $VARIANTS; do
  [ $rc -eq 0 ] && { score_main "$v" "$SLBL" "expertium:w64:$SMALL" $MIN_SMALL || rc=1; }
  [ $rc -eq 0 ] && { score_main "$v" "$BLBL" "fork::$BIGCK" $MIN_BIG || rc=1; }
done
[ $rc -eq 0 ] && { table || rc=1; }
for v in $VARIANTS; do
  [ $rc -eq 0 ] && { score_long "$v" "$SLBL" "expertium:w64:$SMALL" $MIN_LONG_SMALL || rc=1; }
  [ $rc -eq 0 ] && { score_long "$v" "$BLBL" "fork::$BIGCK" $MIN_LONG_BIG || rc=1; }
done
[ $rc -eq 0 ] && { table || rc=1; }
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; else log "done"; fi
else
  log "STOP: ended with exit $rc"
fi
exit $rc
