#!/usr/bin/env bash
# Curve consistency on real rapid Good pairs (ST01a), stage 1 (measure): for the released small model (the leak-free
# small model (SM72s), epoch-25 32k Muon cool-down, label s-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon) and the
# big model (1 leak-free epoch) (label big-leakfree-1ep), on the 723 fit users of the 732 set (the 3 long users are
# left out), scripts/st01a_pairs.py stores the curves of consecutive reviews a -> b (b Good, gap 0-7 days) at the same
# future dates and the next review's outcome. One short GPU ticket per model (~15-20 min), w8+s8, deleted cards
# neighbour, as the scorer. Resumable (finished users are skipped). Respects quiet.json and the daily break.
#   launch   WMI hidden (HANDOFF.md); log artifacts/small-model/st01a/gpu/launch-st01a-s1.log
#   stop     touch C:/Users/jonat/workspace/wt/st01a/artifacts/small-model/st01a/gpu/STOP
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
QUIET=C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json
WT=C:/Users/jonat/workspace/wt/st01a
S=$WT/artifacts/small-model/st01a
G=$S/gpu
LOG=$G/launch-st01a-s1.log
PAIRS="$PY -X utf8 $WT/scripts/st01a_pairs.py"
U732=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json
SMALL=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/scale-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon/expertium/E89-F65e-F77a/master.pth
BIG=C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72-big-leakfree-1ep/expertium/big-leakfree-1ep/master.pth
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS
log() { echo "[st01a $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
stopped() { [ -e "$G/STOP" ] && { log "STOP: STOP file present"; return 0; }; return 1; }

minutes_free() {  # minutes until the next quiet window (scope all/gpu) or daily 20:00 break; 0 inside one
  Q="$QUIET" "$PY" -X utf8 - <<'PYEOF'
import json, os
from datetime import datetime, timedelta
now = datetime.now()
try:
    wins = [(w["start"], w["end"]) for w in json.load(open(os.environ["Q"], encoding="utf-8")) if w.get("scope", "all") in ("all", "gpu")]
except Exception:
    print(0); raise SystemExit
blocks = [(datetime.fromisoformat(s), datetime.fromisoformat(e)) for s, e in wins]
day = now.replace(hour=0, minute=0, second=0, microsecond=0)
blocks += [(day + timedelta(days=d, hours=20), day + timedelta(days=d + 1, hours=1)) for d in range(-1, 3)]
for s, e in sorted(blocks):
    if s <= now < e:
        print(0); raise SystemExit
nxt = min([s for s, e in blocks if s > now], default=now + timedelta(days=9))
print(int((nxt - now).total_seconds() // 60))
PYEOF
}

ticket() {  # ticket MINUTES PURPOSE -- CMD...
  local minutes=$1 purpose=$2; shift 3
  while true; do
    stopped && return 1
    local free; free=$(minutes_free)
    if [ "$free" -ge $(( minutes + 5 )) ]; then break; fi
    log "$purpose needs $minutes min, $free min free; waiting 10 min"; sleep 600
  done
  local maxwait=$(( free - minutes - 3 ))
  log "enqueue: $purpose (expected $minutes min, max wait $maxwait min)"
  "$PY" "$QUEUE" run --owner small-model --purpose "$purpose" --claim gpu --expected-minutes "$minutes" \
    --max-wait-minutes "$maxwait" --launched-by "detached launcher st01a/launch-st01a-s1.sh" -- "$@"
  local rc=$?; log "exit $rc: $purpose"; return $rc
}

log "start"
rows() { local n=0; [ -f "$1" ] && n=$(grep -vc '"error"' "$1"); echo "${n:-0}"; }
run_model() {  # run_model TAG SPEC WHAT: up to 3 tickets until the 723 users have rows
  local pass
  for pass in 1 2 3; do
    [ "$(rows "$G/pairs-$1.jsonl")" -ge 723 ] && return 0
    ticket 25 "curve consistency on rapid Good pairs (ST01a): stage 1 measure, curves of $3 on 723 users of the 732 set (pass $pass)" --       $PAIRS --model "$1=$2" --users-file "$U732" --out "$G/pairs-$1.jsonl" --pairs "$S/work/pairs-$1" --time-limit 1350 >> "$G/pairs-$1.out" 2>&1       || { log "STOP: $1 ticket failed (pass $pass)"; return 1; }
  done
}
run_model small "expertium:w64:$SMALL" "the released small model (the leak-free small model (SM72s), epoch-25 cool-down)" || exit 1
run_model big "fork::$BIG" "the big model (1 leak-free epoch)" || exit 1
for m in small big; do log "$m: $(wc -l < "$G/pairs-$m.jsonl") user rows"; done
log "done"
