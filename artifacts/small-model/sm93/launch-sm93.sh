#!/usr/bin/env bash
# The instant model for RWKV-Instant (SM93): decisive check of the instant-only small model (SM57) vs the release (SM63)
# on the 1,500 validation users (sm93_rule.py). Scores both models (normal inputs, leak-free, w8+s8) on the 768 users of
# the 1,500 pool outside the 732 set (minus 3 above 650k reviews) + 6 check users of the 732 set, then runs the rule.
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm93/artifacts/small-model/sm93/launch-sm93.sh
#   launch   WMI hidden (HANDOFF.md); log gpu/launch-sm93.log; resumable (the scorer skips users already in the rows)
#   stop     touch C:/Users/jonat/workspace/wt/sm93/artifacts/small-model/sm93/gpu/STOP (between tickets)
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
QUIET=C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json
WT=C:/Users/jonat/workspace/wt/sm93
S=$WT/artifacts/small-model/sm93
G=$S/gpu
mkdir -p "$G"
LOG=$G/launch-sm93.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm93-dryrun.log
SC=$WT/scripts/small_model_heldout.py
RULE=$S/sm93_rule.py
TRACK=E89-F65e-F77a
CAND=C:/Users/jonat/workspace/wt/sm57/artifacts/small-model/sm57-inst-e26-dec-1ep32k-muon/expertium/$TRACK/master.pth   # read only
REL=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/scale-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon/expertium/$TRACK/master.pth  # read only
CT=sm57-inst-dec1ep32k
RT=s-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon
USERS=$S/users-sm93-score.json
OUTROWS=$G/sm93-rows.jsonl
OUTREL=artifacts/small-model/sm93/gpu/sm93-rows.jsonl
NAME="the instant model for RWKV-Instant (SM93)"
MIN_PASS=45; TL=2400
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START
log() { echo "[sm93 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
stopped() { [ -e "$G/STOP" ] && { log "STOP: STOP file present"; return 0; }; return 1; }
nap() { [ "$DRYRUN" = 1 ] && return 0; local s=$1; [ "$s" -gt 600 ] && s=600; sleep "$s"; }

block() {  # "NOW <s>" inside a quiet window (quiet.json scope all/gpu, or the daily 20:00-01:00 break), else "NEXT <min>"
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
except (ValueError, KeyError, TypeError):
    wins = re.findall(r'"start"\s*:\s*"([^"]+)"[^{}]*?"end"\s*:\s*"([^"]+)"', text, re.S)
blocks = [(datetime.fromisoformat(s), datetime.fromisoformat(e)) for s, e in wins]
day = now.replace(hour=0, minute=0, second=0, microsecond=0)
blocks += [(day + timedelta(days=d, hours=20), day + timedelta(days=d + 1, hours=1)) for d in range(-1, 14)]
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

wait_free() {  # wait_free MINUTES WHAT: returns once MINUTES fit before the next window (FREE = minutes to it)
  local n=0
  while true; do
    stopped && return 1
    local b; b=$(block)
    case "$b" in
      NOW*) local s=${b#NOW }
            if [ "$DRYRUN" = 1 ]; then FREE=100000; return 0; fi
            [ $((n % 6)) = 0 ] && log "$2: quiet window, ${s}s left"; n=$((n + 1)); nap "$s";;
      NEXT*) FREE=${b#NEXT }
             [ "$FREE" -ge $(( $1 + 3 )) ] && return 0
             if [ "$DRYRUN" = 1 ]; then FREE=100000; return 0; fi
             [ $((n % 6)) = 0 ] && log "$2 needs $1 min, $FREE min before the next window; waiting"; n=$((n + 1)); nap $(( FREE * 60 + 60 ));;
      *) log "window check failed: $b"; nap 600;;
    esac
  done
}

q() {  # q MINUTES PURPOSE -- CMD...
  local minutes=$1 purpose=$2 rc; shift 3
  while true; do
    wait_free "$minutes" "$purpose" || return 1
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $NAME: $purpose (expected $minutes min)"; echo "    cmd: $*" >> "$LOG"; return 0; fi
    local maxwait=$(( FREE - minutes - 3 )); [ "$maxwait" -lt 1 ] && maxwait=1
    log "enqueue: $purpose (expected $minutes min, max wait $maxwait min, $FREE min to the next window)"
    "$PY" "$QUEUE" run --owner small-model --purpose "$NAME: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm93/launch-sm93.sh" -- "$@"
    rc=$?; log "exit $rc: $purpose"
    [ "$rc" = 75 ] && { log "queue wait would reach the next window; re-planning"; sleep 60; continue; }
    return $rc
  done
}
todo() { F="$OUTROWS" U="$USERS" "$PY" -X utf8 -c "
import json, os
from pathlib import Path
p = Path(os.environ['F']); want = set(json.load(open(os.environ['U']))['users'])
got = {}
for r in (map(json.loads, open(p, encoding='utf-8')) if p.exists() else []):
    if 'benchmark' in r: got.setdefault(r['label'], set()).add(r['user_id'])
print(sum(len(want - got.get(lb, set())) for lb in ('$CT/w8+s8', '$RT/w8+s8')))"; }

log "start (dry run $DRYRUN)"
for f in "$CAND" "$REL" "$USERS" "$RULE"; do [ -f "$f" ] || { log "STOP: missing $f"; exit 1; }; done
if [ "$DRYRUN" != 1 ] && [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
  log "STOP: uncommitted source changes in scripts/ expertium/ rwkv/"; exit 1; fi
rc=0; prev=-1; cap=0.8
for pass in 1 2 3 4 5 6; do
  left=$(todo); [ "$DRYRUN" = 1 ] && left=1
  [ "$left" = 0 ] && break
  if [ "$left" = "$prev" ]; then
    if [ "$cap" = 0.8 ]; then cap=0.92; log "no progress ($left label-users left): next pass at cap 0.92 (memory only, same numerics)"
    else log "no progress at cap 0.92 ($left label-users left): the rest do not fit; on to the rule"; break; fi
  fi
  prev=$left
  q $MIN_PASS "scoring the instant-only small model (SM57) and the release (SM63) on normal inputs, the 1,500 validation users outside the 732 set + 6 check users (pass $pass, $left label-users left, cap $cap)" -- \
    env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START HELDOUT_GPU_MEMORY_FRACTION=$cap "$PY" -X utf8 "$SC" --pool 1500 --users-file "$USERS" \
    --model "$CT=expertium:w64:$CAND" --model "$RT=expertium:w64:$REL" --only "$CT/w8+s8" --only "$RT/w8+s8" --inputs normal \
    --time-limit $TL --split-existing --deleted-cards neighbour --out "$OUTREL" >> "$G/score.out" 2>&1 || { log "STOP: scoring ticket failed"; rc=1; break; }
  [ "$DRYRUN" = 1 ] && break
done
if [ $rc = 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: then $PY -X utf8 $RULE decide"; log "DRYRUN OK"; exit 0; fi
  out=$(cd "$WT" && "$PY" -X utf8 "$RULE" decide 2>&1); r=$?
  log "rule exit $r: $(echo "$out" | tail -1)"
  log "done"
else
  log "STOP: ended with exit $rc"
fi
exit $rc
