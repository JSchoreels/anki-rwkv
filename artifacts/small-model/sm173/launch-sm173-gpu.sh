#!/usr/bin/env bash
# Deleted-card note ids (SM173), phase 1b (design.md): score four arms (cur, lanes, fix, reseed) x two models
#   (six-layer Tiny two-head (SM130), Small two-head (SM118)) on the 732 set (723 fit users), one queue ticket per arm and
#   pass (cap 0.8, then 0.92 after a pass without progress), then the CPU report (sm173_report.py).
#   launch   WMI hidden (HANDOFF.md); log artifacts/small-model/sm173/gpu/launch-sm173-gpu.log
#   stop     touch C:/Users/jonat/workspace/wt/sm173/artifacts/small-model/sm173/gpu/STOP
#   resume   remove gpu/STOP, launch again (arms with 723 users are skipped)
# Patterns: done "SM173 DONE", stopped / failed "SM173 STOP" (cause on an earlier "STOP:" line).
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
RT=C:/Users/jonat/workspace/wt/sm173-run
S=C:/Users/jonat/workspace/wt/sm173/artifacts/small-model/sm173
G=$S/gpu
mkdir -p "$G"
LOG=$G/launch-sm173-gpu.log
ROWS=$G/sm173-rows-732.jsonl
STOPF=$G/STOP
USERS=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json
BLOCK=$RT/artifacts/small-model/sm135/sm135_block.py
W=C:/Users/jonat/workspace/wt
T=E89-F65e-F77a
TINY=$W/sm130/artifacts/small-model/sm130-th-dec2ep32k-p2/expertium/$T/master.pth
SMALL=$W/sm118/artifacts/small-model/sm118-e26-dec2ep32k-t91s2-p2/expertium/$T/master.pth
WANT=723
NAME="Deleted-card note ids (SM173)"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset RWKV_DELETED_CARDS EXPERTIUM_END_TO_START SM11_INPUTS HELDOUT_GPU_MEMORY_FRACTION
log() { echo "[sm173 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
stopped() { [ -e "$STOPF" ] && { log "STOP: STOP file present"; return 0; }; return 1; }

wait_free() {  # wait_free MINUTES: return once MINUTES fit before the next quiet window (scope all / gpu)
  local b
  while true; do
    stopped && return 1
    b=$("$PY" -X utf8 "$BLOCK" | tr -d '\r')
    case "$b" in
      NOW*) log "quiet window, ${b#NOW }s left"; sleep "${b#NOW }";;
      NEXT*) local m=${b#NEXT }; [ "$m" -ge $(( $1 + 5 )) ] && return 0
             log "next quiet window in $m min (< $1 + 5): waiting for it"; sleep $(( (m + 1) * 60 ));;
      *) log "block helper said '$b'; going on"; return 0;;
    esac
  done
}

min_users() {  # min_users LABEL...: users with a finite or nonfinite row for every label
  F="$ROWS" "$PY" -X utf8 -c "
import json, os, sys
from pathlib import Path
p = Path(os.environ['F']); got = {}
if p.exists():
    for raw in open(p, 'rb'):
        try:
            r = json.loads(raw.decode('utf-8', errors='replace'))
        except ValueError:
            continue
        if 'benchmark' in r or ('nonfinite' in r and 'error' not in r):
            got.setdefault(r.get('label'), set()).add(r['user_id'])
print(min(len(got.get(l, ())) for l in sys.argv[1:]))" "$@" | tr -d '\r'; }

score_arm() {  # score_arm ARM (reseed = cur with --seed-offset 1)
  local arm=$1 hook=$1 extra=() cap=0.8 pass have prev=-1 minutes=12 rc
  [ "$arm" = reseed ] && { hook=cur; extra=(--seed-offset 1); }
  local labels=("sm130-th-$arm/fp32@e2s-query" "sm118-$arm/fp32@e2s-query")
  for pass in 1 2 3; do
    have=$(min_users "${labels[@]}")
    [ "$have" -ge "$WANT" ] && break
    if [ "$have" = "$prev" ]; then
      if [ "$cap" = 0.8 ]; then cap=0.92; log "$arm: no progress ($have of $WANT users), next pass at cap 0.92 (memory only)"
      else log "$arm: no progress at cap 0.92 ($have of $WANT); going on"; break; fi
    fi
    prev=$have; [ "$pass" -gt 1 ] && minutes=5
    wait_free "$minutes" || return 1
    log "enqueue: $arm pass $pass ($have done, cap $cap, expected $minutes min)"
    "$PY" "$QUEUE" run --owner sm173 --purpose "$NAME: score arm $arm (Tiny SM130 + Small SM118 two-head, fp32, 732 set, pass $pass)" \
      --claim gpu --expected-minutes "$minutes" --launched-by "detached launcher sm173/launch-sm173-gpu.sh" -- \
      env SM173_ARM=$hook HELDOUT_GPU_MEMORY_FRACTION=$cap PYTORCH_JIT=0 "$PY" -X utf8 "$S/sm173_score.py" \
      --model "sm130-th-$arm=expertium:w64-c1-d2p1u1:$TINY" --model "sm118-$arm=expertium:w64:$SMALL" \
      --pool 732 --users-file "$USERS" --inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows \
      "${extra[@]}" --time-limit 1800 --out "$ROWS" >> "$G/score.out" 2>&1
    rc=$?
    log "exit $rc: $arm pass $pass ($(min_users "${labels[@]}") of $WANT users)"
    [ "$rc" = 0 ] || { log "STOP: scoring ticket failed ($arm pass $pass, see $G/score.out)"; return 1; }
  done
  return 0
}

log "start (run tree $(git -C "$RT" rev-parse --short HEAD), hook $(git -C "$S" log -1 --format=%h -- sm173_score.py))"
rc=0
for arm in cur lanes fix reseed; do
  stopped && { rc=1; break; }
  score_arm "$arm" || { rc=1; break; }
done
if [ $rc = 0 ]; then
  (cd "$S" && OMP_NUM_THREADS=2 "$PY" -X utf8 sm173_report.py report > "$G/report.out" 2>&1); rrc=$?
  out=$(tr -d '\r' < "$G/report.out")
  log "report exit $rrc: $(echo "$out" | grep -m1 VERDICT)"
  log "SM173 DONE"
else
  log "SM173 STOP"
fi
