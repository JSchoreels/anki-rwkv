#!/usr/bin/env bash
# Fast-recall prediction (SM94) stage 1 (artifacts/small-model/sm94/DESIGN.md):
#   1. GPU ticket (owner sm94, cap 0.8): per-review dump of the released small model's instant grade probabilities +
#      the shipped answer-time head (SM83)'s per-grade time distributions on the 723 users of the 732 set
#      (scripts/sm94_fastpass.py dump; the SM82c scorer imported read only from the sm82c worktree). Resumable, 2 passes.
#   2. CPU: scripts/sm94_fastpass.py analyze -> stage1-result.json (fixed plan of DESIGN.md).
# Holds: quiet.json windows (scope all/gpu), the daily 20:00-01:00 break, Sunday from 16:00, and free RAM < 3 GB.
#   launch   WMI hidden, below-normal priority (HANDOFF.md); log artifacts/small-model/sm94/gpu/launch-sm94.log
#   resume   the launch command again (dumped users are skipped)
#   stop     touch C:/Users/jonat/workspace/wt/sm94/artifacts/small-model/sm94/gpu/STOP (checked between steps)
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
QUIET=C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json
WT=C:/Users/jonat/workspace/wt/sm94
S=$WT/artifacts/small-model/sm94
G=$S/gpu
DUMP=$S/work/dump
mkdir -p "$G" "$DUMP"
LOG=$G/launch-sm94.log
FP="$PY -X utf8 $WT/scripts/sm94_fastpass.py"
NAME="fast-recall prediction (SM94)"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS
log() { echo "[sm94 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
stopped() { [ -e "$G/STOP" ] && { log "STOP: STOP file present"; return 0; }; return 1; }

block() {  # "NOW <s>" inside a quiet window, else "NEXT <min>"
  Q="$QUIET" "$PY" -X utf8 - <<'PYEOF'
import json, os, re
from datetime import datetime, timedelta
now = datetime.now()
try:
    text = open(os.environ["Q"], encoding="utf-8").read()
except OSError as error:
    print(f"ERR {error}"); raise SystemExit
try:
    wins = [(w["start"], w["end"]) for w in json.loads(text) if w.get("scope", "all") in ("all", "gpu")]
except (ValueError, KeyError, TypeError):
    wins = re.findall(r'"start"\s*:\s*"([^"]+)"[^{}]*?"end"\s*:\s*"([^"]+)"', text, re.S)
blocks = [(datetime.fromisoformat(s), datetime.fromisoformat(e)) for s, e in wins]
day = now.replace(hour=0, minute=0, second=0, microsecond=0)
# the daily break; quiet.json entries for a night override it when they overlap (an early end is written there)
daily = [(day + timedelta(days=d, hours=20), day + timedelta(days=d + 1, hours=1)) for d in range(-1, 9)]
daily += [(day + timedelta(days=d, hours=16), day + timedelta(days=d, hours=20)) for d in range(-1, 9)
          if (day + timedelta(days=d)).weekday() == 6]
for s, e in daily:
    if not any(bs < e and be > s for bs, be in blocks if (be - bs) < timedelta(days=1)):
        blocks.append((s, e))
blocks.sort()
for s, e in blocks:
    if s <= now < e:
        print(f"NOW {int((e - now).total_seconds()) + 60}"); raise SystemExit
for s, e in blocks:
    if s > now:
        print(f"NEXT {int((s - now).total_seconds() // 60)}"); raise SystemExit
print("NEXT 100000")
PYEOF
}

free_gb() { powershell.exe -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)" 2>/dev/null | tr -d '\r'; }

wait_free() {  # wait_free MINUTES WHAT
  local n=0
  while true; do
    stopped && return 1
    local b; b=$(block)
    case "$b" in
      NOW*) [ $((n % 6)) = 0 ] && log "$2: quiet window"; n=$((n + 1)); sleep 600; continue;;
      NEXT*) FREE=${b#NEXT }
             if [ "$FREE" -lt $(( $1 + 3 )) ]; then [ $((n % 6)) = 0 ] && log "$2 needs $1 min, $FREE min to the next window"; n=$((n + 1)); sleep 600; continue; fi;;
      *) log "window check failed: $b"; sleep 600; continue;;
    esac
    local ram; ram=$(free_gb)
    if [ -n "$ram" ] && awk "BEGIN{exit !($ram < 3.0)}"; then
      [ $((n % 6)) = 0 ] && log "$2: hold, free RAM $ram GB < 3 GB"; n=$((n + 1)); sleep 300; continue; fi
    return 0
  done
}

dumped() { [ -f "$DUMP/dump-rows.jsonl" ] && F="$DUMP/dump-rows.jsonl" "$PY" -X utf8 -c "
import json, os
print(len({r['user_id'] for r in map(json.loads, open(os.environ['F'], encoding='utf-8')) if 'error' not in r}))" || echo 0; }

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1)"
rc=0
for pass in 1 2; do
  [ "$(dumped)" -ge 723 ] && break
  wait_free 25 "dump pass $pass" || { rc=1; break; }
  maxwait=$(( FREE - 25 - 3 )); [ "$maxwait" -lt 1 ] && maxwait=1
  log "enqueue dump pass $pass ($(dumped) users done, $FREE min to the next window)"
  "$PY" "$QUEUE" run --owner sm94 --purpose "$NAME: per-review dump of the released small model's grade probabilities and the shipped answer-time head (SM83) on the 732 set (723 users, pass $pass)" \
    --claim gpu --expected-minutes 25 --max-wait-minutes "$maxwait" --launched-by "detached launcher sm94/launch-sm94.sh" -- \
    $FP dump --out-dir "$DUMP" --workers 2 >> "$G/dump.out" 2>&1
  r=$?; log "exit $r: dump pass $pass ($(dumped) users)"
  [ "$r" = 75 ] && { sleep 60; continue; }
  [ "$r" = 0 ] || { log "STOP: dump ticket failed (see $G/dump.out)"; rc=1; break; }
done
if [ $rc = 0 ]; then
  got=$(dumped)
  [ "$got" -ge 722 ] || { log "STOP: only $got of 723 users dumped"; rc=1; }
fi
if [ $rc = 0 ]; then
  wait_free 30 "analysis" || rc=1
  if [ $rc = 0 ]; then
    log "analysis start (free RAM $(free_gb) GB)"
    $FP analyze --dump-dir "$DUMP" --out "$S/stage1-result.json" >> "$G/analyze.out" 2>&1 || { log "STOP: analysis failed (see $G/analyze.out)"; rc=1; }
  fi
fi
log "done (rc $rc)"
exit $rc
