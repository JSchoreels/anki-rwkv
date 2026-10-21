#!/usr/bin/env bash
# SM39 launcher (2026-10-21): wait until SM55 (wt/sm55: ~3 h replay export 10-21, ~6.6 h 8k decay from 10-22 01:00) has
# finished or stopped, then run runbook-sm39.sh (which waits out quiet windows itself before each ticket).
#   GO when one of:
#     - wt/sm55 does not exist;
#     - the LAST "[sm55 ...]" line of SM55's runbook log (wt/sm55/artifacts/small-model/runbook-*sm55*.log, not the dry
#       run) is its final `done` (printed only after the [sm55-summary] block, HANDOFF-sm55.md) or a STOP / "ended with
#       exit" line, the log has been idle for 5 min and the queue shows no live SM55 ticket;
#     - the file artifacts/small-model/sm39/GO exists (manual override by the main session).
#   STOP file artifacts/small-model/sm39/STOP: exit without enqueuing anything.
#   launch: nohup bash <this file> > C:/Users/jonat/workspace/wt/sm39/artifacts/small-model/sm39/launch-sm39.log 2>&1 & disown
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
O=C:/Users/jonat/workspace/wt/sm39/artifacts/small-model/sm39
SM55=C:/Users/jonat/workspace/wt/sm55
log() { echo "[sm39-launch $(date '+%m-%d %H:%M:%S')] $*"; }

sm55_state() {  # prints "go <reason>" or "wait <reason>"
  SM55="$SM55" QUEUE_STATUS="$("$PY" "$QUEUE" status 2>&1)" "$PY" -X utf8 - <<'PYEOF'
import os, re, time
from pathlib import Path
root = Path(os.environ["SM55"])
if not root.exists():
    print("go wt/sm55 does not exist"); raise SystemExit
live = [l for l in os.environ["QUEUE_STATUS"].splitlines() if re.search(r"sm55", l, re.I) and re.search(r"running|waiting|reserved", l)]
logs = []
base = root / "artifacts" / "small-model"
for path in sorted(base.glob("runbook-*sm55*.log")) if base.exists() else []:
    if "dryrun" in path.name.lower():
        continue
    try:
        lines = [l.rstrip() for l in path.read_text(encoding="utf-8", errors="replace").splitlines() if re.match(r"\[sm55", l, re.I)]
    except OSError:
        continue
    if lines:
        logs.append((path, lines, path.stat().st_mtime))
for p, lines, mtime in logs:
    last = lines[-1]
    finished = re.search(r"\]\s+done\s*$", last) or re.search(r"STOP|ended with exit", last)
    if finished and not live and time.time() - mtime > 300:
        print(f"go SM55 finished or stopped: {p.name}: {last[:160]}"); raise SystemExit
print(f"wait SM55 running or not finished (logs with [sm55 lines: {len(logs)}; live SM55 tickets: {len(live)}; "
      f"last: {(logs[-1][1][-1][:120] if logs else 'none')})")
PYEOF
}

log "started; waiting for SM55 (watch $SM55)"
last=""
n=0
while :; do
  if [ -f "$O/STOP" ]; then log "STOP file found; nothing enqueued"; exit 0; fi
  if [ -f "$O/GO" ]; then log "GO file found (manual override)"; break; fi
  state=$(sm55_state) || state="wait state check failed"
  case "$state" in
    go*) log "${state#go }"; break;;
  esac
  if [ "$state" != "$last" ] || [ $((n % 6)) -eq 0 ]; then log "${state#wait }"; fi
  last=$state; n=$((n + 1))
  sleep 600
done
log "running runbook-sm39.sh"
bash "$O/runbook-sm39.sh" > "$O/runbook-sm39.log" 2>&1
rc=$?
log "runbook exit $rc (log $O/runbook-sm39.log)"
tail -1 "$O/runbook-sm39.log"
exit $rc
