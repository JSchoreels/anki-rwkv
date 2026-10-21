#!/usr/bin/env bash
# SM50 detached launcher: the curve arm, then the instant arm (runbook-sm50.sh STAGE=arm), one queue ticket each.
# Before each enqueue it waits while [now, now + expected minutes] touches a quiet window of
# artifacts/coordination/queue/quiet.json or the evening break 21:00-01:00 (brief: both arms end before 21:00 or wait
# for 01:00). The instant arm runs even if the curve arm failed (logged).
#   MIN_CURVE=N MIN_INSTANT=N nohup bash launch-sm50.sh > launch-sm50.log 2>&1 & disown
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
S=C:/Users/jonat/workspace/wt/sm50/artifacts/small-model/sm50
QUIET=C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json
log() { echo "[sm50-launch $(date '+%m-%d %H:%M:%S')] $*"; }
blocked() {  # exit 0 (blocked) when a ticket of $1 minutes started now would touch a quiet window or 21:00-01:00
  MINUTES=$1 QUIET="$QUIET" "$PY" -X utf8 -c "
import datetime as dt, json, os, sys
now = dt.datetime.now(); end = now + dt.timedelta(minutes=float(os.environ['MINUTES']))
windows = []
try:
    windows = [(dt.datetime.fromisoformat(w['start']), dt.datetime.fromisoformat(w['end'])) for w in json.load(open(os.environ['QUIET']))]
except Exception:
    pass
for day in (now.date() - dt.timedelta(days=1), now.date()):
    start = dt.datetime.combine(day, dt.time(21, 0))
    windows.append((start, start + dt.timedelta(hours=4)))
sys.exit(0 if any(s < end and now < e for s, e in windows) else 1)"
}
for arm in curve instant; do
  var=MIN_${arm^^}; minutes=${!var:?}
  said=0
  while blocked "$minutes"; do
    [ $said = 1 ] || log "waiting: $arm ($minutes min) would reach a quiet window"
    said=1; sleep 120
  done
  log "start arm $arm (expected $minutes min)"
  STAGE=arm ARM=$arm MIN=$minutes bash "$S/runbook-sm50.sh" > "$S/runbook-sm50-$arm.log" 2>&1
  log "arm $arm exit $?"
done
log "all arms finished"
