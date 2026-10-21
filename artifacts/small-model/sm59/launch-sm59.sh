#!/usr/bin/env bash
# SM59 detached launcher: waits for SM50's launcher to finish (or fail), then runs kd-control, kd-090, kd-100 (runbook-sm59.sh),
# one queue ticket each (110 expected min). Before each arm it checks quiet.json via quiet_wait.py (a 110-min arm must not reach a
# quiet window); if it would, it logs "deferred: <arm>" and stops cleanly (the main session reschedules). Check D must have passed.
#   nohup bash launch-sm59.sh > launch-sm59.log 2>&1 & disown
# Done lines: "[sm59-<arm> ...] done" in runbook-sm59-<arm>.log and here "[sm59-launch ...] <arm> done"; final "all arms finished".
# STOP file artifacts/small-model/sm59/STOP: exit without enqueuing. A failed arm stops the chain (STOP line).
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
S=C:/Users/jonat/workspace/wt/sm59/artifacts/small-model/sm59
SM50LOG=C:/Users/jonat/workspace/wt/sm50/artifacts/small-model/sm50/launch-sm50.log
MIN=${MIN_ARM:-110}
log() { echo "[sm59-launch $(date '+%m-%d %H:%M:%S')] $*"; }
log "started; waiting for SM50 ($SM50LOG)"
n=0
while :; do
  [ -f "$S/STOP" ] && { log "STOP file found; nothing enqueued"; exit 0; }
  if grep -qE '\[sm50-launch [^]]*\] all arms finished' "$SM50LOG" 2>/dev/null; then log "SM50 all arms finished"; break; fi
  if grep -qE 'STOP|ended with exit' "$SM50LOG" 2>/dev/null; then log "SM50 log shows STOP / 'ended with exit': proceeding anyway"; break; fi
  [ $((n % 36)) -eq 0 ] && log "waiting for SM50 ($(tail -1 "$SM50LOG" 2>/dev/null | cut -c1-120))"
  n=$((n + 1)); sleep 300
done
"$PY" -c "import json,sys; sys.exit(0 if json.load(open(r'$S/check-head-only-d.json')).get('passed') else 1)" \
  || { log "STOP: check D has not passed"; exit 2; }
for arm in kd-control kd-090 kd-100; do
  [ -f "$S/STOP" ] && { log "STOP file found"; exit 0; }
  qw=$("$PY" "$S/quiet_wait.py" "$MIN"); wait=${qw%% *}; reason=${qw#* }
  if [ "${wait:-0}" -gt 0 ] 2>/dev/null; then log "deferred: $arm ($MIN min would reach a quiet window: $reason, wait ${wait}s)"; exit 0; fi
  log "start arm $arm (expected $MIN min; quiet check: $reason)"
  STAGE=arm ARM=$arm MIN=$MIN bash "$S/runbook-sm59.sh" > "$S/runbook-sm59-$arm.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || ! tail -3 "$S/runbook-sm59-$arm.log" | grep -qE '\] done'; then log "STOP: arm $arm failed (exit $rc, see runbook-sm59-$arm.log)"; exit 1; fi
  log "$arm done"
done
"$PY" -X utf8 "$S/compare-sm59.py" > "$S/compare-sm59.log" 2>&1 && log "summary-table.md written" || log "compare step failed (see compare-sm59.log)"
log "all arms finished"
