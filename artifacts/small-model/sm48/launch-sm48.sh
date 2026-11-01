#!/usr/bin/env bash
# SM48 launcher: wait until the SM47 launcher's ticket has ended (its log line "[sm47 <date>] queue exit N"), then
# while the GPU-break HOLD-chain file exists, then one --timing queue ticket for job-sm48.sh.
# DRYRUN=1: report the current wait state and dry-run the job, no waiting and no ticket.
SM47_LOG=C:/Users/jonat/workspace/wt/sm45/artifacts/small-model/sm47/launch-sm47.log
HOLD=C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/HOLD-chain
JOB=C:/Users/jonat/workspace/wt/sm48/artifacts/small-model/sm48/job-sm48.sh
log() { echo "[sm48 $(date '+%m-%d %H:%M:%S')] $*"; }
log "launcher start; waiting for the SM47 ticket to end ($SM47_LOG: '[sm47 ...] queue exit')"
if [ -n "$DRYRUN" ]; then
  [ -e "$SM47_LOG" ] && log "DRYRUN SM47 log present; ended: $(grep -acE '^\[sm47 [^]]*\] queue exit' "$SM47_LOG")" \
    || log "DRYRUN SM47 log missing"
  [ -e "$HOLD" ] && log "DRYRUN HOLD-chain present" || log "DRYRUN HOLD-chain absent"
  bash -n "$JOB" && DRYRUN=1 bash "$JOB"
  exit $?
fi
until grep -aEq '^\[sm47 [^]]*\] queue exit' "$SM47_LOG" 2>/dev/null; do sleep 120; done
log "SM47: $(grep -aE '^\[sm47 [^]]*\] (done|STOP|queue exit)' "$SM47_LOG" | tail -2 | tr '\n' ' ')"
until [ ! -e "$HOLD" ]; do sleep 120; done
log "enqueue"
C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py run --owner small-model \
  --purpose "SM48: held-out scoring speed A/B (dict vs numpy metrics, --lanes 2/4), 40 users of the 732 pool, SM36 w8+s8/w8+s4c8n" \
  --claim gpu --timing --expected-minutes 30 \
  --launched-by "detached launcher sm48" -- bash "$JOB"
log "queue exit $?"
