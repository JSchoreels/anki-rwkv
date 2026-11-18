#!/usr/bin/env bash
# 2026-10-18 order (user: carried state higher in the plan): after the SM26 runbook ends,
#   (1) SM11 fork full-history replay (resumable, ~3 h; touch <OUT>/STOP to stop between tickets),
#   (2) SM21 carried-state experiment, only if its runbook exists and its dry run printed DRYRUN OK,
#   (3) SM40 one-epoch decay at 32k from e10 + 464-user scoring.
# Replaces launch-sm11-probe-and-replay.sh (its probe ran: ticket 000434) and launch-dec1ep32k-after-sm26.sh.
S26=/c/Users/jonat/workspace/wt/small-model/artifacts/small-model/runbook-2026-10-18-sm26.log
SM11=/c/Users/jonat/workspace/wt/sm11/artifacts/small-model
SC=/c/Users/jonat/workspace/wt/state-carry/artifacts/small-model
SM=/c/Users/jonat/workspace/wt/small-model/artifacts/small-model
log() { echo "[orchestrate $(date '+%m-%d %H:%M:%S')] $*"; }
log "waiting for the SM26 runbook to end"
until grep -q "SM26 runbook exit" $S26 2>/dev/null; do sleep 120; done
log "$(grep -h 'SM26 runbook exit' $S26 | tail -1)"
log "(1) SM11 replay"
(cd /c/Users/jonat/workspace/wt/sm11 && bash $SM11/runbook-sm11-replay.sh >> $SM11/runbook-sm11-replay.log 2>&1)
log "SM11 replay runbook exit $? ($(tail -1 $SM11/runbook-sm11-replay.log))"
if [ -f $SC/runbook-2026-10-18-sm21.sh ] && grep -q "DRYRUN OK" $SC/runbook-2026-10-18-sm21-dryrun.log 2>/dev/null \
   && grep -q "export complete" $SM11/runbook-sm11-replay.log; then
  log "(2) SM21 carried-state experiment"
  (cd /c/Users/jonat/workspace/wt/state-carry && bash $SC/runbook-2026-10-18-sm21.sh >> $SC/runbook-2026-10-18-sm21.log 2>&1)
  log "SM21 runbook exit $?"
else
  log "(2) SM21 skipped: runbook, its DRYRUN OK or the complete replay is missing"
fi
log "(3) SM40"
bash $SM/runbook-2026-10-18-dec1ep32k.sh >> $SM/runbook-2026-10-18-dec1ep32k.log 2>&1
log "SM40 runbook exit $?"
log "done"
