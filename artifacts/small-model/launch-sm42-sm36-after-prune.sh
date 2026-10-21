#!/usr/bin/env bash
# SM42 (control) then SM36 (Muon) decays, chained after the running pruning runbook (wt/prune). NOT started by the builder.
#   1. wait until the prune launcher log has a final runbook line "[prune MM-DD HH:MM:SS] done" or "... ended with exit N"
#      (runbook-prune-depth.sh prints one of them last; "[prune-launch ...]" lines do not match);
#   2. before EACH arm wait while the small-model HOLD-chain file exists (the user's pause switch: delete it to continue);
#   3. run ARM=sm42, then ARM=sm36 (the Muon arm runs even if the control failed; the failure is logged).
# Launch: nohup bash C:/Users/jonat/workspace/wt/sm42/artifacts/small-model/launch-sm42-sm36-after-prune.sh \
#           > C:/Users/jonat/workspace/wt/sm42/artifacts/small-model/launch-sm42-sm36-after-prune.log 2>&1 & disown
PRUNE_LOG=C:/Users/jonat/workspace/wt/prune/artifacts/small-model/launch-prune-after-chain.log
HOLD=C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/HOLD-chain
WT=C:/Users/jonat/workspace/wt/sm42
SM=$WT/artifacts/small-model
log() { echo "[sm42-launch $(date '+%m-%d %H:%M:%S')] $*"; }
hold() { local said=0; while [ -e "$HOLD" ]; do [ "$said" = 0 ] && log "HOLD-chain present: waiting ($1)" && said=1; sleep 120; done; }
log "waiting for the prune runbook's final line in $PRUNE_LOG"
until grep -aEq '^\[prune [0-9: -]+\] (done|ended with exit)' "$PRUNE_LOG" 2>/dev/null; do sleep 120; done
log "prune finished: $(grep -aE '^\[prune [0-9: -]+\] (done|ended with exit)' "$PRUNE_LOG" | tail -1)"
for arm in sm42 sm36; do
  hold "$arm"
  log "starting ARM=$arm ($(git -C "$WT" log --oneline -1))"
  ARM=$arm bash "$SM/runbook-2026-10-19-sm42-sm36.sh" >> "$SM/runbook-2026-10-19-$arm.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then log "ARM=$arm finished OK"; else log "ARM=$arm ended with exit $rc (next arm still runs)"; fi
done
log "chain done"
