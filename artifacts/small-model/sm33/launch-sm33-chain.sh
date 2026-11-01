#!/usr/bin/env bash
# Grade-order sanity (SM33): wait for the stage-1 launcher (launch-sm33-stage1.sh, already running) to finish, then run
# the stage-2 launcher (launch-sm33-stage2.sh), whose gate reads stage 1's result. Polls the stage-1 log every 5 min.
#   launch   WMI hidden (artifacts/small-model/sm33/HANDOFF.md); log artifacts/small-model/sm33/gpu/launch-sm33-chain.log
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
S=C:/Users/jonat/workspace/wt/sm33/artifacts/small-model/sm33
G=$S/gpu
L1=$G/launch-stage1.log
LOG=$G/launch-sm33-chain.log
log() { echo "[sm33-chain $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
log "waiting for stage 1 ($L1)"
for i in $(seq 1 120); do   # <= 10 h
  [ -e "$G/STOP" ] && { log "STOP file present"; exit 0; }
  last=$(tail -1 "$L1" 2>/dev/null)
  case "$last" in
    *"] done"*) log "stage 1 done: $(tail -2 "$L1" | head -1)"; break;;
    *"STOP"*|*"ticket exit "[1-9]*|*"report exit "[1-9]*) log "stage 1 ended without a result: $last"; exit 1;;
  esac
  sleep 300
done
[ -f "$S/sm33-stage1.json" ] || { log "no stage-1 result after the wait; stopping"; exit 1; }
log "starting stage 2"
bash "$S/launch-sm33-stage2.sh" >> "$G/launch-sm33-stage2.nohup" 2>&1
log "stage 2 exit $?"
