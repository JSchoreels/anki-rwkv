#!/usr/bin/env bash
# SM15: start the depth-cut GPU runbook (stage e1) once the small-model GPU chain has finished. NOT started by the builder.
#   1. wait until the chain log contains "chain done" (the 2,225-pool step is the last of that chain);
#   2. wait while the HOLD file exists (the user's pause switch: delete it to continue);
#   3. refuse unless the self-test report passed and the prune worktree's scripts/ expertium/ rwkv/ are clean (SourceWatch);
#   4. run `ARCH=<chosen> bash runbook-prune-depth.sh e1`  (its tickets are queued with owner small-model, purposes "SM15 ...").
# ARCH is the CPU screen's choice (artifacts/small-model/prune-screen/report.md): override with ARCH=... in the environment.
# Launch: nohup bash C:/Users/jonat/workspace/wt/prune/artifacts/small-model/launch-prune-after-chain.sh \
#           > C:/Users/jonat/workspace/wt/prune/artifacts/small-model/launch-prune-after-chain.log 2>&1 & disown
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/prune
SM=$WT/artifacts/small-model
CHAIN_SM=C:/Users/jonat/workspace/wt/small-model/artifacts/small-model
CHAIN_LOG=$CHAIN_SM/launch-chain-after-sm40v2.log
HOLD=$CHAIN_SM/HOLD-chain
export ARCH=${ARCH:-w64-d2p1u1}
log() { echo "[prune-launch $(date '+%m-%d %H:%M:%S')] $*"; }
log "waiting for 'chain done' in $CHAIN_LOG (ARCH=$ARCH)"
until grep -aq "chain done" "$CHAIN_LOG" 2>/dev/null; do sleep 120; done
log "chain finished"
said=0; while [ -e "$HOLD" ]; do [ "$said" = 0 ] && log "HOLD-chain present: waiting" && said=1; sleep 120; done
"$PY" -c "
import json, sys
r = json.load(open(r'$SM/check-prune.json'))
sys.exit(0 if r.get('passed') and not r.get('fails') else 1)" || { log "STOP: check-prune.json has not passed"; exit 2; }
if [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then log "STOP: uncommitted source changes in the prune worktree"; exit 2; fi
cd "$WT" || exit 1
log "starting the e1 runbook ($(git log --oneline -1))"
exec bash "$SM/runbook-prune-depth.sh" e1
