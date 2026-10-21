#!/usr/bin/env bash
# SM49 requeue with the discrete GPU-check rule (user approved 2026-10-21 ~09:40), placed so it does not delay SM55's
# 8k decay (that decay must start by ~12:10 to end before the break, else it moves to 01:00).
# GO when one of: an SM55 decay ticket is running or waiting in the queue (SM49 then queues behind it); SM55 stopped;
# or it is past 12:35 with no SM55 ticket in the queue (decay moved to 01:00, GPU free this afternoon).
# Then run launch-sm49.sh (which still waits for HOLD-chain and quiet windows).
#   launch: nohup bash <this file> >> .../sm49/launch-sm49.log 2>&1 & disown
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
SM55LOG=C:/Users/jonat/workspace/wt/sm55/artifacts/small-model/runbook-2026-10-21-sm55.log
log() { echo "[sm49 $(date '+%m-%d %H:%M:%S')] $*"; }
log "after-sm55 waiter start"
while true; do
  q=$("$PY" "$QUEUE" status 2>&1)
  if echo "$q" | grep -qiE "SM55[^)]*decay"; then log "GO: SM55 decay ticket in the queue"; break; fi
  if grep -q "STOP" "$SM55LOG"; then log "GO: SM55 stopped"; break; fi
  if [ "$(date +%H%M)" -ge 1235 ] && ! echo "$q" | grep -q "SM55"; then log "GO: past 12:35, no SM55 ticket (decay moved to 01:00)"; break; fi
  sleep 60
done
export SM49_GPU_RULE=discrete
bash C:/Users/jonat/workspace/wt/sm49/artifacts/small-model/sm49/launch-sm49.sh
