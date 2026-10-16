#!/usr/bin/env bash
# Q8 (2026-10-16): wait for the Q7 ctx2048 cont runbook to finish (a 2nd "done: cont"/"cont ended" line in its log),
# then launch the ctx8192 cont stage (SM21 option 5). Epochs 7-8 if launched by 11:45, else epoch 7 only (21:30 window).
SM=C:/Users/jonat/workspace/wt/small-model/artifacts/small-model
Q7LOG=$SM/runbook-2026-10-16b-cont.log
LOG=$SM/jobs/launch-cont8192-after-q7.log
echo "[$(date +%H:%M:%S)] waiting for Q7 to finish" >> "$LOG"
while [ "$(grep -c 'done: cont\|cont ended' "$Q7LOG")" -lt 2 ]; do sleep 60; done
echo "[$(date +%H:%M:%S)] Q7 finished: $(grep 'done: cont\|cont ended' "$Q7LOG" | tail -1)" >> "$LOG"
TO=8; [ "$(date +%H%M)" -gt 1145 ] && TO=7
[ "$(date +%H%M)" -gt 1330 ] && { echo "[$(date +%H:%M:%S)] too late for ctx8192 (past 13:30), not launching" >> "$LOG"; exit 0; }
echo "[$(date +%H:%M:%S)] launching ctx8192 cont FROM=6 TO=$TO" >> "$LOG"
cd C:/Users/jonat/workspace/wt/small-model || exit 1
TIER=gate FROM=6 TO=$TO CHUNK=8192 "C:/Program Files/Git/usr/bin/bash.exe" $SM/runbook-2026-10-16b.sh cont > $SM/runbook-2026-10-16b-cont8192.log 2>&1
echo "[$(date +%H:%M:%S)] ctx8192 cont exited $?" >> "$LOG"
