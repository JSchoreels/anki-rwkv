#!/usr/bin/env bash
# Q9 (2026-10-16): wait for the ctx8192 cont runbook to finish ("done: cont" / "cont ended" in its log), then launch the
# longdecay stage (16k-chunk decay from ctx8192 e8, SM22). Refuses to launch after 19:15.
SM=C:/Users/jonat/workspace/wt/small-model/artifacts/small-model
Q8LOG=$SM/runbook-2026-10-16b-cont8192.log
LOG=$SM/jobs/launch-longdecay-after-q8.log
echo "[$(date +%H:%M:%S)] waiting for the ctx8192 cont runbook to finish" >> "$LOG"
while ! grep -q 'done: cont\|cont ended' "$Q8LOG" 2>/dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] ctx8192 cont finished: $(grep 'done: cont\|cont ended' "$Q8LOG" | tail -1)" >> "$LOG"
[ "$(date +%H%M)" -gt 1915 ] && { echo "[$(date +%H:%M:%S)] too late for longdecay (past 19:15), not launching" >> "$LOG"; exit 0; }
echo "[$(date +%H:%M:%S)] launching longdecay (SRC_NAME=scale-w64kd-gate-cached-ctx8192-cont EPOCH=8 DCHUNK=16384)" >> "$LOG"
cd C:/Users/jonat/workspace/wt/small-model || exit 1
DRYRUN=0 "C:/Program Files/Git/usr/bin/bash.exe" $SM/runbook-2026-10-16d.sh longdecay > $SM/runbook-2026-10-16d-longdecay.log 2>&1
echo "[$(date +%H:%M:%S)] longdecay exited $?" >> "$LOG"
