#!/usr/bin/env bash
# 2026-10-16 (user: "test without window"): after the 16k longdecay runbook finishes ("done: longdecay" / "longdecay
# ended"), run the same stage at 32,768 x 1 (the fork's context): decay from ctx8192 e8 + 464-user scoring.
SM=C:/Users/jonat/workspace/wt/small-model/artifacts/small-model
L16=$SM/runbook-2026-10-16d-longdecay.log
LOG=$SM/jobs/launch-32k-after-16k.log
echo "[$(date +%H:%M:%S)] waiting for the 16k longdecay runbook to finish" >> "$LOG"
while ! grep -q 'done: longdecay\|longdecay ended' "$L16" 2>/dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] 16k finished: $(grep 'done: longdecay\|longdecay ended' "$L16" | tail -1)" >> "$LOG"
echo "[$(date +%H:%M:%S)] launching longdecay at 32768 x 1" >> "$LOG"
cd C:/Users/jonat/workspace/wt/small-model || exit 1
SRC_NAME=scale-w64kd-gate-cached-ctx8192-cont EPOCH=8 DCHUNK=32768 DBATCH=1 DRYRUN=0 \
  "C:/Program Files/Git/usr/bin/bash.exe" $SM/runbook-2026-10-16d.sh longdecay > $SM/runbook-2026-10-16d-longdecay32k.log 2>&1
echo "[$(date +%H:%M:%S)] longdecay 32k exited $?" >> "$LOG"
