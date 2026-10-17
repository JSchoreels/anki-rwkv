#!/usr/bin/env bash
# Restart of SM30 after it was stopped at 21:43 on 2026-10-17 (it had started at 21:27, into the 22:00-02:00 break).
# Waits for 02:00 on 2026-10-18, then runs the rep-flags runbook (cont), appending to its log.
LOG=/c/Users/jonat/workspace/wt/rep-flags/artifacts/small-model/runbook-2026-10-18-repflags.log
until [ "$(date +%Y%m%d%H%M)" -ge 202610180200 ]; do sleep 60; done
echo "[launcher $(date '+%m-%d %H:%M:%S')] 02:00, starting SM30 (restart after the break)" >> $LOG
bash /c/Users/jonat/workspace/wt/rep-flags/artifacts/small-model/runbook-2026-10-18-repflags.sh cont >> $LOG 2>&1
echo "[launcher $(date '+%m-%d %H:%M:%S')] SM30 runbook exit $?" >> $LOG
