#!/usr/bin/env bash
# Starts the SM30 paired GPU test once the night runbook f has finished (keeps the queue order and the 20:00-02:00 break).
until grep -q "\] done" /c/Users/jonat/workspace/wt/small-model/artifacts/small-model/runbook-2026-10-17f-evening-night.log; do sleep 120; done
echo "[launcher $(date '+%m-%d %H:%M:%S')] night runbook done, starting SM30"
bash /c/Users/jonat/workspace/wt/rep-flags/artifacts/small-model/runbook-2026-10-18-repflags.sh cont > /c/Users/jonat/workspace/wt/rep-flags/artifacts/small-model/runbook-2026-10-18-repflags.log 2>&1
echo "[launcher $(date '+%m-%d %H:%M:%S')] SM30 runbook exit $?"
