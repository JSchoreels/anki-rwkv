#!/usr/bin/env bash
# Starts the SM40 runbook once the SM26 runbook has finished (its launcher logs "SM26 runbook exit N").
S26=/c/Users/jonat/workspace/wt/small-model/artifacts/small-model/runbook-2026-10-18-sm26.log
LOG=/c/Users/jonat/workspace/wt/small-model/artifacts/small-model/runbook-2026-10-18-dec1ep32k.log
until grep -q "SM26 runbook exit" $S26 2>/dev/null; do sleep 120; done
echo "[launcher $(date '+%m-%d %H:%M:%S')] $(grep -h 'SM26 runbook exit' $S26 | tail -1); starting SM40" >> $LOG
bash /c/Users/jonat/workspace/wt/small-model/artifacts/small-model/runbook-2026-10-18-dec1ep32k.sh >> $LOG 2>&1
echo "[launcher $(date '+%m-%d %H:%M:%S')] SM40 runbook exit $?" >> $LOG
