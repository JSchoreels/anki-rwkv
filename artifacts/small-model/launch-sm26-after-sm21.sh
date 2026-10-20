#!/usr/bin/env bash
# SM26 retry (scripted-model fix cb7bb22) after the SM21 carried-state step of the orchestrator (user: carried state first).
O=/c/Users/jonat/workspace/wt/small-model/artifacts/small-model/orchestrate-after-sm26.log
LOG=/c/Users/jonat/workspace/wt/small-model/artifacts/small-model/runbook-2026-10-18-sm26.log
until grep -qE "SM21 runbook exit|\(2\) SM21 skipped" $O 2>/dev/null; do sleep 120; done
echo "[launcher $(date '+%m-%d %H:%M:%S')] $(grep -hE 'SM21 runbook exit|SM21 skipped' $O | tail -1); retrying SM26" >> $LOG
bash /c/Users/jonat/workspace/wt/small-model/artifacts/small-model/runbook-2026-10-18-sm26.sh >> $LOG 2>&1
echo "[launcher $(date '+%m-%d %H:%M:%S')] SM26 retry runbook exit $?" >> $LOG
