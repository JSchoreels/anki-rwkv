#!/usr/bin/env bash
# 2026-10-18 reorder (user OK): carried state v2 (strided lanes) right after the v1 experiment (arms A and B), before SM40.
# Waits for the orchestrator's "SM21 runbook exit" and for the v2 GPU check ticket to end; runs the v2 runbook only if the
# check report passed (the runbook re-checks it).
O=/c/Users/jonat/workspace/wt/small-model/artifacts/small-model/orchestrate-after-sm26.log
SM=C:/Users/jonat/workspace/wt/sm21v2/artifacts/small-model  # Windows Python opens the report: no /c/ path
log() { echo "[launcher $(date '+%m-%d %H:%M:%S')] $*" >> $SM/runbook-2026-10-18-sm21v2.log; }
log "waiting for the v1 runbook (arms A and B) and the v2 GPU check"
until grep -qE "SM21 runbook exit" $O 2>/dev/null && grep -q "sm21v2-gpu-check\] exit" $SM/sm21v2-gpu-check.log 2>/dev/null; do sleep 120; done
log "$(grep -h 'SM21 runbook exit' $O | tail -1); $(grep -h 'sm21v2-gpu-check\] exit' $SM/sm21v2-gpu-check.log | tail -1)"
if ! C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -c "
import json, sys
r = json.load(open(r'$SM/check-state-carry-lanes-gpu.json'))
sys.exit(0 if r.get('fails') in ([], {}, 0, None) and r.get('passed') else 1)"; then
  log "v2 GPU check did not pass: v2 run not started"
  exit 1
fi
cd /c/Users/jonat/workspace/wt/sm21v2 && bash $SM/runbook-2026-10-18-sm21v2.sh >> $SM/runbook-2026-10-18-sm21v2.log 2>&1
log "SM21 v2 runbook exit $?"
