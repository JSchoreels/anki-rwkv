#!/usr/bin/env bash
# (1) now: one 10-min slice of the real SM11 export on the GPU (queued; runs after the current ticket) to test the GPU path
#     (memory, rate) early; the export resumes per user, so the slice is kept. (2) after the SM40 runbook ends: the full
#     replay runbook (4-5 tickets of 95 min; touch <OUT>/STOP to stop between tickets).
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
SM=/c/Users/jonat/workspace/wt/sm11/artifacts/small-model
OUT=C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/fork-fp32-gate5k
S40=/c/Users/jonat/workspace/wt/small-model/artifacts/small-model/runbook-2026-10-18-dec1ep32k.log
log() { echo "[sm11-launch $(date '+%m-%d %H:%M:%S')] $*"; }
cd /c/Users/jonat/workspace/wt/sm11 || exit 1
log "probe: enqueue a 10-min export slice"
"$PY" "$QUEUE" run --owner small-model --purpose "small-model: SM11 replay probe (10-min slice of the fork full-history export)" \
  --claim gpu --expected-minutes 12 -- "$PY" -X utf8 C:/Users/jonat/workspace/wt/sm11/scripts/sm11_replay_export.py export \
  --out $OUT --run C:/Users/jonat/workspace/wt/sm11/artifacts/accuracy/experiments/scale-w64kd-gate/run/run.json --device cuda --time-limit 600
log "probe exit $? (3 = not complete yet, expected)"
"$PY" -X utf8 C:/Users/jonat/workspace/wt/sm11/scripts/sm11_replay_export.py status --out $OUT \
  --run C:/Users/jonat/workspace/wt/sm11/artifacts/accuracy/experiments/scale-w64kd-gate/run/run.json 2>&1 | tail -5
log "waiting for the SM40 runbook to end"
until grep -q "SM40 runbook exit" $S40 2>/dev/null; do sleep 120; done
log "$(grep -h 'SM40 runbook exit' $S40 | tail -1); starting the SM11 replay runbook"
bash $SM/runbook-sm11-replay.sh >> $SM/runbook-sm11-replay.log 2>&1
log "SM11 replay runbook exit $?"
