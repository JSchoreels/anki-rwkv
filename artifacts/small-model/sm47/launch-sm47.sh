#!/usr/bin/env bash
# SM47 launcher: wait until the SM46 runbook has ended (its log line "[sm46 <date>] done" or "... STOP..."), then
# while the GPU-break HOLD-chain file exists, then one queue ticket for job-sm47.sh.
SM46_LOG=C:/Users/jonat/workspace/wt/sm46/artifacts/small-model/sm46/runbook-sm46.log
HOLD=C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/HOLD-chain
log() { echo "[sm47 $(date '+%m-%d %H:%M:%S')] $*"; }
log "launcher start; waiting for the SM46 runbook to end ($SM46_LOG)"
until grep -aEq '^\[sm46 [^]]*\] (done|STOP)' "$SM46_LOG" 2>/dev/null; do sleep 120; done
log "SM46: $(grep -aE '^\[sm46 [^]]*\] (done|STOP)' "$SM46_LOG" | tail -1)"
until [ ! -e "$HOLD" ]; do sleep 120; done
log "enqueue"
C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py run --owner small-model \
  --purpose "SM47: f16 row scales + int8 shift vectors (6 int-state variants) for SM36, 732 pool (GPU check first)" \
  --claim gpu --expected-minutes 115 \
  --launched-by "detached launcher sm47" -- bash C:/Users/jonat/workspace/wt/sm45/artifacts/small-model/sm47/job-sm47.sh
log "queue exit $?"
