#!/usr/bin/env bash
# SM49 launcher: wait while the HOLD-chain file exists or a quiet window (artifacts/coordination/queue/quiet.json, read
# by quiet_wait.py, which also copes with a malformed file) is active, then one queue ticket for job-sm49.sh (the queue itself also holds a ticket whose expected end reaches a window).
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
HOLD=C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/HOLD-chain
log() { echo "[sm49 $(date '+%m-%d %H:%M:%S')] $*"; }
in_quiet() { [ "$("$PY" -X utf8 C:/Users/jonat/workspace/wt/sm49/artifacts/small-model/sm49/quiet_wait.py 0)" != 0 ]; }
log "launcher start"
while [ -e "$HOLD" ] || in_quiet; do sleep 120; done
log "enqueue"
"$PY" C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py run --owner small-model \
  --purpose "SM49: int4 notes, int4 shift vectors, per-head scales (4 int-state variants) for SM36, 732 pool (GPU check first)" \
  --claim gpu --expected-minutes 60 \
  --launched-by "detached launcher sm49" -- bash C:/Users/jonat/workspace/wt/sm49/artifacts/small-model/sm49/job-sm49.sh
log "queue exit $?"
