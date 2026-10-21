#!/usr/bin/env bash
# SM48b launcher: one --timing queue ticket for job-sm48b.sh (checks the quiet windows via the queue itself).
log() { echo "[sm48b $(date '+%m-%d %H:%M:%S')] $*"; }
log "enqueue"
C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py run --owner small-model \
  --purpose "SM48b: --lanes 1/2/4 + lane review budgets with SM47 int state variants (no lock), 40 users of the 732 pool, SM36" \
  --claim gpu --timing --expected-minutes 35 \
  --launched-by "detached launcher sm48b" -- bash C:/Users/jonat/workspace/wt/sm48b/artifacts/small-model/sm48b/job-sm48b.sh
log "queue exit $?"
