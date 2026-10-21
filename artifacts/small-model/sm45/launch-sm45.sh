#!/usr/bin/env bash
# SM45 launcher: wait while the GPU-break HOLD-chain file exists, then one queue ticket for job-sm45.sh.
HOLD=C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/HOLD-chain
until [ ! -e "$HOLD" ]; do sleep 120; done
echo "[sm45 $(date '+%m-%d %H:%M:%S')] enqueue"
C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py run --owner small-model \
  --purpose "SM45: int4 card + int8 note state for SM36, 732 pool (GPU check first)" --claim gpu --expected-minutes 60 \
  --launched-by "detached launcher sm45" -- bash C:/Users/jonat/workspace/wt/sm45/artifacts/small-model/sm45/job-sm45.sh
echo "[sm45 $(date '+%m-%d %H:%M:%S')] queue exit $?"
