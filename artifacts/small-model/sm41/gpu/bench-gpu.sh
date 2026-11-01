#!/usr/bin/env bash
# The small model in the review simulator (SM41): one short GPU ticket, the batched-step throughput benchmark.
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/sm41
O=$WT/artifacts/small-model/sm41
cd "$WT"
"$PY" "$QUEUE" run --owner small-model --purpose "the small model in the review simulator (SM41): batched recurrent-step throughput benchmark (random states, B = 1 to 16,384 learners, ~3 min)" \
  --claim gpu --expected-minutes 5 --max-wait-minutes 120 --launched-by "sm41 artifacts/small-model/sm41/gpu/bench-gpu.sh" -- \
  "$PY" -X utf8 $WT/scripts/sm41_sim.py bench --device cuda --batch 1,64,1024,4096,16384 --reps 20 --out $O/bench.json
echo "exit $? $(date +%H:%M:%S)"
