#!/usr/bin/env bash
# The small model in the review simulator (SM41) stage 2 part 2: engine CPU proof (P1-P3), 1 process x 1 thread.
PY=/c/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=/c/Users/jonat/workspace/wt/sm41
OUT=$WT/artifacts/small-model/sm41/engine
echo "proof start $(date '+%F %T')" >> "$OUT/runner.log"
"$PY" -X utf8 "$WT/scripts/sm41_engine.py" proof --days 20 --users 3876:225,2503:330 --p3 3876:225 \
  --out "$OUT/proof-cpu.json" > "$OUT/proof-cpu.out" 2>&1
echo "proof done $(date '+%F %T')" >> "$OUT/runner.log"
