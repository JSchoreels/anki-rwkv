#!/usr/bin/env bash
# The small model in the review simulator (SM41) stage 2 part 2: engine validation, 8 users x 1 split x 60 days,
# 17 lanes (scripts/sm41_engine.py DEFAULT_LANES), 3 CPU processes x 1 thread (the proof is the 4th), each job holds in quiet.json windows.
# Resumable: a (user, split, days) row already in rows/ is skipped.
set -u
PY=/c/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=/c/Users/jonat/workspace/wt/sm41
OUT=$WT/artifacts/small-model/sm41/engine
mkdir -p "$OUT/rows" "$OUT/logs"
echo "start $(date '+%F %T') pid $$" >> "$OUT/runner.log"
worker() {
  for job in "$@"; do
    u=${job%%:*}; d=${job##*:}
    "$PY" -X utf8 "$WT/scripts/sm41_engine.py" run --user "$u" --split "$d" --days 60 \
      --out "$OUT/rows/$u-60.jsonl" >> "$OUT/logs/$u-60.log" 2>&1
  done
}
worker 2941:30 4813:135 1070:90 &
worker 2626:45 2207:210 &
worker 1380:90 89:150 2503:330 &
wait
"$PY" -X utf8 "$WT/scripts/sm41_engine.py" report > "$OUT/report.out" 2>&1
echo "done $(date '+%F %T')" >> "$OUT/runner.log"
