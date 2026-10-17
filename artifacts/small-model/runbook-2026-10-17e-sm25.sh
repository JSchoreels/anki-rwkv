#!/usr/bin/env bash
# SM25 on all 464 users (after SM24, runbook d): the fork with no-steps review-type labels (feature-screen/sm25.py),
# scores only (sm25-scores.jsonl, heldout.jsonl untouched), up to 2 passes; then feature-screen/sm25_464.py.
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/small-model
SM=$WT/artifacts/small-model
TEACHER=C:/Users/jonat/workspace/anki/qt/aqt/rwkv_inference/RWKV_trained_on_5000_10000.pth
until grep -q "\[sm24 .*\] done" $SM/runbook-2026-10-17d-sm24.log; do sleep 120; done
for pass in 1 2; do
  "$PY" "$QUEUE" run --owner small-model --purpose "small-model: SM25 fork with no-steps review-type labels, 464 users (pass $pass)" \
    --claim gpu --expected-minutes 40 -- "$PY" -X utf8 $SM/feature-screen/sm25.py --model forknosteps=fork::$TEACHER \
    --only forknosteps/w8+s8 --time-limit 3300 --out $SM/sm25-scores.jsonl
  n=$(cat $SM/sm25-scores.jsonl 2>/dev/null | wc -l); echo "[sm25 $(date +%H:%M:%S)] rows after pass $pass: $n"
  [ "$n" -ge 464 ] && break
done
"$PY" -X utf8 $SM/feature-screen/sm25_464.py > $SM/feature-screen/sm25-464.txt 2>&1
echo "[sm25 $(date +%H:%M:%S)] done"
