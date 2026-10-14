#!/usr/bin/env bash
# A08: score expertium-catch-up's epoch-10 decay checkpoint on the 1k screen users (fast) and on the collection.
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
AL=C:/Users/jonat/workspace/wt/accuracy-loop/scripts/accuracy_loop.py
echo "[runbook $(date +%H:%M:%S)] start"
"$PY" "$QUEUE" run --owner accuracy-loop --purpose "accuracy-loop: fast eval of the epoch-10 decay checkpoint (A08) on 1k screen users (~7 min)" --claim gpu -- \
  "$PY" -X utf8 "$AL" evaluate decay-e10-q25 --max-minutes 52 || exit 1
"$PY" -X utf8 "$AL" collection decay-e10-q25 --threads 2 2>&1 | grep -v -i warn
"$PY" -X utf8 "$AL" report decay-e10-q25
"$PY" -X utf8 "$AL" compare fullrun-e10 decay-e10-q25
echo "[runbook $(date +%H:%M:%S)] done"
