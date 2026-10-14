#!/usr/bin/env bash
# One queue ticket: refuse unless the full run is complete (exit 64), then the evaluation-harness check and both
# screen-baseline trainings. Each step is resumable; a rerun skips completed epochs and evaluated users.
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
AL=C:/Users/jonat/workspace/wt/accuracy-loop/scripts/accuracy_loop.py
RUN=C:/Users/jonat/workspace/anki-rwkv/artifacts/full-run/run-8k-20261013
grep -q '"status": "complete"' "$RUN/epoch-10/expertium/E89-F65e-F77a/result.json" 2>/dev/null || { echo "[job] full run not complete"; exit 64; }
echo "[job $(date +%H:%M:%S)] harness check"
"$PY" -X utf8 "$AL" evaluate check-e07 check-e06 --max-minutes 15
echo "[job $(date +%H:%M:%S)] check exit $?"
"$PY" -X utf8 "$AL" train screen-w32-s2025 || exit 1
"$PY" -X utf8 "$AL" train screen-w32-s2026 || exit 1
echo "[job $(date +%H:%M:%S)] done"
