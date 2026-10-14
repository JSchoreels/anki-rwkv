#!/usr/bin/env bash
# Queue job (wide-kernels): short R01 speed runs, same 160 users as artifacts/expertium-catch-up/speed-20261014.
# The queue runs commands from the main checkout; this cds into the worktree. Finished cases are skipped.
# Stops starting new cases after BUDGET_MINUTES (default 26) so the ticket fits its --expected-minutes.
cd /c/Users/jonat/workspace/wt/wide-kernels || exit 1
PY=/c/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
OUT=artifacts/wide-kernels/speed-20261014
CASES="${CASES:-w32 w80-b256-wide w64-b256-wide w80-b256 w64-b256 w80-sequential-b256-wide w64-wide}"
BUDGET_MINUTES="${BUDGET_MINUTES:-26}"
START=$(date +%s)
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >> "$OUT/all.log" 2>&1
for CASE in $CASES; do
  if [ -e "$OUT/$CASE" ]; then continue; fi
  ELAPSED=$(( ($(date +%s) - START) / 60 ))
  if [ "$ELAPSED" -ge "$BUDGET_MINUTES" ]; then echo "budget reached at ${ELAPSED} min; not started: $CASE" >> "$OUT/all.log"; continue; fi
  "$PY" -X utf8 scripts/expertium_speed.py case --out "$OUT" --case "$CASE"
  echo "$CASE exit $? after $(( ($(date +%s) - START) / 60 )) min" >> "$OUT/all.log"
done
"$PY" -X utf8 scripts/expertium_speed.py report --out "$OUT"
