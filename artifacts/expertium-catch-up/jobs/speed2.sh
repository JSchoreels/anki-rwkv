#!/usr/bin/env bash
# Queue job: GPU parity checks + short R01 speed runs (cwd must be the worktree).
set -e
cd /c/Users/jonat/workspace/wt/expertium-catch-up
PY=/c/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
CASES="${CASES:-w32,w64-sequential,w80-sequential-b256,w32-generic-b256,w80-b256}"
exec "$PY" -X utf8 scripts/expertium_speed.py all --out artifacts/expertium-catch-up/speed-20261014 --budget-minutes 36 --cases "$CASES"
