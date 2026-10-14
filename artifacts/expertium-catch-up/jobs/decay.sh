#!/usr/bin/env bash
# Queue job: decay-only 0.25 epoch from the 8k run's epoch-10 checkpoint, scored on its 300 quick users.
set -e
cd /c/Users/jonat/workspace/wt/expertium-catch-up
PY=/c/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
exec "$PY" -X utf8 scripts/expertium_decay.py auto --source-run C:/Users/jonat/workspace/anki-rwkv/artifacts/full-run/run-8k-20261013 --epoch 10 --out artifacts/expertium-catch-up/decay-8k-e10-q25 --fraction 0.25 --shape one_minus_sin --evaluate quick
