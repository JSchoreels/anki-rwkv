#!/usr/bin/env bash
# Wider simulator search with time-aware scheduling (SM86): CPU only, 6 processes, below-normal priority.
cd C:/Users/jonat/workspace/wt/sm84 || exit 1
LOG=artifacts/small-model/sm84/logs/sm86.log
echo "start $(date -Iseconds) commit $(git rev-parse --short HEAD)" >> $LOG
PYTHONPATH=. C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 scripts/sm86_search.py run --procs 6 >> $LOG 2>&1
echo "done (rc $?) $(date -Iseconds)" >> $LOG
