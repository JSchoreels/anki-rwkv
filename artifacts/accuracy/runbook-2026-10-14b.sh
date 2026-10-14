#!/usr/bin/env bash
# A15 order-noise screens (seed 2025, order seeds 1 and 2) and a third training seed (2027), then one fast shared
# evaluation. Three queue tickets, each under 60 minutes.
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
AL=C:/Users/jonat/workspace/wt/accuracy-loop/scripts/accuracy_loop.py
log() { echo "[runbook $(date +%H:%M:%S)] $*"; }
q() { local purpose=$1; shift; "$PY" "$QUEUE" run --owner accuracy-loop --purpose "$purpose" --claim gpu -- "$@"; }

log "start"
q "accuracy-loop: A15 order-noise screen trainings (seed 2025, order seeds 1 and 2; 2 x 13 min)" \
  bash -c "'$PY' -X utf8 '$AL' train screen-w32-s2025-o1 && '$PY' -X utf8 '$AL' train screen-w32-s2025-o2" || exit 1
log "order-noise trainings done"
q "accuracy-loop: third-seed screen training (seed 2027; 13 min)" "$PY" -X utf8 "$AL" train screen-w32-s2027 || exit 1
log "third seed done"
q "accuracy-loop: fast eval of 3 screen checkpoints on 1k validation users (~19 min)" \
  "$PY" -X utf8 "$AL" evaluate screen-w32-s2025-o1 screen-w32-s2025-o2 screen-w32-s2027 --max-minutes 52 || exit 1
for name in screen-w32-s2025-o1 screen-w32-s2025-o2 screen-w32-s2027; do
  "$PY" -X utf8 "$AL" collection "$name" --threads 2 2>&1 | grep -v -i warn
done
for name in screen-w32-s2025-o1 screen-w32-s2025-o2 screen-w32-s2027; do "$PY" -X utf8 "$AL" report "$name"; done
"$PY" -X utf8 "$AL" noise screen-w32-s2025 screen-w32-s2026 screen-w32-s2027
"$PY" -X utf8 "$AL" noise screen-w32-s2025 screen-w32-s2025-o1 screen-w32-s2025-o2
log "done"
