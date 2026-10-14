#!/usr/bin/env bash
# First accuracy-loop measurements (2026-10-14): harness check, screen baseline x 2 seeds, full-run epoch-10 score.
# Every GPU step is a ticket in the main checkout's shared queue (FIFO), each under 60 minutes. The first ticket is
# registered at once (to keep a FIFO place behind the full run) but its command refuses to run (exit 64) unless the
# full run's epoch 10 is complete, so no GPU work starts while the full run is unfinished.
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
AL=C:/Users/jonat/workspace/wt/accuracy-loop/scripts/accuracy_loop.py
HERE=C:/Users/jonat/workspace/wt/accuracy-loop/artifacts/accuracy
RUN=C:/Users/jonat/workspace/anki-rwkv/artifacts/full-run/run-8k-20261013
log() { echo "[runbook $(date +%H:%M:%S)] $*"; }
q() { local purpose=$1; shift; "$PY" "$QUEUE" run --owner accuracy-loop --purpose "$purpose" --claim gpu -- "$@"; }
# Repeat a resumable job (exit 3 = time budget reached, more to do) as fresh tickets until done.
until_done() { local code=3; while [ $code -eq 3 ]; do q "$@"; code=$?; log "${*:2} exit $code"; done; return $code; }

log "start"
for attempt in 1 2 3; do
  q "accuracy-loop: eval harness check (12 users x 2 ckpts) + screen train w32 seeds 2025 and 2026 (2k users x 1 epoch each), ~40 min" \
    bash "$HERE/job-check-and-train.sh"
  code=$?
  log "check+train ticket exit $code"
  [ $code -eq 64 ] || break
  log "full run not complete when the ticket started; waiting for it before re-queueing"
  until grep -q '"status": "complete"' "$RUN/epoch-10/expertium/E89-F65e-F77a/result.json" 2>/dev/null; do sleep 60; done
done
[ $code -eq 0 ] || exit 1
"$PY" -X utf8 "$AL" check-eval check-e07 "$RUN/epoch-07/expertium/E89-F65e-F77a/result.json" | tail -20
"$PY" -X utf8 "$AL" check-eval check-e06 "$RUN/epoch-06/expertium/E89-F65e-F77a/result.json" | tail -20

until_done "accuracy-loop: screen eval w32 seeds 2025+2026 (1k validation users, shared pass, resumable, <=52 min)" \
  "$PY" -X utf8 "$AL" evaluate screen-w32-s2025 screen-w32-s2026 --max-minutes 52 || exit 1
until_done "accuracy-loop: score full-run epoch 10 on the 1k screen users (resumable, <=52 min)" \
  "$PY" -X utf8 "$AL" evaluate fullrun-e10 --max-minutes 52 || exit 1

for name in screen-w32-s2025 screen-w32-s2026 fullrun-e10; do
  "$PY" -X utf8 "$AL" collection "$name" --threads 2 2>&1 | grep -v -i warn
done
for name in screen-w32-s2025 screen-w32-s2026 fullrun-e10; do "$PY" -X utf8 "$AL" report "$name"; done
"$PY" -X utf8 "$AL" noise screen-w32-s2025 screen-w32-s2026
"$PY" -X utf8 "$AL" compare screen-w32-s2025 fullrun-e10
log "done"
