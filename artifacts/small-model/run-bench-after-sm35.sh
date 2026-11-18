#!/usr/bin/env bash
# S09 micro-benchmark (state-VQ QAT kernels, ~4-6 min GPU, no training) in the gap after SM35 + its scoring, before the
# 22:00 break; otherwise from 02:00. Waits until f2 has nothing queued (it then sits in wait_slot or is done).
LOG=/c/Users/jonat/workspace/wt/small-model/artifacts/small-model/runbook-2026-10-17f-evening-night.log
WT=/c/Users/jonat/workspace/wt/state-vq; PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
log() { echo "[bench $(date '+%m-%d %H:%M:%S')] $*"; }
until grep -aqE "exit [0-9]+: SM35 scoring|SM35: no decayed checkpoint" $LOG; do sleep 60; done
log "SM35 scoring finished; waiting for a 10-min slot before 22:00 or after 02:00"
while :; do
  now=$((10#$(date +%H) * 60 + 10#$(date +%M)))
  if [ "$now" -ge 120 ] && [ $((now + 10)) -le 1320 ]; then break; fi
  sleep 60
done
log "enqueue bench"
"$PY" C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py run --owner small-model --timing --claim gpu --expected-minutes 10 \
  --purpose "small-model: S09 state-VQ QAT micro-benchmark (kernel correctness + per-update timing, no training)" \
  -- "C:/Program Files/Git/usr/bin/bash.exe" -c "cd $WT && '$PY' -X utf8 scripts/check_state_vq_qat.py bench > artifacts/small-model/state-vq/qat/bench.log 2>&1"
log "bench exit $?"
