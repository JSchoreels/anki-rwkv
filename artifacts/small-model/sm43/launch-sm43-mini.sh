#!/usr/bin/env bash
# SM43 mini (data diversity vs epochs, equal reviews seen), after SM42, respecting the 21:00-01:00 GPU break.
#   1. wait for SM42's end line in the sm42/sm36 launcher log;
#   2. each step (train A, train B, score both on the 732 pool) runs now only if its expected end is <= 20:55;
#      otherwise it waits until the break guard's HOLD-chain file is gone (01:00) and then runs.
# Launch: nohup bash C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/sm43/launch-sm43-mini.sh \
#           > C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/sm43/launch-sm43-mini.log 2>&1 & disown
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/small-model
LOOP=$WT/scripts/accuracy_loop.py
SC_POOL=$WT/scripts/small_model_heldout.py
EXP=$WT/artifacts/accuracy/experiments
OUTD=$WT/artifacts/small-model/sm43
SM42_LOG=C:/Users/jonat/workspace/wt/sm42/artifacts/small-model/launch-sm42-sm36-after-prune.log
HOLD=$WT/artifacts/small-model/HOLD-chain
CKPT=expertium/E89-F65e-F77a/master.pth
A=sm43a-1000u-1ep; B=sm43b-100u-10ep
LA=s-w64kd-sm43a-1000u-1ep; LB=s-w64kd-sm43b-100u-10ep
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION
log() { echo "[sm43 $(date '+%m-%d %H:%M:%S')] $*"; }
q() {  # q MINUTES PURPOSE -- CMD...
  local minutes=$1 purpose=$2; shift 3
  log "enqueue: $purpose (expected $minutes min)"
  "$PY" "$QUEUE" run --owner small-model --purpose "$purpose" --claim gpu --expected-minutes "$minutes" -- "$@"
  local rc=$?; log "exit $rc: $purpose"; return $rc
}
fits() {  # fits MINUTES: now + MINUTES <= 20:55 today (and not inside the break)
  local now end; now=$(date +%s); end=$(date -d "today 20:55" +%s)
  [ $((now + $1 * 60)) -le "$end" ] && [ "$(date +%H)" -ge 6 ]
}
slot() {  # slot MINUTES STEP: run now if it fits before the break, else wait for the break guard to lift HOLD-chain
  if fits "$1"; then return 0; fi
  log "$2 ($1 min) does not fit before 20:55: waiting for the end of the break (HOLD-chain gone)"
  until [ ! -e "$HOLD" ] && [ "$(date +%H)" -ge 1 ] && [ "$(date +%H)" -lt 20 ]; do sleep 120; done
  log "break over: $2"
}
log "waiting for SM42's end line in $SM42_LOG"
until grep -aEq 'ARM=sm42 (finished OK|ended with exit)' "$SM42_LOG" 2>/dev/null; do sleep 60; done
log "SM42 done: $(grep -aE 'ARM=sm42 (finished OK|ended with exit)' "$SM42_LOG" | tail -1)"
cd "$WT" || exit 1
slot 16 "train A" && q 16 "SM43 mini A: 1,000 users x 1 epoch (2,048 KD recipe)" -- "$PY" -X utf8 "$LOOP" train "$A" || { log "A failed"; exit 1; }
slot 20 "train B" && q 20 "SM43 mini B: 100 users x 10 epochs (same reviews seen)" -- "$PY" -X utf8 "$LOOP" train "$B" || { log "B failed"; exit 1; }
for pass in 1 2; do
  slot $((pass == 1 ? 30 : 10)) "scoring pass $pass" && \
  q $((pass == 1 ? 30 : 10)) "SM43 mini scoring A and B, 732 pool (pass $pass)" -- "$PY" -X utf8 "$SC_POOL" --pool 732 \
    --model "$LA=expertium:w64:$EXP/$A/run/epoch-01/$CKPT" --model "$LB=expertium:w64:$EXP/$B/run/epoch-10/$CKPT" \
    --only "$LA/w8+s8" --only "$LB/w8+s8" --time-limit 3000 --out "$OUTD/heldout-sm43-pools.jsonl" || log "scoring pass $pass exit $?"
done
log "done"
