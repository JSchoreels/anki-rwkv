#!/usr/bin/env bash
# SM43 full (data diversity vs epochs at equal reviews seen, with LR decay), 10-20.
#   A: 1,250 gate users x (10 constant-LR + 1 linear-decay epoch)    B: 125 of them x (100 + 10 decay epochs)
#   then both scored on the 732 pool (w8+s8, two passes like the mini) and paired on CPU (sm43_full_summary.py).
# Before each GPU ticket: wait while artifacts/small-model/HOLD-chain exists (GPU break guard). A failed training
# ticket is retried once (accuracy_loop train resumes at the first incomplete epoch).
# Launch: nohup bash C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/sm43/launch-sm43-full.sh \
#           > C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/sm43/launch-sm43-full.log 2>&1 & disown
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/small-model
LOOP=$WT/scripts/accuracy_loop.py
SC_POOL=$WT/scripts/small_model_heldout.py
EXP=$WT/artifacts/accuracy/experiments
OUTD=$WT/artifacts/small-model/sm43
HOLD=$WT/artifacts/small-model/HOLD-chain
CKPT=expertium/E89-F65e-F77a/master.pth
A=sm43full-a-1250u-11ep; B=sm43full-b-125u-110ep
LA=s-w64kd-$A; LB=s-w64kd-$B
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION  # scorer default 0.8; training options carry gpu_memory_fraction 0.8
log() { echo "[sm43full $(date '+%m-%d %H:%M:%S')] $*"; }
hold() {  # wait while the break guard holds the chain
  if [ -e "$HOLD" ]; then
    log "HOLD-chain present: waiting before $1"
    while [ -e "$HOLD" ]; do sleep 120; done
    log "HOLD-chain gone: $1"
  fi
}
q() {  # q MINUTES PURPOSE -- CMD...
  local minutes=$1 purpose=$2; shift 3
  hold "$purpose"
  log "enqueue: $purpose (expected $minutes min)"
  "$PY" "$QUEUE" run --owner small-model --purpose "$purpose" --claim gpu --expected-minutes "$minutes" -- "$@"
  local rc=$?; log "exit $rc: $purpose"; return $rc
}
train() {  # train MINUTES PURPOSE NAME: one ticket, one retry from the first incomplete epoch
  q "$1" "$2" -- "$PY" -X utf8 "$LOOP" train "$3" && return 0
  log "$3 failed once: retrying from the first incomplete epoch"
  q "$1" "$2 (retry)" -- "$PY" -X utf8 "$LOOP" train "$3"
}
cd "$WT" || exit 1
log "start: A=$A B=$B"
train 180 "SM43 full A: 1,250 users x (10 + 1 decay) epochs (2,048 KD recipe)" "$A" || { log "A failed"; exit 1; }
train 210 "SM43 full B: 125 users x (100 + 10 decay) epochs (same reviews seen)" "$B" || { log "B failed"; exit 1; }
for pass in 1 2; do
  q $((pass == 1 ? 30 : 10)) "SM43 full scoring A and B, 732 pool (pass $pass)" -- "$PY" -X utf8 "$SC_POOL" --pool 732 \
    --model "$LA=expertium:w64:$EXP/$A/run/epoch-11/$CKPT" --model "$LB=expertium:w64:$EXP/$B/run/epoch-110/$CKPT" \
    --only "$LA/w8+s8" --only "$LB/w8+s8" --time-limit 3000 --out "$OUTD/heldout-sm43full-pools.jsonl" || log "scoring pass $pass exit $?"
done
log "pairing (CPU): sm43_full_summary.py"
"$PY" -X utf8 "$OUTD/sm43_full_summary.py" || log "summary exit $?"
log "done"
