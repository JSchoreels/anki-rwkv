#!/usr/bin/env bash
# Small GPU jobs while the full retrain (SM56) is paused after epoch 13 (user 10-24 ~14:50: stop after e13, wait for the user's word).
#   A: score epoch 13 on the 464 set (458 users) into heldout-sm56.jsonl (the launcher skips it on resume)
#   B: decay from epoch 10 on the 3 long users of the 732 pool at the 0.92 memory cap -> 726-user row
export PATH=/usr/bin:/mingw64/bin:$PATH
WT=C:/Users/jonat/workspace/wt/sm56; SM=$WT/artifacts/small-model; S=$SM/sm56; J=$S/smalljobs
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe; QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
SC=$WT/scripts/small_model_heldout.py; RUN=$WT/artifacts/accuracy/experiments/scale-w64kd-final-sc-lanes/run
CKPT=expertium/E89-F65e-F77a/master.pth; LOG=$J/run-smalljobs.log
log() { echo "[smalljobs $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
log "started; waiting for the SM56 launcher to stop after epoch 13"
until grep -q "STOP: STOP file present" <(sed -n '/epoch 12: curve/,$p' "$S/launch-sm56.log"); do
  [ -e "$J/CANCEL" ] && { log "CANCEL"; exit 1; }; sleep 60; done
[ -f "$RUN/epoch-13/$CKPT" ] || { log "STOP: no epoch-13 checkpoint"; exit 1; }
q() { local m=$1 p=$2; shift 3; log "enqueue: $p"
  "$PY" "$QUEUE" run --owner small-model --purpose "small-model: $p" --claim gpu --expected-minutes "$m" --max-wait-minutes 60 \
    --launched-by "detached runbook sm56/smalljobs/run-smalljobs.sh" -- "$@"; local rc=$?; log "exit $rc: $p"; return $rc; }
q 14 "SM56 scoring s-w64kd-final-sc-lanes-e13 on the 464 set (458 fit users), paused run" -- "$PY" -X utf8 "$SC" --pool 464 \
  --users-file "$S/eval-users-464-fit.json" --model "s-w64kd-final-sc-lanes-e13=expertium:w64:$RUN/epoch-13/$CKPT" \
  --only "s-w64kd-final-sc-lanes-e13/w8+s8" --time-limit 2400 --out "$SM/heldout-sm56.jsonl"
D=$SM/scale-w64kd-final-sc-lanes-dec10-1ep32k-muon; L=s-w64kd-final-sc-lanes-dec10-1ep32k-muon
q 15 "SM56 decay from e10 on the 3 long 732-pool users (650, 1601, 4501) at the 0.92 cap" -- env HELDOUT_GPU_MEMORY_FRACTION=0.92 "$PY" -X utf8 "$SC" --pool 732 \
  --users-file "$J/users-732-retry3.json" --model "$L=expertium:w64:$D/$CKPT" --only "$L/w8+s8" --time-limit 2400 --out "$SM/heldout-sm56-pools.jsonl"
log "done"
