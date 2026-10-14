#!/usr/bin/env bash
# Small-model track, night of 2026-10-15 (docs/SMALL_MODEL.md, artifacts/small-model/HANDOFF.md).
# Paired k=3 screens (2,000 users, 1 epoch, order seeds 0/1/2) of w64 and w64 + KD from the fork, then full-history
# scoring of the fork (fp32, int8 weights + int8 state) and our checkpoints on the 464 fork-unseen screen users.
# Each step is its own queue ticket (< 60 min); a failed training step does not stop the chain, scoring skips
# checkpoints that do not exist. Launched detached from C:\Users\jonat\workspace\anki-rwkv.
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/small-model
AL=$WT/scripts/accuracy_loop.py
SC=$WT/scripts/small_model_heldout.py
EXP=$WT/artifacts/accuracy/experiments
ALW=C:/Users/jonat/workspace/wt/accuracy-loop/artifacts/accuracy/experiments
CK=run/epoch-01/expertium/E89-F65e-F77a/master.pth
FORK=C:/Users/jonat/workspace/anki/qt/aqt/rwkv_inference/RWKV_trained_on_5000_10000.pth
DECAY=C:/Users/jonat/workspace/wt/expertium-catch-up/artifacts/expertium-catch-up/decay-8k-e10-q25/expertium/E89-F65e-F77a/master.pth
E10=C:/Users/jonat/workspace/anki-rwkv/artifacts/full-run/run-8k-20261013/epoch-10/expertium/E89-F65e-F77a/master.pth
OUT=artifacts/small-model/heldout.jsonl

log() { echo "[runbook $(date +%H:%M:%S)] $*"; }
q() {  # q MINUTES PURPOSE -- command...
  local minutes=$1 purpose=$2; shift 3
  log "enqueue: $purpose (expected $minutes min)"
  if [ -n "${DRY:-}" ]; then echo "  CMD: $*"; return 0; fi
  "$PY" "$QUEUE" run --owner small-model --purpose "small-model: $purpose" --claim gpu --expected-minutes "$minutes" -- "$@"
  local rc=$?
  log "exit $rc: $purpose"
  return $rc
}
train() {  # train MINUTES NAME
  q "$1" "train $2 (screen tier, 1 epoch)" -- "$PY" -X utf8 "$AL" train "$2"
}
# Model specs (TAG=KIND:ARCH:PATH); only existing checkpoints are passed.
spec() { [ -f "$4" ] && echo "--model $1=$2:$3:$4"; }
only() { for label in "$@"; do echo "--only $label"; done; }
score() {  # score MINUTES LIMIT_SECONDS PURPOSE LABEL...
  local minutes=$1 limit=$2 purpose=$3; shift 3
  local models labels=() tag
  models="$(spec fork fork - "$FORK") $(spec decay expertium iter36 "$DECAY") $(spec fullrun-e10 expertium iter36 "$E10")"
  models="$models $(spec w32-o0 expertium iter36 "$ALW/screen-w32-s2025/$CK") $(spec w32-o1 expertium iter36 "$ALW/screen-w32-s2025-o1/$CK") $(spec w32-o2 expertium iter36 "$ALW/screen-w32-s2025-o2/$CK")"
  models="$models $(spec w64-o0 expertium w64 "$EXP/screen-w64-s2025/$CK") $(spec w64-o1 expertium w64 "$EXP/screen-w64-s2025-o1/$CK") $(spec w64-o2 expertium w64 "$EXP/screen-w64-s2025-o2/$CK")"
  models="$models $(spec w64kd-o0 expertium w64 "$EXP/screen-w64kd-s2025/$CK") $(spec w64kd-o1 expertium w64 "$EXP/screen-w64kd-s2025-o1/$CK") $(spec w64kd-o2 expertium w64 "$EXP/screen-w64kd-s2025-o2/$CK")"
  for label in "$@"; do
    tag=${label%%/*}
    case " $models " in *" --model $tag="*) labels+=("$label") ;; *) log "skip $label: no checkpoint" ;; esac
  done
  [ ${#labels[@]} -gt 0 ] || { log "nothing to score for: $purpose"; return 0; }
  # shellcheck disable=SC2046,SC2086
  q "$minutes" "$purpose" -- "$PY" -X utf8 "$SC" $models $(only "${labels[@]}") --time-limit "$limit" --out "$OUT"
}

log "start"
train 28 screen-w64-s2025
train 36 screen-w64kd-s2025
score 50 2700 "full-history held-out: fork fp32 and int8 weights+int8 state, 464 fork-unseen users" fork/fp32 fork/w8+s8
score 55 3000 "full-history held-out: w32 decay/e10/screen-o0, w64-o0, w64kd-o0" \
  decay/fp32 decay/w8+s8 fullrun-e10/fp32 w32-o0/fp32 w64-o0/fp32 w64kd-o0/fp32
train 28 screen-w64-s2025-o1
train 36 screen-w64kd-s2025-o1
train 28 screen-w64-s2025-o2
train 36 screen-w64kd-s2025-o2
score 55 3000 "full-history held-out: order seeds 1/2 of w32, w64, w64kd" \
  w32-o1/fp32 w32-o2/fp32 w64-o1/fp32 w64-o2/fp32 w64kd-o1/fp32 w64kd-o2/fp32
score 50 2700 "full-history held-out: resume truncated labels + int8 of w64/w64kd o0" \
  fork/fp32 fork/w8+s8 decay/fp32 decay/w8+s8 fullrun-e10/fp32 w32-o0/fp32 w64-o0/fp32 w64kd-o0/fp32 \
  w32-o1/fp32 w32-o2/fp32 w64-o1/fp32 w64-o2/fp32 w64kd-o1/fp32 w64kd-o2/fp32 w64-o0/w8+s8 w64kd-o0/w8+s8
log "done"
