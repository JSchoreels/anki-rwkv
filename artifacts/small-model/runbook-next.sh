#!/usr/bin/env bash
# Small-model next step (design: docs/SMALL_MODEL.md section 5, 2026-10-15). NOT launched by the planning session:
# each stage needs the user's go (GPU). Times are accuracy-lane estimates from logged rates, not benchmarks.
#
# usage (detached, from C:\Users\jonat\workspace\anki-rwkv; one stage per launch):
#   nohup "C:/Program Files/Git/usr/bin/bash.exe" C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/runbook-next.sh <stage> \
#     > C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/runbook-next-<stage>.log 2>&1 & disown
#
# Stages (calibration screen = paired k=3 0.25-epoch decays of scale-w64kd-gate epoch 3, decay order seeds 1-3):
#   check    (~10 min) GPU parity of the WKV recurrence beyond 383 steps (K 16/32, T up to 2,100) and a probe of one
#            32,768-review update as 256 x 128 vs 32 x 1,024 chunks (scripts/check_long_wkv.py). Gate for `context`.
#   control  (~1.9 h) completes fork/w8+s8 on the 34 long-history users cut by time limits (464-user frontier), scores
#            the existing scale-w64kd-gate epoch 1/2 checkpoints (5k-user epoch slope), then control decays seeds 2-3
#            (decay_alpha 0.5, 128-review chunks; seed 1 = existing scale-w64kd-dec) and their scoring.
#   context  (~2.7 h; after `check` passed) arm B, the recommended arm: the same decays at 1,024-review training chunks
#            (batch_chunks 32, same 32,768 reviews per update), seeds 1-3; then scoring.
#   kd09     (~1.6 h, optional, low expected value: section 5.1 window result) arm A: decay_alpha 0.9.
#   scale    cached-KD run of the screen winner: TIER=gate|final EPOCHS=<n> EPOCH1_MIN=<min> EPOCH_MIN=<min>
#            [RUN_TAG=<tag>] [DECAY_ALPHA=0.5] [EXTRA='<json merged into the options>', e.g. '{"chunk_reviews": 1024,
#            "batch_chunks": 32}'] [CONTINUE=1]. Trains epoch 1 and scores it (slope point), then STOPS unless
#            CONTINUE=1; re-launch with CONTINUE=1 to resume epochs 2..n, the 0.25-epoch decay and scoring.
#
# Every GPU step is one queue ticket (`--claim gpu`, absolute paths; docs/R01_PARALLEL_SESSIONS.md); CPU steps
# (prepare) run outside the queue and claim nothing. Every training ticket starts with a 60-120 s smoke of the same
# options (`expertium_decay.py smoke`: 2 users x 512 reviews, scratch copy, no teacher cache) and only trains if it
# passes. Re-launching a stage resumes: finished decays/epochs are skipped, scoring resumes per (user, label).
set -u
STAGE=${1:?stage check, control, context, kd09 or scale}
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
BASH_EXE="C:/Program Files/Git/usr/bin/bash.exe"
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/small-model
AL=$WT/scripts/accuracy_loop.py
DEC_PY=$WT/scripts/expertium_decay.py
SC=$WT/scripts/small_model_heldout.py
SRC=$WT/artifacts/accuracy/experiments/scale-w64kd-gate/run   # epoch-03 = the scale-up's last constant-LR epoch
SM=$WT/artifacts/small-model
CKPT=expertium/E89-F65e-F77a/master.pth
TEACHER=C:/Users/jonat/workspace/anki/qt/aqt/rwkv_inference/RWKV_trained_on_5000_10000.pth
# Teacher cache (bf16 fork teacher, training_seed 2025): main checkout, gitignored, survives the worktree. ~3.7 GB for
# the 5k gate users, ~5.8 GB for 8k. Sub-folders per batch-relevant options, so 128- and 1,024-review chunks never mix.
CACHE=C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-teacher/fork-bf16-s2025
OUT=$SM/heldout.jsonl  # absolute; the complete file lives in the worktree (resume reads it)
CTX='"chunk_reviews": 1024, "batch_chunks": 32'
log() { echo "[next $(date +%H:%M:%S)] $*"; }
q() {
  local minutes=$1 purpose=$2; shift 3
  log "enqueue: $purpose (expected $minutes min)"
  "$PY" "$QUEUE" run --owner small-model --purpose "small-model: $purpose" --claim gpu --expected-minutes "$minutes" -- "$@"
  local rc=$?; log "exit $rc: $purpose"; return $rc
}
kd() {  # distillation options with the cache; $1 = decay_alpha
  echo "{\"teacher\": \"$TEACHER\", \"alpha\": 0.9, \"decay_alpha\": $1, \"cache\": \"$CACHE\"}"
}
# decay NAME SEED DECAY_ALPHA MINUTES [EXTRA_JSON_FIELDS]: 0.25-epoch one_minus_sin decay of $SRC epoch 3.
decay() {
  local name=$1 seed=$2 alpha=$3 minutes=$4 extra=${5:+, $5}
  local out=$SM/$name
  [ -f "$out/$CKPT" ] && { log "$name done, skipped"; return 0; }
  if [ ! -d "$out" ]; then
    "$PY" -X utf8 "$DEC_PY" prepare --source-run "$SRC" --epoch 3 --out "$out" --fraction 0.25 --shape one_minus_sin \
      --evaluate none --seed "$seed" --extra-options "{\"distillation\": $(kd "$alpha")$extra}" || return 1
  fi
  q "$minutes" "calibration screen: decay $name (smoke first)" -- "$BASH_EXE" -c \
    "'$PY' -X utf8 '$DEC_PY' smoke --out '$out' && '$PY' -X utf8 '$DEC_PY' run --out '$out'"
  [ -f "$out/$CKPT" ] || { log "no checkpoint for $name"; return 1; }
}
score() {  # score LABEL... (w8+s8 only: int8 is free for w64, measured twice); resume ticket for time-limit cuts
  local models="" only="" n=0
  for name in "$@"; do
    local D=$SM/$name/$CKPT
    [ -f "$D" ] && { models="$models --model $name=expertium:w64:$D"; only="$only --only $name/w8+s8"; n=$((n + 1)); }
  done
  [ -n "$models" ] || { log "nothing to score"; return 1; }
  # about 11 min per label on 464 users (000353: 3 labels in 32 min)
  # shellcheck disable=SC2086
  q $(( 11 * n + 5 )) "full-history scoring: $*" -- "$PY" -X utf8 "$SC" $models $only --time-limit 3000 --out "$OUT"
  # shellcheck disable=SC2086
  q 15 "full-history scoring resume: $*" -- "$PY" -X utf8 "$SC" $models $only --time-limit 1500 --out "$OUT"
}
cd "$WT" || exit 1
case "$STAGE" in
check)
  q 10 "long-context WKV parity (T 512-2100, K 16/32) + 32x1024 vs 256x128 update probe" -- \
    "$PY" -X utf8 "$WT/scripts/check_long_wkv.py" --output "$SM/check-long-wkv.json"
  ;;
control)
  E=$SRC/epoch-0
  # shellcheck disable=SC2086
  q 40 "complete fork on 464 users; score scale-w64kd-gate epochs 1-2 (slope)" -- "$PY" -X utf8 "$SC" \
    --model "fork=fork::$TEACHER" --model "scale-w64kd-dec=expertium:w64:$SM/scale-w64kd-gate-decay/$CKPT" \
    --model "scale-w64kd-e1=expertium:w64:${E}1/$CKPT" --model "scale-w64kd-e2=expertium:w64:${E}2/$CKPT" \
    --only fork/w8+s8 --only scale-w64kd-dec/w8+s8 --only scale-w64kd-e1/w8+s8 --only scale-w64kd-e2/w8+s8 \
    --time-limit 3000 --out "$OUT"
  decay cal-ctl-s2 2 0.5 24 || exit 1
  decay cal-ctl-s3 3 0.5 24 || exit 1
  score cal-ctl-s2 cal-ctl-s3
  ;;
kd09)
  for seed in 1 2 3; do decay "cal-kd09-s$seed" "$seed" 0.9 24 || exit 1; done
  score cal-kd09-s1 cal-kd09-s2 cal-kd09-s3
  ;;
context)
  "$PY" -c "import json, sys; r = json.load(open(r'$SM/check-long-wkv.json')); sys.exit(1 if r['failed'] or 'probe' not in r['parts'] else 0)" 2>/dev/null || {
    log "run the 'check' stage first and make sure it passed ($SM/check-long-wkv.json); stop"; exit 2; }
  for seed in 1 2 3; do
    decay "cal-ctx1024-s$seed" "$seed" 0.5 40 "$CTX" || exit 1
  done
  score cal-ctx1024-s1 cal-ctx1024-s2 cal-ctx1024-s3
  ;;
scale)
  TIER=${TIER:?gate or final}; EPOCHS=${EPOCHS:?number of epochs}
  EPOCH1_MIN=${EPOCH1_MIN:?first-epoch minutes}; EPOCH_MIN=${EPOCH_MIN:?later-epoch minutes}
  DECAY_ALPHA=${DECAY_ALPHA:-0.5}; EXTRA=${EXTRA:-'{}'}; CONTINUE=${CONTINUE:-0}
  SUFFIX=${RUN_TAG:+-$RUN_TAG}
  NAME=scale-w64kd-$TIER-cached$SUFFIX
  EXP=$WT/artifacts/accuracy/experiments/$NAME
  DEC=$SM/$NAME-decay
  LBL=s-w64kd-$TIER$SUFFIX
  OPTS=$("$PY" -c "
import json
o = json.load(open(r'$WT/artifacts/accuracy/experiments/screen-w64kd-s2025/experiment.json'))['option_overrides']
o['distillation'] = json.loads('''$(kd "$DECAY_ALPHA")''')
o.update(json.loads('''$EXTRA'''))
print(json.dumps(o))")
  if [ ! -f "$EXP/experiment.json" ]; then
    "$PY" -X utf8 "$AL" prepare "$NAME" --tier "$TIER" --seed 2025 --epochs "$EPOCHS" --options "$OPTS" \
      --note "small-model cached-KD run: w64kd, $TIER tier, $EPOCHS epochs + 0.25-epoch decay (docs/SMALL_MODEL.md 5)" || exit 1
  else
    log "resuming existing $NAME"
  fi
  # Smoke source: a scratch decay of $SRC epoch 3 with this run's options (CPU prepare, re-made every launch).
  SMOKE=$SM/smoke-prep-$NAME
  rm -rf "$SMOKE"
  "$PY" -X utf8 "$DEC_PY" prepare --source-run "$SRC" --epoch 3 --out "$SMOKE" --fraction 0.01 --evaluate none --seed 1 \
    --extra-options "$OPTS" > /dev/null || exit 1
  for epoch in $(seq 1 "$EPOCHS"); do
    EP=$EXP/run/epoch-$(printf "%02d" "$epoch")/$CKPT
    if [ "$epoch" -eq 2 ] && [ "$CONTINUE" != 1 ]; then
      log "epoch 1 trained and scored ($LBL-e1); inspect the slope, then re-launch with CONTINUE=1"; exit 0
    fi
    if [ ! -f "$EP" ]; then
      m=$EPOCH_MIN; [ "$epoch" -eq 1 ] && m=$EPOCH1_MIN
      q $(( m + 2 )) "cached-KD run epoch $epoch/$EPOCHS ($TIER, smoke first)" -- "$BASH_EXE" -c \
        "'$PY' -X utf8 '$DEC_PY' smoke --out '$SMOKE' && '$PY' -X utf8 '$AL' train '$NAME' --epochs-per-job 1"
      rc=$?
      [ $rc -eq 0 ] || [ $rc -eq 3 ] || { log "training failed (exit $rc), stop"; exit 1; }
    fi
    if [ "$epoch" -eq 1 ]; then
      q 16 "cached-KD run: score epoch 1 (slope point)" -- "$PY" -X utf8 "$SC" --model "$LBL-e1=expertium:w64:$EP" \
        --only "$LBL-e1/w8+s8" --time-limit 1500 --out "$OUT"
    fi
  done
  EL=$(printf "%02d" "$EPOCHS")
  EN=$EXP/run/epoch-$EL/$CKPT
  [ -f "$EN" ] || { log "no epoch-$EL checkpoint"; exit 1; }
  D=$DEC/$CKPT
  if [ -d "$DEC" ] || "$PY" -X utf8 "$DEC_PY" prepare --source-run "$EXP/run" --epoch "$EPOCHS" --out "$DEC" \
      --fraction 0.25 --shape one_minus_sin --evaluate none --seed 1; then
    [ -f "$D" ] || q $(( EPOCH_MIN / 3 + 7 )) "cached-KD run: 0.25-epoch decay (smoke first)" -- "$BASH_EXE" -c \
      "'$PY' -X utf8 '$DEC_PY' smoke --out '$DEC' && '$PY' -X utf8 '$DEC_PY' run --out '$DEC'"
  fi
  MODELS="--model $LBL-e$EPOCHS=expertium:w64:$EN"; LABELS="--only $LBL-e$EPOCHS/w8+s8"
  [ -f "$D" ] && { MODELS="$MODELS --model $LBL-dec=expertium:w64:$D"; LABELS="--only $LBL-dec/w8+s8 $LABELS"; }
  # shellcheck disable=SC2086
  q 30 "cached-KD run: full-history scoring" -- "$PY" -X utf8 "$SC" $MODELS $LABELS --time-limit 2400 --out "$OUT"
  # shellcheck disable=SC2086
  q 20 "cached-KD run: scoring resume" -- "$PY" -X utf8 "$SC" $MODELS $LABELS --time-limit 1200 --out "$OUT"
  ;;
*) echo "unknown stage $STAGE"; exit 2 ;;
esac
log "done: $STAGE"
