#!/usr/bin/env bash
# Small-model scale-up (user, 2026-10-15: "if that model seems promising, start training it on more users/epochs;
# budget 6h"). NOT launched by the planning agent: the main session launches it only when the HANDOFF threshold is met.
# usage (detached, from C:\Users\jonat\workspace\anki-rwkv):
#   EPOCH_MIN=<per-epoch minutes, <= 57> nohup "C:/Program Files/Git/usr/bin/bash.exe" \
#     C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/runbook-scaleup.sh <w64|w64kd> \
#     > C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/runbook-scaleup.log 2>&1 & disown
# Gate tier: 5,000 training users (contains the screen's 2,000), 369,477,375 reviews per epoch, 4 epochs at constant
# LR, then a 0.25-epoch one_minus_sin decay (A08), then full-history scoring on the 464 fork-unseen users.
# Budget: 4 x EPOCH_MIN (<= 228) + decay 35 + scoring 55 + 40 = <= 358 min.
# Outputs (all kept; nothing here deletes or overwrites a finished checkpoint):
#   every epoch   $EXP/run/epoch-0N/expertium/E89-F65e-F77a/{master,optimizer}.pth (a retried epoch moves the old
#                 folder to epoch-0N-attempt-K first; accuracy_loop.train)
#   decayed final $DEC/expertium/E89-F65e-F77a/master.pth
# Re-launching with the same recipe RESUMES (finished epochs and a finished decay are skipped). For a separate,
# fresh scale-up of the same recipe set RUN_TAG=<tag>: experiment, decay folder and held-out labels all get -<tag>,
# so earlier checkpoints and heldout.jsonl rows are never reused or shadowed.
# After it ends: package the decayed checkpoint (scripts/small_model_export.py package; HANDOFF follow-ups).
set -u
R=${1:?recipe w64 or w64kd}
EPOCH_MIN=${EPOCH_MIN:?set EPOCH_MIN = 369477375 / screen reviews_per_second / 60 + 4 (must be <= 57)}
[ "$EPOCH_MIN" -le 57 ] || { echo "EPOCH_MIN $EPOCH_MIN > 57: an epoch does not fit one ticket; see HANDOFF fallback"; exit 2; }
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/small-model
AL=$WT/scripts/accuracy_loop.py
SC=$WT/scripts/small_model_heldout.py
SUFFIX=${RUN_TAG:+-$RUN_TAG}
NAME=scale-$R-gate$SUFFIX
LBL=scale-$R$SUFFIX
EXP=$WT/artifacts/accuracy/experiments/$NAME
DEC=$WT/artifacts/small-model/$NAME-decay
OUT=C:/Users/jonat/workspace/anki-rwkv/artifacts/small-model/heldout.jsonl  # absolute: the queue runs from the main checkout (same file as runbook-2026-10-15.sh)
log() { echo "[scaleup $(date +%H:%M:%S)] $*"; }
q() {
  local minutes=$1 purpose=$2; shift 3
  log "enqueue: $purpose (expected $minutes min)"
  "$PY" "$QUEUE" run --owner small-model --purpose "small-model: $purpose" --claim gpu --expected-minutes "$minutes" -- "$@"
  local rc=$?; log "exit $rc: $purpose"; return $rc
}
cd "$WT" || exit 1
[ -f "$EXP/experiment.json" ] && log "resuming existing $NAME (set RUN_TAG for a fresh run)"
if [ ! -f "$EXP/experiment.json" ]; then
  OPTS=$("$PY" -c "import json;print(json.dumps(json.load(open(r'$WT/artifacts/accuracy/experiments/screen-$R-s2025/experiment.json'))['option_overrides']))")
  "$PY" -X utf8 "$AL" prepare "$NAME" --tier gate --seed 2025 --epochs 4 --options "$OPTS" \
    --note "small-model scale-up of $R: gate tier 5,000 users, 4 epochs + 0.25-epoch decay (user budget 6 h)" || exit 1
fi
for epoch in 1 2 3 4; do
  q "$EPOCH_MIN" "scale-up $R epoch $epoch/4 (gate, 5,000 users)" -- "$PY" -X utf8 "$AL" train "$NAME" --epochs-per-job 1
  rc=$?
  [ $rc -eq 0 ] || [ $rc -eq 3 ] || { log "training failed (exit $rc), stop"; exit 1; }
done
E4=$EXP/run/epoch-04/expertium/E89-F65e-F77a/master.pth
[ -f "$E4" ] || { log "no epoch-4 checkpoint"; exit 1; }
MODELS="--model $LBL-e4=expertium:w64:$E4"
D=$DEC/expertium/E89-F65e-F77a/master.pth
# prepare refuses an existing folder, so a resumed runbook skips it; a missing decayed checkpoint is (re)run.
if [ -d "$DEC" ] || "$PY" -X utf8 scripts/expertium_decay.py prepare --source-run "$EXP/run" --epoch 4 --out "$DEC"      --fraction 0.25 --shape one_minus_sin --evaluate none --seed 1; then
  [ -f "$D" ] || q 35 "scale-up $R: 0.25-epoch decay from epoch 4" -- "$PY" -X utf8 "$WT/scripts/expertium_decay.py" run --out "$DEC"
fi
[ -f "$D" ] && MODELS="$MODELS --model $LBL-dec=expertium:w64:$D"
LABELS="--only $LBL-e4/fp32"
[ -f "$D" ] && LABELS="--only $LBL-dec/fp32 --only $LBL-dec/w8+s8 $LABELS"
# shellcheck disable=SC2086
q 55 "scale-up $R: full-history held-out scoring (464 users)" -- "$PY" -X utf8 "$SC" $MODELS $LABELS --time-limit 3000 --out "$OUT"
# shellcheck disable=SC2086
q 45 "scale-up $R: full-history scoring resume" -- "$PY" -X utf8 "$SC" $MODELS $LABELS --time-limit 2500 --out "$OUT"
log "done"
