#!/usr/bin/env bash
# SM40 (2026-10-18): one-epoch final decay at 32k from e10 (pairs with dec10, the quarter-epoch decay from the same e10),
# then 464-user scoring. Motivation: SM35 at 2,048 (1 epoch vs 0.25) gained -0.00033 curve / -0.00019 instant.
# Estimate: dec10 ran 0.25 epoch at 23,908 reviews/s (3,865 s) -> 1 epoch ~258 min + smoke; queued as 290.
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
BASH_EXE="C:/Program Files/Git/usr/bin/bash.exe"
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/small-model
SM=$WT/artifacts/small-model
SC=$WT/scripts/small_model_heldout.py
DEC_PY=$WT/scripts/expertium_decay.py
EXPS=$WT/artifacts/accuracy/experiments
CKPT=expertium/E89-F65e-F77a/master.pth
OUT=$SM/heldout.jsonl
log() { echo "[sm40 $(date '+%m-%d %H:%M:%S')] $*"; }
q() {  # q MINUTES PURPOSE -- CMD...
  local minutes=$1 purpose=$2; shift 3
  log "enqueue: $purpose (expected $minutes min)"
  "$PY" "$QUEUE" run --owner small-model --purpose "small-model: $purpose" --claim gpu --expected-minutes "$minutes" -- "$@"
  local rc=$?; log "exit $rc: $purpose"; return $rc
}
label_users() { OUT="$OUT" L="$1" "$PY" -X utf8 -c "
import json, os
print(len({r['user_id'] for r in map(json.loads, open(os.environ['OUT'], encoding='utf-8')) if r.get('label') == os.environ['L'] and 'benchmark' in r}))"; }
cd "$WT" || exit 1
log "started: $(git -C $WT log --oneline -1)"
D=$SM/scale-w64kd-gate-cached-ctx32768-cont-decay1ep; L=s-w64kd-gate-ctx32768c-dec10-1ep
[ -d "$D" ] || "$PY" -X utf8 $DEC_PY prepare --source-run $EXPS/scale-w64kd-gate-cached-ctx32768-cont/run --epoch 10 \
  --out $D --fraction 1.0 --shape one_minus_sin --evaluate none --seed 1 || { log "prepare failed"; exit 1; }
[ -f "$D/$CKPT" ] || q 290 "SM40 1-epoch decay at 32k from e10 (smoke first)" -- "$BASH_EXE" -c \
  "'$PY' -X utf8 '$DEC_PY' smoke --out '$D' && '$PY' -X utf8 '$DEC_PY' run --out '$D'"
[ -f "$D/$CKPT" ] || { log "no SM40 checkpoint; stop"; exit 1; }
for pass in 1 2; do
  [ "$(label_users $L/w8+s8)" -ge 464 ] && break
  q 25 "SM40 scoring $L, 464 users (pass $pass)" -- "$PY" -X utf8 $SC --model $L=expertium:w64:$D/$CKPT \
    --only $L/w8+s8 --time-limit 3000 --out $OUT
done
log "users scored: $(label_users $L/w8+s8)"
log "done"
