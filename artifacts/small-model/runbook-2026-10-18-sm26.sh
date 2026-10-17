#!/usr/bin/env bash
# SM26 (2026-10-18): instant-weighted (x2) 0.25-epoch decay at 32k from e10, paired with dec10, then 464-user scoring.
# Same steps as runbook f2's SM26 block; the check now compares against the parent of the commit adding the option (24644db).
# No break window: the user cancelled the 22:00-02:00 break at 23:50 ("don't block work for me, I'll ask you to stop").
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
log() { echo "[sm26 $(date '+%m-%d %H:%M:%S')] $*"; }
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
if [ -n "$(git -C $WT status --porcelain -- scripts expertium rwkv)" ]; then log "uncommitted source changes; stop"; exit 2; fi
"$PY" -X utf8 $WT/scripts/check_instant_loss_weight.py > $SM/check-instant-loss-weight-night.txt 2>&1
grep -q "ALL OK" $SM/check-instant-loss-weight-night.txt || { log "SM26 check did not print ALL OK; stop"; exit 2; }
log "SM26 check passed"
D2=$SM/scale-w64kd-gate-cached-ctx32768-cont-decay-ilw2; L2=s-w64kd-gate-ctx32768c-dec10-ilw2
[ -d "$D2" ] || "$PY" -X utf8 $DEC_PY prepare --source-run $EXPS/scale-w64kd-gate-cached-ctx32768-cont/run --epoch 10 \
  --out $D2 --fraction 0.25 --shape one_minus_sin --evaluate none --seed 1 --extra-options '{"instant_loss_weight": 2.0}' \
  || { log "SM26 prepare failed"; exit 1; }
[ -f "$D2/$CKPT" ] || q 100 "SM26 instant-weighted (x2) decay at 32k from e10 (smoke first)" -- "$BASH_EXE" -c \
  "'$PY' -X utf8 '$DEC_PY' smoke --out '$D2' && '$PY' -X utf8 '$DEC_PY' run --out '$D2'"
[ -f "$D2/$CKPT" ] || { log "no SM26 checkpoint; stop"; exit 1; }
for pass in 1 2; do
  [ "$(label_users $L2/w8+s8)" -ge 464 ] && break
  q 25 "SM26 scoring $L2, 464 users (pass $pass)" -- "$PY" -X utf8 $SC --model $L2=expertium:w64:$D2/$CKPT \
    --only $L2/w8+s8 --time-limit 3000 --out $OUT
done
log "users scored: $(label_users $L2/w8+s8)"
log "done"
