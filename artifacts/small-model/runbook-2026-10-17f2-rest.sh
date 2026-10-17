#!/usr/bin/env bash
# Evening + night 2026-10-17/18 (replaces runbooks d and e, which only waited). User: break with no GPU work between
# f2 (user 2026-10-17 19:05: "You can work until 10pm"): the rest of runbook f (SM35, SM26) with the break moved to
# 22:00-02:00. Started by switch-to-f2.sh once f has finished the speed profile and sits in wait_slot. Logs to f's log.
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
BASH_EXE="C:/Program Files/Git/usr/bin/bash.exe"
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/small-model
FR=C:/Users/jonat/workspace/wt/fork-reference
SM=$WT/artifacts/small-model
SC=$WT/scripts/small_model_heldout.py
DEC_PY=$WT/scripts/expertium_decay.py
EXPS=$WT/artifacts/accuracy/experiments
TEACHER=C:/Users/jonat/workspace/anki/qt/aqt/rwkv_inference/RWKV_trained_on_5000_10000.pth
CKPT=expertium/E89-F65e-F77a/master.pth
OUT=$SM/heldout.jsonl
log() { echo "[f2 $(date '+%m-%d %H:%M:%S')] $*"; }
# wait_slot MINUTES: return when now is in [02:00, 22:00) and now + MINUTES <= 22:00
wait_slot() {
  local m=$1 said=0
  while :; do
    local now=$((10#$(date +%H) * 60 + 10#$(date +%M)))
    if [ "$now" -ge 120 ] && [ $((now + m)) -le 1320 ]; then return 0; fi
    [ "$said" = 0 ] && log "break window: waiting for 02:00 before a ${m}-min ticket" && said=1
    sleep 120
  done
}
q() {  # q MINUTES PURPOSE [--timing] -- CMD...
  local minutes=$1 purpose=$2; shift 2; local extra=()
  [ "$1" = "--timing" ] && { extra=(--timing); shift; }
  shift  # --
  wait_slot "$minutes"
  log "enqueue: $purpose (expected $minutes min)"
  "$PY" "$QUEUE" run --owner small-model --purpose "small-model: $purpose" --claim gpu "${extra[@]}" --expected-minutes "$minutes" -- "$@"
  local rc=$?; log "exit $rc: $purpose"; return $rc
}
rows() { cat "$1" 2>/dev/null | wc -l; }
label_users() { OUT="$OUT" L="$1" "$PY" -X utf8 -c "
import json, os
print(len({r['user_id'] for r in map(json.loads, open(os.environ['OUT'], encoding='utf-8')) if r.get('label') == os.environ['L'] and 'benchmark' in r}))"; }
log "f2 started (break now 22:00-02:00)"
# SM35: 1-epoch decay at 2,048 from ctx2048c e8 (pairs with s-w64kd-gate-ctx2048c-dec8, the 0.25-epoch decay)
D1=$SM/scale-w64kd-gate-cached-ctx2048-cont-decay1ep; L1=s-w64kd-gate-ctx2048c-dec8-1ep
if [ ! -d "$D1" ]; then
  "$PY" -X utf8 $DEC_PY prepare --source-run $EXPS/scale-w64kd-gate-cached-ctx2048-cont/run --epoch 8 --out $D1 \
    --fraction 1.0 --shape one_minus_sin --evaluate none --seed 1 || log "SM35 prepare failed"
fi
if [ -d "$D1" ]; then
  [ -f "$D1/$CKPT" ] || q 125 "SM35 1-epoch decay at 2,048 from ctx2048c e8 (smoke first)" -- "$BASH_EXE" -c \
    "'$PY' -X utf8 '$DEC_PY' smoke --out '$D1' && '$PY' -X utf8 '$DEC_PY' run --out '$D1'"
  if [ -f "$D1/$CKPT" ]; then
    for pass in 1 2; do
      [ "$(label_users $L1/w8+s8)" -ge 464 ] && break
      q 25 "SM35 scoring $L1, 464 users (pass $pass)" -- "$PY" -X utf8 $SC --model $L1=expertium:w64:$D1/$CKPT \
        --only $L1/w8+s8 --time-limit 3000 --out $OUT
    done
  else log "SM35: no decayed checkpoint, skipping scoring"; fi
fi

# SM26: instant-weighted 0.25-epoch decay at 32k from e10 (pairs with dec10). Its code (branch sm26, commit 6c68cf6:
# code + check only, no docs) is cherry-picked here, between tickets (nothing runs from this worktree at this point).
if ! grep -q instant_loss_weight $WT/scripts/expertium_options.py; then
  if git -C $WT -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" cherry-pick -x 6c68cf6 >/dev/null 2>&1; then
    git -C $WT push -q && log "SM26 code cherry-picked into small-model and pushed: $(git -C $WT log --oneline -1)"
    "$PY" -X utf8 $WT/scripts/check_instant_loss_weight.py > $SM/check-instant-loss-weight-night.txt 2>&1       && log "SM26 check passed" || log "SM26 check FAILED, skipping SM26"
  else
    git -C $WT cherry-pick --abort 2>/dev/null; log "SM26 cherry-pick failed (aborted), skipping SM26"
  fi
fi
if grep -q instant_loss_weight $WT/scripts/expertium_options.py && grep -q "ALL OK" $SM/check-instant-loss-weight-night.txt 2>/dev/null; then
  D2=$SM/scale-w64kd-gate-cached-ctx32768-cont-decay-ilw2; L2=s-w64kd-gate-ctx32768c-dec10-ilw2
  [ -d "$D2" ] || "$PY" -X utf8 $DEC_PY prepare --source-run $EXPS/scale-w64kd-gate-cached-ctx32768-cont/run --epoch 10 \
    --out $D2 --fraction 0.25 --shape one_minus_sin --evaluate none --seed 1 --extra-options '{"instant_loss_weight": 2.0}' \
    || log "SM26 prepare failed"
  if [ -d "$D2" ]; then
    [ -f "$D2/$CKPT" ] || q 100 "SM26 instant-weighted (x2) decay at 32k from e10 (smoke first)" -- "$BASH_EXE" -c \
      "'$PY' -X utf8 '$DEC_PY' smoke --out '$D2' && '$PY' -X utf8 '$DEC_PY' run --out '$D2'"
    if [ -f "$D2/$CKPT" ]; then
      for pass in 1 2; do
        [ "$(label_users $L2/w8+s8)" -ge 464 ] && break
        q 25 "SM26 scoring $L2, 464 users (pass $pass)" -- "$PY" -X utf8 $SC --model $L2=expertium:w64:$D2/$CKPT \
          --only $L2/w8+s8 --time-limit 3000 --out $OUT
      done
    fi
  fi
else
  log "SM26 skipped: code not in small-model or its check did not print ALL OK"
fi
log "done"
