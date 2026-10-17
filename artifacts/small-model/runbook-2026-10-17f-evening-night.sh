#!/usr/bin/env bash
# Evening + night 2026-10-17/18 (replaces runbooks d and e, which only waited). User: break with no GPU work between
# 20:00 and 02:00. Every GPU ticket starts only if it is expected to end by 20:00, or after 02:00 (wait_slot).
#   evening (after chain b's "cont exit"): own-collection CPU scoring of dec10 -> SM24 dumps + breakdown -> SM25 on 464
#   night (from 02:00): whatever of SM24/SM25 is left -> speed profile (--timing, fork-reference worktree) ->
#   1-epoch decay at 2,048 from ctx2048c e8 (SM35) + 464 scoring -> instant-weighted decay at 32k from e10 (SM26,
#   only if branch sm26 is merged into small-model) + 464 scoring.
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
log() { echo "[f $(date '+%m-%d %H:%M:%S')] $*"; }
# wait_slot MINUTES: return when now is in [02:00, 20:00) and now + MINUTES <= 20:00
wait_slot() {
  local m=$1 said=0
  while :; do
    local now=$((10#$(date +%H) * 60 + 10#$(date +%M)))
    if [ "$now" -ge 120 ] && [ $((now + m)) -le 1200 ]; then return 0; fi
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

until grep -q "cont exit" $SM/runbook-2026-10-17b-chain.log; do sleep 120; done
log "chain b: $(grep 'cont exit' $SM/runbook-2026-10-17b-chain.log | tail -1)"
DEC10=$SM/scale-w64kd-gate-cached-ctx32768-cont-decay/$CKPT
ST=$DEC10; NAME=ctx32768c-dec10
[ -f "$ST" ] || { ST=$SM/scale-w64kd-gate-cached-ctx16384-cont-decay-L32k/$CKPT; NAME=ctx16384c-dec9-L32k; }
log "student for SM24 / collection: $NAME = $ST"

# own collection (CPU, ~11 min, aggregate only); skipped if it would run into the break
now=$((10#$(date +%H) * 60 + 10#$(date +%M)))
if [ "$NAME" = ctx32768c-dec10 ] && [ "$now" -lt 1185 ]; then
  (cd $SM/collection && "$PY" -X utf8 score_collection_small.py --device cpu --model dec10=expertium:w64:$ST --only dec10/w8+s8 \
    --out C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/collection/coll-dec10.jsonl > coll-dec10.log 2>&1 &)
  log "collection scoring of dec10 started (CPU)"
fi

# SM24
for pass in 1 2; do
  [ "$(rows $SM/sm24-scores.jsonl)" -ge 928 ] && break
  q 60 "SM24 per-review dumps, fork + $NAME, 464 users (pass $pass)" -- "$PY" -X utf8 $SC --model fork=fork::$TEACHER \
    --model st=expertium:w64:$ST --only fork/w8+s8 --only st/w8+s8 --time-limit 3300 --out $SM/sm24-scores.jsonl --dump $SM/sm24-dump
  log "SM24 rows after pass $pass: $(rows $SM/sm24-scores.jsonl)"
done
"$PY" -X utf8 $SM/feature-screen/sm24_bytype.py > $SM/feature-screen/sm24-bytype.txt 2>&1; log "SM24 breakdown written ($NAME)"

# SM25 on 464 users
for pass in 1 2; do
  [ "$(rows $SM/sm25-scores.jsonl)" -ge 464 ] && break
  q 40 "SM25 fork with no-steps review-type labels, 464 users (pass $pass)" -- "$PY" -X utf8 $SM/feature-screen/sm25.py \
    --model forknosteps=fork::$TEACHER --only forknosteps/w8+s8 --time-limit 3300 --out $SM/sm25-scores.jsonl
  log "SM25 rows after pass $pass: $(rows $SM/sm25-scores.jsonl)"
done
"$PY" -X utf8 $SM/feature-screen/sm25_464.py > $SM/feature-screen/sm25-464.txt 2>&1; log "SM25 summary written"

# speed profile (quiet GPU, --timing): long windows + fork-size cases
SPD=$FR/artifacts/fork-reference/speed-20261018
q 50 "speed profile: w64 at 2,048/8,192/16,384/32,768 windows + fork128 b128/b64 (fork-reference)" --timing -- \
  "$PY" -X utf8 $FR/scripts/expertium_speed.py all --out $SPD \
  --cases w64-ctx2048-b16,w64-ctx8192-b4,w64-ctx32768-b1,fork128-b128,fork128-b64,w64-ctx16384-b2 --budget-minutes 45
"$PY" -X utf8 $FR/scripts/expertium_speed.py report --out $SPD > $SPD/report.txt 2>&1; log "speed report written"

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

# SM26: instant-weighted 0.25-epoch decay at 32k from e10 (pairs with dec10); needs branch sm26 merged
if git -C $WT merge-base --is-ancestor 6c68cf6 HEAD 2>/dev/null; then
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
  log "SM26 skipped: branch sm26 (6c68cf6) is not merged into small-model"
fi
log "done"
