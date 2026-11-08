#!/usr/bin/env bash
# The grade order by construction (SM95): retry of the 29 users the rectified scoring lost to GPU out-of-memory
# (launch-sm95.sh ran it with --workers 3 at cap 0.8; the rectified rows are ~2.5x). Same numerics: same scorer, model,
# flags and users, only --workers 1 and the 0.92 memory cap (memory-only). Then the rule. Never edits launch-sm95.sh.
#   dry run  DRYRUN=1 bash .../gpu/retry-rect.sh      log gpu/retry-rect.log (dry run: retry-rect-dryrun.log)
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/sm95
SM=$WT/artifacts/small-model
S=$SM/sm95
G=$S/gpu
LOG=$G/retry-rect.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/retry-rect-dryrun.log
SC=$WT/scripts/small_model_heldout.py
BLOCK="$PY -X utf8 $S/sm95_block.py"
TRACK=E89-F65e-F77a
DARM=$SM/sm95-arm-e21-dec-q32k-muon
DREF=C:/Users/jonat/workspace/wt/sm77/artifacts/small-model/sm77-ref-e21-dec-q32k-muon
ARLBL=sm95-arm-e21-q32k-rect
RRLBL=sm77-ref-e21-q32k-rect
OUTROWS=$G/sm95-neighbour.jsonl
UF=$G/retry-users.json
NEED=29
NAME="the grade order by construction (SM95)"
MIN=45
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START
log() { echo "[sm95-retry $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
nap() { [ "$DRYRUN" = 1 ] && return 0; local s=$1; [ "$s" -gt 600 ] && s=600; sleep "$s"; }
wait_free() {  # returns once $1 minutes fit before the next break (sets FREE)
  local n=0
  while true; do
    [ -e "$G/STOP" ] && { log "STOP: STOP file present"; return 1; }
    local b; b=$($BLOCK)
    case "$b" in
      NOW*) local s=${b#NOW }
            if [ "$DRYRUN" = 1 ]; then log "DRYRUN: $2 would wait inside a break (${s}s left)"; FREE=100000; return 0; fi
            [ $((n % 6)) = 0 ] && log "$2: break, ${s}s left"; n=$((n + 1)); nap "$s";;
      NEXT*) FREE=${b#NEXT }
             [ "$FREE" -ge $(( $1 + 3 )) ] && return 0
             if [ "$DRYRUN" = 1 ]; then log "DRYRUN: $2 ($1 min) would wait: $FREE min to the next break"; FREE=100000; return 0; fi
             [ $((n % 6)) = 0 ] && log "$2 needs $1 min, $FREE min before the next break; waiting"; n=$((n + 1)); nap $(( FREE * 60 + 60 ));;
      *) log "window check failed: $b"; nap 600;;
    esac
  done
}
label_users() { F="$1" LB="$2" "$PY" -X utf8 -c "
import json, os
from pathlib import Path
p = Path(os.environ['F'])
print(len({r['user_id'] for r in map(json.loads, open(p, encoding='utf-8')) if r.get('label') == os.environ['LB'] and 'benchmark_existing' in r}) if p.exists() else 0)"; }
retry_users() {  # retry_users LABEL: how many of the retry users still lack a good row
  F="$OUTROWS" LB="$1/w8+s8" U="$UF" "$PY" -X utf8 -c "
import json, os
ids = set(json.load(open(os.environ['U']))['users'])
have = {r['user_id'] for r in map(json.loads, open(os.environ['F'], encoding='utf-8')) if r.get('label') == os.environ['LB'] and 'benchmark_existing' in r}
print(len(ids - have))"; }
score_retry() {  # score_retry LABEL CKPT: up to 3 tickets, each only while retry users are missing
  local lb=$1 ck=$2 pass rc maxwait
  for pass in 1 2 3; do
    if [ "$DRYRUN" != 1 ] && [ "$(retry_users "$lb")" = 0 ]; then break; fi
    wait_free $MIN "scoring $lb retry" || return 1
    maxwait=$(( FREE - MIN - 3 )); [ "$maxwait" -lt 1 ] && maxwait=1
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $NAME: retry scoring $lb (expected $MIN min)"
      echo "    cmd: env -u RWKV_DELETED_CARDS HELDOUT_GPU_MEMORY_FRACTION=0.92 $PY -X utf8 $SC --pool 732 --users-file $UF --model $lb=expertium:w64:$ck --only $lb/w8+s8 --time-limit 2400 --split-existing --deleted-cards neighbour --grade-rectify --workers 1 --out $OUTROWS" >> "$LOG"; return 0; fi
    log "enqueue: retry scoring $lb (pass $pass, $(retry_users "$lb") of $NEED users missing, expected $MIN min, max wait $maxwait min, $FREE min to the next break)"
    "$PY" "$QUEUE" run --owner sm95 --purpose "$NAME: retry of the rectified scoring of $lb on the $NEED users lost to GPU out-of-memory (1 worker, cap 0.92, pass $pass)" \
      --claim gpu --expected-minutes "$MIN" --max-wait-minutes "$maxwait" --launched-by "sm95/gpu/retry-rect.sh" -- \
      env -u RWKV_DELETED_CARDS HELDOUT_GPU_MEMORY_FRACTION=0.92 "$PY" -X utf8 "$SC" --pool 732 --users-file "$UF" --model "$lb=expertium:w64:$ck" \
      --only "$lb/w8+s8" --time-limit 2400 --split-existing --deleted-cards neighbour --grade-rectify --workers 1 --out "$OUTROWS" >> "$G/score.out" 2>&1
    rc=$?; log "exit $rc: retry scoring $lb (pass $pass); $(retry_users "$lb") of $NEED users still missing"
    [ "$rc" = 75 ] && { sleep 60; continue; }
  done
  return 0
}
cd "$WT" || exit 1
log "started (DRYRUN=$DRYRUN): $NEED users per rectified label, $(label_users "$OUTROWS" "$ARLBL/w8+s8") arm / $(label_users "$OUTROWS" "$RRLBL/w8+s8") reference rows present"
rc=0
score_retry "$ARLBL" "$DARM/expertium/$TRACK/master.pth" || rc=1
[ $rc -eq 0 ] && { score_retry "$RRLBL" "$DREF/expertium/$TRACK/master.pth" || rc=1; }
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: would run the rule"; else
    out=$(OMP_NUM_THREADS=2 "$PY" -X utf8 "$S/sm95_decide.py" --rows "$OUTROWS" \
      --ref-rows C:/Users/jonat/workspace/wt/sm77/artifacts/small-model/sm77/gpu/sm77-neighbour.jsonl \
      --sm33-rows C:/Users/jonat/workspace/wt/sm33/artifacts/small-model/sm33/gpu/sm33-neighbour.jsonl \
      --arm-measure "$S/work/measure-arm" --ref-measure C:/Users/jonat/workspace/wt/sm33/artifacts/small-model/sm33/work/stage2-reference \
      --sm33-measure C:/Users/jonat/workspace/wt/sm33/artifacts/small-model/sm33/work/stage2-arm --derived "$S/work/derived" 2>&1); r=$?
    echo "$out" >> "$G/rule.out"
    log "rule exit $r: $(echo "$out" | tail -1 | cut -c1-300)"; [ "$r" = 0 ] || rc=1
  fi
fi
if [ $rc -eq 0 ]; then log "done"; else log "STOP: ended with exit $rc"; fi
exit $rc
