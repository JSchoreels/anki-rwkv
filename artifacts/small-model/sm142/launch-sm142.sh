#!/usr/bin/env bash
# Answer-time expressiveness (SM142) (design: artifacts/small-model/sm142/design.md, pre-registered 11-21 before any GPU ticket).
#   GPU dump only, no training: per review the card id, grade, duration, P(grade), the head's and the ablation's 4 x 16 answer-time
#   distributions (scripts/sm142_expr.py dump, SM117's forward pass and rows) for five models, each checked against its recorded
#   scoring rows (verify, CPU). The CPU analysis is NOT in this launcher (fresh agent, HANDOFF.md).
#   Order: the six-layer Tiny two-head model (SM130), the six-layer Tiny Instant-Only model (SM130), the longer Small Instant-Only
#   model (SM113), the show-time two-head cool-down for Small (SM118), the show-time instant-only big model (SM99).
#   Answer time = front + back ("within S s" = answered within S s, reading the back included).
# Breaks: quiet.json windows of scope all / gpu only (sm142_block.py); no planned daily or Sunday break.
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm142/artifacts/small-model/sm142/launch-sm142.sh
#   launch   WMI hidden (HANDOFF.md); log artifacts/small-model/sm142/gpu/launch-sm142.log
#   resume   remove gpu/STOP, then the launch command again (dumped users and verified models are skipped)
#   stop     touch C:/Users/jonat/workspace/wt/sm142/artifacts/small-model/sm142/gpu/STOP (checked between steps)
# Read only: wt/sm117, wt/sm119, wt/sm138 (heads, recorded rows), the model masters. Do not edit scripts/ expertium/ rwkv/ of wt/sm142 while it runs.
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/sm142
S=$WT/artifacts/small-model/sm142
G=$S/gpu
W=$S/work
mkdir -p "$G" "$W"
LOG=$G/launch-sm142.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm142-dryrun.log
X="$PY -X utf8 $WT/scripts/sm142_expr.py"
SM119HEADS=C:/Users/jonat/workspace/wt/sm119/artifacts/small-model
M118=C:/Users/jonat/workspace/wt/sm118/artifacts/small-model/sm118-e26-dec2ep32k-t91s2-p2/expertium/E89-F65e-F77a/master.pth
USERS=$W/users-722.json
NUSERS=722
MODELS="sm130th sm130io sm113 sm118 sm99"
NAME="Answer-time expressiveness (SM142)"
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START   # cap 0.8
log() { echo "[sm142 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
stopped() { [ -e "$G/STOP" ] && { log "STOP: STOP file present"; return 0; }; return 1; }
nap() { [ "$DRYRUN" = 1 ] && return 0; local s=$1; [ "$s" -gt 600 ] && s=600; sleep "$s"; }

BLOCK=$S/sm142_block.py
block() { "$PY" -X utf8 "$BLOCK" | tr -d '\r'; }

wait_free() {  # wait_free MINUTES WHAT: returns once MINUTES fit before the next window (sets FREE = minutes to it)
  local n=0
  while true; do
    stopped && return 1
    local b; b=$(block)
    case "$b" in
      NOW*) local s=${b#NOW }
            if [ "$DRYRUN" = 1 ]; then log "DRYRUN: $2 would wait inside a quiet window (${s}s left)"; FREE=100000; return 0; fi
            [ $((n % 6)) = 0 ] && log "$2: quiet window, ${s}s left"; n=$((n + 1)); nap "$s";;
      NEXT*) FREE=${b#NEXT }
             [ "$FREE" -ge $(( $1 + 3 )) ] && return 0
             if [ "$DRYRUN" = 1 ]; then log "DRYRUN: $2 ($1 min) would wait: $FREE min to the next window"; FREE=100000; return 0; fi
             [ $((n % 6)) = 0 ] && log "$2 needs $1 min, $FREE min before the next window; waiting"; n=$((n + 1)); nap $(( FREE * 60 + 60 ));;
      *) log "window check failed: $b"; nap 600;;
    esac
  done
}

q() {  # q MINUTES PURPOSE -- CMD...   one ticket that must end before the next window
  local minutes=$1 purpose=$2 rc; shift 3
  while true; do
    wait_free "$minutes" "$purpose" || return 1
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $NAME: $purpose (expected $minutes min)"; echo "    cmd: $*" >> "$LOG"; return 0; fi
    local maxwait=$(( FREE - minutes - 3 )); [ "$maxwait" -lt 1 ] && maxwait=1
    log "enqueue: $purpose (expected $minutes min, max wait $maxwait min, $FREE min to the next window)"
    "$PY" "$QUEUE" run --owner sm142 --purpose "$NAME: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm142/launch-sm142.sh" -- "$@"
    rc=$?; log "exit $rc: $purpose"
    [ "$rc" = 75 ] && { log "queue wait would reach the next window; re-planning"; sleep 60; continue; }
    return $rc
  done
}

train_done() { [ -f "$1/head.pth" ] && [ -f "$1/ablation-head.pth" ] && [ -f "$1/master.pth" ] && J="$1/train.json" "$PY" -X utf8 -c "
import json, os, sys
sys.exit(0 if json.load(open(os.environ['J'])).get('status') == 'complete' else 1)" 2>/dev/null; }
minfo() { $X info --model "$1" | tr -d '\r'; }
head_dir() { ( eval "$(minfo "$1")"; echo "$HEADDIR" ); }
mname() { ( eval "$(minfo "$1")"; echo "$MNAME" ); }
recorded() { ( eval "$(minfo "$1")"; echo "$RECORDED" ); }
dumped_users() { J="$1" "$PY" -X utf8 -c "
import json, os
from pathlib import Path
p = Path(os.environ['J'])
print(len({r['user_id'] for r in map(json.loads, open(p, encoding='utf-8')) if 'error' not in r}) if p.exists() else 0)" | tr -d '\r'; }

preconditions() {
  local key hd
  "$PY" -X utf8 "$BLOCK" selftest >> "$LOG" 2>&1 || { log "STOP: block selftest failed"; return 1; }
  R2="$S/check-cpu.json" "$PY" -X utf8 -c "import json, os, sys; sys.exit(0 if json.load(open(os.environ['R2']))['pass'] else 1)" \
    || { log "STOP: the SM142 CPU check did not pass ($S/check-cpu.json)"; return 1; }
  [ -s "$USERS" ] || { log "STOP: no users file $USERS (run: $X users)"; return 1; }
  export SM119_SM118_SHA256
  SM119_SM118_SHA256=$(J="$SM119HEADS/sm119-sm118-head/train.json" "$PY" -X utf8 -c "import json, os; print(json.load(open(os.environ['J']))['released_master_sha256'])" | tr -d '\r')
  [ "$(sha256sum "$M118" | cut -d' ' -f1)" = "$SM119_SM118_SHA256" ] || { log "STOP: SM118's master sha256 differs from its head's train.json"; return 1; }
  for key in $MODELS; do
    hd=$(head_dir "$key")
    [ -n "$hd" ] || { log "STOP: $key: no head directory"; return 1; }
    train_done "$hd" || { log "STOP: $key head missing or incomplete at $hd"; return 1; }
    [ -s "$(recorded "$key")" ] || { log "STOP: $key: no recorded scoring rows"; return 1; }
    ( cd "$WT" && "$PY" -X utf8 -c "
import sys
sys.path.insert(0, '.')
from scripts.sm117_head import verified_master
verified_master('$key')" ) >> "$LOG" 2>&1 || { log "STOP: $key master missing or its sha256 differs"; return 1; }
  done
  if [ "$DRYRUN" != 1 ] && [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    log "STOP: uncommitted source changes in scripts/ expertium/ rwkv/"; return 1; fi
  return 0
}

dump_model() {  # pass 1 at cap 0.8, pass 2 (missing users only) at 0.92, then the CPU verify against the recorded scoring rows
  local key=$1 IX=$G/dump-$1.jsonl hd got nm minutes
  [ -f "$G/dump-ok-$key" ] && { log "$key already dumped and verified"; return 0; }
  hd=$(head_dir "$key"); nm=$(mname "$key")
  minutes=25; [ "$key" = sm99 ] && minutes=35
  if [ "$DRYRUN" = 1 ] || [ "$(dumped_users "$IX")" -lt $NUSERS ]; then
    q $minutes "per-review answer-time distributions of $nm on the 732 set ($NUSERS users, pass 1, cap 0.8)" -- \
      env -u RWKV_DELETED_CARDS $X dump --model "$key" --head-dir "$hd" --users-file "$USERS" >> "$G/dump-$key.out" 2>&1 \
      || { log "STOP: $key dump ticket failed (see $G/dump-$key.out)"; return 1; }
  fi
  if [ "$DRYRUN" = 1 ] || [ "$(dumped_users "$IX")" -lt $NUSERS ]; then
    q 15 "per-review answer-time distributions of $nm (missing users, pass 2, cap 0.92)" -- \
      env -u RWKV_DELETED_CARDS HELDOUT_GPU_MEMORY_FRACTION=0.92 $X dump --model "$key" --head-dir "$hd" --users-file "$USERS" >> "$G/dump-$key.out" 2>&1 \
      || { log "STOP: $key dump pass 2 failed (see $G/dump-$key.out)"; return 1; }
  fi
  [ "$DRYRUN" = 1 ] && { log "DRYRUN: then $X verify --model $key"; return 0; }
  got=$(dumped_users "$IX")
  log "$key dumped users: $got of $NUSERS ($(du -sh "$W/dump-$key" 2>/dev/null | cut -f1))"
  [ "$got" -ge 700 ] || { log "STOP: $key dump has only $got of $NUSERS users after 2 passes"; return 1; }
  local out; out=$($X verify --model "$key" 2>&1 | tr -d '\r'); local r=$?
  log "$key verify exit $r: $(echo "$out" | tail -1 | cut -c1-300)"
  [ "$r" = 0 ] || { log "STOP: $key dump does not match its recorded scoring rows (see $G/verify-$key.json)"; return 1; }
  touch "$G/dump-ok-$key"
  return 0
}

final_check() {  # every expected output exists and is not empty before done
  local f key
  for key in $MODELS; do
    for f in "$G/dump-$key.jsonl" "$G/verify-$key.json" "$G/dump-ok-$key"; do
      [ -e "$f" ] || { log "STOP: expected output missing: $f"; return 1; }
    done
    [ -s "$G/dump-$key.jsonl" ] || { log "STOP: expected output empty: $G/dump-$key.jsonl"; return 1; }
    [ "$(ls "$W/dump-$key" | grep -c '^[0-9]*\.npz$')" -ge 700 ] || { log "STOP: fewer than 700 user dumps in $W/dump-$key"; return 1; }
  done
  return 0
}

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1) (DRYRUN=$DRYRUN)"
rc=0
wait_free 5 "preparation" || rc=1
[ $rc -eq 0 ] && { preconditions || rc=1; }
for key in $MODELS; do
  [ $rc -eq 0 ] && { stopped && rc=1; }
  [ $rc -eq 0 ] && { dump_model "$key" || rc=1; }
done
[ $rc -eq 0 ] && [ "$DRYRUN" != 1 ] && { final_check || rc=1; }
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; else log "done"; fi
else
  log "STOP: ended with exit $rc"
fi
exit $rc
