#!/usr/bin/env bash
# Rolling recalibration for the six-layer Tiny (SM140) (design: artifacts/small-model/sm140/design.md, pre-registered 11-21 before any GPU
#   ticket). Scoring only, no training: SM128's recipe and go rule (scripts/sm125_recal.py, SM125_TAG=sm140) on the six-layer Tiny two-head
#   model (SM130, key sm130th) and the six-layer Tiny Instant-Only model (SM130, key sm130io) with the SM138 heads (346 inputs).
#   Order: 1. GPU scoring of SM120's recorded sums for the two models (the baselines; SM120 never scored them), cap 0.8, missing users at
#   0.92; 2. GPU smoke (3 users, sm130th dump vs those sums within 1e-5); 3. per-review dump of each model; 4. CPU recalibration of both
#   models (2 processes, 1 thread, < 6 GB RAM each, started behind the first dump); 5. sm140-result.{md,json}; 6. final check, done.
#   Answer time = front + back ("within S s" = answered within S s, reading the back included).
# Breaks: quiet.json windows of scope all / gpu only (sm140_block.py); no planned daily or Sunday break.
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm140/artifacts/small-model/sm140/launch-sm140.sh
#   launch   WMI hidden (HANDOFF.md); log artifacts/small-model/sm140/gpu/launch-sm140.log
#   resume   remove gpu/STOP, then the launch command again (scored, dumped and recalibrated users are skipped)
#   stop     touch C:/Users/jonat/workspace/wt/sm140/artifacts/small-model/sm140/gpu/STOP (checked between steps)
# Read only: wt/sm138 (heads). Do not edit scripts/ expertium/ rwkv/ of wt/sm140 while it runs.
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
QUIET=C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json
WT=C:/Users/jonat/workspace/wt/sm140
S=$WT/artifacts/small-model/sm140
G=$S/gpu
W=$S/work
mkdir -p "$G" "$W" "$S/rows"
LOG=$G/launch-sm140.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm140-dryrun.log
H="$PY -X utf8 $WT/scripts/sm117_head.py"
R="$PY -X utf8 $WT/scripts/sm125_recal.py"
F120="$PY -X utf8 $WT/scripts/sm120_fast.py"
HD120=$WT/artifacts/small-model/sm120
SM138HEADS=C:/Users/jonat/workspace/wt/sm138/artifacts/small-model
CHECK125=$WT/artifacts/small-model/sm125/check-cpu.json   # SM125 CPU proofs (inherited, recipe untouched)
MODELS="sm130th sm130io"
NAME="Rolling recalibration for the six-layer Tiny (SM140)"
export DRYRUN=${DRYRUN:-0}
export SM125_TAG=sm140 SM125_MODELS="sm130th sm130io"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START   # cap 0.8
log() { echo "[sm140 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
stopped() { [ -e "$G/STOP" ] && { log "STOP: STOP file present"; return 0; }; return 1; }
nap() { [ "$DRYRUN" = 1 ] && return 0; local s=$1; [ "$s" -gt 600 ] && s=600; sleep "$s"; }

BLOCK=$S/sm140_block.py
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
    "$PY" "$QUEUE" run --owner sm140 --purpose "$NAME: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm140/launch-sm140.sh" -- "$@"
    rc=$?; log "exit $rc: $purpose"
    [ "$rc" = 75 ] && { log "queue wait would reach the next window; re-planning"; sleep 60; continue; }
    return $rc
  done
}

train_done() { [ -f "$1/head.pth" ] && [ -f "$1/master.pth" ] && J="$1/train.json" "$PY" -X utf8 -c "
import json, os, sys
sys.exit(0 if json.load(open(os.environ['J'])).get('status') == 'complete' else 1)" 2>/dev/null; }
head_dir() { echo "$SM138HEADS/sm138-$1-head"; }
dumped_users() { J="$1" "$PY" -X utf8 -c "
import json, os
from pathlib import Path
p = Path(os.environ['J'])
print(len({r['user_id'] for r in map(json.loads, open(p, encoding='utf-8')) if 'error' not in r}) if p.exists() else 0)" | tr -d '\r'; }
mname() { ( eval "$($H info --model "$1" | tr -d '\r')"; echo "$MNAME" ); }
BG=""
start_recal() {  # background CPU recalibration of one model right after its dump (1 thread, < 6 GB RAM), stopped 5 min before the next window
  local key=$1 b free until
  [ -f "$G/recal-ok-$key" ] && return 0
  [ "$DRYRUN" = 1 ] && { log "DRYRUN: background CPU recalibration of $key"; return 0; }
  b=$(block)
  case "$b" in NEXT*) free=${b#NEXT };; *) log "$key recalibration deferred (inside a window)"; return 0;; esac
  [ "$free" -lt 25 ] && { log "$key recalibration deferred ($free min to the next window)"; return 0; }
  until=$(( $(date +%s) + (free - 5) * 60 ))
  log "background CPU recalibration of $key (until $(date -d @$until '+%m-%d %H:%M'))"
  recal_proc "$key" "$until" &
  BG="$BG $!"
}
recal_proc() {  # recal_proc KEY UNTIL: exit code in work/recal-KEY.rc, marker gpu/recal-ok-KEY on success
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $R recal --model "$1" --until "$2" >> "$G/recal-$1.out" 2>&1
  local rc=$?; echo "$rc" > "$W/recal-$1.rc"; [ "$rc" = 0 ] && touch "$G/recal-ok-$1"; return 0
}

preconditions() {
  local key hd
  [ -f "$BLOCK" ] && "$PY" -X utf8 "$BLOCK" selftest >> "$LOG" 2>&1 || { log "STOP: block selftest failed"; return 1; }
  R2="$CHECK125" "$PY" -X utf8 -c "import json, os, sys; sys.exit(0 if json.load(open(os.environ['R2']))['pass'] else 1)" \
    || { log "STOP: the SM125 CPU check did not pass ($CHECK125)"; return 1; }
  for key in $MODELS; do
    hd=$(head_dir "$key")
    train_done "$hd" || { log "STOP: $key head missing or incomplete at $hd"; return 1; }
    $H info --model "$key" > /dev/null 2>> "$LOG" || { log "STOP: $key master missing or its sha256 differs"; return 1; }
    [ -f "$HD120/priors.json" ] || { log "STOP: no SM120 priors"; return 1; }
  done
  if [ "$DRYRUN" != 1 ] && [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    log "STOP: uncommitted source changes in scripts/ expertium/ rwkv/"; return 1; fi
  return 0
}

smoke() {  # 3 users (2388, 1952, 3922), the six-layer Tiny two-head model (SM130), GPU: sums equal the SM120 sums scored in step 1 within 1e-5
  local si=$W/smoke-sm130th.jsonl
  [ -f "$G/smoke-ok" ] && { log "GPU smoke already passed"; return 0; }
  rm -f "$si"
  "$PY" -X utf8 -c "import json; u = [2388, 1952, 3922]; json.dump({'users': u}, open(r'$W/smoke-users.json', 'w'))" \
    || { log "STOP: no check users ($W/check/users.json)"; return 1; }
  q 12 "GPU smoke: dump of the six-layer Tiny two-head model (SM130) on 3 users vs SM120's recorded sums (cap 0.8)" -- \
    env -u RWKV_DELETED_CARDS $R dump --model sm130th --head-dir "$(head_dir sm130th)" --users-file "$W/smoke-users.json" \
    --out-dir "$W/smoke-dump-sm130th" --index "$si" >> "$G/smoke.out" 2>&1 || { log "STOP: GPU smoke ticket failed (see $G/smoke.out)"; return 1; }
  [ "$DRYRUN" = 1 ] && return 0
  ( cd "$WT" && "$PY" -X utf8 -c "
import json, sys
sys.path.insert(0, '.')
from scripts.sm125_recal import sm120_rows, dump_matches_sm120
old = sm120_rows('sm130th')
res = [dump_matches_sm120(json.loads(l), old[json.loads(l)['user_id']], tol=1e-5) for l in open(r'$si')]
print('smoke', res)
sys.exit(0 if res and all(ok for ok, _ in res) else 1)" >> "$G/smoke.out" 2>&1 ) || { log "STOP: GPU smoke sums differ from SM120 (see $G/smoke.out)"; return 1; }
  touch "$G/smoke-ok"; log "GPU smoke passed: $(tail -1 "$G/smoke.out" | cut -c1-200)"
  return 0
}

scored120() { J="$1" "$PY" -X utf8 -c "
import json, os
from pathlib import Path
p = Path(os.environ['J'])
print(len({r['user_id'] for r in map(json.loads, open(p, encoding='utf-8')) if 'error' not in r}) if p.exists() else 0)" | tr -d '\r'; }

score120() {  # SM120's scoring step (recorded sums = baselines + dump check) for a model SM120 never scored; pass 1 cap 0.8, pass 2 (missing users) cap 0.92
  local key=$1 OUTR=$HD120/gpu/sm120-$1-rows.jsonl got nm
  nm=$(mname "$key"); mkdir -p "$HD120/gpu"
  if [ "$DRYRUN" = 1 ] || [ "$(scored120 "$OUTR")" -lt 723 ]; then
    q 30 "SM120 scoring of $nm: P(Good or Easy), P(Easy), P(set and answered within S s) on the 732 set (723 users, pass 1, cap 0.8; the baselines of the recalibration)" --       env -u RWKV_DELETED_CARDS $F120 score --model "$key" --head-dir "$(head_dir "$key")" --out "$OUTR" >> "$G/score120-$key.out" 2>&1       || { log "STOP: $key SM120 scoring ticket failed (see $G/score120-$key.out)"; return 1; }
  fi
  if [ "$DRYRUN" = 1 ] || [ "$(scored120 "$OUTR")" -lt 723 ]; then
    q 15 "SM120 scoring of $nm (missing users, pass 2, cap 0.92)" --       env -u RWKV_DELETED_CARDS HELDOUT_GPU_MEMORY_FRACTION=0.92 $F120 score --model "$key" --head-dir "$(head_dir "$key")" --out "$OUTR" >> "$G/score120-$key.out" 2>&1       || { log "STOP: $key SM120 scoring pass 2 failed (see $G/score120-$key.out)"; return 1; }
  fi
  if [ "$DRYRUN" != 1 ]; then
    got=$(scored120 "$OUTR"); log "$key SM120 sums scored users: $got of 723"
    [ "$got" -ge 700 ] || { log "STOP: $key SM120 scoring has only $got of 723 users after 2 passes"; return 1; }
  fi
  return 0
}

final_check() {  # every expected output exists and is not empty before done
  local f key
  [ -e "$G/smoke-ok" ] || { log "STOP: expected output missing: $G/smoke-ok"; return 1; }
  for key in $MODELS; do
    for f in "$HD120/gpu/sm120-$key-rows.jsonl" "$G/dump-$key.jsonl" "$S/rows/sm140-$key-recal.jsonl"; do
      [ -s "$f" ] || { log "STOP: expected output missing or empty: $f"; return 1; }
    done
    [ -e "$G/recal-ok-$key" ] || { log "STOP: expected output missing: $G/recal-ok-$key"; return 1; }
  done
  for f in "$S/sm140-result.md" "$S/sm140-result.json"; do
    [ -s "$f" ] || { log "STOP: expected output missing or empty: $f"; return 1; }
  done
  return 0
}

dump_model() {  # pass 1 at cap 0.8, pass 2 (missing users only) at 0.92
  local key=$1 IX=$G/dump-$1.jsonl hd got nm
  hd=$(head_dir "$key"); nm=$(mname "$key")
  if [ "$DRYRUN" = 1 ] || [ "$(dumped_users "$IX")" -lt 723 ]; then
    q 30 "per-review dump of $nm on the 732 set (723 users, pass 1, cap 0.8)" -- \
      env -u RWKV_DELETED_CARDS $R dump --model "$key" --head-dir "$hd" >> "$G/dump-$key.out" 2>&1 \
      || { log "STOP: $key dump ticket failed (see $G/dump-$key.out)"; return 1; }
  fi
  if [ "$DRYRUN" = 1 ] || [ "$(dumped_users "$IX")" -lt 723 ]; then
    q 15 "per-review dump of $nm on the 732 set (missing users, pass 2, cap 0.92)" -- \
      env -u RWKV_DELETED_CARDS HELDOUT_GPU_MEMORY_FRACTION=0.92 $R dump --model "$key" --head-dir "$hd" >> "$G/dump-$key.out" 2>&1 \
      || { log "STOP: $key dump pass 2 failed (see $G/dump-$key.out)"; return 1; }
  fi
  if [ "$DRYRUN" != 1 ]; then
    got=$(dumped_users "$IX")
    log "$key dumped users: $got of 723 ($(du -sh "$W/dump-$key" 2>/dev/null | cut -f1))"
    [ "$got" -ge 700 ] || { log "STOP: $key dump has only $got of 723 users after 2 passes"; return 1; }
  fi
  return 0
}

recal_all() {  # CPU, the pending models in parallel (1 process each, 1 thread, < 6 GB RAM), stopped 5 min before every window, resumed after it
  local rounds=0 key pending rc
  while true; do
    stopped && return 1
    pending=""
    for key in $MODELS; do [ -f "$G/recal-ok-$key" ] || pending="$pending $key"; done
    [ -z "$pending" ] && return 0
    wait_free 20 "CPU recalibration ($pending )" || return 1
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN: CPU recalibration of$pending"; return 0; fi
    local until=$(( $(date +%s) + (FREE - 5) * 60 ))
    log "CPU recalibration of$pending (until $(date -d @$until '+%m-%d %H:%M'))"
    local pids=""
    for key in $pending; do
      recal_proc "$key" "$until" &
      pids="$pids $!"
    done
    wait $pids
    for key in $pending; do
      rc=$(cat "$W/recal-$key.rc" 2>/dev/null || echo 1)
      case "$rc" in
        0) log "$key recalibrated: $(tail -1 "$G/recal-$key.out" | cut -c1-160)";;
        3) log "$key recalibration paused before the window";;
        *) log "STOP: $key recalibration failed (exit $rc, see $G/recal-$key.out)"; return 1;;
      esac
    done
    rounds=$((rounds + 1)); [ "$rounds" -gt 20 ] && { log "STOP: recalibration did not finish in 20 rounds"; return 1; }
  done
}

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1) (DRYRUN=$DRYRUN)"
rm -f "$W"/recal-*.rc
rc=0
wait_free 5 "preparation" || rc=1
[ $rc -eq 0 ] && { preconditions || rc=1; }
for key in $MODELS; do
  [ $rc -eq 0 ] && { score120 "$key" || rc=1; }
done
[ $rc -eq 0 ] && { smoke || rc=1; }
for key in $MODELS; do
  [ $rc -eq 0 ] && { dump_model "$key" && start_recal "$key" || rc=1; }
done
if [ -n "$BG" ]; then
  log "waiting for the background recalibrations"; wait $BG
  for key in $MODELS; do
    [ -f "$W/recal-$key.rc" ] || continue
    case "$(cat "$W/recal-$key.rc")" in
      0) log "$key recalibrated: $(tail -1 "$G/recal-$key.out" | cut -c1-160)";;
      3) log "$key recalibration paused before the window";;
      *) log "STOP: $key background recalibration failed (see $G/recal-$key.out)"; rc=1;;
    esac
  done
fi
[ $rc -eq 0 ] && { recal_all || rc=1; }
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then
    log "DRYRUN: then $R report"
  else
    wait_free 20 "report" || rc=1
    if [ $rc -eq 0 ]; then out=$($R report 2>&1); r=$?
      log "report exit $r: $(echo "$out" | tail -1 | cut -c1-400)"; [ "$r" = 0 ] || rc=1; fi
  fi
fi
[ $rc -eq 0 ] && [ "$DRYRUN" != 1 ] && { final_check || rc=1; }
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; else log "done"; fi
else
  log "STOP: ended with exit $rc"
fi
exit $rc
