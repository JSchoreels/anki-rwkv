#!/usr/bin/env bash
# Rolling recalibration on the 732 set (SM125) (design: artifacts/small-model/sm125/design.md). Scoring only, no training:
#   1. GPU: per-review raw outputs of five release family candidates (the longer Small Instant-Only model (SM113), the
#      longer cool-down for Tiny (SM115), the show-time two-head cool-down for Small (SM118), the Small Instant-Only
#      model (SM98), the Tiny two-head model (SM106)) re-dumped with fast Good/Easy prediction (SM120)'s code on the 732
#      set (723 users): 33 event probabilities + labels per review, work/dump-<key>/<user>.npz (~4.3 GB per model,
#      gitignored, never committed). First ticket: a 3-user GPU smoke checked against SM120's recorded sums.
#   2. CPU: per-user rolling Platt (7-day refits on the latest 30,000 earlier reviews, raw before 1,000), rows/ + work/.
#   3. CPU: sm125-result.{md,json}.
#   Answer time = front + back ("within S s" = answered within S s, reading the back included).
# Nothing is enqueued until the launcher of the show-time two-head cool-down for Tiny (SM124) has ended (the last line of
#   its log is a "done" or "STOP" line; read only, never touched).
# Breaks: quiet.json (scope all/gpu) plus the daily 20:00-01:00 break and Sunday from 16:00, re-read at every check
#   (<= 10 min); every GPU ticket starts only when its budget fits before the next window; the CPU recalibration stops
#   itself before the next window (--until) and resumes after it. GPU cap 0.8 (missing users at 0.92).
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm125/artifacts/small-model/sm125/launch-sm125.sh
#   launch   WMI hidden (artifacts/small-model/sm125/HANDOFF.md); log artifacts/small-model/sm125/gpu/launch-sm125.log
#   resume   the launch command again (dumped and recalibrated users are skipped)
#   stop     touch C:/Users/jonat/workspace/wt/sm125/artifacts/small-model/sm125/gpu/STOP (checked between steps)
# Read only: wt/sm117 + wt/sm119 (heads), wt/sm118 (master), wt/sm124 (log). Do not edit scripts/ of wt/sm125 while it runs.
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
QUIET=C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json
WT=C:/Users/jonat/workspace/wt/sm125
S=$WT/artifacts/small-model/sm125
G=$S/gpu
W=$S/work
mkdir -p "$G" "$W" "$S/rows"
LOG=$G/launch-sm125.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm125-dryrun.log
H="$PY -X utf8 $WT/scripts/sm117_head.py"
R="$PY -X utf8 $WT/scripts/sm125_recal.py"
SM117HEADS=C:/Users/jonat/workspace/wt/sm117/artifacts/small-model
SM119HEADS=C:/Users/jonat/workspace/wt/sm119/artifacts/small-model
SM124LOG=C:/Users/jonat/workspace/wt/sm124/artifacts/small-model/sm124/gpu/launch-sm124.log   # read only
M118=C:/Users/jonat/workspace/wt/sm118/artifacts/small-model/sm118-e26-dec2ep32k-t91s2-p2/expertium/E89-F65e-F77a/master.pth
MODELS="sm113 sm115 sm118 sm98 sm106"
NAME="Rolling recalibration on the 732 set (SM125)"
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START   # cap 0.8
log() { echo "[sm125 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
stopped() { [ -e "$G/STOP" ] && { log "STOP: STOP file present"; return 0; }; return 1; }
nap() { [ "$DRYRUN" = 1 ] && return 0; local s=$1; [ "$s" -gt 600 ] && s=600; sleep "$s"; }

block() {  # -> "NOW <seconds to its end>" inside a quiet window (quiet.json scope all/gpu, or the daily break), else "NEXT <minutes>"
  Q="$QUIET" "$PY" -X utf8 - <<'PYEOF'
import json, os, re
from datetime import datetime, timedelta
now = datetime.now()
try:
    text = open(os.environ["Q"], encoding="utf-8").read()
except OSError as error:
    print(f"ERR quiet.json unreadable: {error}"); raise SystemExit
try:
    wins = [(w["start"], w["end"]) for w in json.loads(text) if w.get("scope", "all") in ("all", "gpu")]
except (ValueError, KeyError, TypeError):  # malformed JSON: take every start/end pair textually, scope assumed "all"
    wins = re.findall(r'"start"\s*:\s*"([^"]+)"[^{}]*?"end"\s*:\s*"([^"]+)"', text, re.S)
blocks = [(datetime.fromisoformat(s), datetime.fromisoformat(e)) for s, e in wins]
day = now.replace(hour=0, minute=0, second=0, microsecond=0)
blocks += [(day + timedelta(days=d, hours=20), day + timedelta(days=d + 1, hours=1)) for d in range(-1, 9)]  # daily break
blocks += [(day + timedelta(days=d, hours=16), day + timedelta(days=d, hours=20))                          # Sunday from 16:00
           for d in range(-1, 9) if (day + timedelta(days=d)).weekday() == 6]
blocks.sort()
merged = []
for s, e in blocks:
    if merged and s <= merged[-1][1]:
        merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    else:
        merged.append((s, e))
for s, e in merged:
    if s <= now < e:
        print(f"NOW {int((e - now).total_seconds()) + 60}"); raise SystemExit
    if s > now:
        print(f"NEXT {int((s - now).total_seconds() // 60)}"); raise SystemExit
print("NEXT 100000")
PYEOF
}

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
    "$PY" "$QUEUE" run --owner small-model --purpose "$NAME: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm125/launch-sm125.sh" -- "$@"
    rc=$?; log "exit $rc: $purpose"
    [ "$rc" = 75 ] && { log "queue wait would reach the next window; re-planning"; sleep 60; continue; }
    return $rc
  done
}

train_done() { [ -f "$1/head.pth" ] && [ -f "$1/master.pth" ] && J="$1/train.json" "$PY" -X utf8 -c "
import json, os, sys
sys.exit(0 if json.load(open(os.environ['J'])).get('status') == 'complete' else 1)" 2>/dev/null; }
head_dir() { case "$1" in sm113|sm115|sm118) echo "$SM119HEADS/sm119-$1-head";; *) echo "$SM117HEADS/sm117-$1-head";; esac; }
dumped_users() { J="$1" "$PY" -X utf8 -c "
import json, os
from pathlib import Path
p = Path(os.environ['J'])
print(len({r['user_id'] for r in map(json.loads, open(p, encoding='utf-8')) if 'error' not in r}) if p.exists() else 0)"; }
mname() { ( eval "$($H info --model "$1")"; echo "$MNAME" ); }
BG=""
start_recal() {  # background CPU recalibration of one model right after its dump (1 thread, < 1.5 GB RAM), stopped 5 min before the next window
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

after_sm124() {  # nothing is enqueued until the show-time two-head cool-down for Tiny (SM124)'s launcher has ended
  local n=0 last
  while true; do
    stopped && return 1
    last=$(tail -1 "$SM124LOG" 2>/dev/null)
    if [[ "$last" =~ ^\[sm124\ [^]]*\]\ (done|STOP) ]]; then log "the show-time two-head cool-down for Tiny (SM124) has ended: ${last:0:220}"; return 0; fi
    [ "${SKIP_SM124_WAIT:-0}" = 1 ] && { log "SKIP_SM124_WAIT=1: not waiting for the show-time two-head cool-down for Tiny (SM124)"; return 0; }
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN: would wait for the show-time two-head cool-down for Tiny (SM124) to end (last line: ${last:0:140})"; return 0; fi
    [ $((n % 12)) = 0 ] && log "waiting for the show-time two-head cool-down for Tiny (SM124) to end (no done / STOP last line in its launcher log yet; last: ${last:0:140})"
    n=$((n + 1)); sleep 300
  done
}

preconditions() {
  local key hd
  R2="$S/check-cpu.json" "$PY" -X utf8 -c "import json, os, sys; sys.exit(0 if json.load(open(os.environ['R2']))['pass'] else 1)" \
    || { log "STOP: the CPU check did not pass ($S/check-cpu.json)"; return 1; }
  export SM119_SM118_SHA256
  SM119_SM118_SHA256=$(J="$SM119HEADS/sm119-sm118-head/train.json" "$PY" -X utf8 -c "import json, os; print(json.load(open(os.environ['J']))['released_master_sha256'])")
  [ "$(sha256sum "$M118" | cut -d' ' -f1)" = "$SM119_SM118_SHA256" ] || { log "STOP: SM118's master sha256 differs from its head's train.json"; return 1; }
  for key in $MODELS; do
    hd=$(head_dir "$key")
    train_done "$hd" || { log "STOP: $key head missing or incomplete at $hd"; return 1; }
    $H info --model "$key" > /dev/null 2>> "$LOG" || { log "STOP: $key master missing or its sha256 differs"; return 1; }
    [ -f "$WT/artifacts/small-model/sm120/gpu/sm120-$key-rows.jsonl" ] || { log "STOP: no SM120 rows for $key"; return 1; }
  done
  if [ "$DRYRUN" != 1 ] && [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    log "STOP: uncommitted source changes in scripts/ expertium/ rwkv/"; return 1; fi
  return 0
}

smoke() {  # 3 users of the CPU check, Tiny two-head model (SM106), GPU: rows and sums equal SM120's GPU sums within 1e-5
  local si=$W/smoke-sm106.jsonl
  [ -f "$G/smoke-ok" ] && { log "GPU smoke already passed"; return 0; }
  rm -f "$si"
  "$PY" -X utf8 -c "import json; u = json.load(open(r'$W/check/users.json'))['users'][:3]; json.dump({'users': u}, open(r'$W/smoke-users.json', 'w'))" \
    || { log "STOP: no check users ($W/check/users.json)"; return 1; }
  q 6 "GPU smoke: dump of the Tiny two-head model (SM106) on 3 users vs SM120's recorded sums (cap 0.8)" -- \
    env -u RWKV_DELETED_CARDS $R dump --model sm106 --head-dir "$(head_dir sm106)" --users-file "$W/smoke-users.json" \
    --out-dir "$W/smoke-dump-sm106" --index "$si" >> "$G/smoke.out" 2>&1 || { log "STOP: GPU smoke ticket failed (see $G/smoke.out)"; return 1; }
  [ "$DRYRUN" = 1 ] && return 0
  ( cd "$WT" && "$PY" -X utf8 -c "
import json, sys
sys.path.insert(0, '.')
from scripts.sm125_recal import sm120_rows, dump_matches_sm120
old = sm120_rows('sm106')
res = [dump_matches_sm120(json.loads(l), old[json.loads(l)['user_id']], tol=1e-5) for l in open(r'$si')]
print('smoke', res)
sys.exit(0 if res and all(ok for ok, _ in res) else 1)" >> "$G/smoke.out" 2>&1 ) || { log "STOP: GPU smoke sums differ from SM120 (see $G/smoke.out)"; return 1; }
  touch "$G/smoke-ok"; log "GPU smoke passed: $(tail -1 "$G/smoke.out" | cut -c1-200)"
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

recal_all() {  # CPU, 3 processes at a time (each < 1.5 GB RAM, 1 thread), stopped 5 min before every window, resumed after it
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
    set -- $pending
    while [ $# -gt 0 ]; do
      local grp="$1 ${2:-} ${3:-}"; shift; [ $# -gt 0 ] && shift; [ $# -gt 0 ] && shift
      local pids=""
      for key in $grp; do
        recal_proc "$key" "$until" &
        pids="$pids $!"
      done
      wait $pids
      for key in $grp; do
        rc=$(cat "$W/recal-$key.rc" 2>/dev/null || echo 1)
        case "$rc" in
          0) log "$key recalibrated: $(tail -1 "$G/recal-$key.out" | cut -c1-160)";;
          3) log "$key recalibration paused before the window";;
          *) log "STOP: $key recalibration failed (exit $rc, see $G/recal-$key.out)"; return 1;;
        esac
      done
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
[ $rc -eq 0 ] && { after_sm124 || rc=1; }
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
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; else log "done"; fi
else
  log "STOP: ended with exit $rc"
fi
exit $rc
