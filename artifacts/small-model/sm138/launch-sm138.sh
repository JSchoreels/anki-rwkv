#!/usr/bin/env bash
# Answer-time heads for the six-layer Tiny (SM138): exactly the recipe of the time heads for the new candidates (SM119),
#   which is the recipe of the time heads for the family (SM117) (answer-time head MLP -> 128 -> 4 grades x 16 log-time
#   bins, zero-init output, trained ALONE on a frozen copy of the model, the explicit-only ablation in the same pass,
#   show-time inputs (e2s-query), same-day reviews included), on the two final models of the six-layer Tiny (SM130):
#   the two-head model (sm130-th-dec2ep32k-p2, scorer label sm130-th-dec2ep32k-t91s2) and the Instant-Only model
#   (sm130-io-dec2ep32k-p2, label sm130-io-dec2ep32k-t99), arch w64-c1-d2p1u1 (346 head inputs: 3 snapshots x 64 + 128 + 26).
#   Same steps per model as SM119: smoke -> head training -> grade-unchanged check (bitwise, separate scorer processes);
#   then gate scoring + P(recall within S seconds) at S = 3,5,8,10,15,20,30 on the 732 set, per-model `decide` and
#   `report` on CPU (outputs in artifacts/small-model/sm138/).
#   Answer time = front + back, question shown to grade press, capped at the max answer seconds; recall within S s = recalled
#   (Hard/Good/Easy) and answered within S s, reading the back included.
# Breaks: only quiet.json windows (scope all/gpu), re-read at every check (<= 10 min); no daily or Sunday break (no
#   planned breaks since 11-19). Every ticket starts only when its budget fits before the next window (the training is
#   resumable). GPU cap 0.8. Queue owner sm138.
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm138/artifacts/small-model/sm138/launch-sm138.sh
#   launch   WMI hidden (artifacts/small-model/sm138/HANDOFF.md); log artifacts/small-model/sm138/gpu/launch-sm138.log
#   resume   the launch command again (finished steps are skipped; a head training resumes from its resume.pt)
#   stop     touch C:/Users/jonat/workspace/wt/sm138/artifacts/small-model/sm138/gpu/STOP (checked between tickets)
# Do not edit scripts/ expertium/ rwkv/ of the sm138 worktree while it runs. Never writes to the sm130 worktree (masters read only).
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
QUIET=C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json
WT=C:/Users/jonat/workspace/wt/sm138
SM=$WT/artifacts/small-model
S=$SM/sm138
SIN=$SM/sm117   # SM117 inputs (plan, priors, baselines, user lists), read only; SM138 writes only to $S and $SM/sm138-*-head
G=$S/gpu
W=$S/work
mkdir -p "$G" "$W"
LOG=$G/launch-sm138.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm138-dryrun.log
H="$PY -X utf8 $WT/scripts/sm117_head.py"
SC=$WT/scripts/small_model_heldout.py
USERS732=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json   # read only (723 users)
MODELS="sm130th sm130io"   # the six-layer Tiny (SM130): two-head, then Instant-Only
P=sm138; [ "${DRYRUN:-0}" = 1 ] && P=sm138-dryrun
NAME="Answer-time heads for the six-layer Tiny (SM138)"
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START   # cap 0.8
log() { echo "[sm138 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
stopped() { [ -e "$G/STOP" ] && { log "STOP: STOP file present"; return 0; }; return 1; }
nap() { [ "$DRYRUN" = 1 ] && return 0; local s=$1; [ "$s" -gt 600 ] && s=600; sleep "$s"; }

block() {  # -> "NOW <seconds to its end>" inside a quiet window (quiet.json scope all/gpu only; no daily / Sunday break), else "NEXT <minutes>"
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
    local b; b=$(block | tr -d '\r')
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
    "$PY" "$QUEUE" run --owner sm138 --purpose "$NAME: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm138/launch-sm138.sh" -- "$@"
    rc=$?; log "exit $rc: $purpose"
    [ "$rc" = 75 ] && { log "queue wait would reach the next window; re-planning"; sleep 60; continue; }
    return $rc
  done
}
cpu_step() { wait_free "$1" "$2"; }   # CPU work also stays out of the windows
scored_users() { F="$1" "$PY" -X utf8 -c "
import json, os
from pathlib import Path
p = Path(os.environ['F'])
print(len({r['user_id'] for r in map(json.loads, open(p, encoding='utf-8')) if 'error' not in r}) if p.exists() else 0)" | tr -d '\r'; }
train_done() { [ -f "$1/head.pth" ] && R="$1/train.json" "$PY" -X utf8 -c "
import json, os, sys
sys.exit(0 if json.load(open(os.environ['R'])).get('status') == 'complete' else 1)" 2>/dev/null; }
budget() {  # budget KIND STEP -> minutes (fork = the Big models)
  case "$1:$2" in
    fork:smoke) echo 20;; *:smoke) echo 12;;
    fork:train) echo ${MIN_TRAIN_BIG:-170};; *:train) echo ${MIN_TRAIN:-90};;
    fork:grade) echo 15;; *:grade) echo 10;;
    fork:score) echo 45;; *:score) echo 30;;
    *:retry) echo 15;;
  esac
}

preconditions() {
  for f in "$SIN/plan.json" "$SIN/priors.json" "$SIN/time-baselines.jsonl" "$SIN/grade-users.json" "$SIN/smoke-users.json" "$USERS732" \
           "C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72/gpu-big/bigleak-neighbour.jsonl"; do
    [ -f "$f" ] || { log "STOP: missing $f"; return 1; }
  done
  for key in $MODELS; do
    $H info --model "$key" > /dev/null 2>> "$LOG" || { log "STOP: $key master missing or its sha256 differs from the package manifest"; return 1; }
  done
  if [ "$DRYRUN" != 1 ] && [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    log "STOP: uncommitted source changes in scripts/ expertium/ rwkv/"; return 1; fi
  return 0
}

prepare() {  # SM138's own CPU proof (the SM117 proof on both six-layer models, 11-21): check-cpu-sm138.json must have passed
  log "plan (SM117's, read only): $(R="$SIN/plan.json" "$PY" -X utf8 -c "import json, os; p = json.load(open(os.environ['R'])); print(f\"{p['users']:,} users, {p['reviews_used']:,} of {p['reviews_all']:,} reviews (cap {p['cap']:,}), {p['labelled_rows']:,} labelled rows ({p['same_day_rows']:,} same-day)\")")"
  local cp; cp=$(R="$S/check-cpu-sm138.json" "$PY" -X utf8 -c "import json, os; r = json.load(open(os.environ['R'])); print('PASS' if r['passed'] and not r['partial'] and r['models'] == ['sm130th', 'sm130io'] else 'FAIL', len(r['checks']), 'checks,', r['seconds'], 's')" 2>&1 | tr -d '\r')
  [[ "$cp" == PASS* ]] || { log "STOP: the CPU proof check-cpu-sm138.json did not pass: $cp"; return 1; }
  log "CPU proof: $cp (check-cpu-sm138.json)"
  return 0
}

model_steps() {  # model_steps KEY: smoke -> head training -> grade check (each once)
  # grade checks score the release and the trunk in two scorer processes (each first in its process): in one process
  # the second big (fork) model differs from the first by ~1e-7 (same weights; 2026-11-15 SM99 smoke, see design.md)
  local key=$1 rc=0
  eval "$($H info --model "$key" | tr -d '\r')"
  local TR=$SM/$P-$key-head SMK=$W/smoke-$P-$key
  local BUF_ARG="--buffer-rows $BUFFER"
  local SHIP_ARG=""; [ "$key" = sm92 ] && SHIP_ARG="--shipped-head $SHIPPED"
  if [ ! -e "$G/smoke-ok-$P-$key" ]; then
    rm -rf "$SMK-train" "$SMK-time.jsonl" "$SMK-time.hist" "$SMK-grade-rows.jsonl" "$SMK-grade-dump"
    echo '{"users": [2388, 4676]}' > "$W/smoke-grade-users.json"
    q "$(budget "$KIND" smoke)" "$MNAME: smoke (head training on 4 users: 3 small + the longest cut at 400k; time scoring of 3 users; grade check of 2 users)" -- \
      bash -c "$H train --model $key --out $SMK-train --max-users 3 --smoke-longest --buffer-rows 20000 --batch 1024 --log-every 5 \
        && $H score --model $key --head-dir $SMK-train --users-file $SIN/smoke-users.json --out $SMK-time.jsonl $SHIP_ARG \
        && env -u RWKV_DELETED_CARDS $PY -X utf8 $SC --pool 732 --users-file $W/smoke-grade-users.json --model $key-rel=$SPEC --only $key-rel/w8+s8 --inputs e2s-query --time-limit 2400 --split-existing --deleted-cards neighbour --dump $SMK-grade-dump --out $SMK-grade-rows.jsonl && env -u RWKV_DELETED_CARDS $PY -X utf8 $SC --pool 732 --users-file $W/smoke-grade-users.json --model $key-trunk=$SPECPFX$SMK-train/master.pth --only $key-trunk/w8+s8 --inputs e2s-query --time-limit 2400 --split-existing --deleted-cards neighbour --dump $SMK-grade-dump --out $SMK-grade-rows.jsonl" \
      >> "$G/check-$key.out" 2>&1 || { log "STOP: $key smoke failed (see $G/check-$key.out)"; return 1; }
    if [ "$DRYRUN" != 1 ]; then
      local gline sk first
      gline=$($H gradecompare --model "$key" --rows "$SMK-grade-rows.jsonl" --dump "$SMK-grade-dump" --rel-tag "$key-rel" --new-tag "$key-trunk" \
        --users-file "$W/smoke-grade-users.json" --out "$G/smoke-grade-check-$key.json" 2>&1 | tail -1)
      [[ "$gline" == *"bitwise True"* ]] || { log "STOP: $key smoke grade check not bitwise equal: $gline"; return 1; }
      first=$(grep '\[sm117-train\] step 1:' "$G/check-$key.out" | tail -1)
      [ -n "$first" ] || { log "STOP: $key smoke printed no [sm117-train] step 1 line"; return 1; }
      train_done "$SMK-train" || { log "STOP: $key smoke head training did not complete"; return 1; }
      sk=$(R="$SMK-train/train.json" "$PY" -X utf8 -c "import json, os; t = json.load(open(os.environ['R'])); print(len(t['skipped']), t['peak_gpu_gib'], t['seconds_this_run'], t['skipped'][:1])")
      if [ "${sk%% *}" != 0 ]; then
        if [ "$KIND" = fork ]; then log "WARNING: $key smoke skipped users (the training records skips; reported): $sk"
        else log "STOP: $key smoke head training skipped users (OOM or preparation): $sk"; return 1; fi
      fi
      log "$key smoke head training: skipped / peak GiB / seconds: $sk"
      [ "$(scored_users "$SMK-time.jsonl")" = 3 ] || { log "STOP: $key time-scorer smoke did not score 3 users"; return 1; }
      touch "$G/smoke-ok-$P-$key"; log "$key smoke: $first; $gline"
    fi
  fi
  if ! train_done "$TR"; then
    q "$(budget "$KIND" train)" "$MNAME: head training (1,991 users, first 400k reviews each, show-time inputs, same-day reviews included, head + explicit-only ablation)" -- \
      env -u RWKV_DELETED_CARDS $H train --model "$key" --out "$TR" $BUF_ARG >> "$G/train-$key.out" 2>&1 \
      || { log "STOP: $key head training ticket failed (see $G/train-$key.out)"; return 1; }
    if [ "$DRYRUN" != 1 ]; then
      train_done "$TR" || { log "STOP: $key head training did not complete (relaunch resumes from $TR/resume.pt)"; return 1; }
      log "$key head training: $(grep '\[sm117-train\] complete' "$G/train-$key.out" | tail -1 | cut -c1-300)"
    fi
  fi
  if [ ! -e "$G/grade-ok-$P-$key" ]; then
    q "$(budget "$KIND" grade)" "$MNAME: grade check (the release vs the trunk after head training, 5 users, show-time scorer rows and dumps bitwise)" -- \
      bash -c "env -u RWKV_DELETED_CARDS $PY -X utf8 $SC --pool 732 --users-file $SIN/grade-users.json --model $key-rel=$SPEC --only $key-rel/w8+s8 --inputs e2s-query --time-limit 2400 --split-existing --deleted-cards neighbour --dump $G/grade-dump-$key --out $G/grade-rows-$key.jsonl && env -u RWKV_DELETED_CARDS $PY -X utf8 $SC --pool 732 --users-file $SIN/grade-users.json --model $key-trunk=$SPECPFX$TR/master.pth --only $key-trunk/w8+s8 --inputs e2s-query --time-limit 2400 --split-existing --deleted-cards neighbour --dump $G/grade-dump-$key --out $G/grade-rows-$key.jsonl" \
      >> "$G/grade-$key.out" 2>&1 || { log "STOP: $key grade check ticket failed (see $G/grade-$key.out)"; return 1; }
    if [ "$DRYRUN" != 1 ]; then
      local line
      line=$($H gradecompare --model "$key" --rows "$G/grade-rows-$key.jsonl" --dump "$G/grade-dump-$key" --rel-tag "$key-rel" \
        --new-tag "$key-trunk" --users-file "$SIN/grade-users.json" --out "$G/grade-check-$key.json" 2>&1 | tail -1)
      log "$line"
      [[ "$line" == *"bitwise True"* ]] && touch "$G/grade-ok-$P-$key" || log "$key grade check: NOT bitwise equal (the rule reports fail-grade)"
    fi
  fi
  return 0
}

score_model() {  # score_model KEY: pass 1 at cap 0.8, pass 2 (missing users only) at 0.92
  local key=$1
  eval "$($H info --model "$key" | tr -d '\r')"
  local TR=$SM/$P-$key-head OUT=$G/$P-$key-rows.jsonl
  local SHIP_ARG=""; [ "$key" = sm92 ] && SHIP_ARG="--shipped-head $SHIPPED"
  if [ "$DRYRUN" = 1 ] || [ "$(scored_users "$OUT")" -lt 723 ]; then
    q "$(budget "$KIND" score)" "scoring $MNAME: answer time and P(recall within S seconds) on the 732 set (723 users, pass 1, cap 0.8)" -- \
      env -u RWKV_DELETED_CARDS $H score --model "$key" --head-dir "$TR" --out "$OUT" $SHIP_ARG >> "$G/score-$key.out" 2>&1 \
      || { log "STOP: $key scoring ticket failed (see $G/score-$key.out)"; return 1; }
  fi
  if [ "$DRYRUN" = 1 ] || [ "$(scored_users "$OUT")" -lt 723 ]; then
    q "$(budget "$KIND" retry)" "scoring $MNAME: answer time and P(recall within S seconds) on the 732 set (missing users, pass 2, cap 0.92)" -- \
      env -u RWKV_DELETED_CARDS HELDOUT_GPU_MEMORY_FRACTION=0.92 $H score --model "$key" --head-dir "$TR" --out "$OUT" $SHIP_ARG >> "$G/score-$key.out" 2>&1 \
      || { log "STOP: $key scoring pass 2 failed (see $G/score-$key.out)"; return 1; }
  fi
  if [ "$DRYRUN" != 1 ]; then
    local got; got=$(scored_users "$OUT")
    log "$key scored users: $got of 723"
    [ "$got" -ge 700 ] || { log "STOP: $key scoring has only $got of 723 users after 2 passes (see $G/score-$key.out)"; return 1; }
  fi
  return 0
}

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1) (DRYRUN=$DRYRUN; models $MODELS)"
rc=0
cpu_step 10 "preparation" || rc=1
[ $rc -eq 0 ] && { preconditions || rc=1; }
[ $rc -eq 0 ] && { prepare || rc=1; }
for key in $MODELS; do
  [ $rc -eq 0 ] && { model_steps "$key" || rc=1; }
done
for key in $MODELS; do
  [ $rc -eq 0 ] && { score_model "$key" || rc=1; }
done
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then
    for key in $MODELS; do
      log "DRYRUN: then $H decide --model $key --rows $G/$P-$key-rows.jsonl --grade-check $G/grade-check-$key.json --train-json $SM/$P-$key-head/train.json"
    done
    log "DRYRUN: then $H report"
  else
    cpu_step 10 "decision" || rc=1
    for key in $MODELS; do
      out=$($H decide --model "$key" --rows "$G/$P-$key-rows.jsonl" --grade-check "$G/grade-check-$key.json" \
        --train-json "$SM/$P-$key-head/train.json" 2>&1); r=$?
      log "decide $key exit $r: $(echo "$out" | tail -1)"
    done
    out=$($H report 2>&1); r=$?
    log "report exit $r: $(echo "$out" | tail -1)"; [ "$r" = 0 ] || rc=1
  fi
fi
if [ "$DRYRUN" = 1 ]; then rm -rf "$SM"/sm138-dryrun-*-head; fi
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; else log "done"; fi
else
  log "STOP: ended with exit $rc"
fi
exit $rc
