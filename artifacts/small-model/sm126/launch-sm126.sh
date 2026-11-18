#!/usr/bin/env bash
# Time head for the show-time two-head cool-down for Tiny (SM126) (design: artifacts/small-model/sm126/design.md). Two chained parts:
#   A. SM126's recipe for the single model the show-time two-head cool-down for Tiny (SM124) (key sm124, Tiny two-head, 474 head
#      inputs, final master = its pass-2 folder, sha256 pinned in scripts/sm117_head.py): smoke -> head training (alone on the frozen
#      model, show-time inputs, same-day reviews included, explicit-only ablation in the same pass) -> grade-unchanged check
#      (bitwise, separate scorer processes) -> gate scoring + P(recall within S seconds) on the 732 set -> `decide` + `report` (CPU).
#   B. SM125's recipe for the same model with its new head: SM120's scoring of fast Good/Easy prediction (its rows hold the
#      baselines, which depend on the model's own grades), a 3-user GPU smoke of the dump (vs those rows within 1e-5 and vs step A's
#      recall-within-S sums), the per-review dump on the 732 set (723 users), the CPU rolling Platt recalibration, the report.
#   Answer time = front + back, question shown to grade press, capped at the max answer seconds; "within S s" = answered within S s,
#   reading the back included.
# No other launcher is waited for (the show-time two-head cool-down for Tiny (SM124) is finished: verdict go, 11-18 04:46).
# Breaks: quiet.json (scope all/gpu) plus the daily 20:00-01:00 break and Sunday from 16:00, re-read at every check
#   (<= 10 min); every GPU ticket starts only when its budget fits before the next window (the head training is resumable; the
#   CPU recalibration stops itself 5 min before a window and resumes after it). GPU cap 0.8 (missing users at 0.92).
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm126/artifacts/small-model/sm126/launch-sm126.sh
#   launch   WMI hidden (artifacts/small-model/sm126/HANDOFF.md); log artifacts/small-model/sm126/gpu/launch-sm126.log
#   resume   the launch command again (finished steps are skipped; a head training resumes from its resume.pt)
#   stop     touch C:/Users/jonat/workspace/wt/sm126/artifacts/small-model/sm126/gpu/STOP (checked between tickets and steps)
# Do not edit scripts/ expertium/ rwkv/ of the sm126 worktree while it runs. Read only: wt/sm124 (master, rows), wt/sm117, wt/sm126,
# wt/sm125 (never touched).
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
QUIET=C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json
WT=C:/Users/jonat/workspace/wt/sm126
SM=$WT/artifacts/small-model
S=$SM/sm126
SIN=$SM/sm117   # SM117 inputs (plan, priors, baselines, user lists), read only; SM126 writes only to $S and $SM/sm126-*-head
G=$S/gpu
W=$S/work
mkdir -p "$G" "$W"
LOG=$G/launch-sm126.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm126-dryrun.log
H="$PY -X utf8 $WT/scripts/sm117_head.py"
SC=$WT/scripts/small_model_heldout.py
USERS732=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json   # read only (723 users)
MODELS="sm124"   # the show-time two-head cool-down for Tiny (SM124)
P=sm126; [ "${DRYRUN:-0}" = 1 ] && P=sm126-dryrun
NAME="time head for the show-time two-head cool-down for Tiny (SM126)"
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export SM117_OUT=artifacts/small-model/sm126 SM125_OUT=artifacts/small-model/sm126   # outputs of both scripts land here
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START   # cap 0.8
log() { echo "[sm126 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
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
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm126/launch-sm126.sh" -- "$@"
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
print(len({r['user_id'] for r in map(json.loads, open(p, encoding='utf-8')) if 'error' not in r}) if p.exists() else 0)"; }
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

model_steps() {  # model_steps KEY: smoke -> head training -> grade check (each once)
  # grade checks score the release and the trunk in two scorer processes (each first in its process): in one process
  # the second big (fork) model differs from the first by ~1e-7 (same weights; 2026-11-15 SM99 smoke, see design.md)
  local key=$1 rc=0
  eval "$($H info --model "$key")"
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
  eval "$($H info --model "$key")"
  local TR=$SM/$P-$key-head OUT=$G/sm126-$key-rows.jsonl
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

prepare() {  # no new CPU proof: SM126 only adds one entry (sm124) to MODELS of the SM117 head script (SM117's proof: PASS, 22 checks)
  log "plan (SM117's, read only): $(R="$SIN/plan.json" "$PY" -X utf8 -c "import json, os; p = json.load(open(os.environ['R'])); print(f\"{p['users']:,} users, {p['reviews_used']:,} of {p['reviews_all']:,} reviews (cap {p['cap']:,}), {p['labelled_rows']:,} labelled rows ({p['same_day_rows']:,} same-day)\")")"
  log "CPU proof: SM117's ($(cat "$SIN/check-cpu-ok" 2>/dev/null)); only the model table differs"
  return 0
}

# ------------------------------------------------------------------ part B (SM125's recipe for SM124 with its new head)
R="$PY -X utf8 $WT/scripts/sm125_recal.py"
F="$PY -X utf8 $WT/scripts/sm120_fast.py"
HD=$SM/sm126-sm124-head                  # the head of part A (its master.pth = the released model, checked by the dump)
S120=$G/sm120-sm124-rows.jsonl           # SM120's code on SM124 + its new head: baselines (they depend on the model's own grades) and reference sums
IX=$G/dump-sm124.jsonl
MN124=$(eval "$($H info --model sm124)"; echo "$MNAME")

preconditions_b() {
  R2="$SM/sm125/check-cpu.json" "$PY" -X utf8 -c "import json, os, sys; sys.exit(0 if json.load(open(os.environ['R2']))['pass'] else 1)" \
    || { log "STOP: SM125's CPU check did not pass ($SM/sm125/check-cpu.json)"; return 1; }
  [ -f "$SM/sm120/priors.json" ] || { log "STOP: missing $SM/sm120/priors.json"; return 1; }
  if [ "$DRYRUN" != 1 ]; then
    { [ -f "$HD/head.pth" ] && [ -f "$HD/master.pth" ]; } || { log "STOP: part A head missing at $HD"; return 1; }
    [ -e "$G/grade-ok-$P-sm124" ] || { log "STOP: the grade check of part A is not bitwise equal: no head to dump"; return 1; }
    [ -z "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ] || { log "STOP: uncommitted source changes in scripts/ expertium/ rwkv/"; return 1; }
  fi
  return 0
}

sm120_score() {  # SM120's scoring (grade, fast events, baselines) of SM124 with the new head: pass 1 at cap 0.8, pass 2 (missing users) at 0.92
  if [ "$DRYRUN" = 1 ] || [ "$(scored_users "$S120")" -lt 723 ]; then
    q 30 "$MN124: scoring with fast Good/Easy prediction (SM120)'s code, P(Good or Easy), P(Easy), P(set and answered within S s) and the baselines on the 732 set (723 users, pass 1, cap 0.8)" -- \
      env -u RWKV_DELETED_CARDS $F score --model sm124 --head-dir "$HD" --out "$S120" >> "$G/sm120-score.out" 2>&1 \
      || { log "STOP: SM120-code scoring ticket failed (see $G/sm120-score.out)"; return 1; }
  fi
  if [ "$DRYRUN" = 1 ] || [ "$(scored_users "$S120")" -lt 723 ]; then
    q 15 "$MN124: scoring with fast Good/Easy prediction (SM120)'s code (missing users, pass 2, cap 0.92)" -- \
      env -u RWKV_DELETED_CARDS HELDOUT_GPU_MEMORY_FRACTION=0.92 $F score --model sm124 --head-dir "$HD" --out "$S120" >> "$G/sm120-score.out" 2>&1 \
      || { log "STOP: SM120-code scoring pass 2 failed (see $G/sm120-score.out)"; return 1; }
  fi
  if [ "$DRYRUN" != 1 ]; then
    local got; got=$(scored_users "$S120"); log "sm124 SM120-code scored users: $got of 723"
    [ "$got" -ge 700 ] || { log "STOP: SM120-code scoring has only $got of 723 users (see $G/sm120-score.out)"; return 1; }
  fi
  return 0
}

smoke_b() {  # 3 users: the dump's float64 sums equal the SM120-code rows (within 1e-5) and step A's recall-within-S sums (within 1e-5)
  local si=$W/smoke-dump-sm124.jsonl
  [ -f "$G/smoke-b-ok" ] && { log "part B GPU smoke already passed"; return 0; }
  rm -rf "$si" "$W/smoke-dump-sm124"
  q 6 "$MN124: GPU smoke of the per-review dump on 3 users vs the SM120-code rows and the sums of the head scoring (cap 0.8)" -- \
    env -u RWKV_DELETED_CARDS $R dump --model sm124 --head-dir "$HD" --users-file "$SIN/smoke-users.json" \
    --out-dir "$W/smoke-dump-sm124" --index "$si" >> "$G/smoke-b.out" 2>&1 || { log "STOP: part B GPU smoke ticket failed (see $G/smoke-b.out)"; return 1; }
  [ "$DRYRUN" = 1 ] && return 0
  $R stepa --model sm124 --index "$si" --rows "$G/sm126-sm124-rows.jsonl" >> "$G/smoke-b.out" 2>&1 \
    || { log "STOP: part B smoke sums differ from the head scoring of part A (see $G/smoke-b.out)"; return 1; }
  ( cd "$WT" && "$PY" -X utf8 -c "
import json, sys
sys.path.insert(0, '.')
from scripts.sm125_recal import sm120_rows, dump_matches_sm120
old = sm120_rows('sm124')
res = [dump_matches_sm120(json.loads(l), old[json.loads(l)['user_id']], tol=1e-5) for l in open(r'$si')]
print('smoke vs SM120-code rows', res)
sys.exit(0 if res and all(ok for ok, _ in res) else 1)" >> "$G/smoke-b.out" 2>&1 ) \
    || { log "STOP: part B smoke sums differ from the SM120-code rows (see $G/smoke-b.out)"; return 1; }
  touch "$G/smoke-b-ok"; log "part B GPU smoke passed: $(tail -2 "$G/smoke-b.out" | cut -c1-200 | tr '\n' ' ')"
  return 0
}

dump_b() {  # pass 1 at cap 0.8, pass 2 (missing users only) at 0.92
  local got out r
  if [ "$DRYRUN" = 1 ] || [ "$(scored_users "$IX")" -lt 723 ]; then
    q 30 "per-review dump of $MN124 on the 732 set (723 users, pass 1, cap 0.8)" -- \
      env -u RWKV_DELETED_CARDS $R dump --model sm124 --head-dir "$HD" >> "$G/dump-sm124.out" 2>&1 \
      || { log "STOP: dump ticket failed (see $G/dump-sm124.out)"; return 1; }
  fi
  if [ "$DRYRUN" = 1 ] || [ "$(scored_users "$IX")" -lt 723 ]; then
    q 15 "per-review dump of $MN124 on the 732 set (missing users, pass 2, cap 0.92)" -- \
      env -u RWKV_DELETED_CARDS HELDOUT_GPU_MEMORY_FRACTION=0.92 $R dump --model sm124 --head-dir "$HD" >> "$G/dump-sm124.out" 2>&1 \
      || { log "STOP: dump pass 2 failed (see $G/dump-sm124.out)"; return 1; }
  fi
  if [ "$DRYRUN" != 1 ]; then
    got=$(scored_users "$IX")
    log "sm124 dumped users: $got of 723 ($(du -sh "$W/dump-sm124" 2>/dev/null | cut -f1))"
    [ "$got" -ge 700 ] || { log "STOP: dump has only $got of 723 users after 2 passes"; return 1; }
    out=$($R stepa --model sm124 --index "$IX" --rows "$G/sm126-sm124-rows.jsonl" 2>&1); r=$?
    log "dump vs head scoring (all users) exit $r: $(echo "$out" | tail -1 | cut -c1-300)"
    [ "$r" = 0 ] || { log "STOP: the full dump differs from the head scoring of part A"; return 1; }
  fi
  return 0
}

recal_proc() {  # recal_proc UNTIL: exit code in work/recal-sm124.rc, marker gpu/recal-ok-sm124 on success
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $R recal --model sm124 --until "$1" >> "$G/recal-sm124.out" 2>&1
  local rc=$?; echo "$rc" > "$W/recal-sm124.rc"; [ "$rc" = 0 ] && touch "$G/recal-ok-sm124"; return 0
}
recal_b() {  # CPU, 1 process (1 thread, < 1.5 GB RAM, ~1.2 h), stopped 5 min before every window, resumed after it
  local rounds=0 until rc
  while true; do
    stopped && return 1
    [ -f "$G/recal-ok-sm124" ] && return 0
    wait_free 20 "CPU recalibration" || return 1
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN: CPU recalibration of sm124"; return 0; fi
    until=$(( $(date +%s) + (FREE - 5) * 60 ))
    log "CPU recalibration of sm124 (until $(date -d @$until '+%m-%d %H:%M'))"
    rm -f "$W/recal-sm124.rc"; recal_proc "$until"
    rc=$(cat "$W/recal-sm124.rc" 2>/dev/null || echo 1)
    case "$rc" in
      0) log "sm124 recalibrated: $(tail -1 "$G/recal-sm124.out" | cut -c1-160)";;
      3) log "sm124 recalibration paused before the window";;
      *) log "STOP: sm124 recalibration failed (exit $rc, see $G/recal-sm124.out)"; return 1;;
    esac
    rounds=$((rounds + 1)); [ "$rounds" -gt 20 ] && { log "STOP: recalibration did not finish in 20 rounds"; return 1; }
  done
}

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1) (DRYRUN=$DRYRUN; models $MODELS)"
rc=0
cpu_step 10 "preparation" || rc=1
[ $rc -eq 0 ] && { preconditions || rc=1; }
[ $rc -eq 0 ] && { prepare || rc=1; }
# ---- part A: the time head
for key in $MODELS; do [ $rc -eq 0 ] && { model_steps "$key" || rc=1; }; done
for key in $MODELS; do [ $rc -eq 0 ] && { score_model "$key" || rc=1; }; done
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then
    for key in $MODELS; do
      log "DRYRUN: then $H decide --model $key --rows $G/sm126-$key-rows.jsonl --grade-check $G/grade-check-$key.json --train-json $SM/$P-$key-head/train.json"
    done
    log "DRYRUN: then $H report"
  else
    cpu_step 10 "decision" || rc=1
    for key in $MODELS; do
      out=$($H decide --model "$key" --rows "$G/sm126-$key-rows.jsonl" --grade-check "$G/grade-check-$key.json" \
        --train-json "$SM/$P-$key-head/train.json" 2>&1); r=$?
      log "decide $key exit $r: $(echo "$out" | tail -1 | cut -c1-400)"
    done
    out=$($H report 2>&1); r=$?
    log "report (part A) exit $r: $(echo "$out" | tail -1 | cut -c1-300)"; [ "$r" = 0 ] || rc=1
    [ $rc -eq 0 ] && log "PART A done (time head of sm124 trained, grade-checked and scored)"
  fi
fi
# ---- part B: rolling recalibration (needs part A's head and rows; runs whatever the gate verdict of part A)
[ $rc -eq 0 ] && { preconditions_b || rc=1; }
[ $rc -eq 0 ] && { sm120_score || rc=1; }
[ $rc -eq 0 ] && { smoke_b || rc=1; }
[ $rc -eq 0 ] && { dump_b || rc=1; }
[ $rc -eq 0 ] && { recal_b || rc=1; }
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then
    log "DRYRUN: then $R report --models sm124"
  else
    wait_free 20 "report" || rc=1
    if [ $rc -eq 0 ]; then out=$($R report --models sm124 2>&1); r=$?
      log "report (part B) exit $r: $(echo "$out" | tail -1 | cut -c1-400)"; [ "$r" = 0 ] || rc=1; fi
  fi
fi
if [ "$DRYRUN" = 1 ]; then rm -rf "$SM"/sm126-dryrun-*-head; fi
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; else log "done"; fi
else
  log "STOP: ended with exit $rc"
fi
exit $rc
