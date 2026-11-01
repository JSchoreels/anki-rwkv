#!/usr/bin/env bash
# Frozen-model answer-time head (SM82b): the response-time prediction (SM82) answer-time head (preset w64-sm82: per grade
#   16 log-time bins, Linear 128 -> 64 on the instant head's hidden layer, zero init) trained ALONE on a frozen copy of
#   the released small model (the leak-free small model (SM72s) 25 epochs + the 1-epoch 32k Muon cool-down, label
#   s-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon). The trunk runs forward only (time scorer path, fp32, leak-free
#   inputs, torch.no_grad, every parameter requires_grad False); only the head is in the optimizer (AdamW); the written
#   master = the released master's tensors + the head (scripts/sm82b_head.py).
# Data: the 1,991 training users of the SM82 quarter-epoch cool-down, each history's first 400k reviews (136.1M reviews,
#   81.9M time-labelled instant rows, one pass, 10k steps of 8,192 rows).
# Rule (scripts/sm82b_head.py decide, CPU): pass if the head beats the card's last time and the user's running median per
#   grade on the censored time LogLoss and the mean |ln error| (95% CI clear of zero, 722 users) AND grade is unchanged
#   (released vs frozen-with-head scorer rows and per-review dumps bitwise equal on 5 users). Also vs SM82's joint head.
#   Writes artifacts/small-model/sm82b/sm82b-decision.{md,json}.
# Steps: CPU preparation (extend, plan, CPU proof) -> ticket 0 (smoke: head training on 4 users + time scorer 3 users +
#   grade check 2 users) -> head training -> grade check (5 users) -> time scoring (723 users) -> rule.
# Breaks: quiet.json (scope all/gpu) plus the daily 20:00-01:00 break, re-read at every check (<= 10 min); every ticket
#   starts only when it fits before the next window (the head training is resumable from its last buffer pass).
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm82b/artifacts/small-model/sm82b/launch-sm82b.sh
#   launch   WMI hidden (artifacts/small-model/sm82b/HANDOFF.md); log artifacts/small-model/sm82b/gpu/launch-sm82b.log
#   resume   the launch command again (finished steps are skipped; the head training resumes from resume.pt)
#   stop     touch C:/Users/jonat/workspace/wt/sm82b/artifacts/small-model/sm82b/gpu/STOP (checked between tickets)
# Do not edit scripts/ expertium/ rwkv/ of the sm82b worktree while it runs.
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
QUIET=C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json
WT=C:/Users/jonat/workspace/wt/sm82b
SM=$WT/artifacts/small-model
S=$SM/sm82b
G=$S/gpu
mkdir -p "$G" "$S/work"
LOG=$G/launch-sm82b.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm82b-dryrun.log
H="$PY -X utf8 $WT/scripts/sm82b_head.py"
T82="$PY -X utf8 $WT/scripts/sm82_time.py"
SC=$WT/scripts/small_model_heldout.py
SMALL=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/scale-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon/expertium/E89-F65e-F77a/master.pth  # read only
SLBL=s-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon
S72=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s                                                  # read only
SOLD=$S72/gpu/small-lf-neighbour.jsonl
USERS732=$S72/eval-users-732-fit.json
S82=C:/Users/jonat/workspace/wt/sm82/artifacts/small-model                                                        # read only
PROTO=$S82/sm82-arm-e21-dec-q32k-muon/protocol.json
BASE=$S82/sm82/time-baselines.jsonl
SM82T=$S82/sm82/gpu/sm82-time.jsonl
P=sm82b; [ "${DRYRUN:-0}" = 1 ] && P=sm82b-dryrun
EXT=$SM/$P-ext-released             # the released master + the zero head
TRAIN=$SM/$P-head-released          # the head training (master.pth = released tensors + the trained head)
PLAN=$S/plan.json
TROWS=$G/sm82b-time.jsonl
NLBL=sm82b-frozen-head
MIN_SMOKE=10; MIN_TRAIN=${MIN_TRAIN:-80}; MIN_GRADE=10; MIN_TIME=25
NAME="frozen-model answer-time head (SM82b)"
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS   # cap 0.8 (scorers and the head training)
log() { echo "[sm82b $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
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
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm82b/launch-sm82b.sh" -- "$@"
    rc=$?; log "exit $rc: $purpose"
    [ "$rc" = 75 ] && { log "queue wait would reach the next window; re-planning"; sleep 60; continue; }
    return $rc
  done
}
cpu_step() { wait_free "$1" "$2"; }   # CPU work also stays out of the windows
time_users() { F="$1" "$PY" -X utf8 -c "
import json, os
from pathlib import Path
p = Path(os.environ['F'])
print(len({r['user_id'] for r in map(json.loads, open(p, encoding='utf-8')) if 'error' not in r}) if p.exists() else 0)"; }
train_done() { [ -f "$1/master.pth" ] && R="$1/train.json" "$PY" -X utf8 -c "
import json, os, sys
sys.exit(0 if json.load(open(os.environ['R'])).get('status') == 'complete' else 1)" 2>/dev/null; }

preconditions() {
  for f in "$SMALL" "$SOLD" "$USERS732" "$PROTO" "$BASE" "$SM82T" "$S/grade-users.json" "$S/smoke-users.json" \
           "C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72/gpu-big/bigleak-neighbour.jsonl"; do
    [ -f "$f" ] || { log "STOP: missing $f"; return 1; }
  done
  if [ "$DRYRUN" != 1 ] && [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    log "STOP: uncommitted source changes in scripts/ expertium/ rwkv/"; return 1; fi
  return 0
}

prepare() {  # CPU: the extended start, the plan, the CPU proof
  if [ ! -f "$EXT/extend.json" ]; then
    CUDA_VISIBLE_DEVICES= $T82 extend --master "$SMALL" --out "$EXT" >> "$G/prepare.out" 2>&1 || { log "STOP: extend failed"; return 1; }
    log "extended: $EXT (the released small model + the zero answer-time head)"
  fi
  if [ ! -f "$PLAN" ]; then
    $H plan --protocol "$PROTO" --cap 400000 --out "$S" >> "$G/prepare.out" 2>&1 || { log "STOP: plan failed"; return 1; }
  fi
  log "plan: $(R="$PLAN" "$PY" -X utf8 -c "import json, os; p = json.load(open(os.environ['R'])); print(f\"{p['users']:,} users, {p['reviews_used']:,} of {p['reviews_all']:,} reviews (cap {p['cap']:,}, {p['users_cut']} users cut), {p['labelled_rows']:,} labelled rows\")")"
  if [ ! -e "$G/check-cpu-ok-$P" ]; then
    local line
    line=$(CUDA_VISIBLE_DEVICES= $H check --ext "$EXT" --plan "$PLAN" --out "$G/check-cpu-$P.json" 2>&1 | tail -1)
    log "$line"
    [[ "$line" == "[sm82b-check] PASS"* ]] || { log "STOP: CPU proof failed (see $G/check-cpu-$P.json)"; return 1; }
    touch "$G/check-cpu-ok-$P"
  fi
  return 0
}

score_time() {  # the head on the 723 users (fp32, leak-free inputs), resumable, 2 passes
  local pass
  for pass in 1 2; do
    if [ "$DRYRUN" = 1 ] || [ "$(time_users "$TROWS")" -lt 723 ]; then
      q $([ $pass = 1 ] && echo $MIN_TIME || echo 10) "time scoring of the head on the 732 pool minus 9 never-fit (723, pass $pass)" -- \
        env -u RWKV_DELETED_CARDS $T82 score --master "$TRAIN/master.pth" --baselines "$BASE" --users-file "$USERS732" \
        --out "$TROWS" >> "$G/time.out" 2>&1 || { log "STOP: time scoring ticket failed"; return 1; }
      [ "$DRYRUN" = 1 ] && return 0
    fi
  done
  local got; got=$(time_users "$TROWS")
  [ "$got" -ge 722 ] || { log "STOP: time scoring has $got of 723 users after 2 passes"; return 1; }
  return 0
}

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1) (DRYRUN=$DRYRUN, head on the frozen released small model: $SMALL)"
rc=0
cpu_step 10 "preparation" || rc=1
[ $rc -eq 0 ] && { preconditions || rc=1; }
[ $rc -eq 0 ] && { prepare || rc=1; }
# ticket 0: head training on 4 small users + the time scorer (3 users) + the grade check (2 users) with the smoke head, once
if [ $rc -eq 0 ] && [ ! -e "$G/smoke-ok-$P" ]; then
  W=$S/work; rm -rf "$W/smoke-train-$P" "$W/smoke-time-$P.jsonl" "$W/smoke-grade-rows-$P.jsonl" "$W/smoke-grade-dump-$P"
  echo '{"users": [2388, 4676]}' > "$W/smoke-grade-users.json"
  q $MIN_SMOKE "smoke: head training on 4 users (3 small + the longest, cut at 400k) + time scoring (3 users) + grade check (2 users) on the frozen released small model" --     bash -c "$H train --start $EXT --plan $PLAN --out $W/smoke-train-$P --max-users 3 --smoke-longest --buffer-rows 20000 --batch 1024 --log-every 5       && env -u RWKV_DELETED_CARDS $T82 score --master $W/smoke-train-$P/master.pth --baselines $BASE --users-file $S/smoke-users.json --out $W/smoke-time-$P.jsonl       && env -u RWKV_DELETED_CARDS $PY -X utf8 $SC --pool 732 --users-file $W/smoke-grade-users.json --model $SLBL=expertium:w64:$SMALL --model $NLBL=expertium:w64-sm82:$W/smoke-train-$P/master.pth --only $SLBL/w8+s8 --only $NLBL/w8+s8 --time-limit 2400 --split-existing --deleted-cards neighbour --dump $W/smoke-grade-dump-$P --out $W/smoke-grade-rows-$P.jsonl"     >> "$G/check.out" 2>&1 || { log "STOP: smoke failed (see $G/check.out)"; rc=1; }
  if [ $rc -eq 0 ] && [ "$DRYRUN" != 1 ]; then
    gline=$($H gradecompare --rows "$W/smoke-grade-rows-$P.jsonl" --dump "$W/smoke-grade-dump-$P" --rel-tag "$SLBL" --new-tag "$NLBL"       --users-file "$W/smoke-grade-users.json" --existing "$SOLD" --existing-label "$SLBL/w8+s8" --out "$G/smoke-grade-check.json" 2>&1 | tail -1)
    [[ "$gline" == *"bitwise True"* ]] || { log "STOP: smoke grade check not bitwise equal: $gline"; rc=1; }
  fi
  if [ $rc -eq 0 ] && [ "$DRYRUN" != 1 ]; then
    first=$(grep '\[sm82b-train\] step 1:' "$G/check.out" | tail -1)
    [ -n "$first" ] || { log "STOP: the smoke printed no [sm82b-train] step 1 line"; rc=1; }
    train_done "$W/smoke-train-$P" || { log "STOP: the smoke head training did not complete"; rc=1; }
    sk=$(R="$W/smoke-train-$P/train.json" "$PY" -X utf8 -c "import json, os; t = json.load(open(os.environ['R'])); print(len(t['skipped']), t['peak_gpu_gib'], t['seconds_this_run'])")
    [ "${sk%% *}" = 0 ] || { log "STOP: the smoke head training skipped users (OOM or preparation): $sk"; rc=1; }
    log "smoke head training: skipped / peak GiB / seconds: $sk"
    [ "$(time_users "$W/smoke-time-$P.jsonl")" = 3 ] || { log "STOP: the time-scorer smoke did not score 3 users"; rc=1; }
    [ $rc -eq 0 ] && { touch "$G/smoke-ok-$P"; log "smoke: $(grep '\[sm82b-train\] complete' "$G/check.out" | tail -1 | cut -c1-220); $first; $gline"; }
  fi
fi
if [ $rc -eq 0 ] && ! train_done "$TRAIN"; then
  q $MIN_TRAIN "head training on the frozen released small model (1,991 users of the quarter-epoch cool-down, first 400k reviews each, 10k AdamW steps)" -- \
    env -u RWKV_DELETED_CARDS $H train --start "$EXT" --plan "$PLAN" --out "$TRAIN" >> "$G/train.out" 2>&1 \
    || { log "STOP: head training ticket failed (see $G/train.out)"; rc=1; }
  if [ $rc -eq 0 ] && [ "$DRYRUN" != 1 ]; then
    train_done "$TRAIN" || { log "STOP: head training did not complete (relaunch resumes from $TRAIN/resume.pt)"; rc=1; }
    [ $rc -eq 0 ] && log "head training: $(grep '\[sm82b-train\] complete' "$G/train.out" | tail -1 | cut -c1-260)"
  fi
fi
if [ $rc -eq 0 ] && [ ! -e "$G/grade-ok-$P" ]; then
  q $MIN_GRADE "grade check: the released small model vs the frozen model with the head (5 users, scorer rows and dumps bitwise)" -- \
    bash -c "env -u RWKV_DELETED_CARDS $PY -X utf8 $SC --pool 732 --users-file $S/grade-users.json --model $SLBL=expertium:w64:$SMALL --model $NLBL=expertium:w64-sm82:$TRAIN/master.pth --only $SLBL/w8+s8 --only $NLBL/w8+s8 --time-limit 2400 --split-existing --deleted-cards neighbour --dump $G/grade-dump --out $G/grade-check-rows.jsonl" \
    >> "$G/grade.out" 2>&1 || { log "STOP: grade check ticket failed (see $G/grade.out)"; rc=1; }
  if [ $rc -eq 0 ] && [ "$DRYRUN" != 1 ]; then
    line=$($H gradecompare --rows "$G/grade-check-rows.jsonl" --dump "$G/grade-dump" --rel-tag "$SLBL" --new-tag "$NLBL" \
      --users-file "$S/grade-users.json" --existing "$SOLD" --existing-label "$SLBL/w8+s8" --out "$G/grade-check.json" 2>&1 | tail -1)
    log "$line"
    [[ "$line" == *"bitwise True"* ]] && touch "$G/grade-ok-$P" || log "grade check: NOT bitwise equal (the rule reports fail-grade)"
  fi
fi
[ $rc -eq 0 ] && { score_time || rc=1; }
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then
    log "DRYRUN: then $H decide --time-rows $TROWS --grade-check $G/grade-check.json --train-json $TRAIN/train.json"
  else
    cpu_step 3 "rule" || rc=1
    out=$($H decide --time-rows "$TROWS" --grade-check "$G/grade-check.json" --train-json "$TRAIN/train.json" 2>&1); r=$?
    line=$(echo "$out" | tail -1)
    log "rule exit $r: $line"; [ "$r" = 0 ] || rc=1
  fi
fi
if [ "$DRYRUN" = 1 ]; then rm -rf "$EXT" "$TRAIN"; rm -f "$G/check-cpu-ok-$P" "$G/check-cpu-$P.json"; fi
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; else log "done"; fi
else
  log "STOP: ended with exit $rc"
fi
exit $rc
