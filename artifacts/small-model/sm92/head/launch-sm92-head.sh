#!/usr/bin/env bash
# Show-time retrain of the release (SM92): answer-time head on richer frozen features (SM82c) retrained on the frozen
#   SM92 model. The shipped head (SM83) reads the frozen released model's internal states, so it is invalid for the new
#   weights. Exact SM82c recipe (scripts/sm82c_head.py of the sm82c worktree, read only, unchanged): MLP 474 -> 128 -> 4
#   grades x 16 log-time bins, zero-init output, preset w64-sm82c time_mlps.0, trained ALONE on a frozen copy of the
#   show-time retrain of the release (SM92) master (sm92-e26-dec-1ep32k-muon, label sm92-e26-dec1ep32k); the same plan
#   (1,991 users, first 400k reviews each), buffer, batch, LR, seed; the explicit-only ablation in the same pass.
#   Only the frozen master, the recorded grade rows (sm92-inputs.jsonl, normal inputs) and the output folders change.
# Gate (sm92_head_gate.py, CPU): time LogLoss on the same 722-user evaluation <= SM82c's 1.9568 x 1.01; plus SM82c's own
#   rule (grade unchanged bitwise on 5 users, head beats both baselines), written as sm92-head-decision.{md,json}.
# Steps: CPU proof -> ticket 0 (smoke) -> head training -> grade check (5 users) -> time scoring (723 users) -> gate.
# Breaks: quiet.json (scope all/gpu) plus the daily 20:00-01:00 break and Sunday from 16:00, re-read at every check.
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm92/artifacts/small-model/sm92/head/launch-sm92-head.sh
#   launch   WMI hidden (artifacts/small-model/sm92/head/HANDOFF.md); log artifacts/small-model/sm92/head/gpu/launch-sm92-head.log
#   resume   the launch command again (finished steps are skipped; the head training resumes from resume.pt)
#   stop     touch C:/Users/jonat/workspace/wt/sm92/artifacts/small-model/sm92/head/gpu/STOP (checked between tickets)
# Do not edit scripts/ expertium/ rwkv/ of the sm82c worktree while it runs.
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
QUIET=C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json
WT=C:/Users/jonat/workspace/wt/sm82c                      # scripts (read only)
S82C=$WT/artifacts/small-model/sm82c                      # SM82c plan / user files (read only)
SM=C:/Users/jonat/workspace/wt/sm92/artifacts/small-model
S=$SM/sm92/head
G=$S/gpu
mkdir -p "$G" "$S/work"
LOG=$G/launch-sm92-head.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm92-head-dryrun.log
H="$PY -X utf8 $WT/scripts/sm82c_head.py"
GATE="$PY -X utf8 $S/sm92_head_gate.py"
SC=$WT/scripts/small_model_heldout.py
SMALL=$SM/sm92-e26-dec-1ep32k-muon/expertium/E89-F65e-F77a/master.pth   # the SM92 master, read only
SLBL=sm92-e26-dec1ep32k
S72=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s                                                  # read only
SOLD=$SM/sm92/gpu/sm92-inputs.jsonl     # SM92's recorded normal-input scores (label sm92-e26-dec1ep32k/w8+s8)
USERS732=$S72/eval-users-732-fit.json
S82=C:/Users/jonat/workspace/wt/sm82/artifacts/small-model                                                        # read only
BASE=$S82/sm82/time-baselines.jsonl
SM82T=$S82/sm82/gpu/sm82-time.jsonl
SM82BT=C:/Users/jonat/workspace/wt/sm82b/artifacts/small-model/sm82b/gpu/sm82b-time.jsonl                         # read only
P=sm92; [ "${DRYRUN:-0}" = 1 ] && P=sm92-dryrun
TRAIN=$SM/$P-head                   # the head training (master.pth = SM92 tensors + the trained head; ablation-head.pth)
PLAN=$S82C/plan.json                # = SM82c's plan (same users, order, cap)
TROWS=$G/sm92-head-time.jsonl
AROWS=$G/sm92-head-time-ablation.jsonl
NLBL=sm92-frozen-head
MIN_SMOKE=12; MIN_TRAIN=${MIN_TRAIN:-90}; MIN_GRADE=10; MIN_TIME=30
NAME="show-time retrain of the release (SM92): answer-time head (SM82c recipe)"
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS   # cap 0.8 (scorers and the head training)
log() { echo "[sm92-head $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
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
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm92/head/launch-sm92-head.sh" -- "$@"
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
both_users() { local a b; a=$(time_users "$TROWS"); b=$(time_users "$AROWS"); [ "$a" -lt "$b" ] && echo "$a" || echo "$b"; }
train_done() { [ -f "$1/master.pth" ] && R="$1/train.json" "$PY" -X utf8 -c "
import json, os, sys
sys.exit(0 if json.load(open(os.environ['R'])).get('status') == 'complete' else 1)" 2>/dev/null; }

preconditions() {
  for f in "$SMALL" "$SOLD" "$USERS732" "$BASE" "$SM82T" "$SM82BT" "$PLAN" "$S82C/grade-users.json" "$S82C/smoke-users.json" \
           "C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72/gpu-big/bigleak-neighbour.jsonl"; do
    [ -f "$f" ] || { log "STOP: missing $f"; return 1; }
  done
  if [ "$DRYRUN" != 1 ] && [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    log "STOP: uncommitted source changes in scripts/ expertium/ rwkv/"; return 1; fi
  return 0
}

prepare() {  # CPU: the CPU proof
  log "plan: $(R="$PLAN" "$PY" -X utf8 -c "import json, os; p = json.load(open(os.environ['R'])); print(f\"{p['users']:,} users, {p['reviews_used']:,} of {p['reviews_all']:,} reviews (cap {p['cap']:,}, {p['users_cut']} users cut), {p['labelled_rows']:,} labelled rows\")")"
  if [ ! -e "$G/check-cpu-ok-$P" ]; then
    local line
    line=$(CUDA_VISIBLE_DEVICES= $H check --released "$SMALL" --plan "$PLAN" --out "$G/check-cpu-$P.json" 2>&1 | tail -1)
    log "$line"
    [[ "$line" == "[sm82c-check] PASS"* ]] || { log "STOP: CPU proof failed (see $G/check-cpu-$P.json)"; return 1; }
    touch "$G/check-cpu-ok-$P"
  fi
  return 0
}

score_time() {  # both heads on the 723 users (fp32, leak-free inputs), resumable, 2 passes
  local pass
  for pass in 1 2; do
    if [ "$DRYRUN" = 1 ] || [ "$(both_users)" -lt 723 ]; then
      q $([ $pass = 1 ] && echo $MIN_TIME || echo 10) "time scoring of the head and the ablation on the 732 pool minus 9 never-fit (723, pass $pass)" -- \
        env -u RWKV_DELETED_CARDS $H score --master "$TRAIN/master.pth" --ablation "$TRAIN/ablation-head.pth" --baselines "$BASE" \
        --users-file "$USERS732" --out "$TROWS" --out-ablation "$AROWS" >> "$G/time.out" 2>&1 || { log "STOP: time scoring ticket failed"; return 1; }
      [ "$DRYRUN" = 1 ] && return 0
    fi
  done
  local got; got=$(both_users)
  [ "$got" -ge 722 ] || { log "STOP: time scoring has $got of 723 users after 2 passes"; return 1; }
  return 0
}

cd "$WT" || exit 1
log "started: scripts $(git -C "$WT" log --oneline -1) (DRYRUN=$DRYRUN, head on the frozen SM92 model: $SMALL)"
rc=0
cpu_step 10 "preparation" || rc=1
[ $rc -eq 0 ] && { preconditions || rc=1; }
[ $rc -eq 0 ] && { prepare || rc=1; }
# ticket 0: head training on 4 users + the time scorer (3 users) + the grade check (2 users) with the smoke head, once
if [ $rc -eq 0 ] && [ ! -e "$G/smoke-ok-$P" ]; then
  W=$S/work; rm -rf "$W/smoke-train-$P" "$W/smoke-time-$P.jsonl" "$W/smoke-time-abl-$P.jsonl" "$W/smoke-grade-rows-$P.jsonl" "$W/smoke-grade-dump-$P"
  echo '{"users": [2388, 4676]}' > "$W/smoke-grade-users.json"
  q $MIN_SMOKE "smoke: head training on 4 users (3 small + the longest, cut at 400k) + time scoring of both heads (3 users) + grade check (2 users) on the frozen SM92 model" -- \
    bash -c "$H train --released $SMALL --plan $PLAN --out $W/smoke-train-$P --max-users 3 --smoke-longest --buffer-rows 20000 --batch 1024 --log-every 5 \
      && env -u RWKV_DELETED_CARDS $H score --master $W/smoke-train-$P/master.pth --ablation $W/smoke-train-$P/ablation-head.pth --baselines $BASE --users-file $S82C/smoke-users.json --out $W/smoke-time-$P.jsonl --out-ablation $W/smoke-time-abl-$P.jsonl \
      && env -u RWKV_DELETED_CARDS $PY -X utf8 $SC --pool 732 --users-file $W/smoke-grade-users.json --model $SLBL=expertium:w64:$SMALL --model $NLBL=expertium:w64-sm82c:$W/smoke-train-$P/master.pth --only $SLBL/w8+s8 --only $NLBL/w8+s8 --time-limit 2400 --split-existing --deleted-cards neighbour --dump $W/smoke-grade-dump-$P --out $W/smoke-grade-rows-$P.jsonl" \
    >> "$G/check.out" 2>&1 || { log "STOP: smoke failed (see $G/check.out)"; rc=1; }
  if [ $rc -eq 0 ] && [ "$DRYRUN" != 1 ]; then
    gline=$($H gradecompare --rows "$W/smoke-grade-rows-$P.jsonl" --dump "$W/smoke-grade-dump-$P" --rel-tag "$SLBL" --new-tag "$NLBL" \
      --users-file "$W/smoke-grade-users.json" --existing "$SOLD" --existing-label "$SLBL/w8+s8" --out "$G/smoke-grade-check.json" 2>&1 | tail -1)
    [[ "$gline" == *"bitwise True"* ]] || { log "STOP: smoke grade check not bitwise equal: $gline"; rc=1; }
  fi
  if [ $rc -eq 0 ] && [ "$DRYRUN" != 1 ]; then
    first=$(grep '\[sm82c-train\] step 1:' "$G/check.out" | tail -1)
    [ -n "$first" ] || { log "STOP: the smoke printed no [sm82c-train] step 1 line"; rc=1; }
    train_done "$W/smoke-train-$P" || { log "STOP: the smoke head training did not complete"; rc=1; }
    sk=$(R="$W/smoke-train-$P/train.json" "$PY" -X utf8 -c "import json, os; t = json.load(open(os.environ['R'])); print(len(t['skipped']), t['peak_gpu_gib'], t['seconds_this_run'])")
    [ "${sk%% *}" = 0 ] || { log "STOP: the smoke head training skipped users (OOM or preparation): $sk"; rc=1; }
    log "smoke head training: skipped / peak GiB / seconds: $sk"
    if [ "$(time_users "$W/smoke-time-$P.jsonl")" != 3 ] || [ "$(time_users "$W/smoke-time-abl-$P.jsonl")" != 3 ]; then
      log "STOP: the time-scorer smoke did not score 3 users with both heads"; rc=1; fi
    [ $rc -eq 0 ] && { touch "$G/smoke-ok-$P"; log "smoke: $(grep '\[sm82c-train\] complete' "$G/check.out" | tail -1 | cut -c1-240); $first; $gline"; }
  fi
fi
if [ $rc -eq 0 ] && ! train_done "$TRAIN"; then
  q $MIN_TRAIN "head training on the frozen SM92 model (1,991 users of the quarter-epoch cool-down, first 400k reviews each, 10k AdamW steps, head + explicit-only ablation)" -- \
    env -u RWKV_DELETED_CARDS $H train --released "$SMALL" --plan "$PLAN" --out "$TRAIN" >> "$G/train.out" 2>&1 \
    || { log "STOP: head training ticket failed (see $G/train.out)"; rc=1; }
  if [ $rc -eq 0 ] && [ "$DRYRUN" != 1 ]; then
    train_done "$TRAIN" || { log "STOP: head training did not complete (relaunch resumes from $TRAIN/resume.pt)"; rc=1; }
    [ $rc -eq 0 ] && log "head training: $(grep '\[sm82c-train\] complete' "$G/train.out" | tail -1 | cut -c1-280)"
  fi
fi
if [ $rc -eq 0 ] && [ ! -e "$G/grade-ok-$P" ]; then
  q $MIN_GRADE "grade check: the SM92 model vs the exported model with the head (5 users, scorer rows and dumps bitwise)" -- \
    bash -c "env -u RWKV_DELETED_CARDS $PY -X utf8 $SC --pool 732 --users-file $S82C/grade-users.json --model $SLBL=expertium:w64:$SMALL --model $NLBL=expertium:w64-sm82c:$TRAIN/master.pth --only $SLBL/w8+s8 --only $NLBL/w8+s8 --time-limit 2400 --split-existing --deleted-cards neighbour --dump $G/grade-dump --out $G/grade-check-rows.jsonl" \
    >> "$G/grade.out" 2>&1 || { log "STOP: grade check ticket failed (see $G/grade.out)"; rc=1; }
  if [ $rc -eq 0 ] && [ "$DRYRUN" != 1 ]; then
    line=$($H gradecompare --rows "$G/grade-check-rows.jsonl" --dump "$G/grade-dump" --rel-tag "$SLBL" --new-tag "$NLBL" \
      --users-file "$S82C/grade-users.json" --existing "$SOLD" --existing-label "$SLBL/w8+s8" --out "$G/grade-check.json" 2>&1 | tail -1)
    log "$line"
    [[ "$line" == *"bitwise True"* ]] && touch "$G/grade-ok-$P" || log "grade check: NOT bitwise equal (the rule reports fail-grade)"
  fi
fi
[ $rc -eq 0 ] && { score_time || rc=1; }
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then
    log "DRYRUN: then $GATE --time-rows $TROWS --ablation-rows $AROWS --grade-check $G/grade-check.json --train-json $TRAIN/train.json"
  else
    cpu_step 3 "gate" || rc=1
    out=$($GATE --time-rows "$TROWS" --ablation-rows "$AROWS" --grade-check "$G/grade-check.json" --train-json "$TRAIN/train.json" 2>&1); r=$?
    line=$(echo "$out" | tail -1)
    log "rule exit $r: $line"; [ "$r" = 0 ] || rc=1
  fi
fi
if [ "$DRYRUN" = 1 ]; then rm -rf "$TRAIN"; rm -f "$G/check-cpu-ok-$P" "$G/check-cpu-$P.json"; fi
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; else log "done"; fi
else
  log "STOP: ended with exit $rc"
fi
exit $rc
