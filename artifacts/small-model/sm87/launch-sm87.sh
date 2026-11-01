#!/usr/bin/env bash
# Frozen-model grade side head (SM87): a side head (MLP 494 -> 128 -> 4, zero-init output, 63,876 values) that adds a
#   correction to the released instant (grade) logits of a FROZEN copy of the released small model (the leak-free small
#   model (SM72s) 25 epochs + the 1-epoch 32k Muon cool-down, label s-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon);
#   trunk, curve and the released instant head untouched. Inputs: the SM82c frozen trunk states (5 x 64 + head_p 128), the
#   released instant prediction (5), 41 explicit causal grade features. Ablation without the model states in the same
#   pass (46 -> 128 -> 4). scripts/sm87_head.py.
# Data: the SM82c plan users (1,991 users of the SM82 quarter-epoch cool-down, first 400k reviews each), every labelled
#   instant row (117.7M), one pass, ~14.4k AdamW steps of 8,192 rows.
# Rule (scripts/sm87_head.py decide, CPU): go if the instant LogLoss improves by >= 0.15 points of the big model (1
#   leak-free epoch) with the 95% CI clear of zero (722 users, w8+s8, leak-free scorer path); stop if < 0.05; in between
#   report. Curve unchanged (checked). Writes artifacts/small-model/sm87/sm87-decision.{md,json}.
# Steps: CPU proof -> ONE GPU ticket (`launch-sm87.sh gpu`): smoke (training on 4 users + scoring 3 users with the
#   apply-check) -> side head training -> scoring (723 users, released + corrected + ablation rows, apply-check on 5
#   users) -> (pass 2 ticket only if users are missing) -> compare -> rule (CPU).
# Breaks: quiet.json (scope all/gpu) plus the daily 20:00-01:00 break and Sunday from 16:00, re-read at every check
#   (<= 10 min); every ticket starts only when it fits before the next window (the training is resumable).
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm87/artifacts/small-model/sm87/launch-sm87.sh
#   launch   WMI hidden (artifacts/small-model/sm87/HANDOFF.md); log artifacts/small-model/sm87/gpu/launch-sm87.log
#   resume   the launch command again (finished steps are skipped; the training resumes from resume.pt)
#   stop     touch C:/Users/jonat/workspace/wt/sm87/artifacts/small-model/sm87/gpu/STOP (checked between tickets)
# Do not edit scripts/ expertium/ rwkv/ of the sm87 worktree while it runs.
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
QUIET=C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json
WT=C:/Users/jonat/workspace/wt/sm87
SM=$WT/artifacts/small-model
S=$SM/sm87
G=$S/gpu
mkdir -p "$G" "$S/work"
LOG=$G/launch-sm87.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm87-dryrun.log
H="$PY -X utf8 $WT/scripts/sm87_head.py"
SMALL=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/scale-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon/expertium/E89-F65e-F77a/master.pth  # read only
SLBL=s-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon
S72=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s                                                  # read only
SOLD=$S72/gpu/small-lf-neighbour.jsonl
USERS732=$S72/eval-users-732-fit.json
P=sm87; [ "${DRYRUN:-0}" = 1 ] && P=sm87-dryrun
TRAIN=$SM/$P-side-head             # side-head.pth, ablation-head.pth, train.json, resume.pt
PLAN=$S/plan.json
ROWS=$G/sm87-rows.jsonl
MIN_SMOKE=12; MIN_TRAIN=${MIN_TRAIN:-75}; MIN_SCORE=35
NAME="frozen-model grade side head (SM87)"
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS   # cap 0.8
log() { echo "[sm87 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
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
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm87/launch-sm87.sh" -- "$@"
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
lab = {}
for r in (map(json.loads, open(p, encoding='utf-8')) if p.exists() else []):
    if 'error' not in r:
        lab.setdefault(r['user_id'], set()).add(r['label'])
print(sum(len(v) == 3 for v in lab.values()))"; }
train_done() { [ -f "$1/side-head.pth" ] && R="$1/train.json" "$PY" -X utf8 -c "
import json, os, sys
sys.exit(0 if json.load(open(os.environ['R'])).get('status') == 'complete' else 1)" 2>/dev/null; }

preconditions() {
  for f in "$SMALL" "$SOLD" "$USERS732" "$PLAN" "$S/apply-users.json" "$S/smoke-users.json" \
           "C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72/gpu-big/bigleak-neighbour.jsonl"; do
    [ -f "$f" ] || { log "STOP: missing $f"; return 1; }
  done
  if [ "$DRYRUN" != 1 ] && [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    log "STOP: uncommitted source changes in scripts/ expertium/ rwkv/"; return 1; fi
  return 0
}

prepare() {  # CPU: the CPU proof
  log "plan: $(R="$PLAN" "$PY" -X utf8 -c "import json, os; p = json.load(open(os.environ['R'])); print(f\"{p['users']:,} users, {p['reviews_used']:,} of {p['reviews_all']:,} reviews (cap {p['cap']:,}), {p['labelled_rows']:,} labelled instant rows\")")"
  [ "$DRYRUN" = 1 ] && { log "DRYRUN: CPU proof skipped (run by hand: $H check ...)"; return 0; }
  if [ ! -e "$G/check-cpu-ok-$P" ]; then
    local line
    line=$(CUDA_VISIBLE_DEVICES= $H check --released "$SMALL" --plan "$PLAN" --out "$G/check-cpu-$P.json" 2>&1 | tail -1)
    log "$line"
    [[ "$line" == "[sm87-check] PASS"* ]] || { log "STOP: CPU proof failed (see $G/check-cpu-$P.json)"; return 1; }
    touch "$G/check-cpu-ok-$P"
  fi
  return 0
}

gpu_steps() {  # inside the one GPU ticket: the smoke (once) -> the side head training -> scoring pass 1
  local W=$S/work cline first sk
  if [ ! -e "$G/smoke-ok-$P" ]; then
    rm -rf "$W/smoke-train-$P" "$W/smoke-rows-$P.jsonl"
    log "gpu: smoke (training on 4 users, scoring 3 users with the apply-check)"
    { $H train --released "$SMALL" --plan "$PLAN" --out "$W/smoke-train-$P" --max-users 3 --smoke-longest --buffer-rows 20000 \
        --batch 1024 --log-every 5 \
      && env -u RWKV_DELETED_CARDS $H score --released "$SMALL" --heads "$W/smoke-train-$P" --users-file "$S/smoke-users.json" \
        --apply-check-users "$S/smoke-users.json" --out "$W/smoke-rows-$P.jsonl"; } >> "$G/check.out" 2>&1 \
      || { log "STOP: smoke failed (see $G/check.out)"; return 1; }
    cline=$($H compare --rows "$W/smoke-rows-$P.jsonl" --users-file "$S/smoke-users.json" --existing "$SOLD" --existing-label "$SLBL/w8+s8" \
      --need-apply --out "$G/smoke-compare.json" 2>&1 | tail -1)
    [[ "$cline" == *"PASS" ]] || { log "STOP: smoke compare failed: $cline"; return 1; }
    first=$(grep '\[sm87-train\] step 1:' "$G/check.out" | tail -1)
    [ -n "$first" ] || { log "STOP: the smoke printed no [sm87-train] step 1 line"; return 1; }
    train_done "$W/smoke-train-$P" || { log "STOP: the smoke training did not complete"; return 1; }
    sk=$(R="$W/smoke-train-$P/train.json" "$PY" -X utf8 -c "import json, os; t = json.load(open(os.environ['R'])); print(len(t['skipped']), t['peak_gpu_gib'], t['seconds_this_run'], t['first_corrected_equals_released'], len(t['rows_planned_mismatch']), t['label_mismatch'])")
    log "smoke training: skipped / peak GiB / seconds / first corrected = released / plan mismatches / label mismatches: $sk"
    [ "${sk%% *}" = 0 ] || { log "STOP: the smoke training skipped users (OOM or preparation): $sk"; return 1; }
    [ "$(scored_users "$W/smoke-rows-$P.jsonl")" = 3 ] || { log "STOP: the smoke scoring did not write 3 users x 3 rows"; return 1; }
    touch "$G/smoke-ok-$P"
    log "smoke: $(grep '\[sm87-train\] complete' "$G/check.out" | tail -1 | cut -c1-260); $first; $cline"
  fi
  if ! train_done "$TRAIN"; then
    stopped && return 1
    log "gpu: side head training"
    env -u RWKV_DELETED_CARDS $H train --released "$SMALL" --plan "$PLAN" --out "$TRAIN" >> "$G/train.out" 2>&1 \
      || { log "STOP: side head training failed (see $G/train.out; a relaunch resumes from $TRAIN/resume.pt)"; return 1; }
    train_done "$TRAIN" || { log "STOP: side head training did not complete (relaunch resumes from $TRAIN/resume.pt)"; return 1; }
    log "side head training: $(grep '\[sm87-train\] complete' "$G/train.out" | tail -1 | cut -c1-300)"
  fi
  if [ "$(scored_users "$ROWS")" -lt 723 ]; then
    stopped && return 1
    log "gpu: scoring (pass 1)"
    env -u RWKV_DELETED_CARDS $H score --released "$SMALL" --heads "$TRAIN" --users-file "$USERS732" \
      --apply-check-users "$S/apply-users.json" --out "$ROWS" >> "$G/score.out" 2>&1 || { log "STOP: scoring failed (see $G/score.out)"; return 1; }
    log "scoring pass 1: $(scored_users "$ROWS") users"
  fi
  return 0
}

cd "$WT" || exit 1
if [ "${1:-}" = gpu ]; then gpu_steps; exit $?; fi
log "started: $(git -C "$WT" log --oneline -1) (DRYRUN=$DRYRUN, side head on the frozen released small model: $SMALL)"
rc=0
cpu_step 10 "preparation" || rc=1
[ $rc -eq 0 ] && { preconditions || rc=1; }
[ $rc -eq 0 ] && { prepare || rc=1; }
# one GPU ticket (so no other ticket lands between the steps): smoke (once) -> side head training -> scoring pass 1
if [ $rc -eq 0 ] && { [ ! -e "$G/smoke-ok-$P" ] || ! train_done "$TRAIN" || [ "$(scored_users "$ROWS")" -lt 723 ]; }; then
  m=$MIN_SCORE; train_done "$TRAIN" || m=$(( m + MIN_TRAIN )); [ -e "$G/smoke-ok-$P" ] || m=$(( m + MIN_SMOKE ))
  q $m "smoke (side head training on 4 users + scoring 3 users), then side head training on the frozen released small model (1,991 users of the quarter-epoch cool-down, first 400k reviews each, 117.7M instant rows, 14.4k AdamW steps, side head + ablation without the model states), then scoring on the 732 pool minus 9 never-fit (723 users, w8+s8 leak-free: released, corrected and ablation rows; correction applied inside the model on 5 users)" -- \
    bash "$S/launch-sm87.sh" gpu || { log "STOP: the GPU ticket failed (see the gpu: lines above)"; rc=1; }
fi
if [ $rc -eq 0 ] && [ "$DRYRUN" != 1 ] && [ "$(scored_users "$ROWS")" -lt 723 ]; then
  q 15 "scoring on the 732 pool minus 9 never-fit (723 users, pass 2: users missing after pass 1)" -- \
    env -u RWKV_DELETED_CARDS $H score --released "$SMALL" --heads "$TRAIN" --users-file "$USERS732" --apply-check-users "$S/apply-users.json" \
    --out "$ROWS" >> "$G/score.out" 2>&1 || { log "STOP: scoring pass 2 failed (see $G/score.out)"; rc=1; }
fi
if [ $rc -eq 0 ] && [ "$DRYRUN" != 1 ]; then
  got=$(scored_users "$ROWS")
  [ "$got" -ge 722 ] || { log "STOP: scoring has $got of 723 users after 2 passes"; rc=1; }
fi
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then
    log "DRYRUN: then $H compare --rows $ROWS --users-file $S/apply-users.json --need-apply ... and $H decide --rows $ROWS --check $G/apply-compare.json --train-json $TRAIN/train.json"
  else
    cpu_step 3 "rule" || rc=1
    cline=$($H compare --rows "$ROWS" --users-file "$S/apply-users.json" --existing "$SOLD" --existing-label "$SLBL/w8+s8" \
      --need-apply --out "$G/apply-compare.json" 2>&1 | tail -1)
    log "$cline"
    out=$($H decide --rows "$ROWS" --check "$G/apply-compare.json" --train-json "$TRAIN/train.json" 2>&1); r=$?
    line=$(echo "$out" | tail -1)
    log "rule exit $r: $line"; [ "$r" = 0 ] || rc=1
  fi
fi
if [ "$DRYRUN" = 1 ]; then rm -rf "$TRAIN"; fi
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; else log "done"; fi
else
  log "STOP: ended with exit $rc"
fi
exit $rc
