#!/usr/bin/env bash
# Grade-order sanity (SM33), stage 2: the soft grade-order hinge in the paired quarter-epoch 32k Muon cool-down.
#   arm        the leak-free small model (SM72s) epoch 21 + the quarter-epoch 32k Muon cool-down with the reference's
#              options, users, order and seeds plus "grade_order" (scripts/gpu_r01_samples.add_grade_probes: 4 what-if
#              rows before a seeded 4% of the reviews; expertium/model/variant_model.grade_order_penalty: weight 0.1 x
#              mean relu(logit R_g(t) - logit R_{g+1}(t)) over adjacent probe pairs at 12 horizons 1 d .. 3650 d);
#              CPU proof scripts/check_grade_order.py (artifacts/small-model/sm33/check-cpu-grade-order.json, PASS).
#   reference  the tiny architecture additions (SM77) reference cool-down (read only, already scored):
#              wt/sm77/artifacts/small-model/sm77-ref-e21-dec-q32k-muon, label sm77-ref-e21-q32k/w8+s8.
# Gate (stage 1, artifacts/small-model/sm33/sm33-stage1.json): proceed only if the released small model's displayed
#   intervals at 0.90 (before the fork's PAVA) break the order on >= 5% of reviews or PAVA changes Good's displayed
#   interval on >= 5% of reviews.
# Rule (artifacts/small-model/sm33/sm33_decide.py, CPU): 722 users, points of % of the big model (1 leak-free epoch):
#   both heads (arm - reference) <= +0.03 points AND the review-weighted displayed break rate at 0.90 (any pair) drops
#   with the paired user bootstrap CI clear of zero -> go; it drops but a head is > +0.03 -> cost; else no effect.
# Steps: gate -> CPU prepare + pair check -> smoke ticket -> cool-down (segments, resumable over breaks) -> 732 scoring
#   -> what-if break measure of arm and reference (sm33_grade_order.py, 723 users x 3,000 reviews) -> rule.
#   launch   WMI hidden (artifacts/small-model/sm33/HANDOFF.md); log artifacts/small-model/sm33/gpu/launch-sm33-stage2.log
#   resume   the launch command again (finished steps are skipped; an interrupted cool-down resumes from its snapshot)
#   stop     touch C:/Users/jonat/workspace/wt/sm33/artifacts/small-model/sm33/gpu/STOP
# Do not edit scripts/ expertium/ rwkv/ of the sm33 worktree while it runs (SourceWatch stops the training).
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
QUIET=C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json
WT=C:/Users/jonat/workspace/wt/sm33
SM=$WT/artifacts/small-model
S=$SM/sm33
G=$S/gpu
mkdir -p "$G"
LOG=$G/launch-sm33-stage2.log
WARM="$PY -X utf8 $WT/scripts/sm77_warm.py"
DEC_PY=$WT/scripts/expertium_decay.py
SC=$WT/scripts/small_model_heldout.py
GO_PY=$WT/scripts/sm33_grade_order.py
TRACK=E89-F65e-F77a
K1=21
RUN72S=C:/Users/jonat/workspace/wt/sm72s/artifacts/accuracy/experiments/scale-w64kd-final-sc-lanes-leakfree/run   # read only
S72=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s                                                  # read only
SM77=C:/Users/jonat/workspace/wt/sm77/artifacts/small-model                                                        # read only
DREF=$SM77/sm77-ref-e21-dec-q32k-muon
REFROWS=$SM77/sm77/gpu/sm77-neighbour.jsonl
RLBL=sm77-ref-e21-q32k
BIGCK=C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72-big-leakfree-1ep/expertium/big-leakfree-1ep/master.pth
REPLAY=C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/big-leakfree-1ep-final8k-neighbour
USERS732=$S72/eval-users-732-fit.json; ULONG=$S72/users-732-long.json
DARM=$SM/sm33-arm-e$K1-dec-q32k-muon
ALBL=sm33-arm-e$K1-q32k
OUTROWS=$G/sm33-neighbour.jsonl
MODE=neighbour
LF="env RWKV_DELETED_CARDS=$MODE"
SPEC='{"lanes": "strided"}'
DIST="{\"teacher\": \"$BIGCK\", \"alpha\": 0.9, \"decay_alpha\": 0.5, \"kd_source\": \"replay\", \"replay\": \"$REPLAY\"}"
GRADE='{"weight": 0.1, "density": 0.04, "horizons_days": [1, 3, 7, 15, 30, 60, 120, 240, 480, 960, 1920, 3650], "seed": 33}'
EXTRA="{\"distillation\": $DIST, \"state_carry\": $SPEC, \"chunk_reviews\": 32768, \"batch_chunks\": 1, \"optimizer\": \"expertium-iter53\", \"grade_order\": $GRADE}"  # = the SM77 reference's + grade_order
FRACTION=0.25
MIN_DECAY=${MIN_DECAY:-170}   # reference: ~115-125 min training at ~20k reviews/s; +8% rows for the probes; + setup
MIN_CHECK=12; MIN_SCORE732=12; MIN_LONG=6; MIN_MEASURE=40
NAME="grade-order sanity (SM33)"
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS   # scorer cap 0.8; trainer cap 0.8 (options)
log() { echo "[sm33 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
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
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $purpose (expected $minutes min)"; echo "    cmd: $*" >> "$LOG"; return 0; fi
    local maxwait=$(( FREE - minutes - 3 )); [ "$maxwait" -lt 1 ] && maxwait=1
    log "enqueue: $purpose (expected $minutes min, max wait $maxwait min, $FREE min to the next window)"
    "$PY" "$QUEUE" run --owner small-model --purpose "$NAME: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm33/launch-sm33-stage2.sh" -- "$@"
    rc=$?; log "exit $rc: $purpose"
    [ "$rc" = 75 ] && { log "queue wait would reach the next window; re-planning"; sleep 60; continue; }
    return $rc
  done
}
cpu_step() { wait_free "$1" "$2"; }   # CPU work also stays out of the windows
label_users() { F="$1" LB="$2" "$PY" -X utf8 -c "
import json, os
from pathlib import Path
p = Path(os.environ['F'])
print(len({r['user_id'] for r in map(json.loads, open(p, encoding='utf-8')) if r.get('label') == os.environ['LB'] and 'benchmark_existing' in r}) if p.exists() else 0)"; }
done_in() { [ -f "$1/expertium/$TRACK/master.pth" ] && R="$1/expertium/$TRACK/result.json" "$PY" -X utf8 -c "
import json, os, sys
r = json.load(open(os.environ['R']))
sys.exit(0 if r.get('status') == 'complete' and not r.get('smoke_only') else 1)" 2>/dev/null; }


gate() {  # stage 1 says the order is broken often enough to train against it
  F="$S/sm33-stage1.json" "$PY" -X utf8 - <<'PYEOF'
import json, os, sys
r = json.load(open(os.environ["F"]))
broken = r["targets"]["0.90"]["any_pair_displayed"]["review_weighted"]
good = r["fork_display_0.90"]["Good"]["displayed_days_changed"]
ok = broken >= 0.05 or good >= 0.05
print(f"[sm33-gate] {'PASS' if ok else 'STOP'}: displayed order broken on {broken:.1%} of reviews at 0.90, PAVA changes Good's displayed interval on {good:.1%} ({r['users']} users)")
sys.exit(0 if ok else 1)
PYEOF
}

preconditions() {
  local e; e=$RUN72S/epoch-$K1/expertium/$TRACK
  for f in "$e/master.pth" "$e/result.json" "$RUN72S/run.json" "$BIGCK" "$REPLAY/manifest.json" "$USERS732" "$ULONG" \
           "$DREF/decay.json" "$DREF/protocol.json" "$DREF/expertium/$TRACK/master.pth" "$REFROWS" "$S/check-cpu-grade-order.json"; do
    [ -f "$f" ] || { log "STOP: missing $f"; return 1; }
  done
  done_in "$DREF" || { log "STOP: the reference cool-down $DREF is not complete"; return 1; }
  grep -q '"ok": true' "$S/check-cpu-grade-order.json" || { log "STOP: the CPU proof is not PASS"; return 1; }
  if [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    log "STOP: uncommitted source changes in scripts/ expertium/ rwkv/ (SourceWatch refuses them)"; return 1; fi
  return 0
}

prepare_decay() {  # the arm cool-down from SM72s's own epoch 21 (the reference's start), with grade_order
  $LF $WARM cooldown --epoch $K1 --start "$RUN72S/epoch-$K1" --out "${DARM#$WT/}" --arch w64 --fraction $FRACTION --extra-options "$EXTRA" >> "$G/prepare.out" 2>&1 \
    || { log "cool-down prepare failed ($DARM)"; return 1; }
  $LF "$PY" -X utf8 "$WT/scripts/check_replay_label_windows.py" --decay "$DARM" --trim-unlabelled-tail >/dev/null || { log "label windows (trim pass) failed ($DARM)"; return 1; }
  $LF "$PY" -X utf8 "$WT/scripts/check_replay_label_windows.py" --decay "$DARM" --out "$G/check-replay-label-windows-$(basename "$DARM").json" >/dev/null \
    || { log "label windows check failed ($DARM)"; return 1; }
  CUDA_VISIBLE_DEVICES= $LF "$PY" -X utf8 "$WT/scripts/check_muon_fresh_decay.py" --decay "$DARM" --out "$G/check-muon-fresh-$(basename "$DARM").json" \
    | grep -c PASS | sed "s/^/[sm33-muon] $(basename "$DARM") PASS lines: /" >> "$LOG"
  [ "${PIPESTATUS[0]}" = 0 ] || { log "Muon fresh-state CPU check failed ($DARM)"; return 1; }
  return 0
}

pair_decay() {  # arm vs reference: identical except the grade_order option and paths (same start weights, users, order, seeds)
  "$PY" -X utf8 "$S/sm33_pair.py" --arm "$DARM" --ref "$DREF"
}

measure() {  # measure TAG MODEL: what-if break measure (723 users x 3,000 reviews, fp32), then its report
  local out=$S/work/stage2-$1
  if [ ! -f "$out/sm33-stage1.json" ]; then
    q $MIN_MEASURE "stage 2, what-if intervals of the $1 on the 732 pool (723 users, 3,000 reviews each)" -- \
      "$PY" -X utf8 "$GO_PY" measure --model "$2" --device cuda --users-file "$USERS732" --probes 3000 --time-limit 2100 --out "$out" \
      >> "$G/measure2.out" 2>&1 || { log "STOP: measure ticket failed ($1)"; return 1; }
    [ "$DRYRUN" = 1 ] && return 0
    "$PY" -X utf8 "$GO_PY" report --out "$out" >> "$G/measure2.out" 2>&1 || { log "STOP: report failed ($1)"; return 1; }
    cp "$out/sm33-stage1.json" "$S/sm33-stage2-$1.json"
  fi
  return 0
}
run_folder() {  # run_folder DIR WHAT FIRST_MINUTES SEG_MAX: expertium_decay.py run in segments; resumable from snapshots
  local D=$1 what=$2 first=$3 segmax=$4 rc qpid stop_sent b need
  while ! done_in "$D"; do
    stopped && return 1
    need=30; [ ! -d "$D/expertium" ] && [ ! -d "$D/attempts" ] && need=$first   # a fresh start must fit as a whole
    wait_free "$need" "$what" || return 1
    local seg=$(( FREE - 5 )); [ "$seg" -gt "$segmax" ] && seg=$segmax
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $what (segment to the next window or the end, expected $seg min of ~$segmax)"
      echo "    cmd: $LF $PY -X utf8 $DEC_PY run --out $D" >> "$LOG"; return 0; fi
    rm -f "$D/control/STOP"
    log "enqueue: $what (expected $seg min of ~$segmax, $FREE min to the next window)"
    "$PY" "$QUEUE" run --owner small-model --purpose "$NAME: $what (segment to the next window or the end)" --claim gpu \
      --expected-minutes "$seg" --max-wait-minutes $(( FREE - 20 > 1 ? FREE - 20 : 1 )) --launched-by "detached launcher sm33/launch-sm33-stage2.sh" -- \
      $LF "$PY" -X utf8 "$DEC_PY" run --out "$D" >> "$G/train.out" 2>&1 &
    qpid=$!
    stop_sent=0
    while kill -0 "$qpid" 2>/dev/null; do
      b=$(block)
      if [ -e "$G/STOP" ] || [[ "$b" == NOW* ]] || { [[ "$b" == NEXT* ]] && [ "${b#NEXT }" -le 2 ]; }; then
        [ "$stop_sent" = 0 ] && log "requesting a snapshot stop ($b)"
        "$PY" -X utf8 "$DEC_PY" stop --out "$D" > /dev/null 2>&1; stop_sent=1
      fi
      sleep 30
    done
    wait "$qpid"; rc=$?; log "exit $rc: $what segment"
    [ "$rc" = 75 ] && { sleep 60; continue; }
    done_in "$D" && break
    [ "$stop_sent" = 0 ] && { log "STOP: $what segment ended without completing and without a stop request (exit $rc); see $G/train.out"; return 1; }
  done
  [ "$DRYRUN" = 1 ] || log "$what complete: $("$PY" -X utf8 -c "import json; r=json.load(open(r'$D/expertium/$TRACK/result.json')); print(f\"{r.get('review_presentations', 0):,} reviews, {r.get('training_seconds', 0)/60:.1f} min training, {r.get('reviews_per_second', 0):,.0f} reviews/s\")")"
  return 0
}

score() {  # score USERS_FILE OUT LABEL MODEL_SPEC MIN1 MIN2 WANT WHAT [ENV]: leak-free scoring, 2 passes
  local pass
  for pass in 1 2; do
    if [ "$DRYRUN" = 1 ] || [ "$(label_users "$2" "$3/w8+s8")" -lt "$7" ]; then
      q $([ $pass = 1 ] && echo "$5" || echo "$6") "scoring $3 on $8 (pass $pass)" -- \
        env -u RWKV_DELETED_CARDS ${9:-} "$PY" -X utf8 "$SC" --pool 732 --users-file "$1" --model "$3=$4" --only "$3/w8+s8" \
        --time-limit 2400 --split-existing --deleted-cards $MODE --out "$2" >> "$G/score.out" 2>&1 || { log "STOP: scoring ticket failed ($3)"; return 1; }
      [ "$DRYRUN" = 1 ] && return 0
    fi
  done
  local got; got=$(label_users "$2" "$3/w8+s8")
  [ "$got" -ge "$7" ] || { log "STOP: $3 scored on $got of $7 users after 2 passes"; return 1; }
  return 0
}
score732() {  # score732 LABEL MODEL_SPEC: 723 users at the 0.8 cap, then the 3 long users at 0.92 (726 in all)
  score "$USERS732" "$OUTROWS" "$1" "$2" $MIN_SCORE732 8 723 "the 732 pool minus 9 never-fit (723)" || return 1
  if [ "$DRYRUN" = 1 ] || [ "$(label_users "$OUTROWS" "$1/w8+s8")" -lt 726 ]; then
    q $MIN_LONG "scoring $1 on the 3 long users of the 732 pool (cap 0.92)" -- \
      env -u RWKV_DELETED_CARDS HELDOUT_GPU_MEMORY_FRACTION=0.92 "$PY" -X utf8 "$SC" --pool 732 --users-file "$ULONG" --model "$1=$2" \
      --only "$1/w8+s8" --time-limit 2400 --split-existing --deleted-cards $MODE --out "$OUTROWS" >> "$G/score.out" 2>&1 \
      || log "long users ticket failed ($1); continuing with $(label_users "$OUTROWS" "$1/w8+s8") users"
  fi
  return 0
}

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1) (DRYRUN=$DRYRUN; arm $DARM; reference $DREF)"
rc=0
line=$(gate 2>&1 | tail -1); log "$line"
[[ "$line" == "[sm33-gate] PASS"* ]] || { log "STOP: stage 1 does not call for training (or is missing)"; exit 0; }
cpu_step 10 "preparation" || rc=1
[ $rc -eq 0 ] && { preconditions || rc=1; }
if [ $rc -eq 0 ] && [ ! -f "$DARM/decay.json" ]; then
  prepare_decay || { rm -rf "$DARM"; rc=1; }
  [ $rc -eq 0 ] && log "cool-down prepared: $(basename "$DARM")"
fi
[ $rc -eq 0 ] && { pair_decay >> "$LOG" 2>&1 || { log "STOP: the cool-down does not match the reference's"; rc=1; }; }
if [ $rc -eq 0 ] && [ ! -e "$G/smoke-ok" ]; then
  q $MIN_CHECK "trainer smoke of the cool-down with the grade-order hinge (2 users x 512 reviews)" -- \
    $LF "$PY" -X utf8 "$DEC_PY" smoke --out "$DARM" >> "$G/check.out" 2>&1 \
    || { log "STOP: smoke failed (see $G/check.out)"; rc=1; }
  [ $rc -eq 0 ] && [ "$DRYRUN" != 1 ] && { touch "$G/smoke-ok"; log "smoke: $(grep '"smoke"' "$G/check.out" | tail -1 | cut -c1-200)"; }
fi
[ $rc -eq 0 ] && { run_folder "$DARM" "arm quarter-epoch 32k Muon cool-down from SM72s epoch $K1 with the grade-order hinge (w64, the reference's options and order)" 30 $MIN_DECAY || rc=1; }
[ $rc -eq 0 ] && { score732 "$ALBL" "expertium:w64:$DARM/expertium/$TRACK/master.pth" || rc=1; }
[ $rc -eq 0 ] && { measure arm "$DARM/expertium/$TRACK/master.pth" || rc=1; }
[ $rc -eq 0 ] && { measure reference "$DREF/expertium/$TRACK/master.pth" || rc=1; }
if [ $rc -eq 0 ] && [ "$DRYRUN" != 1 ]; then
  cpu_step 3 "rule" || rc=1
  out=$("$PY" -X utf8 "$S/sm33_decide.py" --arm-rows "$OUTROWS" --arm-label "$ALBL/w8+s8" --ref-rows "$REFROWS" --ref-label "$RLBL/w8+s8" \
        --arm-breaks "$S/work/stage2-arm" --ref-breaks "$S/work/stage2-reference" 2>&1); r=$?
  log "rule exit $r: $(echo "$out" | tail -1)"; [ "$r" = 0 ] || rc=1
fi
if [ $rc -eq 0 ]; then log "done"; else log "STOP: ended with exit $rc"; fi
exit $rc
