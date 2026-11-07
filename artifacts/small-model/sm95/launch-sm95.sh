#!/usr/bin/env bash
# The grade order by construction (SM95), stage 1 (artifacts/small-model/sm95/HANDOFF.md): the in-model grade rectifier
#   in the paired quarter-epoch 32k Muon cool-down.
#   arm        the leak-free small model (SM72s) epoch 21 + the quarter-epoch 32k Muon cool-down with the reference's
#              options, users, order and seeds plus "grade_rectify": {"density": 0.08, "seed": 95}
#              (scripts/gpu_r01_samples.add_rectify_probes: the three other-grade what-if rows before a seeded 8% of the
#              labelled reviews; expertium/model/grade_rectify.rectify_curve: those reviews' curve = its grade's value
#              in the isotonic 4-vector of the four what-if curves at the label time). CPU proof
#              scripts/check_grade_rectify.py -> artifacts/small-model/sm95/check-cpu-grade-rectify.json (PASS).
#   reference  the tiny architecture additions (SM77) reference cool-down (read only, already scored raw):
#              wt/sm77/artifacts/small-model/sm77-ref-e21-dec-q32k-muon, label sm77-ref-e21-q32k/w8+s8.
#   Scored probability: the rectified curve (small_model_heldout.py --grade-rectify), the scheduler's curve; the
#   reference is also scored rectified; the arm also raw (diagnostic).
# Rule (pre-registered, artifacts/small-model/sm95/sm95_decide.py): go = zero breaks of the arm's rectified curves AND
#   both heads (arm rectified - reference raw) <= +0.05 points of % of the big model (1 leak-free epoch); cost = order
#   ok but a head > +0.05; fail = any break (a bug); incomplete = < 700 users.
# Steps: wait for the instant-only small model (SM57) stage 4 to log "] done" (or $G/GO-NOW) -> CPU prepare + pair
#   check -> GPU smoke (trainer smoke; rectified scorer smoke on 2 users) -> cool-down (segments, snapshot stop before
#   a break, resume after) -> 732 scoring: arm raw, arm rectified, reference rectified -> what-if measure of the arm
#   (scripts/sm33_grade_order.py, same 3,000 sampled reviews as SM33's) -> CPU derive + rule.
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm95/artifacts/small-model/sm95/launch-sm95.sh
#   prepare  PREPARE_ONLY=1 bash ... (CPU prepare + pair check, then exit)
#   launch   WMI hidden (HANDOFF.md); log artifacts/small-model/sm95/gpu/launch-sm95.log
#   resume   the launch command again (finished steps are skipped; an interrupted cool-down resumes from its snapshot)
#   stop     touch C:/Users/jonat/workspace/wt/sm95/artifacts/small-model/sm95/gpu/STOP (between tickets; a running
#            cool-down gets a snapshot stop)
# Do not edit scripts/ expertium/ rwkv/ of the sm95 worktree while it runs (SourceWatch stops the training).
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/sm95
SM=$WT/artifacts/small-model
S=$SM/sm95
G=$S/gpu
mkdir -p "$G"
LOG=$G/launch-sm95.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm95-dryrun.log
WARM="$PY -X utf8 $WT/scripts/sm77_warm.py"
DEC_PY=$WT/scripts/expertium_decay.py
SC=$WT/scripts/small_model_heldout.py
GO_PY=$WT/scripts/sm33_grade_order.py
BLOCK="$PY -X utf8 $S/sm95_block.py"
SM57LOG=C:/Users/jonat/workspace/wt/sm57/artifacts/small-model/sm57/gpu/launch-sm57-stage4.log   # read only
TRACK=E89-F65e-F77a
K1=21
RUN72S=C:/Users/jonat/workspace/wt/sm72s/artifacts/accuracy/experiments/scale-w64kd-final-sc-lanes-leakfree/run   # read only
S72=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s                                                  # read only
SM77=C:/Users/jonat/workspace/wt/sm77/artifacts/small-model                                                        # read only
SM33W=C:/Users/jonat/workspace/wt/sm33/artifacts/small-model                                                       # read only
DREF=$SM77/sm77-ref-e21-dec-q32k-muon
REFROWS=$SM77/sm77/gpu/sm77-neighbour.jsonl
RLBL=sm77-ref-e21-q32k
RRLBL=sm77-ref-e21-q32k-rect
SM33ROWS=$SM33W/sm33/gpu/sm33-neighbour.jsonl
REFMEAS=$SM33W/sm33/work/stage2-reference
SM33MEAS=$SM33W/sm33/work/stage2-arm
BIGCK=C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72-big-leakfree-1ep/expertium/big-leakfree-1ep/master.pth
REPLAY=C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/big-leakfree-1ep-final8k-neighbour
USERS732=$S72/eval-users-732-fit.json; ULONG=$S72/users-732-long.json
DARM=$SM/sm95-arm-e$K1-dec-q32k-muon
ALBL=sm95-arm-e$K1-q32k
ARLBL=sm95-arm-e$K1-q32k-rect
OUTROWS=$G/sm95-neighbour.jsonl
SMOKEROWS=$G/smoke-rect.jsonl
ARMMEAS=$S/work/measure-arm
MODE=neighbour
LF="env RWKV_DELETED_CARDS=$MODE"
SPEC='{"lanes": "strided"}'
DIST="{\"teacher\": \"$BIGCK\", \"alpha\": 0.9, \"decay_alpha\": 0.5, \"kd_source\": \"replay\", \"replay\": \"$REPLAY\"}"
RECT='{"density": 0.08, "seed": 95}'
EXTRA="{\"distillation\": $DIST, \"state_carry\": $SPEC, \"chunk_reviews\": 32768, \"batch_chunks\": 1, \"optimizer\": \"expertium-iter53\", \"grade_rectify\": $RECT}"  # = the SM77 reference's + grade_rectify
FRACTION=0.25
MIN_DECAY=${MIN_DECAY:-175}   # SM33 arm: 17.5k reviews/s with +8% rows (~139 min for 145.6M reviews); here ~+11% rows; + setup
MIN_START=30                  # a cool-down segment starts when 30 min fit (snapshot stop before the break, resume after)
MIN_CHECK=12; MIN_SMOKE_SCORE=12; MIN_SCORE732=14; MIN_LONG=8; MIN_RECT=45; MIN_RECT_LONG=25; MIN_MEASURE=40
NAME="the grade order by construction (SM95)"
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START   # scorer cap 0.8; trainer cap 0.8 (options)
log() { echo "[sm95 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
stopped() { [ -e "$G/STOP" ] && { log "STOP: STOP file present"; return 0; }; return 1; }
nap() { [ "$DRYRUN" = 1 ] && return 0; local s=$1; [ "$s" -gt 600 ] && s=600; sleep "$s"; }
block() { $BLOCK; }
free_mib() { awk '/MemFree/ {print int($2 / 1024)}' /proc/meminfo 2>/dev/null || echo 99999; }

wait_free() {  # wait_free MINUTES WHAT: returns once MINUTES fit before the next break (sets FREE = minutes to it)
  local n=0
  while true; do
    stopped && return 1
    local b; b=$(block)
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
ram_gate() {  # hold while < 3 GB RAM is free, at most 30 min, then go on
  local i=0
  while [ "$DRYRUN" != 1 ] && [ "$(free_mib)" -lt 3072 ]; do
    [ $i -ge 6 ] && { log "$1: still $(free_mib) MiB RAM free after 30 min; going on"; return 0; }
    [ $i = 0 ] && log "$1: $(free_mib) MiB RAM free (< 3 GB), holding"; i=$((i + 1)); sleep 300
  done
  return 0
}
cpu_step() { wait_free "$1" "$2" || return 1; ram_gate "$2"; }

wait_sm57() {  # the GPU belongs to the instant-only small model (SM57) stage 4 until its launcher logs "] done"
  local n=0
  while true; do
    stopped && return 1
    [ -e "$G/GO-NOW" ] && { log "GO-NOW file present: not waiting for SM57 stage 4"; return 0; }
    if [ -f "$SM57LOG" ] && grep -q '\] done' "$SM57LOG"; then
      log "SM57 stage 4 finished: $(grep '\] done' "$SM57LOG" | tail -1 | cut -c1-160)"; return 0; fi
    [ "$DRYRUN" = 1 ] && { log "DRYRUN: would wait for SM57 stage 4 (last line: $(tail -1 "$SM57LOG" | cut -c1-120))"; return 0; }
    [ $((n % 12)) = 0 ] && log "waiting for the instant-only small model (SM57) stage 4 to finish (last line: $(tail -1 "$SM57LOG" | cut -c1-120))"
    n=$((n + 1)); sleep 300
  done
}

q() {  # q MINUTES PURPOSE -- CMD...   one ticket that must end before the next break
  local minutes=$1 purpose=$2 rc; shift 3
  while true; do
    wait_free "$minutes" "$purpose" || return 1
    ram_gate "$purpose"
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $NAME: $purpose (expected $minutes min)"; echo "    cmd: $*" >> "$LOG"; return 0; fi
    local maxwait=$(( FREE - minutes - 3 )); [ "$maxwait" -lt 1 ] && maxwait=1
    log "enqueue: $purpose (expected $minutes min, max wait $maxwait min, $FREE min to the next break)"
    "$PY" "$QUEUE" run --owner sm95 --purpose "$NAME: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm95/launch-sm95.sh" -- "$@"
    rc=$?; log "exit $rc: $purpose"
    [ "$rc" = 75 ] && { log "queue wait would reach the next break; re-planning"; sleep 60; continue; }
    return $rc
  done
}
label_users() { F="$1" LB="$2" "$PY" -X utf8 -c "
import json, os
from pathlib import Path
p = Path(os.environ['F'])
print(len({r['user_id'] for r in map(json.loads, open(p, encoding='utf-8')) if r.get('label') == os.environ['LB'] and 'benchmark_existing' in r}) if p.exists() else 0)"; }
done_in() { [ -f "$1/expertium/$TRACK/master.pth" ] && R="$1/expertium/$TRACK/result.json" "$PY" -X utf8 -c "
import json, os, sys
r = json.load(open(os.environ['R']))
sys.exit(0 if r.get('status') == 'complete' and not r.get('smoke_only') else 1)" 2>/dev/null; }

preconditions() {
  local e; e=$RUN72S/epoch-$K1/expertium/$TRACK
  for f in "$e/master.pth" "$e/result.json" "$RUN72S/run.json" "$BIGCK" "$REPLAY/manifest.json" "$USERS732" "$ULONG" \
           "$DREF/decay.json" "$DREF/protocol.json" "$DREF/expertium/$TRACK/master.pth" "$REFROWS" "$SM33ROWS" \
           "$S/check-cpu-grade-rectify.json" "$S/smoke-users.json"; do
    [ -f "$f" ] || { log "STOP: missing $f"; return 1; }
  done
  [ -d "$REFMEAS/users" ] || { log "STOP: missing the reference's what-if measure $REFMEAS"; return 1; }
  done_in "$DREF" || { log "STOP: the reference cool-down $DREF is not complete"; return 1; }
  grep -q '"ok": true' "$S/check-cpu-grade-rectify.json" || { log "STOP: the CPU proof is not PASS"; return 1; }
  if [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    log "STOP: uncommitted source changes in scripts/ expertium/ rwkv/ (SourceWatch refuses them)"; return 1; fi
  return 0
}

prepare_decay() {  # the arm cool-down from SM72s's own epoch 21 (the reference's start), with grade_rectify
  $LF $WARM cooldown --epoch $K1 --start "$RUN72S/epoch-$K1" --out "${DARM#$WT/}" --arch w64 --fraction $FRACTION --extra-options "$EXTRA" >> "$G/prepare.out" 2>&1 \
    || { log "cool-down prepare failed ($DARM)"; return 1; }
  $LF "$PY" -X utf8 "$WT/scripts/check_replay_label_windows.py" --decay "$DARM" --trim-unlabelled-tail >/dev/null || { log "label windows (trim pass) failed ($DARM)"; return 1; }
  $LF "$PY" -X utf8 "$WT/scripts/check_replay_label_windows.py" --decay "$DARM" --out "$G/check-replay-label-windows-$(basename "$DARM").json" >/dev/null \
    || { log "label windows check failed ($DARM)"; return 1; }
  CUDA_VISIBLE_DEVICES= $LF "$PY" -X utf8 "$WT/scripts/check_muon_fresh_decay.py" --decay "$DARM" --out "$G/check-muon-fresh-$(basename "$DARM").json" \
    | grep -c PASS | sed "s/^/[sm95-muon] $(basename "$DARM") PASS lines: /" >> "$LOG"
  [ "${PIPESTATUS[0]}" = 0 ] || { log "Muon fresh-state CPU check failed ($DARM)"; return 1; }
  return 0
}
pair_decay() { "$PY" -X utf8 "$S/sm95_pair.py" --arm "$DARM" --ref "$DREF"; }

run_folder() {  # run_folder DIR WHAT SEG_MAX: expertium_decay.py run in segments; resumable from snapshots
  local D=$1 what=$2 segmax=$3 rc qpid stop_sent b need
  while ! done_in "$D"; do
    stopped && return 1
    need=$MIN_START
    wait_free "$need" "$what" || return 1
    local seg=$(( FREE - 5 )); [ "$seg" -gt "$segmax" ] && seg=$segmax
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $NAME: $what (segment to the next break or the end, expected $seg min of ~$segmax)"
      echo "    cmd: $LF $PY -X utf8 $DEC_PY run --out $D" >> "$LOG"; return 0; fi
    ram_gate "$what"
    rm -f "$D/control/STOP"
    log "enqueue: $what (expected $seg min of ~$segmax, $FREE min to the next break)"
    "$PY" "$QUEUE" run --owner sm95 --purpose "$NAME: $what (segment to the next break or the end)" --claim gpu \
      --expected-minutes "$seg" --max-wait-minutes $(( FREE - 20 > 1 ? FREE - 20 : 1 )) --launched-by "detached launcher sm95/launch-sm95.sh" -- \
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
  [ "$DRYRUN" = 1 ] || log "$what complete: $("$PY" -X utf8 -c "import json; r=json.load(open(r'$D/expertium/$TRACK/result.json')); print(f\"{r.get('review_presentations', 0):,} reviews, {r.get('training_seconds', 0)/60:.1f} min training, {r.get('reviews_per_second', 0):,.0f} reviews/s, peak allocated {r.get('peak_allocated_mib') or 0:,.0f} MiB\")")"
  return 0
}

score() {  # score USERS_FILE LABEL MODEL_SPEC MINUTES PASSES WANT WHAT FLAGS [ENV]: leak-free scoring in passes
  local pass
  for pass in $(seq 1 "$5"); do
    if [ "$DRYRUN" = 1 ] || [ "$(label_users "$OUTROWS" "$2/w8+s8")" -lt "$6" ]; then
      q "$4" "scoring $2 on $7 (pass $pass)" -- \
        env -u RWKV_DELETED_CARDS ${9:-} "$PY" -X utf8 "$SC" --pool 732 --users-file "$1" --model "$2=$3" --only "$2/w8+s8" \
        --time-limit 2400 --split-existing --deleted-cards $MODE $8 --out "$OUTROWS" >> "$G/score.out" 2>&1 || { log "STOP: scoring ticket failed ($2)"; return 1; }
      [ "$DRYRUN" = 1 ] && return 0
    fi
  done
  log "$2: $(label_users "$OUTROWS" "$2/w8+s8") of $6 users scored on $7"
  return 0
}
score732() {  # score732 LABEL MODEL_SPEC FLAGS MIN PASSES MIN_LONG: 723 users at the 0.8 cap, then the 3 long users at 0.92
  score "$USERS732" "$1" "$2" "$4" "$5" 723 "the 732 pool minus 9 never-fit (723)" "$3" || return 1
  if [ "$DRYRUN" = 1 ] || [ "$(label_users "$OUTROWS" "$1/w8+s8")" -lt 726 ]; then
    q "$6" "scoring $1 on the 3 long users of the 732 pool (cap 0.92)" -- \
      env -u RWKV_DELETED_CARDS HELDOUT_GPU_MEMORY_FRACTION=0.92 "$PY" -X utf8 "$SC" --pool 732 --users-file "$ULONG" --model "$1=$2" \
      --only "$1/w8+s8" --time-limit 2400 --split-existing --deleted-cards $MODE $3 --out "$OUTROWS" >> "$G/score.out" 2>&1 \
      || log "long users ticket failed ($1); continuing with $(label_users "$OUTROWS" "$1/w8+s8") users"
  fi
  return 0
}

smoke() {  # trainer smoke of the arm; rectified scorer smoke (reference checkpoint, 2 users) vs the reference's raw rows
  if [ ! -e "$G/smoke-train-ok" ]; then
    q $MIN_CHECK "GPU smoke 1/2: trainer smoke of the rectified cool-down (2 users x 512 reviews)" -- \
      $LF "$PY" -X utf8 "$DEC_PY" smoke --out "$DARM" >> "$G/check.out" 2>&1 || { log "STOP: trainer smoke failed (see $G/check.out)"; return 1; }
    [ "$DRYRUN" = 1 ] || { touch "$G/smoke-train-ok"; log "smoke: $(grep '"smoke"' "$G/check.out" | tail -1 | cut -c1-200)"; }
  fi
  if [ ! -e "$G/smoke-score-ok" ]; then
    q $MIN_SMOKE_SCORE "GPU smoke 2/2: rectified scorer on 2 users of the 732 pool (reference checkpoint, w8+s8, raw and rectified)" -- \
      env -u RWKV_DELETED_CARDS bash "$S/smoke-score.sh" \
      >> "$G/check.out" 2>&1 || { log "STOP: scorer smoke failed (see $G/check.out)"; return 1; }
    [ "$DRYRUN" = 1 ] && return 0
    line=$("$PY" -X utf8 "$S/sm95_smoke.py" --rows "$SMOKEROWS" --ref-rows "$REFROWS" 2>&1 | tail -1); log "$line"
    [[ "$line" == "[sm95-smoke] PASS"* ]] || { log "STOP: scorer smoke check failed"; return 1; }
    touch "$G/smoke-score-ok"
  fi
  return 0
}

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1) (DRYRUN=$DRYRUN; arm $DARM; reference $DREF)"
rc=0
cpu_step 15 "preparation" || rc=1
[ $rc -eq 0 ] && { preconditions || rc=1; }
if [ $rc -eq 0 ] && [ ! -f "$DARM/decay.json" ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: would prepare $DARM"; else
    prepare_decay || { rm -rf "$DARM"; rc=1; }
    [ $rc -eq 0 ] && log "cool-down prepared: $(basename "$DARM")"
  fi
fi
[ $rc -eq 0 ] && [ -f "$DARM/decay.json" ] && { pair_decay >> "$LOG" 2>&1 || { log "STOP: the cool-down does not match the reference's"; rc=1; }; }
if [ "${PREPARE_ONLY:-0}" = 1 ]; then log "PREPARE_ONLY: exit $rc"; exit $rc; fi
[ $rc -eq 0 ] && { wait_sm57 || rc=1; }
[ $rc -eq 0 ] && { smoke || rc=1; }
[ $rc -eq 0 ] && { run_folder "$DARM" "arm quarter-epoch 32k Muon cool-down from SM72s epoch $K1 with the grade rectifier (w64, the reference's options and order)" $MIN_DECAY || rc=1; }
[ $rc -eq 0 ] && { score732 "$ALBL" "expertium:w64:$DARM/expertium/$TRACK/master.pth" "" $MIN_SCORE732 2 $MIN_LONG || rc=1; }
[ $rc -eq 0 ] && { score732 "$ARLBL" "expertium:w64:$DARM/expertium/$TRACK/master.pth" "--grade-rectify --workers 3" $MIN_RECT 4 $MIN_RECT_LONG || rc=1; }
[ $rc -eq 0 ] && { score732 "$RRLBL" "expertium:w64:$DREF/expertium/$TRACK/master.pth" "--grade-rectify --workers 3" $MIN_RECT 4 $MIN_RECT_LONG || rc=1; }
measured() { ls "$ARMMEAS/users" 2>/dev/null | grep -c '\.npz$'; }
for pass in 1 2; do
  [ $rc -eq 0 ] || break
  [ "$DRYRUN" != 1 ] && [ "$(measured)" -ge 722 ] && break
  q $MIN_MEASURE "what-if intervals of the arm on the 732 pool (723 users, 3,000 reviews each, the grade-order sanity check (SM33)'s sampled reviews) (pass $pass)" -- \
    "$PY" -X utf8 "$GO_PY" measure --model "$DARM/expertium/$TRACK/master.pth" --device cuda --users-file "$USERS732" --probes 3000 \
    --time-limit 2100 --out "$ARMMEAS" >> "$G/measure.out" 2>&1 || { log "STOP: measure ticket failed"; rc=1; }
  [ "$DRYRUN" = 1 ] && break
done
[ $rc -eq 0 ] && [ "$DRYRUN" != 1 ] && log "arm measure: $(measured) users"
if [ $rc -eq 0 ] && [ "$DRYRUN" != 1 ]; then
  cpu_step 60 "derive + rule" || rc=1
  out=$(OMP_NUM_THREADS=2 "$PY" -X utf8 "$S/sm95_decide.py" --rows "$OUTROWS" --ref-rows "$REFROWS" --sm33-rows "$SM33ROWS" \
        --arm-measure "$ARMMEAS" --ref-measure "$REFMEAS" --sm33-measure "$SM33MEAS" --derived "$S/work/derived" 2>&1); r=$?
  echo "$out" >> "$G/rule.out"
  log "rule exit $r: $(echo "$out" | tail -1)"; [ "$r" = 0 ] || rc=1
fi
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; else log "done"; fi
else log "STOP: ended with exit $rc"; fi
exit $rc
