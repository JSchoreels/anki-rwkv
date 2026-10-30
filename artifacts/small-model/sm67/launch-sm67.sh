#!/usr/bin/env bash
# Data augmentation (SM67) stage 1 (artifacts/small-model/sm67/DESIGN.md on main): ONE augmented 1-epoch 32k Muon
# cool-down of the leak-free small model (SM72s) from its constant-LR epoch K (START_EPOCH, default 25), compared with
# the existing cool-down from the same epoch (the control, trained and scored by the SM72s launcher; free).
#   arm      the control's exact recipe (SM72s's prepare command and $EXTRA: same 8,000 users, seed-1 order, LR 7e-4 -> 0
#            over the same 581.9M reviews, fresh Muon, replay KD from the big model (1 leak-free epoch), lanes carry,
#            cap 0.8, RWKV_DELETED_CARDS=neighbour) plus "augment": note_split (scripts/augment.py): half of the
#            training users with >= 65,536 reviews that are in no evaluation set are split into two pseudo-users by a
#            random half of their notes, in place in the user order. Every card keeps its whole history; the teacher
#            targets are the original ones (review-number remap); the item seeds stay the real user's.
#   control  wt/sm72s .../scale-w64kd-final-sc-lanes-leakfree-dec$K-1ep32k-muon, label
#            s-w64kd-final-sc-lanes-leakfree-dec$K-1ep32k-muon/w8+s8 in wt/sm72s .../gpu/small-lf-neighbour.jsonl.
# Steps: wait for the tiny architecture additions' (SM77) launcher to end -> CPU prepare (expansion, label-window trims,
#   Muon fresh-state check, pair check vs the control) -> ticket 0: GPU smoke of the pseudo-user path (2 halves x 512
#   reviews, replay remap) -> the cool-down in segments (~8.3 h) -> 732 scoring (723 fit users + the 3 long users at the
#   0.92 cap) -> wait for the control's 732 rows -> rule (scripts/sm67_rule.py decide, 722 users).
# Rule (pre-registered): go if curve <= -0.20 or instant <= -0.15 points of % of the big model (1 leak-free epoch), that
#   head's CI clear of zero, the other head <= +0.03; stop if both > -0.10; grey otherwise. Ship check separate.
# Breaks: quiet.json (scope all/gpu) plus the daily 21:00-02:00 break as a safeguard, re-read at every check (<= 10
#   min). The cool-down runs in segments (snapshot stop <= 2 min before a window, resume after); CPU steps stay out of
#   windows.
#   dry run  DRYRUN=1 [START_EPOCH=K] bash C:/Users/jonat/workspace/wt/sm67/artifacts/small-model/sm67/launch-sm67.sh
#            (runs the real CPU prepare and checks into a scratch folder sm67-dryrun-*, logs the GPU tickets, deletes it)
#   launch   WMI hidden (HANDOFF-sm67.md); log artifacts/small-model/sm67/gpu/launch-sm67.log
#   resume   the launch command again (prepared folder re-checked, an interrupted cool-down resumes from its snapshot,
#            scored users are skipped)
#   stop     touch C:/Users/jonat/workspace/wt/sm67/artifacts/small-model/sm67/gpu/STOP (a running segment is snapshot-stopped)
#   no wait  touch .../sm67/gpu/NO-WAIT (skip the wait for SM77's launcher)
# Do not edit scripts/ expertium/ rwkv/ of the sm67 worktree while it runs (SourceWatch stops the training).
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
BASH_EXE="C:/Program Files/Git/usr/bin/bash.exe"
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
QUIET=C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json
WT=C:/Users/jonat/workspace/wt/sm67
SM=$WT/artifacts/small-model
S=$SM/sm67
G=$S/gpu
mkdir -p "$G"
LOG=$G/launch-sm67.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm67-dryrun.log
DEC_PY=$WT/scripts/expertium_decay.py
SC=$WT/scripts/small_model_heldout.py
RULE=$WT/scripts/sm67_rule.py
CHECK=$WT/scripts/check_augment.py
AUG=$WT/scripts/augment.py
TRACK=E89-F65e-F77a
CKPT=expertium/$TRACK/master.pth
K=${START_EPOCH:-25}
RUN72S=C:/Users/jonat/workspace/wt/sm72s/artifacts/accuracy/experiments/scale-w64kd-final-sc-lanes-leakfree/run   # read only
S72=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s                                                  # read only
CTL=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/scale-w64kd-final-sc-lanes-leakfree-dec$K-1ep32k-muon  # read only: the control
CTL_ROWS=$S72/gpu/small-lf-neighbour.jsonl
CTL_LBL=s-w64kd-final-sc-lanes-leakfree-dec$K-1ep32k-muon/w8+s8
CTL_WIN=$S72/check-replay-label-windows-dec$K.json
SM77LOG=C:/Users/jonat/workspace/wt/sm77/artifacts/small-model/sm77/gpu/launch-sm77.log                            # read only
BIGCK=C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72-big-leakfree-1ep/expertium/big-leakfree-1ep/master.pth
REPLAY=C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/big-leakfree-1ep-final8k-neighbour
USERS732=$S72/eval-users-732-fit.json; ULONG=$S72/users-732-long.json
P=sm67; [ "${DRYRUN:-0}" = 1 ] && P=sm67-dryrun   # the dry run prepares a scratch folder and deletes it
D=$SM/$P-aug-dec$K-1ep32k-muon
SMOKE=$G/$P-smoke-pseudo
LBL=sm67-aug-dec$K-1ep32k-muon
OUTROWS=$G/sm67-neighbour.jsonl
MODE=neighbour
LF="env RWKV_DELETED_CARDS=$MODE"
SPEC='{"lanes": "strided"}'
AUGMENT='{"kind": "note_split", "min_reviews": 65536, "share": 0.5, "seed": 67}'
DIST="{\"teacher\": \"$BIGCK\", \"alpha\": 0.9, \"decay_alpha\": 0.5, \"kd_source\": \"replay\", \"replay\": \"$REPLAY\"}"
EXTRA="{\"distillation\": $DIST, \"state_carry\": $SPEC, \"chunk_reviews\": 32768, \"batch_chunks\": 1, \"optimizer\": \"expertium-iter53\", \"augment\": $AUGMENT}"  # = SM72s's cool-down + augment
MIN_DECAY=${MIN_DECAY:-540}   # SM72s cool-downs ~8 h at ~20.2k reviews/s + 3% updates + setup; segmented
MIN_CHECK=15; MIN_SCORE732=12; MIN_LONG=6
NAME="data augmentation (SM67) stage 1"
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS   # scorer cap 0.8; trainer cap 0.8 (options)
log() { echo "[sm67 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
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
blocks += [(day + timedelta(days=d, hours=21), day + timedelta(days=d + 1, hours=2)) for d in range(-1, 9)]  # daily break
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
    "$PY" "$QUEUE" run --owner small-model --purpose "small-model: $NAME: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm67/launch-sm67.sh" -- "$@"
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
done_in() { [ -f "$1/$CKPT" ] && R="$1/expertium/$TRACK/result.json" "$PY" -X utf8 -c "
import json, os, sys
r = json.load(open(os.environ['R']))
sys.exit(0 if r.get('status') == 'complete' and not r.get('smoke_only') else 1)" 2>/dev/null; }

# ---- wait for the tiny architecture additions' (SM77) launcher: its log's last line is "done" or a failure / STOP line
sm77_last() { tail -1 "$SM77LOG" 2>/dev/null; }
sm77_ended() { local l; l=$(sm77_last)
  [[ "$l" == *"] done" ]] || [[ "$l" == *"] STOP"* ]] || [[ "$l" == *"STOP file present"* ]] || [[ "$l" =~ (failed|stopped|ended\ with\ exit|rule\ exit\ [1-9]) ]]; }
wait_sm77() {
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: would wait for the tiny architecture additions' (SM77) launcher; its last line now: $(sm77_last | cut -c1-200)"; return 0; fi
  local n=0
  until sm77_ended || [ -e "$G/NO-WAIT" ]; do
    stopped && return 1
    [ $(( n % 30 )) = 0 ] && log "waiting for the tiny architecture additions' (SM77) launcher to end; its last line: $(sm77_last | cut -c1-200)"
    n=$(( n + 1 )); sleep 120
  done
  log "SM77 launcher ended (or NO-WAIT): $(sm77_last | cut -c1-200)"
}

preconditions() {
  local e; e=$RUN72S/epoch-$(printf %02d $K)/expertium/$TRACK
  for f in "$e/master.pth" "$e/result.json" "$RUN72S/run.json" "$CTL/decay.json" "$CTL/protocol.json" "$CTL_WIN" "$BIGCK" \
           "$REPLAY/manifest.json" "$USERS732" "$ULONG" "C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72/gpu-big/bigleak-neighbour.jsonl"; do
    [ -f "$f" ] || { log "STOP: missing $f"; return 1; }
  done
  R="$e/result.json" "$PY" -X utf8 -c "import json, os, sys; sys.exit(0 if json.load(open(os.environ['R'])).get('status') == 'complete' else 1)" \
    || { log "STOP: SM72s epoch $K is not complete"; return 1; }
  if [ "$DRYRUN" != 1 ] && [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    log "STOP: uncommitted source changes in scripts/ expertium/ rwkv/ (SourceWatch refuses them)"; return 1; fi
  return 0
}

pair_check() {  # the arm vs the control cool-down (options, start, schedule, users, order, reviews, labels, held-out)
  $LF "$PY" -X utf8 "$CHECK" pair --arm "$D" --control "$CTL" --arm-windows "$S/check-replay-label-windows-$P-dec$K.json" \
    --control-windows "$CTL_WIN" --out "$S/check-pair-$P-dec$K.json" >> "$LOG" 2>&1
}

prepare() {  # CPU (~5 min): SM72s's cool-down prepare command + augment; trims; Muon fresh-state check; pair check
  if [ -f "$D/decay.json" ]; then
    pair_check || { log "STOP: the existing $D does not match the control"; return 1; }
    return 0
  fi
  rm -rf "$D"
  $LF "$PY" -X utf8 "$DEC_PY" prepare --source-run "$RUN72S" --epoch "$K" --out "${D#$WT/}" --fraction 1.0 --shape one_minus_sin \
    --evaluate none --seed 1 --fresh-optimizer --train-tier final --extra-options "$EXTRA" > "$G/prepare.out" 2>&1 \
    || { log "STOP: cool-down prepare failed (see $G/prepare.out)"; rm -rf "$D"; return 1; }
  $LF "$PY" -X utf8 "$WT/scripts/check_replay_label_windows.py" --decay "$D" --trim-unlabelled-tail >> "$G/prepare.out" 2>&1 \
    || { log "STOP: label windows (trim pass) failed"; rm -rf "$D"; return 1; }
  $LF "$PY" -X utf8 "$WT/scripts/check_replay_label_windows.py" --decay "$D" --out "$S/check-replay-label-windows-$P-dec$K.json" >> "$G/prepare.out" 2>&1 \
    || { log "STOP: label windows check failed (see $S/check-replay-label-windows-$P-dec$K.json)"; rm -rf "$D"; return 1; }
  CUDA_VISIBLE_DEVICES= $LF "$PY" -X utf8 "$WT/scripts/check_muon_fresh_decay.py" --decay "$D" --out "$S/check-muon-fresh-$P-dec$K.json" \
    | grep -c PASS | sed "s/^/[sm67-muon] PASS lines: /" >> "$LOG"
  [ "${PIPESTATUS[0]}" = 0 ] || { log "STOP: Muon fresh-state CPU check failed"; rm -rf "$D"; return 1; }
  pair_check || { log "STOP: the arm is not the control's recipe plus augment (see $S/check-pair-$P-dec$K.json)"; rm -rf "$D"; return 1; }
  log "prepared $(basename "$D"): $("$PY" -X utf8 -c "import json; d=json.load(open(r'$D/decay.json')); a=d['augment']; print(f\"{a['split_users']:,} of {a['candidates']:,} candidates split ({a['candidates_excluded_as_held_out']} large users of the 2,225 pool never split), {d['training_rows']:,} rows, {a['split_review_share']:.1%} of {d['training_reviews']:,} reviews in halves, smallest half {a['smallest_half']:,}, trims {[t['user_id'] for t in d.get('trimmed_unlabelled_tails', [])]}\")")"
  return 0
}

run_decay() {  # segments to the next window or the end; snapshot stop before a window; resumable
  local what="augmented 1-epoch 32k Muon cool-down from SM72s epoch $K (note split, 8k final tier, lanes carry, leak-free replay KD)" rc qpid stop_sent b
  while ! done_in "$D"; do
    stopped && return 1
    wait_free 30 "$what" || return 1
    local seg=$(( FREE - 5 )); [ "$seg" -gt "$MIN_DECAY" ] && seg=$MIN_DECAY
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $what (segment to the next window or the end, expected $seg min of ~$MIN_DECAY)"
      echo "    cmd: $LF $PY -X utf8 $DEC_PY smoke --out $D && $LF $PY -X utf8 $DEC_PY run --out $D" >> "$LOG"; return 0; fi
    rm -f "$D/control/STOP"   # a stop flag left by the previous segment would stop the smoke at once
    log "enqueue: $what (expected $seg min of ~$MIN_DECAY, $FREE min to the next window)"
    "$PY" "$QUEUE" run --owner small-model --purpose "small-model: $NAME: $what (segment to the next window or the end)" --claim gpu \
      --expected-minutes "$seg" --max-wait-minutes $(( FREE - 20 > 1 ? FREE - 20 : 1 )) --launched-by "detached launcher sm67/launch-sm67.sh" -- \
      $LF "$BASH_EXE" -c "'$PY' -X utf8 '$DEC_PY' smoke --out '$D' && '$PY' -X utf8 '$DEC_PY' run --out '$D'" >> "$G/decay.out" 2>&1 &
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
    wait "$qpid"; rc=$?; log "exit $rc: cool-down segment"
    [ "$rc" = 75 ] && { sleep 60; continue; }
    done_in "$D" && break
    [ "$stop_sent" = 0 ] && { log "STOP: cool-down segment ended without completing and without a stop request (exit $rc); see $G/decay.out"; return 1; }
  done
  [ "$DRYRUN" = 1 ] || log "cool-down complete: $("$PY" -X utf8 -c "import json; r=json.load(open(r'$D/expertium/$TRACK/result.json')); print(f\"{r.get('review_presentations', 0):,} reviews, {r.get('updates', 0):,} updates, {r.get('training_seconds', 0)/60:.1f} min training, {r.get('reviews_per_second', 0):,.0f} reviews/s\")")"
  return 0
}

score() {  # score USERS_FILE OUT LABEL MODEL_SPEC MIN1 MIN2 WANT WHAT: leak-free scoring, 2 passes
  local pass
  for pass in 1 2; do
    if [ "$DRYRUN" = 1 ] || [ "$(label_users "$2" "$3/w8+s8")" -lt "$7" ]; then
      q $([ $pass = 1 ] && echo "$5" || echo "$6") "scoring $3 on $8 (pass $pass)" -- \
        env -u RWKV_DELETED_CARDS "$PY" -X utf8 "$SC" --pool 732 --users-file "$1" --model "$3=$4" --only "$3/w8+s8" \
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

wait_control() {  # the control's 732 rows are written by the SM72s launcher (~15 min after its cool-down)
  local n=0 got
  while true; do
    got=$(label_users "$CTL_ROWS" "$CTL_LBL")
    [ "$got" -ge 723 ] && return 0
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN: the control $CTL_LBL has $got of 723 rows now (the rule waits for them)"; return 0; fi
    stopped && return 1
    [ $(( n % 30 )) = 0 ] && log "waiting for the control's 732 rows ($CTL_LBL: $got of 723)"
    n=$(( n + 1 )); sleep 120
  done
}

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1) (DRYRUN=$DRYRUN, START_EPOCH=$K, control $(basename "$CTL"))"
rc=0
wait_sm77 || rc=1
[ $rc -eq 0 ] && { cpu_step 15 "preparation" || rc=1; }
[ $rc -eq 0 ] && { preconditions || rc=1; }
[ $rc -eq 0 ] && { prepare || rc=1; }
# ticket 0: GPU smoke of the pseudo-user path (the two halves of one split user, 512 reviews each, replay remap), once
if [ $rc -eq 0 ] && [ ! -e "$G/gpu-smoke-ok" ]; then
  $LF "$PY" -X utf8 "$AUG" smoke-folder --decay "$D" --out "$SMOKE" >> "$G/check.out" 2>&1 || { log "STOP: smoke folder failed"; rc=1; }
  [ $rc -eq 0 ] && { q $MIN_CHECK "GPU smoke of the pseudo-user path (2 halves x 512 reviews, replay remap)" -- \
    $LF "$PY" -X utf8 "$DEC_PY" smoke --out "$SMOKE" >> "$G/check.out" 2>&1 || { log "STOP: GPU smoke failed (see $G/check.out)"; rc=1; }; }
  [ $rc -eq 0 ] && [ "$DRYRUN" != 1 ] && { touch "$G/gpu-smoke-ok"; log "GPU smoke: $(grep '"smoke"' "$G/check.out" | tail -1 | cut -c1-240)"; }
fi
[ $rc -eq 0 ] && { run_decay || rc=1; }
[ $rc -eq 0 ] && [ "$DRYRUN" != 1 ] && { done_in "$D" || { log "STOP: no cool-down checkpoint"; rc=1; }; }
[ $rc -eq 0 ] && { score732 "$LBL" "expertium:w64:$D/$CKPT" || rc=1; }
[ $rc -eq 0 ] && { wait_control || rc=1; }
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then
    log "DRYRUN: then $PY -X utf8 $RULE decide --rows $OUTROWS --arm-label $LBL/w8+s8 --ref-label $CTL_LBL"
  else
    cpu_step 3 "rule" || rc=1
    line=$("$PY" -X utf8 "$RULE" decide --rows "$OUTROWS" --arm-label "$LBL/w8+s8" --ref-label "$CTL_LBL" 2>&1); r=$?
    log "rule exit $r: $line"; [ "$r" = 0 ] || rc=1
  fi
fi
if [ "$DRYRUN" = 1 ]; then rm -rf "$D" "$SMOKE" "$S/check-replay-label-windows-$P-dec$K.json" "$S/check-muon-fresh-$P-dec$K.json"; fi
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; else log "done"; fi
else
  log "STOP: ended with exit $rc"
fi
exit $rc
