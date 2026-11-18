#!/usr/bin/env bash
# Show-time interval retrain (SM91) step 1 (artifacts/small-model/sm91/DESIGN.md):
#   arm        the leak-free small model (SM72s) epoch 20 (w64, as is) + epoch 21 = SM72s's epoch-21 protocol (same users,
#              order, seeds, options, AdamW state; constant LR 7e-4) + the quarter-epoch 32k Muon cool-down with the
#              reference's users / order / schedule / options, all trained with the track option
#              "show_time_intervals": true = the duration-leak scoring (SM31) e2s-query inputs (instant query rows'
#              interval + sin / cos + cumulative interval at show time, curve query time at show time; update rows keep
#              the full interval). KD replay targets unchanged (sm91_pair.py: only the flag, start weights and paths differ).
#   reference  SM72s epoch 21 + the same cool-down on normal inputs (the tiny architecture additions (SM77) reference,
#              wt/sm77/artifacts/small-model/sm77-ref-e21-dec-q32k-muon, read only): re-scored here on e2s-query inputs.
# Rule (sm91_rule.py, CPU): 722 users of the 732 set, arm - reference both on e2s-query inputs: go to step 2 if one head
#   gains >= 0.10 points with its 95% CI below zero; else stop.
# CPU proofs (scripts/sm91.py cpu-check -> artifacts/small-model/sm91/check-cpu-sm91.json): ALL OK required.
# Steps: preconditions -> wait for the smaller model, card stream 1 layer (SM73 stage 2) chain to end -> reference
#   re-scoring on e2s-query (723 users cap 0.8, then 3 long users cap 0.92) -> trainer smoke -> arm epoch 21 (must fit
#   before the next window, snapshot-resumable) -> cool-down prepare + label windows + Muon fresh check + pair check vs
#   the reference -> cool-down (segments, snapshot stop <= 2 min before a window, resumes after) -> arm scoring on
#   e2s-query, then on normal inputs (723 + 3 long users each) -> rule.
# Breaks: quiet.json (scope all/gpu) plus the daily 20:00-01:00 break, re-read at every check (<= 10 min); CPU steps too.
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm91/artifacts/small-model/sm91/launch-sm91.sh
#   launch   WMI hidden (artifacts/small-model/sm91/HANDOFF.md); log artifacts/small-model/sm91/gpu/launch-sm91.log
#   resume   the launch command again (finished steps are skipped; an interrupted epoch / cool-down resumes from its snapshot)
#   stop     touch C:/Users/jonat/workspace/wt/sm91/artifacts/small-model/sm91/gpu/STOP (between tickets; a running
#            epoch / cool-down gets a snapshot stop at the next user boundary)
# Do not edit scripts/ expertium/ rwkv/ of the sm91 worktree while it runs (SourceWatch stops the training).
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
QUIET=C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json
WT=C:/Users/jonat/workspace/wt/sm91
SM=$WT/artifacts/small-model
S=$SM/sm91
G=$S/gpu
mkdir -p "$G"
LOG=$G/launch-sm91.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm91-dryrun.log
WARM="$PY -X utf8 $WT/scripts/sm77_warm.py"
DEC_PY=$WT/scripts/expertium_decay.py
SC=$WT/scripts/small_model_heldout.py
RULE=$S/sm91_rule.py
PAIR=$S/sm91_pair.py
TRACK=E89-F65e-F77a
ARCH=w64
K=20; K1=21
RUN72S=C:/Users/jonat/workspace/wt/sm72s/artifacts/accuracy/experiments/scale-w64kd-final-sc-lanes-leakfree/run   # read only
S72=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s                                                  # read only
SM77=C:/Users/jonat/workspace/wt/sm77/artifacts/small-model                                                        # read only
DREF=$SM77/sm77-ref-e21-dec-q32k-muon
REFNORMAL=$SM77/sm77/gpu/sm77-neighbour.jsonl
BIGCK=C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72-big-leakfree-1ep/expertium/big-leakfree-1ep/master.pth
BIGROWS=C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72/gpu-big/bigleak-neighbour.jsonl
REPLAY=C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/big-leakfree-1ep-final8k-neighbour
SM73LOG=C:/Users/jonat/workspace/wt/sm73s2/artifacts/small-model/sm73s2/gpu/launch-sm73s2.log                    # read only
USERS732=$S72/eval-users-732-fit.json; ULONG=$S72/users-732-long.json
ARM=$SM/sm91-arm-e$K1
DARM=$SM/sm91-arm-e$K1-dec-q32k-muon
ALBL=sm91-arm-e$K1-q32k
RLBL=sm77-ref-e21-q32k
OUTROWS=$G/sm91-inputs.jsonl
MODE=neighbour
LF="env RWKV_DELETED_CARDS=$MODE"
FLAGJSON='{"show_time_intervals": true}'
DIST="{\"teacher\": \"$BIGCK\", \"alpha\": 0.9, \"decay_alpha\": 0.5, \"kd_source\": \"replay\", \"replay\": \"$REPLAY\"}"
EXTRA="{\"distillation\": $DIST, \"state_carry\": {\"lanes\": \"strided\"}, \"chunk_reviews\": 32768, \"batch_chunks\": 1, \"optimizer\": \"expertium-iter53\", \"show_time_intervals\": true}"  # = the reference's + the flag
FRACTION=0.25
MIN_EPOCH=${MIN_EPOCH:-125}   # SM72s epochs: 108-112 min training, ~120 min tickets
MIN_DECAY=${MIN_DECAY:-150}   # reference: 124 min training (145.6M reviews at 19.5k reviews/s) + setup
MIN_SMOKE=15; MIN_SCORE732=32; MIN_SCORE_RETRY=20; MIN_LONG=25   # e2s-query scoring: SM31 pass 1 ~28 min on 723 users
TL732=1800; TLLONG=1380
WAIT_SM73_MAX=${WAIT_SM73_MAX:-300}   # minutes to wait for the SM73 stage 2 chain before going ahead anyway
NAME="show-time interval retrain (SM91)"
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START   # scorer cap 0.8; trainer cap 0.8 (options)
log() { echo "[sm91 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
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
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $NAME: $purpose (expected $minutes min)"; echo "    cmd: $*" >> "$LOG"; return 0; fi
    local maxwait=$(( FREE - minutes - 3 )); [ "$maxwait" -lt 1 ] && maxwait=1
    log "enqueue: $purpose (expected $minutes min, max wait $maxwait min, $FREE min to the next window)"
    "$PY" "$QUEUE" run --owner small-model --purpose "$NAME: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm91/launch-sm91.sh" -- "$@"
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
print(len({r['user_id'] for r in map(json.loads, open(p, encoding='utf-8')) if r.get('label') == os.environ['LB'] and 'benchmark' in r}) if p.exists() else 0)"; }
done_in() { [ -f "$1/expertium/$TRACK/master.pth" ] && R="$1/expertium/$TRACK/result.json" "$PY" -X utf8 -c "
import json, os, sys
r = json.load(open(os.environ['R']))
sys.exit(0 if r.get('status') == 'complete' and not r.get('smoke_only') else 1)" 2>/dev/null; }

preconditions() {
  for f in "$RUN72S/epoch-$K/expertium/$TRACK/master.pth" "$RUN72S/epoch-$K1/protocol.json" "$RUN72S/run.json" "$BIGCK" \
           "$REPLAY/manifest.json" "$USERS732" "$ULONG" "$DREF/decay.json" "$DREF/protocol.json" "$DREF/expertium/$TRACK/master.pth" \
           "$REFNORMAL" "$BIGROWS" "$S/check-cpu-sm91.json"; do
    [ -f "$f" ] || { log "STOP: missing $f"; return 1; }
  done
  done_in "$DREF" || { log "STOP: the reference cool-down $DREF is not complete"; return 1; }
  grep -q '"ok": true' "$S/check-cpu-sm91.json" && ! grep -q '"ok": false' "$S/check-cpu-sm91.json" \
    || { log "STOP: the CPU proofs are not ALL OK ($S/check-cpu-sm91.json)"; return 1; }
  if [ "$DRYRUN" != 1 ] && [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    log "STOP: uncommitted source changes in scripts/ expertium/ rwkv/ (SourceWatch refuses them)"; return 1; fi
  return 0
}

prepare_epoch() {
  if [ ! -f "$ARM/decay.json" ]; then
    $LF $WARM epoch --epoch $K1 --out "${ARM#$WT/}" --extra-options "$FLAGJSON" >> "$G/prepare.out" 2>&1 || { log "STOP: arm epoch prepare failed"; return 1; }
    log "arm epoch prepared: $ARM"
  fi
  "$PY" -X utf8 "$PAIR" epoch --arm "$ARM" --ref "$RUN72S/epoch-$K1" >> "$LOG" 2>&1 || { log "STOP: the arm epoch is not SM72s's epoch-$K1 protocol + the flag"; return 1; }
  return 0
}

wait_sm73() {  # queue behind the smaller model, card stream 1 layer (SM73 stage 2) chain (its scoring tickets come later)
  [ "$DRYRUN" = 1 ] && { log "DRYRUN: would wait for the SM73 stage 2 chain ($SM73LOG ends with done / STOP)"; return 0; }
  [ -e "$G/sm73-waited" ] && return 0
  local start; start=$(date +%s)
  while true; do
    stopped && return 1
    if tail -1 "$SM73LOG" 2>/dev/null | grep -qE '\] (done|STOP: ended)'; then log "the smaller model, card stream 1 layer (SM73 stage 2) chain ended: $(tail -1 "$SM73LOG")"; break; fi
    if [ $(( ($(date +%s) - start) / 60 )) -ge "$WAIT_SM73_MAX" ]; then log "waited $WAIT_SM73_MAX min for the SM73 stage 2 chain; going ahead"; break; fi
    sleep 300
  done
  touch "$G/sm73-waited"
  return 0
}

run_folder() {  # run_folder DIR WHAT FIRST_MINUTES SEG_MAX: expertium_decay.py run in segments; resumable from snapshots
  local D=$1 what=$2 first=$3 segmax=$4 rc qpid stop_sent b need
  while ! done_in "$D"; do
    stopped && return 1
    need=30; [ ! -d "$D/expertium" ] && [ ! -d "$D/attempts" ] && need=$first   # a fresh start must fit as a whole
    wait_free "$need" "$what" || return 1
    local seg=$(( FREE - 5 )); [ "$seg" -gt "$segmax" ] && seg=$segmax
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $NAME: $what (segment to the next window or the end, expected $seg min of ~$segmax)"
      echo "    cmd: $LF $PY -X utf8 $DEC_PY run --out $D" >> "$LOG"; return 0; fi
    rm -f "$D/control/STOP"
    log "enqueue: $what (expected $seg min of ~$segmax, $FREE min to the next window)"
    "$PY" "$QUEUE" run --owner small-model --purpose "$NAME: $what (segment to the next window or the end)" --claim gpu \
      --expected-minutes "$seg" --max-wait-minutes $(( FREE - 20 > 1 ? FREE - 20 : 1 )) --launched-by "detached launcher sm91/launch-sm91.sh" -- \
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
  [ "$DRYRUN" = 1 ] || log "$what complete: $("$PY" -X utf8 -c "import json; r=json.load(open(r'$D/expertium/$TRACK/result.json')); print(f\"{r.get('review_presentations', 0):,} reviews, {r.get('training_seconds', 0)/60:.1f} min training, {r.get('reviews_per_second', 0):,.0f} reviews/s, update offset end {r.get('update_offset_end')}\")")"
  return 0
}

prepare_decay() {  # the arm cool-down from the trained arm epoch (users / schedule from SM72s epoch K1, as the reference)
  $LF $WARM cooldown --epoch $K1 --start "$ARM" --out "${DARM#$WT/}" --arch $ARCH --fraction $FRACTION --extra-options "$EXTRA" >> "$G/prepare.out" 2>&1 \
    || { log "cool-down prepare failed ($DARM)"; return 1; }
  $LF "$PY" -X utf8 "$WT/scripts/check_replay_label_windows.py" --decay "$DARM" --trim-unlabelled-tail >/dev/null || { log "label windows (trim pass) failed ($DARM)"; return 1; }
  $LF "$PY" -X utf8 "$WT/scripts/check_replay_label_windows.py" --decay "$DARM" --out "$G/check-replay-label-windows-$(basename "$DARM").json" >/dev/null \
    || { log "label windows check failed ($DARM)"; return 1; }
  CUDA_VISIBLE_DEVICES= $LF "$PY" -X utf8 "$WT/scripts/check_muon_fresh_decay.py" --decay "$DARM" --out "$G/check-muon-fresh-$(basename "$DARM").json" \
    | grep -c PASS | sed "s/^/[sm91-muon] $(basename "$DARM") PASS lines: /" >> "$LOG"
  [ "${PIPESTATUS[0]}" = 0 ] || { log "Muon fresh-state CPU check failed ($DARM)"; return 1; }
  return 0
}

score() {  # score TAG SPEC INPUTS WHAT: 723 users at cap 0.8 (a pass without progress -> one pass at 0.92), then the 3 long users at 0.92
  local tag=$1 spec=$2 v=$3 what=$4 pass have prev=-1 cap=0.8 minutes lb
  lb="$tag/w8+s8"; [ "$v" != normal ] && lb="$lb@$v"
  for pass in 1 2 3 4; do
    have=$(label_users "$OUTROWS" "$lb")
    [ "$have" -ge 723 ] && break
    if [ "$have" = "$prev" ]; then
      if [ "$cap" = 0.8 ]; then cap=0.92; log "no progress on $lb ($have of 723 users): next pass at cap 0.92 (memory only, same numerics)"
      else log "STOP: no progress on $lb ($have of 723 users)"; return 1; fi
    fi
    prev=$have
    minutes=$MIN_SCORE732; [ "$pass" -gt 1 ] && minutes=$MIN_SCORE_RETRY
    q $minutes "scoring $what on $v inputs, the 732 set minus 9 never-fit (723 users) (pass $pass, $have done, cap $cap)" -- \
      env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START HELDOUT_GPU_MEMORY_FRACTION=$cap "$PY" -X utf8 "$SC" --pool 732 --users-file "$USERS732" \
      --model "$tag=$spec" --only "$tag/w8+s8" --inputs "$v" --time-limit $TL732 --split-existing \
      --deleted-cards $MODE --out "$OUTROWS" >> "$G/score.out" 2>&1 || { log "STOP: scoring ticket failed ($lb)"; return 1; }
    [ "$DRYRUN" = 1 ] && break
  done
  [ "$DRYRUN" = 1 ] || [ "$(label_users "$OUTROWS" "$lb")" -ge 723 ] || { log "STOP: $lb scored on $(label_users "$OUTROWS" "$lb") of 723 users"; return 1; }
  for pass in 1 2; do
    if [ "$DRYRUN" = 1 ] || [ "$(label_users "$OUTROWS" "$lb")" -lt 726 ]; then
      q $MIN_LONG "scoring $what on $v inputs, the 3 long users of the 732 set (pass $pass, cap 0.92)" -- \
        env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START HELDOUT_GPU_MEMORY_FRACTION=0.92 "$PY" -X utf8 "$SC" --pool 732 --users-file "$ULONG" \
        --model "$tag=$spec" --only "$tag/w8+s8" --inputs "$v" --time-limit $TLLONG --split-existing \
        --deleted-cards $MODE --out "$OUTROWS" >> "$G/score.out" 2>&1 \
        || { log "long users ticket failed ($lb); continuing with $(label_users "$OUTROWS" "$lb") users (the rule pool does not need them)"; break; }
      [ "$DRYRUN" = 1 ] && break
    fi
  done
  [ "$DRYRUN" = 1 ] || log "$lb: $(label_users "$OUTROWS" "$lb") users scored"
  return 0
}

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1) (DRYRUN=$DRYRUN; arm w64 + show-time inputs from the leak-free small model (SM72s) epoch $K; reference $DREF)"
rc=0
cpu_step 10 "preparation" || rc=1
[ $rc -eq 0 ] && { preconditions || rc=1; }
[ $rc -eq 0 ] && { prepare_epoch || rc=1; }
[ $rc -eq 0 ] && { wait_sm73 || rc=1; }
[ $rc -eq 0 ] && { score $RLBL "expertium:$ARCH:$DREF/expertium/$TRACK/master.pth" e2s-query "the tiny architecture additions (SM77) reference cool-down (the leak-free small model (SM72s) epoch 21 + quarter-epoch 32k Muon, normal-trained)" || rc=1; }
if [ $rc -eq 0 ] && [ ! -e "$G/smoke-ok" ]; then
  q $MIN_SMOKE "trainer smoke of the arm epoch (w64, show-time inputs, 2 users x 512 reviews)" -- \
    $LF "$PY" -X utf8 "$DEC_PY" smoke --out "$ARM" >> "$G/check.out" 2>&1 \
    || { log "STOP: smoke failed (see $G/check.out)"; rc=1; }
  [ $rc -eq 0 ] && [ "$DRYRUN" != 1 ] && { touch "$G/smoke-ok"; log "smoke: $(grep '"smoke"' "$G/check.out" | tail -1 | cut -c1-200)"; }
fi
[ $rc -eq 0 ] && { run_folder "$ARM" "arm epoch $K1 on show-time inputs (w64 from the leak-free small model (SM72s) epoch $K, its epoch-$K1 protocol: 8k final tier, 2048 x 16 lanes carry, leak-free replay KD, AdamW 7e-4 constant)" $MIN_EPOCH $MIN_EPOCH || rc=1; }
if [ $rc -eq 0 ] && [ ! -f "$DARM/decay.json" ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: the cool-down is prepared after the trained arm epoch"
  else
    cpu_step 15 "cool-down prepare" || rc=1
    [ $rc -eq 0 ] && { prepare_decay || { rm -rf "$DARM"; rc=1; }; }
    [ $rc -eq 0 ] && log "cool-down prepared: $(basename "$DARM")"
  fi
fi
if [ $rc -eq 0 ] && [ "$DRYRUN" != 1 ]; then
  "$PY" -X utf8 "$PAIR" decay --arm "$DARM" --ref "$DREF" >> "$LOG" 2>&1 || { log "STOP: the cool-down does not match the reference's (except the flag)"; rc=1; }
fi
[ $rc -eq 0 ] && { run_folder "$DARM" "arm quarter-epoch 32k Muon cool-down on show-time inputs from arm epoch $K1 (w64, the reference's options and order)" 30 $MIN_DECAY || rc=1; }
[ $rc -eq 0 ] && { score $ALBL "expertium:$ARCH:$DARM/expertium/$TRACK/master.pth" e2s-query "the show-time arm cool-down" || rc=1; }
[ $rc -eq 0 ] && { score $ALBL "expertium:$ARCH:$DARM/expertium/$TRACK/master.pth" normal "the show-time arm cool-down (information)" || rc=1; }
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then
    log "DRYRUN: then $PY -X utf8 $RULE decide"
  else
    cpu_step 3 "rule" || rc=1
    out=$("$PY" -X utf8 "$RULE" decide 2>&1); r=$?
    log "rule exit $r: $(echo "$out" | tail -1)"; [ "$r" = 0 ] || rc=1
  fi
fi
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; else log "done"; fi
else
  log "STOP: ended with exit $rc"
fi
exit $rc
