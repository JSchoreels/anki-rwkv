#!/usr/bin/env bash
# The instant-only small model (SM57) (artifacts/small-model/sm57/DESIGN.md), adapted from launch-sm92.sh:
#   instant-only  the leak-free small model (SM72s) constant-LR epoch 25 (the release's start) + epoch 26 (the run's own
#                 epoch-26 protocol) + the release's exact one-epoch 32k Muon cool-down, both with
#                 "curve_loss_weight": 0.0 (every curve loss term, curve KD included, at weight 0).
#   joint         the same epoch 26 + cool-down with the normal joint loss (also epoch 26 of the "more epochs" run).
# Rule (sm57_rule.py, CPU): 722 users of the 732 set, normal inputs, instant-only - joint on the instant head:
#   go if the 95% CI is entirely below zero; else stop.
# CPU proofs (sm57_check.py -> check-cpu-sm57.json): ALL OK required.
# Steps: preconditions + pair checks -> GPU check (trainer smokes: joint by the wt/sm92 tree == this tree bitwise,
#   instant-only trains) -> instant-only epoch 26 -> joint epoch 26 (snapshot-resumable: start when 30 min fit, snapshot
#   stop <= 2 min before a window, resume after it) -> both cool-downs prepared + label windows + Muon fresh checks + pair
#   checks (vs the release, and instant-only vs joint) -> instant-only cool-down smoke -> instant-only cool-down ->
#   its scoring (normal inputs, 723 users cap 0.8 + 3 long users cap 0.92) -> joint cool-down -> its scoring -> rule.
# Breaks: quiet.json (scope all/gpu) plus the daily 20:00-01:00 break; re-read at every check; CPU steps too.
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm57/artifacts/small-model/sm57/launch-sm57.sh
#   launch   WMI hidden (artifacts/small-model/sm57/HANDOFF.md); log artifacts/small-model/sm57/gpu/launch-sm57.log
#   resume   the launch command again (finished steps are skipped; an interrupted epoch / cool-down resumes from its snapshot)
#   stop     touch C:/Users/jonat/workspace/wt/sm57/artifacts/small-model/sm57/gpu/STOP (between tickets; a running
#            epoch / cool-down gets a snapshot stop at the next user boundary)
# Do not edit scripts/ expertium/ rwkv/ of the sm57 worktree while it runs (SourceWatch stops the training).
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
QUIET=C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json
WT=C:/Users/jonat/workspace/wt/sm57
SM=$WT/artifacts/small-model
S=$SM/sm57
G=$S/gpu
mkdir -p "$G"
LOG=$G/launch-sm57.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm57-dryrun.log
WARM="$PY -X utf8 $WT/scripts/sm77_warm.py"
DEC_PY=$WT/scripts/expertium_decay.py
SC=$WT/scripts/small_model_heldout.py
RULE=$S/sm57_rule.py
PAIR=$S/sm57_pair.py
TRACK=E89-F65e-F77a
ARCH=w64
K=25; K1=26
FLAG='{"curve_loss_weight": 0.0}'
RUN72S=C:/Users/jonat/workspace/wt/sm72s/artifacts/accuracy/experiments/scale-w64kd-final-sc-lanes-leakfree/run   # read only
S72=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s                                                  # read only
REL=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/scale-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon  # the release, read only
BIGROWS=C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72/gpu-big/bigleak-neighbour.jsonl
REPLAY=C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/big-leakfree-1ep-final8k-neighbour
USERS732=$S72/eval-users-732-fit.json; ULONG=$S72/users-732-long.json
IE=$SM/sm57-inst-e$K1;  ID=$SM/sm57-inst-e$K1-dec-1ep32k-muon;  ILBL=sm57-inst-dec1ep32k
JE=$SM/sm57-joint-e$K1; JD=$SM/sm57-joint-e$K1-dec-1ep32k-muon; JLBL=sm57-joint-dec1ep32k
OUTROWS=$G/sm57-rows.jsonl
SEGS=$G/segments.jsonl
MODE=neighbour
LF="env RWKV_DELETED_CARDS=$MODE"
FRACTION=1.0
MIN_EPOCH=${MIN_EPOCH:-125}   # SM72s / SM92 epochs: 108-114 min training, ~120 min tickets
MIN_DECAY=${MIN_DECAY:-560}   # the release's cool-down: ~480-520 min + setup
MIN_START=30                  # a fresh epoch / cool-down starts when 30 min fit (snapshot stop before the window, resume after)
MIN_GPUCHECK=15; MIN_SMOKE=10; MIN_SCORE732=32; MIN_SCORE_RETRY=20; MIN_LONG=25
TL732=1800; TLLONG=1380
NAME="the instant-only small model (SM57)"
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START   # scorer cap 0.8; trainer cap 0.8 (options)
log() { echo "[sm57 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
stopped() { [ -e "$G/STOP" ] && { log "STOP: STOP file present"; return 0; }; return 1; }
nap() { [ "$DRYRUN" = 1 ] && return 0; local s=$1; [ "$s" -gt 600 ] && s=600; sleep "$s"; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'; }

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
blocks += [(day + timedelta(days=d, hours=20), day + timedelta(days=d + 1, hours=1)) for d in range(-1, 14)]  # daily break
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
    "$PY" "$QUEUE" run --owner sm57 --purpose "$NAME: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm57/launch-sm57.sh" -- "$@"
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
  for f in "$RUN72S/epoch-$K/expertium/$TRACK/master.pth" "$RUN72S/epoch-$K/expertium/$TRACK/optimizer.pth" "$RUN72S/epoch-$K/protocol.json" \
           "$RUN72S/run.json" "$REPLAY/manifest.json" "$USERS732" "$ULONG" "$REL/decay.json" "$REL/protocol.json" \
           "$REL/expertium/$TRACK/master.pth" "$BIGROWS" "$S/check-cpu-sm57.json" "$IE/decay.json" "$IE/protocol.json" \
           "$JE/decay.json" "$JE/protocol.json" "$S/extra-options-joint.json" "$S/extra-options-instant.json"; do
    [ -f "$f" ] || { log "STOP: missing $f"; return 1; }
  done
  done_in "$REL" || { log "STOP: the release's cool-down $REL is not complete"; return 1; }
  grep -q '"ok": true' "$S/check-cpu-sm57.json" && ! grep -q '"ok": false' "$S/check-cpu-sm57.json" \
    || { log "STOP: the CPU proofs are not ALL OK ($S/check-cpu-sm57.json)"; return 1; }
  if [ "$DRYRUN" != 1 ] && [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    log "STOP: uncommitted source changes in scripts/ expertium/ rwkv/ (SourceWatch refuses them)"; return 1; fi
  "$PY" -X utf8 "$PAIR" epoch --arm "$JE" --ref "$RUN72S/epoch-$K" >> "$LOG" 2>&1 || { log "STOP: joint epoch $K1 is not the recipe"; return 1; }
  "$PY" -X utf8 "$PAIR" epoch --arm "$IE" --ref "$RUN72S/epoch-$K" --flag "$FLAG" >> "$LOG" 2>&1 || { log "STOP: instant-only epoch $K1 is not the recipe + the flag"; return 1; }
  "$PY" -X utf8 "$PAIR" arms --arm "$IE" --ref "$JE" --flag "$FLAG" >> "$LOG" 2>&1 || { log "STOP: the epoch-$K1 arms differ beyond the flag"; return 1; }
  return 0
}

segment_line() {  # segment_line FOLDER STAGE START_EPOCH END_EPOCH EXIT BASE_MIB MAX_MIB STOP_SENT
  "$PY" -X utf8 -c "
import json, sys
f, stage, a, b, rc, base, mx, stop = sys.argv[1:9]
print(json.dumps({'folder': f, 'stage': stage, 'start_epoch_s': int(a), 'end_epoch_s': int(b), 'minutes': round((int(b) - int(a)) / 60, 1),
                  'exit': int(rc), 'nvidia_smi_baseline_mib': int(base or 0), 'nvidia_smi_max_used_mib': int(mx or 0), 'stop_requested': stop == '1'}))" \
    "$@" >> "$SEGS"
}

run_folder() {  # run_folder DIR WHAT FIRST_MINUTES SEG_MAX: expertium_decay.py run in segments; resumable from snapshots
  local D=$1 what=$2 first=$3 segmax=$4 rc qpid stop_sent b need t0 base mx u
  while ! done_in "$D"; do
    stopped && return 1
    need=30; [ ! -d "$D/expertium" ] && [ ! -d "$D/attempts" ] && need=$first
    wait_free "$need" "$what" || return 1
    local seg=$(( FREE - 5 )); [ "$seg" -gt "$segmax" ] && seg=$segmax
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $NAME: $what (segment to the next window or the end, expected $seg min of ~$segmax)"
      echo "    cmd: $LF $PY -X utf8 $DEC_PY run --out $D" >> "$LOG"; return 0; fi
    rm -f "$D/control/STOP"
    log "enqueue: $what (expected $seg min of ~$segmax, $FREE min to the next window)"
    t0=$(date +%s); base=$(gpu_used); mx=0
    "$PY" "$QUEUE" run --owner sm57 --purpose "$NAME: $what (segment to the next window or the end)" --claim gpu \
      --expected-minutes "$seg" --max-wait-minutes $(( FREE - 20 > 1 ? FREE - 20 : 1 )) --launched-by "detached launcher sm57/launch-sm57.sh" -- \
      $LF "$PY" -X utf8 "$DEC_PY" run --out "$D" >> "$G/train.out" 2>&1 &
    qpid=$!
    stop_sent=0
    while kill -0 "$qpid" 2>/dev/null; do
      u=$(gpu_used); [ -n "$u" ] && [ "$u" -gt "$mx" ] && mx=$u
      b=$(block)
      if [ -e "$G/STOP" ] || [[ "$b" == NOW* ]] || { [[ "$b" == NEXT* ]] && [ "${b#NEXT }" -le 2 ]; }; then
        [ "$stop_sent" = 0 ] && log "requesting a snapshot stop ($b)"
        "$PY" -X utf8 "$DEC_PY" stop --out "$D" > /dev/null 2>&1; stop_sent=1
      fi
      sleep 30
    done
    wait "$qpid"; rc=$?; log "exit $rc: $what segment (nvidia-smi max $mx MiB, baseline $base MiB)"
    segment_line "$D" "$what" "$t0" "$(date +%s)" "$rc" "$base" "$mx" "$stop_sent"
    [ "$rc" = 75 ] && { sleep 60; continue; }
    done_in "$D" && break
    [ "$stop_sent" = 0 ] && { log "STOP: $what segment ended without completing and without a stop request (exit $rc); see $G/train.out"; return 1; }
  done
  [ "$DRYRUN" = 1 ] || log "$what complete: $("$PY" -X utf8 -c "import json; r=json.load(open(r'$D/expertium/$TRACK/result.json')); print(f\"{r.get('review_presentations', 0):,} reviews, {r.get('training_seconds', 0)/60:.1f} min training, {r.get('reviews_per_second', 0):,.0f} reviews/s, peak allocated {r.get('peak_allocated_mib', 0):,.0f} MiB, reserved {r.get('peak_reserved_mib', 0):,.0f} MiB, curve_loss_weight {r.get('curve_loss_weight')}, update offset end {r.get('update_offset_end')}\")")"
  return 0
}

prepare_decay() {  # prepare_decay EPOCH_DIR DECAY_DIR EXTRA_FILE: the release's cool-down recipe from the arm's own epoch 26
  local E=$1 D=$2 X; X=$(cat "$3")
  $LF $WARM cooldown --epoch $K --start "$E" --out "${D#$WT/}" --arch $ARCH --fraction $FRACTION --extra-options "$X" >> "$G/prepare.out" 2>&1 \
    || { log "cool-down prepare failed ($D)"; return 1; }
  $LF "$PY" -X utf8 "$WT/scripts/check_replay_label_windows.py" --decay "$D" --trim-unlabelled-tail >/dev/null || { log "label windows (trim pass) failed ($D)"; return 1; }
  $LF "$PY" -X utf8 "$WT/scripts/check_replay_label_windows.py" --decay "$D" --out "$G/check-replay-label-windows-$(basename "$D").json" >/dev/null \
    || { log "label windows check failed ($D)"; return 1; }
  CUDA_VISIBLE_DEVICES= $LF "$PY" -X utf8 "$WT/scripts/check_muon_fresh_decay.py" --decay "$D" --out "$G/check-muon-fresh-$(basename "$D").json" \
    | grep -c PASS | sed "s/^/[sm57-muon] $(basename "$D") PASS lines: /" >> "$LOG"
  [ "${PIPESTATUS[0]}" = 0 ] || { log "Muon fresh-state CPU check failed ($D)"; return 1; }
  return 0
}

score() {  # score TAG SPEC WHAT: normal inputs; 723 users at cap 0.8 (a pass without progress -> one pass at 0.92), then the 3 long users at 0.92
  local tag=$1 spec=$2 what=$3 pass have prev=-1 cap=0.8 minutes lb
  lb="$tag/w8+s8"
  for pass in 1 2 3 4; do
    have=$(label_users "$OUTROWS" "$lb")
    [ "$have" -ge 723 ] && break
    if [ "$have" = "$prev" ]; then
      if [ "$cap" = 0.8 ]; then cap=0.92; log "no progress on $lb ($have of 723 users): next pass at cap 0.92 (memory only, same numerics)"
      else log "STOP: no progress on $lb ($have of 723 users)"; return 1; fi
    fi
    prev=$have
    minutes=$MIN_SCORE732; [ "$pass" -gt 1 ] && minutes=$MIN_SCORE_RETRY
    q $minutes "scoring $what on normal inputs, the 732 set minus 9 never-fit (723 users) (pass $pass, $have done, cap $cap)" -- \
      env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START HELDOUT_GPU_MEMORY_FRACTION=$cap "$PY" -X utf8 "$SC" --pool 732 --users-file "$USERS732" \
      --model "$tag=$spec" --only "$tag/w8+s8" --inputs normal --time-limit $TL732 --split-existing \
      --deleted-cards $MODE --out "$OUTROWS" >> "$G/score.out" 2>&1 || { log "STOP: scoring ticket failed ($lb)"; return 1; }
    [ "$DRYRUN" = 1 ] && break
  done
  [ "$DRYRUN" = 1 ] || [ "$(label_users "$OUTROWS" "$lb")" -ge 723 ] || { log "STOP: $lb scored on $(label_users "$OUTROWS" "$lb") of 723 users"; return 1; }
  for pass in 1 2; do
    if [ "$DRYRUN" = 1 ] || [ "$(label_users "$OUTROWS" "$lb")" -lt 726 ]; then
      q $MIN_LONG "scoring $what on normal inputs, the 3 long users of the 732 set (pass $pass, cap 0.92)" -- \
        env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START HELDOUT_GPU_MEMORY_FRACTION=0.92 "$PY" -X utf8 "$SC" --pool 732 --users-file "$ULONG" \
        --model "$tag=$spec" --only "$tag/w8+s8" --inputs normal --time-limit $TLLONG --split-existing \
        --deleted-cards $MODE --out "$OUTROWS" >> "$G/score.out" 2>&1 \
        || { log "long users ticket failed ($lb); continuing with $(label_users "$OUTROWS" "$lb") users (the rule pool does not need them)"; break; }
      [ "$DRYRUN" = 1 ] && break
    fi
  done
  [ "$DRYRUN" = 1 ] || log "$lb: $(label_users "$OUTROWS" "$lb") users scored"
  return 0
}

rule() {  # rule WHEN: run the rule (a partial read-out exits 2 until both arms are scored)
  [ "$DRYRUN" = 1 ] && { log "DRYRUN: then $PY -X utf8 $RULE decide ($1)"; return 0; }
  cpu_step 3 "rule" || return 1
  local out r; out=$("$PY" -X utf8 "$RULE" decide 2>&1); r=$?
  log "rule ($1) exit $r: $(echo "$out" | tail -1)"
  [ "$r" = 0 ] || [ "$1" = partial ]
}

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1) (DRYRUN=$DRYRUN; instant-only vs joint epoch $K1 + the release's cool-down from the leak-free small model (SM72s) epoch $K)"
rc=0
cpu_step 5 "preparation" || rc=1
[ $rc -eq 0 ] && { preconditions || rc=1; }
if [ $rc -eq 0 ] && [ ! -e "$G/gpu-check-ok" ]; then
  q $MIN_GPUCHECK "GPU check (trainer smokes 2 users x 512 reviews: joint by the base tree == this tree bitwise, instant-only trains)" -- \
    "$PY" -X utf8 "$S/sm57_gpu_check.py" --out "$G/check-gpu-sm57.json" >> "$G/check.out" 2>&1 \
    || { log "STOP: GPU check failed (see $G/check.out, $G/check-gpu-sm57.json)"; rc=1; }
  [ $rc -eq 0 ] && [ "$DRYRUN" != 1 ] && { touch "$G/gpu-check-ok"; log "GPU check: $(grep 'ALL OK' "$G/check.out" | tail -1 | cut -c1-400)"; }
fi
EP="(w64 from the leak-free small model (SM72s) epoch $K, the release's epoch 21-25 recipe: 8k final tier, 2048 x 16 lanes carry, leak-free replay KD, AdamW 7e-4 constant)"
[ $rc -eq 0 ] && { run_folder "$IE" "instant-only epoch $K1, curve loss weight 0 $EP" $MIN_START $MIN_EPOCH || rc=1; }
[ $rc -eq 0 ] && { run_folder "$JE" "joint epoch $K1 $EP" $MIN_START $MIN_EPOCH || rc=1; }
for arm in inst joint; do
  [ $rc -eq 0 ] || break
  if [ "$arm" = inst ]; then E=$IE; D=$ID; X=$S/extra-options-instant.json; else E=$JE; D=$JD; X=$S/extra-options-joint.json; fi
  if [ ! -f "$D/decay.json" ]; then
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN: the $arm cool-down is prepared after the trained epoch $K1"
    else
      cpu_step 20 "$arm cool-down prepare" || { rc=1; break; }
      prepare_decay "$E" "$D" "$X" || { rm -rf "$D"; rc=1; break; }
      log "cool-down prepared: $(basename "$D")"
    fi
  fi
done
if [ $rc -eq 0 ] && [ "$DRYRUN" != 1 ]; then
  "$PY" -X utf8 "$PAIR" decay --arm "$JD" --ref "$REL" >> "$LOG" 2>&1 || { log "STOP: the joint cool-down does not match the release's"; rc=1; }
  [ $rc -eq 0 ] && { "$PY" -X utf8 "$PAIR" decay --arm "$ID" --ref "$REL" --flag "$FLAG" >> "$LOG" 2>&1 || { log "STOP: the instant-only cool-down is not the release's + the flag"; rc=1; }; }
  [ $rc -eq 0 ] && { "$PY" -X utf8 "$PAIR" arms --arm "$ID" --ref "$JD" --flag "$FLAG" >> "$LOG" 2>&1 || { log "STOP: the cool-down arms differ beyond the flag"; rc=1; }; }
fi
if [ $rc -eq 0 ] && [ ! -e "$G/smoke-inst-decay-ok" ]; then
  q $MIN_SMOKE "trainer smoke of the instant-only cool-down (fresh Muon, curve loss weight 0, 2 users x 512 reviews)" -- \
    $LF "$PY" -X utf8 "$DEC_PY" smoke --out "$ID" >> "$G/check.out" 2>&1 || { log "STOP: instant-only cool-down smoke failed (see $G/check.out)"; rc=1; }
  [ $rc -eq 0 ] && [ "$DRYRUN" != 1 ] && { touch "$G/smoke-inst-decay-ok"; log "smoke: $(grep '"smoke"' "$G/check.out" | tail -1 | cut -c1-200)"; }
fi
DC="the release's one-epoch 32k Muon cool-down from epoch $K1 (w64, the release's users, order, seeds, KD targets)"
[ $rc -eq 0 ] && { run_folder "$ID" "instant-only: $DC, curve loss weight 0" $MIN_START $MIN_DECAY || rc=1; }
[ $rc -eq 0 ] && { score $ILBL "expertium:$ARCH:$ID/expertium/$TRACK/master.pth" "the instant-only model" || rc=1; }
[ $rc -eq 0 ] && { rule partial || rc=1; }
[ $rc -eq 0 ] && { run_folder "$JD" "joint: $DC" $MIN_START $MIN_DECAY || rc=1; }
[ $rc -eq 0 ] && { score $JLBL "expertium:$ARCH:$JD/expertium/$TRACK/master.pth" "the joint model" || rc=1; }
[ $rc -eq 0 ] && { rule final || rc=1; }
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; else log "done"; fi
else
  log "STOP: ended with exit $rc"
fi
exit $rc
