#!/usr/bin/env bash
# The instant-only small model (SM57) stage 2 (artifacts/small-model/sm57/DESIGN.md, "Stage 2 plan"):
#   from scratch on the factory's 2k tier (2,000 users, 146.65M reviews/epoch) with the leak-free small model (SM72s)'s
#   own recipe (2,048-review windows x 16 lanes, leak-free replay KD alpha 0.9, AdamW 7e-4 constant, training_seed 2025),
#   joint vs instant-only ("curve_loss_weight": 0.0), same fresh init, same users / order / seeds / teacher targets.
#   Epochs 1..F (F = 5 unless the time plan after epoch 1 cuts it, 7 h cap), interleaved: inst eN, joint eN, then the
#   scoring of both when N is scored (epochs 1, 3, F), then the rule (sm57_stage2.py rule) at F.
# CPU proofs (sm57s2_check.py -> check-cpu-sm57-stage2.json): ALL OK required. First ticket: the GPU check (smokes).
# Breaks: quiet.json (scope all/gpu) plus the default nightly 20:00-01:00 unless quiet.json has a window overlapping
#   that night (then quiet.json's window replaces the default) -- sm57_stage2.py block; re-read at every check, CPU
#   steps too. Epochs snapshot-stop <= 2 min before a window and resume after it (scripts/expertium_decay.py run).
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm57/artifacts/small-model/sm57/launch-sm57-stage2.sh
#   launch   WMI hidden (artifacts/small-model/sm57/HANDOFF-stage2.md); log artifacts/small-model/sm57/gpu/launch-sm57-stage2.log
#   resume   the launch command again (finished steps are skipped; an interrupted epoch resumes from its snapshot)
#   stop     touch C:/Users/jonat/workspace/wt/sm57/artifacts/small-model/sm57/gpu/STOP2 (between tickets; a running
#            epoch gets a snapshot stop at the next user boundary)
# Do not edit scripts/ expertium/ rwkv/ of the sm57 worktree while it runs (SourceWatch stops the training).
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/sm57
SM=$WT/artifacts/small-model
S=$SM/sm57
G=$S/gpu
mkdir -p "$G"
LOG=$G/launch-sm57-stage2.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm57-stage2-dryrun.log
ST="$PY -X utf8 $S/sm57_stage2.py"
DEC_PY=$WT/scripts/expertium_decay.py
SC=$WT/scripts/small_model_heldout.py
TRACK=E89-F65e-F77a
ARCH=w64
S72=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s                                                  # read only
USERS732=$S72/eval-users-732-fit.json
OUTROWS=$G/sm57s2-rows.jsonl
SEGS=$G/segments-stage2.jsonl
TICKETS=$G/tickets-stage2.jsonl
PLAN=$G/plan-stage2.json
STOPF=$G/STOP2
MODE=neighbour
LF="env RWKV_DELETED_CARDS=$MODE"
MIN_EPOCH=${MIN_EPOCH:-40}    # 2k tier: ~146.65M reviews at ~88k reviews/s = ~28 min training + setup (estimate)
MIN_GPUCHECK=15; MIN_SCORE732=25; MIN_SCORE_RETRY=20
TL732=1800
NAME="the instant-only small model (SM57) stage 2"
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START   # scorer cap 0.8; trainer cap 0.8 (options)
log() { echo "[sm57s2 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
stopped() { [ -e "$STOPF" ] && { log "STOP: STOP2 file present"; return 0; }; return 1; }
nap() { [ "$DRYRUN" = 1 ] && return 0; local s=$1; [ "$s" -gt 600 ] && s=600; sleep "$s"; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'; }
free_mib() { awk '/MemFree/ {print int($2 / 1024)}' /proc/meminfo 2>/dev/null || echo 99999; }
block() { $ST block; }
jline() { "$PY" -X utf8 -c "import json, sys; print(json.dumps(dict(zip(sys.argv[1::2], [int(v) if v.lstrip('-').isdigit() else v for v in sys.argv[2::2]]))))" "$@"; }

wait_free() {  # wait_free MINUTES WHAT: returns once MINUTES fit before the next window (sets FREE = minutes to it)
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

cpu_step() {  # cpu_step MINUTES WHAT: out of the breaks; hold while < 3 GB RAM is free (at most 30 min)
  wait_free "$1" "$2" || return 1
  local i=0
  while [ "$DRYRUN" != 1 ] && [ "$(free_mib)" -lt 3072 ] && [ $i -lt 6 ]; do
    [ $i = 0 ] && log "$2: $(free_mib) MiB RAM free (< 3 GB), holding"; i=$((i + 1)); sleep 300
  done
  return 0
}

q() {  # q MINUTES KIND EPOCH PURPOSE -- CMD...   one ticket that must end before the next window; timed in $TICKETS
  local minutes=$1 kind=$2 ep=$3 purpose=$4 rc t0; shift 5
  while true; do
    wait_free "$minutes" "$purpose" || return 1
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $NAME: $purpose (expected $minutes min)"; echo "    cmd: $*" >> "$LOG"; return 0; fi
    local maxwait=$(( FREE - minutes - 3 )); [ "$maxwait" -lt 1 ] && maxwait=1
    log "enqueue: $purpose (expected $minutes min, max wait $maxwait min, $FREE min to the next break)"
    t0=$(date +%s)
    "$PY" "$QUEUE" run --owner sm57 --purpose "$NAME: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm57/launch-sm57-stage2.sh" -- "$@"
    rc=$?; log "exit $rc: $purpose"
    [ "$rc" = 75 ] && { log "queue wait would reach the next break; re-planning"; sleep 60; continue; }
    jline kind "$kind" epoch "$ep" purpose "$purpose" exit "$rc" minutes "$(( ($(date +%s) - t0 + 30) / 60 ))" >> "$TICKETS"
    return $rc
  done
}

label_users() { F="$1" LB="$2" "$PY" -X utf8 -c "
import json, os
from pathlib import Path
p = Path(os.environ['F'])
print(len({r['user_id'] for r in map(json.loads, open(p, encoding='utf-8')) if r.get('label') == os.environ['LB'] and 'benchmark' in r}) if p.exists() else 0)"; }
done_in() { [ -f "$1/expertium/$TRACK/master.pth" ] && R="$1/expertium/$TRACK/result.json" "$PY" -X utf8 -c "
import json, os, sys
r = json.load(open(os.environ['R']))
sys.exit(0 if r.get('status') == 'complete' and not r.get('smoke_only') else 1)" 2>/dev/null; }
folder() { printf '%s/sm57s2-%s-e%02d' "$SM" "$1" "$2"; }

segment_line() {  # segment_line FOLDER STAGE START_EPOCH END_EPOCH EXIT BASE_MIB MAX_MIB STOP_SENT
  "$PY" -X utf8 -c "
import json, sys
f, stage, a, b, rc, base, mx, stop = sys.argv[1:9]
print(json.dumps({'folder': f, 'stage': stage, 'start_epoch_s': int(a), 'end_epoch_s': int(b), 'minutes': round((int(b) - int(a)) / 60, 1),
                  'exit': int(rc), 'nvidia_smi_baseline_mib': int(base or 0), 'nvidia_smi_max_used_mib': int(mx or 0), 'stop_requested': stop == '1'}))" \
    "$@" >> "$SEGS"
}

run_folder() {  # run_folder DIR WHAT: expertium_decay.py run in segments; resumable from snapshots
  local D=$1 what=$2 rc qpid stop_sent b need t0 base mx u
  while ! done_in "$D"; do
    stopped && return 1
    need=15; [ ! -d "$D/expertium" ] && [ ! -d "$D/attempts" ] && need=$MIN_EPOCH
    wait_free "$need" "$what" || return 1
    local seg=$(( FREE - 5 )); [ "$seg" -gt "$MIN_EPOCH" ] && seg=$MIN_EPOCH
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $NAME: $what (expected $seg min)"
      echo "    cmd: $LF $PY -X utf8 $DEC_PY run --out $D" >> "$LOG"; return 0; fi
    rm -f "$D/control/STOP"
    log "enqueue: $what (expected $seg min, $FREE min to the next break)"
    t0=$(date +%s); base=$(gpu_used); mx=0
    "$PY" "$QUEUE" run --owner sm57 --purpose "$NAME: $what (segment to the next break or the end)" --claim gpu \
      --expected-minutes "$seg" --max-wait-minutes $(( FREE - 20 > 1 ? FREE - 20 : 1 )) --launched-by "detached launcher sm57/launch-sm57-stage2.sh" -- \
      $LF "$PY" -X utf8 "$DEC_PY" run --out "$D" >> "$G/train-stage2.out" 2>&1 &
    qpid=$!
    stop_sent=0
    while kill -0 "$qpid" 2>/dev/null; do
      u=$(gpu_used); [ -n "$u" ] && [ "$u" -gt "$mx" ] && mx=$u
      b=$(block)
      if [ -e "$STOPF" ] || [[ "$b" == NOW* ]] || { [[ "$b" == NEXT* ]] && [ "${b#NEXT }" -le 2 ]; }; then
        [ "$stop_sent" = 0 ] && log "requesting a snapshot stop ($b)"
        "$PY" -X utf8 "$DEC_PY" stop --out "$D" > /dev/null 2>&1; stop_sent=1
      fi
      sleep 30
    done
    wait "$qpid"; rc=$?; log "exit $rc: $what segment (nvidia-smi max $mx MiB, baseline $base MiB)"
    segment_line "$D" "$what" "$t0" "$(date +%s)" "$rc" "$base" "$mx" "$stop_sent"
    [ "$rc" = 75 ] && { sleep 60; continue; }
    done_in "$D" && break
    [ "$stop_sent" = 0 ] && { log "STOP: $what segment ended without completing and without a stop request (exit $rc); see $G/train-stage2.out"; return 1; }
  done
  [ "$DRYRUN" = 1 ] || log "$what complete: $("$PY" -X utf8 -c "import json; r=json.load(open(r'$D/expertium/$TRACK/result.json')); print(f\"{r.get('review_presentations', 0):,} reviews, {r.get('training_seconds', 0)/60:.1f} min training, {r.get('reviews_per_second', 0):,.0f} reviews/s, peak allocated {r.get('peak_allocated_mib', 0):,.0f} MiB, reserved {r.get('peak_reserved_mib', 0):,.0f} MiB, curve_loss_weight {r.get('curve_loss_weight')}, init {str(r.get('initial_parameter_sha256'))[:12]}, update offset end {r.get('update_offset_end')}\")")"
  return 0
}

score() {  # score ARM EPOCH: normal inputs, 723 users of the 732 set at cap 0.8 (a pass without progress -> one pass at 0.92)
  local arm=$1 n=$2 pass have prev=-1 cap=0.8 minutes tag lb D
  D=$(folder "$arm" "$n"); tag=$(basename "$D"); lb="$tag/w8+s8"
  for pass in 1 2 3 4; do
    have=$(label_users "$OUTROWS" "$lb")
    [ "$have" -ge 723 ] && break
    if [ "$have" = "$prev" ]; then
      if [ "$cap" = 0.8 ]; then cap=0.92; log "no progress on $lb ($have of 723 users): next pass at cap 0.92 (memory only, same numerics)"
      else log "STOP: no progress on $lb ($have of 723 users)"; return 1; fi
    fi
    prev=$have
    minutes=$MIN_SCORE732; [ "$pass" -gt 1 ] && minutes=$MIN_SCORE_RETRY
    q $minutes score "$n" "scoring $arm epoch $n on normal inputs, the 732 set minus 9 never-fit (723 users) (pass $pass, $have done, cap $cap)" -- \
      env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START HELDOUT_GPU_MEMORY_FRACTION=$cap "$PY" -X utf8 "$SC" --pool 732 --users-file "$USERS732" \
      --model "$tag=expertium:$ARCH:$D/expertium/$TRACK/master.pth" --only "$lb" --inputs normal --time-limit $TL732 --split-existing \
      --deleted-cards $MODE --out "$OUTROWS" >> "$G/score-stage2.out" 2>&1 || { log "STOP: scoring ticket failed ($lb)"; return 1; }
    [ "$DRYRUN" = 1 ] && break
  done
  [ "$DRYRUN" = 1 ] || [ "$(label_users "$OUTROWS" "$lb")" -ge 723 ] || { log "STOP: $lb scored on $(label_users "$OUTROWS" "$lb") of 723 users"; return 1; }
  [ "$DRYRUN" = 1 ] || log "$lb: $(label_users "$OUTROWS" "$lb") users scored"
  return 0
}

rule() {  # rule WHEN (a partial read-out exits 2 until the final epoch is scored)
  [ "$DRYRUN" = 1 ] && { log "DRYRUN: then $ST rule ($1)"; return 0; }
  cpu_step 3 "rule" || return 1
  local out r; out=$($ST rule 2>&1); r=$?
  log "rule ($1) exit $r: $(echo "$out" | tail -1)"
  [ "$r" = 0 ] || [ "$1" = partial ]
}

final_epoch() { [ -f "$PLAN" ] && "$PY" -X utf8 -c "import json; print(json.load(open(r'$PLAN'))['final_epoch'])" || echo 5; }
scored() { local n=$1 f=$2; [ "$n" = 1 ] || [ "$n" = 3 ] || [ "$n" = "$f" ]; }

preconditions() {
  for f in "$S/check-cpu-sm57-stage2.json" "$USERS732" "$(folder inst 1)/decay.json" "$(folder joint 1)/decay.json"; do
    [ -f "$f" ] || { log "STOP: missing $f"; return 1; }
  done
  grep -q '"ok": true' "$S/check-cpu-sm57-stage2.json" && ! grep -q '"ok": false' "$S/check-cpu-sm57-stage2.json" \
    || { log "STOP: the CPU proofs are not ALL OK ($S/check-cpu-sm57-stage2.json)"; return 1; }
  if [ "$DRYRUN" != 1 ] && [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    log "STOP: uncommitted source changes in scripts/ expertium/ rwkv/ (SourceWatch refuses them)"; return 1; fi
  return 0
}

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1) (DRYRUN=$DRYRUN; the 2k tier from scratch, instant-only vs joint, epochs 1..$(final_epoch))"
rc=0
cpu_step 5 "preparation" || rc=1
[ $rc -eq 0 ] && { preconditions || rc=1; }
if [ $rc -eq 0 ] && [ ! -e "$G/gpu-check-stage2-ok" ]; then
  q $MIN_GPUCHECK check 0 "GPU check (trainer smokes 2 users x 512 reviews of epoch 1: joint by the base tree == this tree bitwise, instant-only trains, same fresh init)" -- \
    "$PY" -X utf8 "$S/sm57s2_gpu_check.py" --out "$G/check-gpu-sm57-stage2.json" >> "$G/check-stage2.out" 2>&1 \
    || { log "STOP: GPU check failed (see $G/check-stage2.out, $G/check-gpu-sm57-stage2.json)"; rc=1; }
  [ $rc -eq 0 ] && [ "$DRYRUN" != 1 ] && { touch "$G/gpu-check-stage2-ok"; log "GPU check: $(grep 'ALL OK' "$G/check-stage2.out" | tail -1 | cut -c1-400)"; }
fi
EP="(w64 from scratch, the 2k tier, the leak-free small model (SM72s) recipe: 2048 x 16 lanes carry, leak-free replay KD, AdamW 7e-4 constant)"
for n in 1 2 3 4 5; do
  [ $rc -eq 0 ] || break
  F=$(final_epoch); [ "$n" -gt "$F" ] && break
  for arm in inst joint; do
    D=$(folder $arm $n)
    if [ ! -f "$D/decay.json" ]; then
      if [ "$DRYRUN" = 1 ]; then log "DRYRUN: $arm epoch $n is prepared after epoch $((n - 1))"; continue; fi
      cpu_step 3 "$arm epoch $n prepare" || { rc=1; break; }
      $ST prepare --arm $arm --epoch $n >> "$G/prepare-stage2.out" 2>&1 || { log "STOP: $arm epoch $n prepare failed"; rm -rf "$D"; rc=1; break; }
      log "prepared: $(basename "$D")"
    fi
  done
  [ $rc -eq 0 ] || break
  if [ "$DRYRUN" != 1 ] || [ "$n" = 1 ]; then
    $ST pair --epoch $n >> "$LOG" 2>&1 || { log "STOP: epoch $n arms differ beyond the flag"; rc=1; break; }
  fi
  run_folder "$(folder inst $n)" "instant-only epoch $n, curve loss weight 0 $EP" || { rc=1; break; }
  run_folder "$(folder joint $n)" "joint epoch $n $EP" || { rc=1; break; }
  if scored $n "$F"; then
    score inst $n || { rc=1; break; }
    score joint $n || { rc=1; break; }
    if [ "$n" = 1 ] && [ "$DRYRUN" != 1 ]; then
      log "plan: $($ST plan 2>&1 | tail -1)"
      F=$(final_epoch)
    fi
    [ "$n" -lt "$F" ] && { rule partial || { rc=1; break; }; }
  fi
done
[ $rc -eq 0 ] && { rule final || rc=1; }
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; else log "done"; fi
else
  log "STOP: ended with exit $rc"
fi
exit $rc
