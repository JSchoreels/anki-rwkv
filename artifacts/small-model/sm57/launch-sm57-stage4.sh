#!/usr/bin/env bash
# The instant-only small model (SM57) stage 4 (artifacts/small-model/sm57/DESIGN.md, "Stage 4 plan"): a better dedicated
#   instant model. Stage 1's instant-only epoch 26 (reused) + instant-only epochs 27..30 at the release's epoch recipe
#   (8k final tier, the run's order rule, AdamW 7e-4 constant, "curve_loss_weight": 0.0) + the release's one-epoch 32k
#   Muon cool-down from epoch 30, instant-only. Rule (sm57_stage4.py rule): candidate - stage 1's instant-only arm,
#   instant head, 722 users: go if the 95% CI is below zero and the gain >= 0.05 points; on go the 1,500 validation users
#   vs the release (sm57_stage4.py pool1500). Report-only extras: constant-LR epochs 26 / 28 / 30 scored.
# CPU proofs (sm57s4_check.py -> check-cpu-sm57-stage4.json): ALL OK required. First ticket: the GPU check (smokes).
# Breaks: quiet.json (scope all/gpu) plus the default nightly 20:00-01:00 (Sundays from 16:00) unless quiet.json has a
#   window overlapping that night (sm57_stage4.py block); re-read at every check, CPU steps too. Epochs and the
#   cool-down snapshot-stop <= 2 min before a break and resume after it (scripts/expertium_decay.py run / stop).
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm57/artifacts/small-model/sm57/launch-sm57-stage4.sh
#   launch   WMI hidden (artifacts/small-model/sm57/HANDOFF-stage4.md); log artifacts/small-model/sm57/gpu/launch-sm57-stage4.log
#   resume   the launch command again (finished steps are skipped; an interrupted epoch / cool-down resumes from its snapshot)
#   stop     touch C:/Users/jonat/workspace/wt/sm57/artifacts/small-model/sm57/gpu/STOP4 (between tickets; a running
#            epoch / cool-down gets a snapshot stop at the next user boundary)
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
LOG=$G/launch-sm57-stage4.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm57-stage4-dryrun.log
ST="$PY -X utf8 $S/sm57_stage4.py"
WARM="$PY -X utf8 $WT/scripts/sm77_warm.py"
DEC_PY=$WT/scripts/expertium_decay.py
SC=$WT/scripts/small_model_heldout.py
PAIR=$S/sm57_pair.py
TRACK=E89-F65e-F77a
ARCH=w64
K=25
FIRST=27; LAST=30
FLAG='{"curve_loss_weight": 0.0}'
S72=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s                                                  # read only
REL=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/scale-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon  # the release, read only
USERS732=$S72/eval-users-732-fit.json; ULONG=$S72/users-732-long.json
USERS1500=$S/stage4/users-sm93-score.json
E26=$SM/sm57-inst-e26
DEC1=$SM/sm57-inst-e26-dec-1ep32k-muon
DEC=$SM/sm57s4-inst-e$LAST-dec-1ep32k-muon
CLBL=sm57s4-inst-e$LAST-dec1ep32k
OUTROWS=$G/sm57s4-rows.jsonl
ROWS1500=$G/sm57s4-rows-1500.jsonl
SEGS=$G/segments-stage4.jsonl
TICKETS=$G/tickets-stage4.jsonl
STOPF=$G/STOP4
MODE=neighbour
LF="env RWKV_DELETED_CARDS=$MODE"
MIN_EPOCH=${MIN_EPOCH:-125}   # stage 1: 111.2 min training at ~87.7k reviews/s, ~113 min tickets
MIN_DECAY=${MIN_DECAY:-560}   # stage 1: 477 min wall at ~20.3k reviews/s + setup
MIN_START=30                  # a fresh epoch / cool-down starts when 30 min fit (snapshot stop before the break, resume after)
MIN_GPUCHECK=15; MIN_SMOKE=10; MIN_SCORE732=32; MIN_SCORE_RETRY=20; MIN_LONG=25; MIN_POOL=45
TL732=1800; TLLONG=1380; TLPOOL=2400
NAME="the instant-only small model (SM57) stage 4"
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START   # scorer cap 0.8; trainer cap 0.8 (options)
log() { echo "[sm57s4 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
stopped() { [ -e "$STOPF" ] && { log "STOP: STOP4 file present"; return 0; }; return 1; }
nap() { [ "$DRYRUN" = 1 ] && return 0; local s=$1; [ "$s" -gt 600 ] && s=600; sleep "$s"; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'; }
free_mib() { awk '/MemFree/ {print int($2 / 1024)}' /proc/meminfo 2>/dev/null || echo 99999; }
block() { $ST block; }
jline() { "$PY" -X utf8 -c "import json, sys; print(json.dumps(dict(zip(sys.argv[1::2], [int(v) if v.lstrip('-').isdigit() else v for v in sys.argv[2::2]]))))" "$@"; }

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

ram_gate() {  # ram_gate WHAT [strict]: hold while < 3 GB RAM is free, at most 30 min (strict: then STOP instead of going on)
  local i=0
  while [ "$DRYRUN" != 1 ] && [ "$(free_mib)" -lt 3072 ]; do
    if [ $i -ge 6 ]; then
      [ "${2:-}" = strict ] && { log "STOP: $1: $(free_mib) MiB RAM free (< 3 GB) for 30 min"; return 1; }
      log "$1: still $(free_mib) MiB RAM free after 30 min; going on"; return 0
    fi
    [ $i = 0 ] && log "$1: $(free_mib) MiB RAM free (< 3 GB), holding"; i=$((i + 1)); sleep 300
  done
  return 0
}
cpu_step() { wait_free "$1" "$2" || return 1; ram_gate "$2"; }

q() {  # q MINUTES KIND PURPOSE -- CMD...   one ticket that must end before the next break; timed in $TICKETS
  local minutes=$1 kind=$2 purpose=$3 rc t0; shift 4
  while true; do
    wait_free "$minutes" "$purpose" || return 1
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $NAME: $purpose (expected $minutes min)"; echo "    cmd: $*" >> "$LOG"; return 0; fi
    local maxwait=$(( FREE - minutes - 3 )); [ "$maxwait" -lt 1 ] && maxwait=1
    log "enqueue: $purpose (expected $minutes min, max wait $maxwait min, $FREE min to the next break)"
    t0=$(date +%s)
    "$PY" "$QUEUE" run --owner sm57 --purpose "$NAME: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm57/launch-sm57-stage4.sh" -- "$@"
    rc=$?; log "exit $rc: $purpose"
    [ "$rc" = 75 ] && { log "queue wait would reach the next break; re-planning"; sleep 60; continue; }
    jline kind "$kind" purpose "$purpose" exit "$rc" minutes "$(( ($(date +%s) - t0 + 30) / 60 ))" >> "$TICKETS"
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
folder() { if [ "$1" = 26 ]; then echo "$E26"; else printf '%s/sm57s4-inst-e%02d' "$SM" "$1"; fi; }
verdict() { "$PY" -X utf8 -c "import json; print(json.load(open(r'$S/sm57-stage4-decision.json')).get('verdict', 'pending'))" 2>/dev/null || echo pending; }

segment_line() {  # segment_line FOLDER STAGE START_EPOCH END_EPOCH EXIT BASE_MIB MAX_MIB STOP_SENT
  "$PY" -X utf8 -c "
import json, sys
f, stage, a, b, rc, base, mx, stop = sys.argv[1:9]
print(json.dumps({'folder': f, 'stage': stage, 'start_epoch_s': int(a), 'end_epoch_s': int(b), 'minutes': round((int(b) - int(a)) / 60, 1),
                  'exit': int(rc), 'nvidia_smi_baseline_mib': int(base or 0), 'nvidia_smi_max_used_mib': int(mx or 0), 'stop_requested': stop == '1'}))" \
    "$@" >> "$SEGS"
}

run_folder() {  # run_folder DIR WHAT SEG_MAX: expertium_decay.py run in segments; resumable from snapshots
  local D=$1 what=$2 segmax=$3 rc qpid stop_sent b need t0 base mx u
  while ! done_in "$D"; do
    stopped && return 1
    need=15; [ ! -d "$D/expertium" ] && [ ! -d "$D/attempts" ] && need=$MIN_START
    wait_free "$need" "$what" || return 1
    ram_gate "$what" || return 1
    wait_free "$need" "$what" || return 1
    local seg=$(( FREE - 5 )); [ "$seg" -gt "$segmax" ] && seg=$segmax
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $NAME: $what (expected $seg min of ~$segmax)"
      echo "    cmd: $LF $PY -X utf8 $DEC_PY run --out $D" >> "$LOG"; return 0; fi
    rm -f "$D/control/STOP"
    log "enqueue: $what (expected $seg min of ~$segmax, $FREE min to the next break, $(free_mib) MiB RAM free)"
    t0=$(date +%s); base=$(gpu_used); mx=0
    "$PY" "$QUEUE" run --owner sm57 --purpose "$NAME: $what (segment to the next break or the end)" --claim gpu \
      --expected-minutes "$seg" --max-wait-minutes $(( FREE - 20 > 1 ? FREE - 20 : 1 )) --launched-by "detached launcher sm57/launch-sm57-stage4.sh" -- \
      $LF "$PY" -X utf8 "$DEC_PY" run --out "$D" >> "$G/train-stage4.out" 2>&1 &
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
    [ "$stop_sent" = 0 ] && { log "STOP: $what segment ended without completing and without a stop request (exit $rc); see $G/train-stage4.out"; return 1; }
  done
  [ "$DRYRUN" = 1 ] || log "$what complete: $("$PY" -X utf8 -c "import json; r=json.load(open(r'$D/expertium/$TRACK/result.json')); print(f\"{r.get('review_presentations', 0):,} reviews, {r.get('training_seconds', 0)/60:.1f} min training, {r.get('reviews_per_second', 0):,.0f} reviews/s, peak allocated {r.get('peak_allocated_mib', 0):,.0f} MiB, reserved {r.get('peak_reserved_mib', 0):,.0f} MiB, curve_loss_weight {r.get('curve_loss_weight')}, init {str(r.get('initial_parameter_sha256'))[:12]}, trained {str(r.get('trained_parameter_sha256'))[:12]}, update offset end {r.get('update_offset_end')}\")")"
  return 0
}

score() {  # score TAG SPEC WHAT ROWS POOL USERS_FILE EXPECT STRICT MIN1 TL [CAP0]: normal inputs, passes at CAP0 (default 0.8; a pass without
           # progress -> cap 0.92; no progress at 0.92: strict -> STOP, else go on with what fits)
  local tag=$1 spec=$2 what=$3 rows=$4 pool=$5 uf=$6 expect=$7 strict=$8 min1=$9 tl=${10} cap=${11:-0.8} pass have prev=-1 minutes lb
  lb="$tag/w8+s8"
  for pass in 1 2 3 4 5; do
    have=$(label_users "$rows" "$lb")
    [ "$have" -ge "$expect" ] && break
    if [ "$have" = "$prev" ]; then
      if [ "$cap" = 0.8 ]; then cap=0.92; log "no progress on $lb ($have of $expect users): next pass at cap 0.92 (memory only, same numerics)"
      elif [ "$strict" = 1 ]; then log "STOP: no progress on $lb ($have of $expect users)"; return 1
      else log "no progress at cap 0.92 on $lb ($have of $expect users): the rest do not fit; going on"; break; fi
    fi
    prev=$have
    minutes=$min1; [ "$pass" -gt 1 ] && minutes=$MIN_SCORE_RETRY
    q $minutes score "scoring $what on normal inputs, ${pool#* } (pass $pass, $have done, cap $cap)" -- \
      env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START HELDOUT_GPU_MEMORY_FRACTION=$cap "$PY" -X utf8 "$SC" --pool "${pool%% *}" --users-file "$uf" \
      --model "$tag=$spec" --only "$lb" --inputs normal --time-limit $tl --split-existing \
      --deleted-cards $MODE --out "$rows" >> "$G/score-stage4.out" 2>&1 || { log "STOP: scoring ticket failed ($lb)"; return 1; }
    [ "$DRYRUN" = 1 ] && break
  done
  [ "$DRYRUN" = 1 ] || [ "$strict" != 1 ] || [ "$(label_users "$rows" "$lb")" -ge "$expect" ] || { log "STOP: $lb scored on $(label_users "$rows" "$lb") of $expect users"; return 1; }
  [ "$DRYRUN" = 1 ] || log "$lb: $(label_users "$rows" "$lb") users scored in $(basename "$rows")"
  return 0
}
score732() {  # score732 TAG DIR WHAT (723 users of the 732 set, strict)
  score "$1" "expertium:$ARCH:$2/expertium/$TRACK/master.pth" "$3" "$OUTROWS" "732 the 732 set minus 9 never-fit (723 users)" "$USERS732" 723 1 $MIN_SCORE732 $TL732
}

rule() {  # rule WHEN (exit 2 while the candidate's rule rows are missing)
  [ "$DRYRUN" = 1 ] && { log "DRYRUN: then $ST rule ($1)"; return 0; }
  cpu_step 3 "rule" || return 1
  local out r; out=$($ST rule 2>&1); r=$?
  log "rule ($1) exit $r: $(echo "$out" | tail -1 | cut -c1-900)"
  [ "$r" = 0 ]
}

preconditions() {
  for f in "$S/check-cpu-sm57-stage4.json" "$USERS732" "$ULONG" "$USERS1500" "$REL/decay.json" "$E26/decay.json" "$DEC1/decay.json" \
           "$G/sm57-rows.jsonl" "$S/stage4/sm93-rows.jsonl" "$S/extra-options-instant.json" "$G/check-gpu-sm57.json" "$(folder $FIRST)/decay.json"; do
    [ -f "$f" ] || { log "STOP: missing $f"; return 1; }
  done
  done_in "$E26" || { log "STOP: stage 1's instant-only epoch 26 is not complete ($E26)"; return 1; }
  done_in "$REL" || { log "STOP: the release's cool-down $REL is not complete"; return 1; }
  grep -q '"ok": true' "$S/check-cpu-sm57-stage4.json" && ! grep -q '"ok": false' "$S/check-cpu-sm57-stage4.json" \
    || { log "STOP: the CPU proofs are not ALL OK ($S/check-cpu-sm57-stage4.json)"; return 1; }
  if [ "$DRYRUN" != 1 ] && [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    log "STOP: uncommitted source changes in scripts/ expertium/ rwkv/ (SourceWatch refuses them)"; return 1; fi
  return 0
}

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1) (DRYRUN=$DRYRUN; instant-only epochs $FIRST..$LAST from stage 1's epoch 26 + the release's cool-down)"
rc=0
cpu_step 5 "preparation" || rc=1
[ $rc -eq 0 ] && { preconditions || rc=1; }
if [ $rc -eq 0 ] && [ ! -e "$G/gpu-check-stage4-ok" ]; then
  ram_gate "GPU check" strict || rc=1
  [ $rc -eq 0 ] && { q $MIN_GPUCHECK check "GPU check (trainer smokes 2 users x 512 reviews: stage 1's epoch 26 == stage 1's GPU check bitwise, epoch 27 trains with curve loss weight 0 from epoch 26)" -- \
    "$PY" -X utf8 "$S/sm57s4_gpu_check.py" --out "$G/check-gpu-sm57-stage4.json" >> "$G/check-stage4.out" 2>&1 \
    || { log "STOP: GPU check failed (see $G/check-stage4.out, $G/check-gpu-sm57-stage4.json)"; rc=1; }; }
  [ $rc -eq 0 ] && [ "$DRYRUN" != 1 ] && { touch "$G/gpu-check-stage4-ok"; log "GPU check: $(grep 'ALL OK' "$G/check-stage4.out" | tail -1 | cut -c1-400)"; }
fi
EP="(w64, the release's epoch recipe: 8k final tier, 2048 x 16 lanes carry, leak-free replay KD, AdamW 7e-4 constant, curve loss weight 0)"
for n in $(seq $FIRST $LAST); do
  [ $rc -eq 0 ] || break
  D=$(folder $n)
  if [ ! -f "$D/decay.json" ]; then
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN: instant-only epoch $n is prepared after epoch $((n - 1))"; continue; fi
    cpu_step 3 "epoch $n prepare" || { rc=1; break; }
    $ST prepare --epoch $n >> "$G/prepare-stage4.out" 2>&1 || { log "STOP: epoch $n prepare failed"; rm -rf "$D"; rc=1; break; }
    log "prepared: $(basename "$D")"
  fi
  $ST pair --epoch $n >> "$LOG" 2>&1 || { log "STOP: epoch $n is not the recipe"; rc=1; break; }
  run_folder "$D" "instant-only epoch $n from epoch $((n - 1)) $EP" $MIN_EPOCH || { rc=1; break; }
  if [ "$n" = 28 ]; then   # report-only constant-LR read-outs (a failure here does not stop the run)
    for m in 26 28; do
      score732 "sm57s4-inst-e$m" "$(folder $m)" "the constant-LR instant-only epoch $m (report only)" || { log "constant-LR epoch $m scoring failed (report only); going on"; stopped && rc=1; }
    done
  fi
done
if [ $rc -eq 0 ] && [ ! -f "$DEC/decay.json" ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: the cool-down is prepared after epoch $LAST"
  else
    cpu_step 20 "cool-down prepare" || rc=1
    if [ $rc -eq 0 ]; then
      X=$(cat "$S/extra-options-instant.json")
      { $LF $WARM cooldown --epoch $K --start "$(folder $LAST)" --out "${DEC#$WT/}" --arch $ARCH --fraction 1.0 --extra-options "$X" >> "$G/prepare-stage4.out" 2>&1 \
        && $LF "$PY" -X utf8 "$WT/scripts/check_replay_label_windows.py" --decay "$DEC" --trim-unlabelled-tail >/dev/null \
        && $LF "$PY" -X utf8 "$WT/scripts/check_replay_label_windows.py" --decay "$DEC" --out "$G/check-replay-label-windows-$(basename "$DEC").json" >/dev/null; } \
        || { log "STOP: cool-down prepare / label windows failed"; rm -rf "$DEC"; rc=1; }
      if [ $rc -eq 0 ]; then
        CUDA_VISIBLE_DEVICES= $LF "$PY" -X utf8 "$WT/scripts/check_muon_fresh_decay.py" --decay "$DEC" --out "$G/check-muon-fresh-$(basename "$DEC").json" \
          | grep -c PASS | sed "s/^/[sm57s4-muon] $(basename "$DEC") PASS lines: /" >> "$LOG"
        [ "${PIPESTATUS[0]}" = 0 ] || { log "STOP: Muon fresh-state CPU check failed"; rm -rf "$DEC"; rc=1; }
      fi
      [ $rc -eq 0 ] && log "cool-down prepared: $(basename "$DEC")"
    fi
  fi
fi
if [ $rc -eq 0 ] && [ "$DRYRUN" != 1 ]; then
  "$PY" -X utf8 "$PAIR" decay --arm "$DEC" --ref "$REL" --flag "$FLAG" >> "$LOG" 2>&1 || { log "STOP: the cool-down is not the release's + the flag"; rc=1; }
  [ $rc -eq 0 ] && { "$PY" -X utf8 "$PAIR" decay --arm "$DEC" --ref "$DEC1" >> "$LOG" 2>&1 || { log "STOP: the cool-down differs from stage 1's beyond the start"; rc=1; }; }
fi
if [ $rc -eq 0 ] && [ ! -e "$G/smoke-stage4-decay-ok" ]; then
  q $MIN_SMOKE check "trainer smoke of the cool-down from epoch $LAST (fresh Muon, curve loss weight 0, 2 users x 512 reviews)" -- \
    $LF "$PY" -X utf8 "$DEC_PY" smoke --out "$DEC" >> "$G/check-stage4.out" 2>&1 || { log "STOP: cool-down smoke failed (see $G/check-stage4.out)"; rc=1; }
  [ $rc -eq 0 ] && [ "$DRYRUN" != 1 ] && { touch "$G/smoke-stage4-decay-ok"; log "smoke: $(grep '"smoke"' "$G/check-stage4.out" | tail -1 | cut -c1-200)"; }
fi
[ $rc -eq 0 ] && { run_folder "$DEC" "the release's one-epoch 32k Muon cool-down from instant-only epoch $LAST (the release's users, order, seeds, KD targets), curve loss weight 0" $MIN_DECAY || rc=1; }
[ $rc -eq 0 ] && { score732 "$CLBL" "$DEC" "the stage-4 candidate (cooled down from epoch $LAST)" || rc=1; }
[ $rc -eq 0 ] && { rule final || rc=1; }
if [ $rc -eq 0 ]; then   # the 3 long users of the 732 set (the 732+long pool and the 1,500-user pool include them)
  score "$CLBL" "expertium:$ARCH:$DEC/expertium/$TRACK/master.pth" "the stage-4 candidate" "$OUTROWS" "732 the 3 long users of the 732 set" "$ULONG" 726 0 $MIN_LONG $TLLONG 0.92 \
    || log "long users scoring failed; going on (the rule pool does not need them)"
  rule "final + long users" || true
fi
if [ $rc -eq 0 ] && { [ "$DRYRUN" = 1 ] || [ "$(verdict)" = go ]; }; then
  [ "$DRYRUN" = 1 ] && log "DRYRUN: on go only"
  score "$CLBL" "expertium:$ARCH:$DEC/expertium/$TRACK/master.pth" "the stage-4 candidate" "$ROWS1500" "1500 the 1,500 validation users outside the 732 set + 6 check users" "$USERS1500" 771 0 $MIN_POOL $TLPOOL \
    || { log "STOP: the 1,500-user scoring failed"; rc=1; }
  if [ $rc -eq 0 ] && [ "$DRYRUN" != 1 ]; then
    cpu_step 3 "pool rule" || rc=1
    out=$($ST pool1500 2>&1); log "pool1500 exit $?: $(echo "$out" | tail -1 | cut -c1-600)"
  fi
fi
if [ $rc -eq 0 ]; then   # report-only: the constant-LR epoch 30
  score732 "sm57s4-inst-e$LAST" "$(folder $LAST)" "the constant-LR instant-only epoch $LAST (report only)" || log "constant-LR epoch $LAST scoring failed (report only)"
  [ "$DRYRUN" = 1 ] || rule "with the constant-LR epoch $LAST" || true
fi
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; else log "done (verdict $(verdict))"; fi
else
  log "STOP: ended with exit $rc"
fi
exit $rc
