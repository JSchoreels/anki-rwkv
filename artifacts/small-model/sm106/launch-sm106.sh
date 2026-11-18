#!/usr/bin/env bash
# The Tiny two-head model (SM106) (artifacts/small-model/sm106/design.md, HANDOFF.md), designed 2026-11-11:
#   the show-time retrain (SM92)'s constant-LR epoch 26 (read only), card stream cut to 1 layer
#   (expertium.variants "w64-c1"; cut checkpoint artifacts/small-model/sm106-cut-e26 from scripts/sm73s2.py cut)
#   -> reference scoring: SM92 epoch 26 (constant LR, w64) on show-time inputs (the early check's reference)
#   -> epoch 27 from the cut, SM92 epoch 26's recipe (sm106_epoch.py: the run's order rule with N, show-time inputs, BOTH
#      heads (SM92's loss weights), leak-free replay KD, AdamW 7e-4 constant), the only change the architecture; trainer smoke first
#   -> EARLY CHECK: epoch 27 scored on show-time inputs; sm106_rule.py early: stop if epoch 27 - SM92 epoch 26 > +0.25% on
#      the curve OR the instant head (point estimates)
#   -> epochs 28, 29 (each prepared after the previous one completes, then pair-checked)
#   -> the release's one-epoch 32k Muon cool-down on show-time inputs from epoch 29 (SM92's cool-down except the start and
#      the architecture; prepared after epoch 29, then sm106_check.py cooldown)
#   -> scoring (show-time 732 + 3 long users, the 1,500 pool show-time) and sm106_rule.py decide:
#      A (pruning bar, the verdict): new - the big model (1 leak-free epoch), show-time inputs: curve 95% CI upper end <= +0.5%
#        AND instant 95% CI upper end <= +0.4%;  B (report only): new - SM92 (both heads), new - SM101 (instant)
# CPU proofs (sm106_check.py -> check-cpu-sm106.json): ALL OK required.
# Breaks: sm106_block.py = quiet.json (scope all/gpu) + the daily 20:00-01:00 (Sunday from 16:00) unless quiet.json
#   overlaps that night; no hard hold. Re-read at every check. Each epoch and the cool-down start when 30 min fit,
#   snapshot-stop <= 2 min before a break and resume after it (scripts/expertium_decay.py run / stop). Scoring tickets
#   must end before the next break.
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm106/artifacts/small-model/sm106/launch-sm106.sh
#   recipe   PREPARE_RECIPE=1 bash .../launch-sm106.sh   (CPU: the cool-down recipe proof from a simulated complete
#            w64-c1 start holding the cut, never trained)
#   launch   WMI hidden (HANDOFF.md); log artifacts/small-model/sm106/gpu/launch-sm106.log
#   resume   the launch command again (finished steps are skipped; an interrupted epoch / cool-down resumes from its snapshot)
#   stop     touch C:/Users/jonat/workspace/wt/sm106/artifacts/small-model/sm106/gpu/STOP (between tickets; a running
#            epoch / cool-down gets a snapshot stop at the next user boundary)
# Do not edit scripts/ expertium/ rwkv/ of the sm106 worktree while it runs (SourceWatch stops the training).
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/sm106
SM=$WT/artifacts/small-model
S=$SM/sm106
G=$S/gpu
mkdir -p "$G"
LOG=$G/launch-sm106.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm106-dryrun.log
[ "${PREPARE_RECIPE:-0}" = 1 ] && LOG=$G/launch-sm106-recipe.log
DEC_PY=$WT/scripts/expertium_decay.py
SC=$WT/scripts/small_model_heldout.py
RULE=$S/sm106_rule.py
EPY=$S/sm106_epoch.py
CHECK=$S/sm106_check.py
DPAIR=$S/sm106_pair.py
WARM="$PY -X utf8 $WT/scripts/sm77_warm.py"
TRACK=E89-F65e-F77a
ARCH=w64-c1
K=25                          # the release's cool-down recipe (sm77_warm.py cooldown --epoch 25: users / order / seeds / KD targets)
S92=C:/Users/jonat/workspace/wt/sm92/artifacts/small-model                         # read only
S72=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s                  # read only
USERS732=$S72/eval-users-732-fit.json; ULONG=$S72/users-732-long.json
USERS1500=C:/Users/jonat/workspace/wt/sm57/artifacts/small-model/sm57/stage4/users-sm93-score.json
E26=$S92/sm92-e26                                                                  # SM92 epoch 26 (show-time, w64, two heads), the cut's source
S92DEC=$S92/sm92-e26-dec-1ep32k-muon                                               # SM92's cool-down (the recipe reference)
CUT=$SM/sm106-cut-e26
EPOCHS="27 28 29"
efold() { echo "$SM/sm106-c1-e$1-st"; }
E27=$(efold 27); E29=$(efold 29)
SIMSTART=$SM/sm106-recipe-start-sim                                               # simulated complete w64-c1 start (the cut), never trained
RECIPE=$SM/sm106-recipe-sim-dec-1ep32k-muon                                       # cool-down recipe proof (never trained)
DEC=$SM/sm106-c1-e29st-st-dec-1ep32k-muon
CLBL=sm106-c1-e29st-st-dec1ep32k
EREF=sm92-e26-const; EARM=sm106-c1-e27-st-const
OUTROWS=$G/sm106-rows.jsonl
ROWS1500=$G/sm106-rows-1500.jsonl
EROWS=$G/sm106-early-rows.jsonl
SEGS=$G/segments.jsonl
TICKETS=$G/tickets.jsonl
STOPF=$G/STOP
MODE=neighbour
LF="env RWKV_DELETED_CARDS=$MODE"
MIN_EPOCH=${MIN_EPOCH:-125}   # SM101's w64-c1 epochs: 105.9-110.1 min wall at 88.1-91.6k reviews/s; SM92's w64 two-head epoch ran at ~86.9k
MIN_DECAY=${MIN_DECAY:-540}   # SM101's w64-c1 cool-down: about 458 min wall at ~20.8-21.2k reviews/s; SM92's w64 two-head cool-down ran at 20.4k
MIN_START=30                  # an epoch / the cool-down starts when 30 min fit (snapshot stop before the break, resume after)
MIN_SMOKE=10; MIN_SCORE732=32; MIN_SCORE_RETRY=20; MIN_LONG=25; MIN_POOL=45
TL732=1800; TLLONG=1380; TLPOOL=2400
NAME="the Tiny two-head model (SM106)"
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START   # scorer cap 0.8; trainer cap 0.8 (options)
log() { echo "[sm106 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
stopped() { [ -e "$STOPF" ] && { log "STOP: STOP file present"; return 0; }; return 1; }
nap() { [ "$DRYRUN" = 1 ] && return 0; local s=$1; [ "$s" -gt 600 ] && s=600; sleep "$s"; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'; }
free_mib() { awk '/MemFree/ {print int($2 / 1024)}' /proc/meminfo 2>/dev/null || echo 99999; }
block() { "$PY" -X utf8 "$S/sm106_block.py"; }
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
    "$PY" "$QUEUE" run --owner sm106 --purpose "$NAME: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm106/launch-sm106.sh" -- "$@"
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
    "$PY" "$QUEUE" run --owner sm106 --purpose "$NAME: $what (segment to the next break or the end)" --claim gpu \
      --expected-minutes "$seg" --max-wait-minutes $(( FREE - 20 > 1 ? FREE - 20 : 1 )) --launched-by "detached launcher sm106/launch-sm106.sh" -- \
      $LF "$PY" -X utf8 "$DEC_PY" run --out "$D" >> "$G/train.out" 2>&1 &
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
    [ "$stop_sent" = 0 ] && { log "STOP: $what segment ended without completing and without a stop request (exit $rc); see $G/train.out"; return 1; }
  done
  [ "$DRYRUN" = 1 ] || log "$what complete: $("$PY" -X utf8 -c "import json; r=json.load(open(r'$D/expertium/$TRACK/result.json')); print(f\"{r.get('review_presentations', 0):,} reviews, {r.get('training_seconds', 0)/60:.1f} min training, {r.get('reviews_per_second', 0):,.0f} reviews/s, peak allocated {r.get('peak_allocated_mib', 0):,.0f} MiB, reserved {r.get('peak_reserved_mib', 0):,.0f} MiB, curve_loss_weight {r.get('curve_loss_weight')}, arch {r.get('options', {}).get('architecture')}, init {str(r.get('initial_parameter_sha256'))[:12]}, trained {str(r.get('trained_parameter_sha256'))[:12]}, update offset end {r.get('update_offset_end')}\")")"
  return 0
}

score() {  # score INPUTS WHAT ROWS POOL USERS_FILE EXPECT STRICT MIN1 TL CAP0 LABEL SPEC: passes at CAP0 (a pass without
           # progress -> cap 0.92; no progress at 0.92: strict -> STOP, else go on with what fits)
  local v=$1 what=$2 rows=$3 pool=$4 uf=$5 expect=$6 strict=$7 min1=$8 tl=$9 cap=${10} slbl=${11} sspec=${12} pass have prev=-1 minutes lb
  lb="$slbl/w8+s8"; [ "$v" != normal ] && lb="$lb@$v"
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
    q $minutes score "scoring $what on $v inputs, ${pool#* } (pass $pass, $have done, cap $cap)" -- \
      env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START HELDOUT_GPU_MEMORY_FRACTION=$cap "$PY" -X utf8 "$SC" --pool "${pool%% *}" --users-file "$uf" \
      --model "$slbl=$sspec" --only "$slbl/w8+s8" --inputs "$v" --time-limit $tl --split-existing \
      --deleted-cards $MODE --out "$rows" >> "$G/score.out" 2>&1 || { log "STOP: scoring ticket failed ($lb)"; return 1; }
    [ "$DRYRUN" = 1 ] && break
  done
  [ "$DRYRUN" = 1 ] || [ "$strict" != 1 ] || [ "$(label_users "$rows" "$lb")" -ge "$expect" ] || { log "STOP: $lb scored on $(label_users "$rows" "$lb") of $expect users"; return 1; }
  [ "$DRYRUN" = 1 ] || log "$lb: $(label_users "$rows" "$lb") users scored in $(basename "$rows")"
  return 0
}
P723="732 the 732 set minus 9 never-fit (723 users)"
score_early() {  # score_early WHAT LABEL SPEC: 723 users of the 732 set on show-time inputs (strict), constant-LR checkpoint
  score e2s-query "$1" "$EROWS" "$P723" "$USERS732" 723 1 $MIN_SCORE732 $TL732 0.8 "$2" "$3"
}
score732() {  # score732 INPUTS: the new model on 723 users of the 732 set (strict), then the 3 long users (not needed by the rule)
  local spec="expertium:$ARCH:$DEC/expertium/$TRACK/master.pth"
  score "$1" "the new model" "$OUTROWS" "$P723" "$USERS732" 723 1 $MIN_SCORE732 $TL732 0.8 "$CLBL" "$spec" || return 1
  score "$1" "the new model" "$OUTROWS" "732 the 3 long users of the 732 set" "$ULONG" 726 0 $MIN_LONG $TLLONG 0.92 "$CLBL" "$spec" \
    || log "long users scoring failed ($1); going on (the rule pool does not need them)"
  return 0
}

rule() {  # rule CMD WHEN
  [ "$DRYRUN" = 1 ] && { log "DRYRUN: then $PY -X utf8 $RULE $1 ($2)"; return 0; }
  cpu_step 3 "rule" || return 1
  local out r; out=$("$PY" -X utf8 "$RULE" "$1" 2>&1); r=$?
  log "rule $1 ($2) exit $r: $(echo "$out" | tail -1 | cut -c1-900)"
  [ "$r" = 0 ]
}

prepare_decay() {  # prepare_decay START OUT: the release's cool-down recipe from START (SM92's prepare command + --arch w64-c1) + checks
  local START=$1 OUT=$2 X; X=$(cat "$S/extra-options.json")
  { $LF $WARM cooldown --epoch $K --start "$START" --out "${OUT#$WT/}" --arch $ARCH --fraction 1.0 --extra-options "$X" >> "$G/prepare.out" 2>&1 \
    && $LF "$PY" -X utf8 "$WT/scripts/check_replay_label_windows.py" --decay "$OUT" --trim-unlabelled-tail >/dev/null \
    && $LF "$PY" -X utf8 "$WT/scripts/check_replay_label_windows.py" --decay "$OUT" --out "$G/check-replay-label-windows-$(basename "$OUT").json" >/dev/null; } \
    || { log "STOP: cool-down prepare / label windows failed ($(basename "$OUT"))"; return 1; }
  CUDA_VISIBLE_DEVICES= $LF "$PY" -X utf8 "$WT/scripts/check_muon_fresh_decay.py" --decay "$OUT" --out "$G/check-muon-fresh-$(basename "$OUT").json" \
    | grep -c PASS | sed "s/^/[sm106-muon] $(basename "$OUT") PASS lines: /" >> "$LOG"
  [ "${PIPESTATUS[0]}" = 0 ] || { log "STOP: Muon fresh-state CPU check failed ($(basename "$OUT"))"; return 1; }
  return 0
}

cd "$WT" || exit 1

if [ "${PREPARE_RECIPE:-0}" = 1 ]; then   # CPU only: the cool-down recipe proof from a simulated complete w64-c1 start (the cut)
  log "recipe: $(git -C "$WT" log --oneline -1)"
  [ -e "$RECIPE" ] && { log "STOP: $RECIPE exists"; exit 1; }
  [ -e "$SIMSTART" ] || SIM="$SIMSTART" E27="$E27" CUT="$CUT" TRACK="$TRACK" "$PY" -X utf8 -c "
import json, os, shutil
from pathlib import Path
sim, e27, cut, track = (Path(os.environ[k]) for k in ('SIM', 'E27', 'CUT', 'TRACK'))
case = sim / 'expertium' / str(track)
case.mkdir(parents=True)
shutil.copy(e27 / 'decay.json', sim / 'decay.json')
shutil.copy(e27 / 'protocol.json', sim / 'protocol.json')
shutil.copy(cut / 'master.pth', case / 'master.pth')
off = json.loads((e27 / 'protocol.json').read_text())['full_run']['update_offset']
(case / 'result.json').write_text(json.dumps({'status': 'complete', 'update_offset_end': off, 'simulated': True}), encoding='utf-8')
" || { log "STOP: simulated start failed"; exit 1; }
  prepare_decay "$SIMSTART" "$RECIPE" || { rm -rf "$RECIPE"; exit 1; }
  log "recipe prepared: $(basename "$RECIPE") from $(basename "$SIMSTART")"
  exit 0
fi

SM99LOG=C:/Users/jonat/workspace/wt/sm99/artifacts/small-model/sm99/gpu/launch-sm99.log   # read only
sm99_gate() {  # the show-time instant-only big model (SM99) must have finished training AND scoring (its launcher logged "done (verdict ...)")
  [ "${SKIP_SM99_GATE:-0}" = 1 ] && { log "SM99 gate skipped (SKIP_SM99_GATE=1)"; return 0; }
  local last; last=$(grep -E '\] (done \(verdict|STOP:)' "$SM99LOG" 2>/dev/null | tail -1)
  case "$last" in
    *"done (verdict"*) log "SM99 finished: ${last:0:160}"; return 0;;
    *) [ "$DRYRUN" = 1 ] && { log "DRYRUN: SM99 not finished yet (${last:-no done line}); a real launch would STOP here"; return 0; }
       log "STOP: the show-time instant-only big model (SM99) has not logged 'done (verdict ...)' (last: ${last:-none}); start SM106 after SM99's training and scoring (or SKIP_SM99_GATE=1 after checking by hand)"; return 1;;
  esac
}

preconditions() {
  sm99_gate || return 1
  for f in "$S/check-cpu-sm106.json" "$USERS732" "$ULONG" "$USERS1500" "$E27/decay.json" "$E27/protocol.json" \
           "$S92DEC/decay.json" "$RECIPE/decay.json" "$E26/expertium/$TRACK/master.pth" "$E26/expertium/$TRACK/optimizer.pth" \
           "$CUT/master.pth" "$CUT/optimizer.pth" "$CUT/extend.json" "$RULE" "$EPY" "$CHECK" "$DPAIR" "$S/extra-options.json" \
           "$S/sm106_block.py" "$WT/artifacts/small-model/sm101/gpu/sm101-rows.jsonl"; do
    [ -f "$f" ] || { log "STOP: missing $f"; return 1; }
  done
  done_in "$E26" || { log "STOP: SM92's constant-LR epoch 26 is not complete ($E26)"; return 1; }
  done_in "$S92DEC" || { log "STOP: SM92's cool-down is not complete ($S92DEC)"; return 1; }
  grep -q '"all_ok": true' "$S/check-cpu-sm106.json" || { log "STOP: the CPU proofs are not ALL OK ($S/check-cpu-sm106.json)"; return 1; }
  if [ "$DRYRUN" != 1 ] && [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    log "STOP: uncommitted source changes in scripts/ expertium/ rwkv/ (SourceWatch refuses them)"; return 1; fi
  "$PY" -X utf8 "$DPAIR" decay --arm "$RECIPE" --ref "$S92DEC" >> "$LOG" 2>&1 \
    || { log "STOP: the cool-down recipe is not SM92's cool-down except the start and the architecture"; return 1; }
  return 0
}

early_decision() { "$PY" -X utf8 -c "import json; print(json.load(open(r'$S/sm106-early.json')).get('decision'))" 2>/dev/null; }

log "started: $(git -C "$WT" log --oneline -1) (DRYRUN=$DRYRUN; SM92 epoch 26 cut to 1 card layer (w64-c1), two-head epochs 27-29 on show-time inputs, early check after 27, then the release's cool-down on show-time inputs from 29, both heads)"
rc=0
cpu_step 5 "preparation" || rc=1
[ $rc -eq 0 ] && { preconditions || rc=1; }
# the early check's reference: SM92's constant-LR epoch 26 (w64) on show-time inputs
[ $rc -eq 0 ] && { score_early "SM92's constant-LR epoch 26 (the early check's reference, w64)" "$EREF" "expertium:w64:$E26/expertium/$TRACK/master.pth" || rc=1; }
EP="(w64-c1, SM92 epoch 26's recipe: 8k final tier, the run's order rule, 2048 x 16 lanes carry, leak-free replay KD, AdamW 7e-4 constant, both heads, show-time inputs)"
STOPPED_EARLY=0
for n in $EPOCHS; do
  [ $rc -eq 0 ] || break
  D=$(efold $n)
  if [ ! -f "$D/decay.json" ]; then
    if [ "$DRYRUN" = 1 ]; then
      log "DRYRUN: epoch $n is prepared after epoch $((n - 1)) completes: $PY -X utf8 $EPY prepare --epoch $n, then pair --epoch $n"
      log "DRYRUN enqueue: $NAME: two-head epoch $n on show-time inputs $EP (expected ~$MIN_EPOCH min, segments to the breaks)"
      echo "    cmd: $LF $PY -X utf8 $DEC_PY run --out $D" >> "$LOG"
      continue
    fi
    cpu_step 3 "epoch $n prepare" || { rc=1; break; }
    "$PY" -X utf8 "$EPY" prepare --epoch $n >> "$LOG" 2>&1 || { log "STOP: epoch $n prepare failed"; rc=1; break; }
  fi
  "$PY" -X utf8 "$EPY" pair --epoch $n >> "$LOG" 2>&1 || { log "STOP: epoch $n is not epoch $((n - 1))'s step (SM92 epoch 26's recipe)"; rc=1; break; }
  if [ $n = 27 ] && [ ! -e "$G/smoke-e27-ok" ] && ! done_in "$D"; then
    ram_gate "smoke" strict || { rc=1; break; }
    q $MIN_SMOKE check "trainer smoke of epoch 27 (w64-c1 from the cut of SM92's epoch 26, both heads, constant LR, show-time inputs, 2 users x 512 reviews)" -- \
      $LF "$PY" -X utf8 "$DEC_PY" smoke --out "$D" >> "$G/check.out" 2>&1 || { log "STOP: epoch 27 smoke failed (see $G/check.out)"; rc=1; break; }
    [ "$DRYRUN" != 1 ] && { touch "$G/smoke-e27-ok"; log "smoke: $(grep '"smoke"' "$G/check.out" | tail -1 | cut -c1-200)"; }
  fi
  run_folder "$D" "two-head epoch $n on show-time inputs from $([ $n = 27 ] && echo "the cut of SM92's epoch 26" || echo "epoch $((n - 1))") $EP" $MIN_EPOCH || { rc=1; break; }
  if [ $n = 27 ]; then   # EARLY CHECK (pre-registered stop rule, sm106_rule.py early)
    score_early "epoch 27 (constant LR, w64-c1; the early check)" "$EARM" "expertium:$ARCH:$E27/expertium/$TRACK/master.pth" || { rc=1; break; }
    rule early "after epoch 27" || { rc=1; break; }
    if [ "$DRYRUN" != 1 ] && [ "$(early_decision)" != continue ]; then
      if [ "${OVERRIDE_EARLY_STOP:-0}" = 1 ]; then
        log "EARLY STOP OVERRIDDEN (OVERRIDE_EARLY_STOP=1; user 11-12 ~13:15: 'Finish SM106 anyway'): early decision $(early_decision), see $S/sm106-early.md; continuing to epochs 28-29 and the cool-down, rule A unchanged"
      else
        log "EARLY STOP: $(early_decision) (epoch 27 - SM92 epoch 26 above +0.25% on show-time curve or instant; see $S/sm106-early.md); no further GPU work"
        STOPPED_EARLY=1; break
      fi
    fi
  fi
done
if [ $STOPPED_EARLY = 1 ]; then log "done (verdict early stop)"; exit 0; fi
if [ $rc -eq 0 ] && [ ! -f "$DEC/decay.json" ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: the cool-down is prepared after epoch 29: prepare_decay $E29 $DEC = $WARM cooldown --epoch $K --start $E29 --out ${DEC#$WT/} --arch $ARCH --fraction 1.0 --extra-options extra-options.json + label windows + Muon fresh"
  else
    cpu_step 20 "cool-down prepare" || rc=1
    [ $rc -eq 0 ] && { prepare_decay "$E29" "$DEC" || { rm -rf "$DEC"; rc=1; }; }
    [ $rc -eq 0 ] && log "cool-down prepared: $(basename "$DEC")"
  fi
fi
if [ $rc -eq 0 ] && [ "$DRYRUN" != 1 ]; then
  "$PY" -X utf8 "$CHECK" cooldown >> "$LOG" 2>&1 \
    || { log "STOP: the cool-down is not SM92's except the start (epoch 29), the architecture and paths (see $S/check-cpu-sm106-cooldown.json)"; rc=1; }
fi
[ "$DRYRUN" = 1 ] && log "DRYRUN: then $PY -X utf8 $CHECK cooldown (SM92's cool-down except the start, the architecture and paths; start == epoch 29)"
if [ $rc -eq 0 ] && [ ! -e "$G/smoke-dec-ok" ]; then
  ram_gate "smoke" strict || rc=1
  [ $rc -eq 0 ] && { q $MIN_SMOKE check "trainer smoke of the w64-c1 show-time two-head cool-down from epoch 29 (fresh Muon, both heads, show-time inputs, 2 users x 512 reviews)" -- \
    $LF "$PY" -X utf8 "$DEC_PY" smoke --out "$DEC" >> "$G/check.out" 2>&1 || { log "STOP: cool-down smoke failed (see $G/check.out)"; rc=1; }; }
  [ $rc -eq 0 ] && [ "$DRYRUN" != 1 ] && { touch "$G/smoke-dec-ok"; log "smoke: $(grep '"smoke"' "$G/check.out" | tail -1 | cut -c1-200)"; }
fi
[ $rc -eq 0 ] && { run_folder "$DEC" "the release's one-epoch 32k Muon cool-down on show-time inputs from epoch 29 (w64-c1; SM92's users, order, seeds, KD targets), both heads" $MIN_DECAY || rc=1; }
[ $rc -eq 0 ] && { score732 e2s-query || rc=1; }
[ $rc -eq 0 ] && { rule decide "732 set" || rc=1; }
[ $rc -eq 0 ] && { score e2s-query "the new model" "$ROWS1500" "1500 the 1,500 validation users outside the 732 set + 6 check users" "$USERS1500" 771 0 $MIN_POOL $TLPOOL 0.8 \
  "$CLBL" "expertium:$ARCH:$DEC/expertium/$TRACK/master.pth" || log "the 1,500-pool scoring failed; going on (secondary only)"; }
[ $rc -eq 0 ] && { rule decide "final" || rc=1; }
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; else log "done (verdict $("$PY" -X utf8 -c "import json; print(json.load(open(r'$S/sm106-decision.json')).get('verdict'))" 2>/dev/null))"; fi
else
  log "STOP: ended with exit $rc"
fi
exit $rc
