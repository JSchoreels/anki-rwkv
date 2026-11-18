#!/usr/bin/env bash
# The show-time instant-only small model (SM96) (artifacts/small-model/sm96/HANDOFF.md), redesigned 2026-11-09 00:20 (user):
#   the instant-only small model (SM57) stage 4's constant-LR instant-only epoch 30 (reused, read only)
#   -> epoch 31: one constant-LR instant-only epoch on show-time inputs (stage 4's epoch recipe, sm96_epoch.py, +
#      "show_time_intervals": true, the switch of the show-time retrain of the release (SM92) as in its epoch 26)
#   -> the release's one-epoch 32k Muon cool-down from epoch 31 with "curve_loss_weight": 0.0 AND "show_time_intervals": true
#      (stage 4's cool-down except the start checkpoint, the switch and paths; prepared after epoch 31, then
#      sm96_check.py cooldown). The same shape as SM92 (epoch 26 + cool-down on show-time inputs).
# Rule (sm96_rule.py decide, pre-registered, unchanged): new - SM92, instant head, show-time inputs, 722 users of the 732 set:
#   go if the 95% CI is below 0 and the gain >= 0.05 points (% of the big model (1 leak-free epoch) normal instant LogLoss).
#   Secondary: new - stage 4 (show-time, normal), new - SM92 normal, the show-time cost, the 1,500 pool (show-time, % of the release).
# CPU proofs (sm96_check.py -> check-cpu-sm96.json): ALL OK required.
# Breaks: sm96_block.py = quiet.json (scope all/gpu) + the daily 20:00-01:00 (Sunday from 16:00) unless quiet.json overlaps
#   that night, + a hard hold until 2026-11-09 01:00; re-read at every check, CPU steps too. Epoch 31 and the cool-down
#   each start when 30 min fit, snapshot-stop <= 2 min before a break and resume after it (scripts/expertium_decay.py run / stop).
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm96/artifacts/small-model/sm96/launch-sm96.sh
#   launch   WMI hidden (HANDOFF.md); log artifacts/small-model/sm96/gpu/launch-sm96.log
#   resume   the launch command again (finished steps are skipped; an interrupted epoch / cool-down resumes from its snapshot)
#   stop     touch C:/Users/jonat/workspace/wt/sm96/artifacts/small-model/sm96/gpu/STOP (between tickets; a running
#            epoch / cool-down gets a snapshot stop at the next user boundary)
# Do not edit scripts/ expertium/ rwkv/ of the sm96 worktree while it runs (SourceWatch stops the training).
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/sm96
SM=$WT/artifacts/small-model
S=$SM/sm96
G=$S/gpu
mkdir -p "$G"
LOG=$G/launch-sm96.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm96-dryrun.log
DEC_PY=$WT/scripts/expertium_decay.py
SC=$WT/scripts/small_model_heldout.py
RULE=$S/sm96_rule.py
EPY=$S/sm96_epoch.py
CHECK=$S/sm96_check.py
WARM="$PY -X utf8 $WT/scripts/sm77_warm.py"
PAIR=C:/Users/jonat/workspace/wt/sm57/artifacts/small-model/sm57/sm57_pair.py     # read only
TRACK=E89-F65e-F77a
ARCH=w64
K=25                          # the release's cool-down recipe (sm77_warm.py cooldown --epoch 25: users / order / seeds / KD targets)
S57=C:/Users/jonat/workspace/wt/sm57/artifacts/small-model                         # read only
E30=$S57/sm57s4-inst-e30
S4DEC=$S57/sm57s4-inst-e30-dec-1ep32k-muon
S72=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s                  # read only
USERS732=$S72/eval-users-732-fit.json; ULONG=$S72/users-732-long.json
USERS1500=$S57/sm57/stage4/users-sm93-score.json
E31=$SM/sm96-inst-e31-st                                                          # epoch 31 (show-time inputs)
RECIPE=$SM/sm96-inst-e30-showtime-dec-1ep32k-muon                                  # cool-down recipe proof from epoch 30 (never trained)
DEC=$SM/sm96-inst-e31st-st-dec-1ep32k-muon
CLBL=sm96-inst-e31st-st-dec1ep32k
SPEC="expertium:$ARCH:$DEC/expertium/$TRACK/master.pth"
OUTROWS=$G/sm96-rows.jsonl
ROWS1500=$G/sm96-rows-1500.jsonl
SEGS=$G/segments.jsonl
TICKETS=$G/tickets.jsonl
STOPF=$G/STOP
MODE=neighbour
LF="env RWKV_DELETED_CARDS=$MODE"
MIN_EPOCH=${MIN_EPOCH:-125}   # stage 4's epochs 27-30: 116-123 min at ~82-88k reviews/s
MIN_DECAY=${MIN_DECAY:-560}   # stage 4's cool-down: 478 min wall at ~20.3k reviews/s + setup
MIN_START=30                  # epoch 31 / the cool-down start when 30 min fit (snapshot stop before the break, resume after)
MIN_SMOKE=10; MIN_SCORE732=32; MIN_SCORE_RETRY=20; MIN_LONG=25; MIN_POOL=45
TL732=1800; TLLONG=1380; TLPOOL=2400
NAME="the show-time instant-only small model (SM96)"
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START   # scorer cap 0.8; trainer cap 0.8 (options)
log() { echo "[sm96 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
stopped() { [ -e "$STOPF" ] && { log "STOP: STOP file present"; return 0; }; return 1; }
nap() { [ "$DRYRUN" = 1 ] && return 0; local s=$1; [ "$s" -gt 600 ] && s=600; sleep "$s"; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'; }
free_mib() { awk '/MemFree/ {print int($2 / 1024)}' /proc/meminfo 2>/dev/null || echo 99999; }
block() { "$PY" -X utf8 "$S/sm96_block.py"; }
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
    "$PY" "$QUEUE" run --owner sm96 --purpose "$NAME: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm96/launch-sm96.sh" -- "$@"
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
    "$PY" "$QUEUE" run --owner sm96 --purpose "$NAME: $what (segment to the next break or the end)" --claim gpu \
      --expected-minutes "$seg" --max-wait-minutes $(( FREE - 20 > 1 ? FREE - 20 : 1 )) --launched-by "detached launcher sm96/launch-sm96.sh" -- \
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
  [ "$DRYRUN" = 1 ] || log "$what complete: $("$PY" -X utf8 -c "import json; r=json.load(open(r'$D/expertium/$TRACK/result.json')); print(f\"{r.get('review_presentations', 0):,} reviews, {r.get('training_seconds', 0)/60:.1f} min training, {r.get('reviews_per_second', 0):,.0f} reviews/s, peak allocated {r.get('peak_allocated_mib', 0):,.0f} MiB, reserved {r.get('peak_reserved_mib', 0):,.0f} MiB, curve_loss_weight {r.get('curve_loss_weight')}, init {str(r.get('initial_parameter_sha256'))[:12]}, trained {str(r.get('trained_parameter_sha256'))[:12]}, update offset end {r.get('update_offset_end')}\")")"
  return 0
}

score() {  # score INPUTS WHAT ROWS POOL USERS_FILE EXPECT STRICT MIN1 TL [CAP0]: passes at CAP0 (default 0.8; a pass without
           # progress -> cap 0.92; no progress at 0.92: strict -> STOP, else go on with what fits)
  local v=$1 what=$2 rows=$3 pool=$4 uf=$5 expect=$6 strict=$7 min1=$8 tl=$9 cap=${10:-0.8} pass have prev=-1 minutes lb
  lb="$CLBL/w8+s8"; [ "$v" != normal ] && lb="$lb@$v"
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
      --model "$CLBL=$SPEC" --only "$CLBL/w8+s8" --inputs "$v" --time-limit $tl --split-existing \
      --deleted-cards $MODE --out "$rows" >> "$G/score.out" 2>&1 || { log "STOP: scoring ticket failed ($lb)"; return 1; }
    [ "$DRYRUN" = 1 ] && break
  done
  [ "$DRYRUN" = 1 ] || [ "$strict" != 1 ] || [ "$(label_users "$rows" "$lb")" -ge "$expect" ] || { log "STOP: $lb scored on $(label_users "$rows" "$lb") of $expect users"; return 1; }
  [ "$DRYRUN" = 1 ] || log "$lb: $(label_users "$rows" "$lb") users scored in $(basename "$rows")"
  return 0
}
score732() {  # score732 INPUTS: 723 users of the 732 set (strict), then the 3 long users (cap 0.92, not needed by the rule)
  score "$1" "the new model" "$OUTROWS" "732 the 732 set minus 9 never-fit (723 users)" "$USERS732" 723 1 $MIN_SCORE732 $TL732 || return 1
  score "$1" "the new model" "$OUTROWS" "732 the 3 long users of the 732 set" "$ULONG" 726 0 $MIN_LONG $TLLONG 0.92 \
    || log "long users scoring failed ($1); going on (the rule pool does not need them)"
  return 0
}

rule() {  # rule WHEN
  [ "$DRYRUN" = 1 ] && { log "DRYRUN: then $PY -X utf8 $RULE decide ($1)"; return 0; }
  cpu_step 3 "rule" || return 1
  local out r; out=$("$PY" -X utf8 "$RULE" decide 2>&1); r=$?
  log "rule ($1) exit $r: $(echo "$out" | tail -1 | cut -c1-900)"
  [ "$r" = 0 ]
}

preconditions() {
  for f in "$S/check-cpu-sm96.json" "$USERS732" "$ULONG" "$USERS1500" "$E31/decay.json" "$E31/protocol.json" "$S4DEC/decay.json" \
           "$RECIPE/decay.json" "$E30/expertium/$TRACK/master.pth" "$E30/expertium/$TRACK/optimizer.pth" "$PAIR" "$RULE" "$EPY" "$CHECK" \
           "$S/extra-options.json" "$S/sm96_block.py"; do
    [ -f "$f" ] || { log "STOP: missing $f"; return 1; }
  done
  done_in "$E30" || { log "STOP: stage 4's constant-LR epoch 30 is not complete ($E30)"; return 1; }
  done_in "$S4DEC" || { log "STOP: stage 4's cool-down is not complete ($S4DEC)"; return 1; }
  grep -q '"all_ok": true' "$S/check-cpu-sm96.json" || { log "STOP: the CPU proofs are not ALL OK ($S/check-cpu-sm96.json)"; return 1; }
  if [ "$DRYRUN" != 1 ] && [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    log "STOP: uncommitted source changes in scripts/ expertium/ rwkv/ (SourceWatch refuses them)"; return 1; fi
  "$PY" -X utf8 "$EPY" pair >> "$LOG" 2>&1 || { log "STOP: epoch 31 is not stage 4's epoch-30 step + the show-time switch"; return 1; }
  "$PY" -X utf8 "$PAIR" decay --arm "$RECIPE" --ref "$S4DEC" --same-start --flag '{"show_time_intervals": true}' >> "$LOG" 2>&1 \
    || { log "STOP: the cool-down recipe is not stage 4's + the show-time switch"; return 1; }
  return 0
}

prepare_decay() {  # the release's cool-down recipe from epoch 31 (the command that wrote the recipe folder from epoch 30) + checks
  local X; X=$(cat "$S/extra-options.json")
  { $LF $WARM cooldown --epoch $K --start "$E31" --out "${DEC#$WT/}" --arch $ARCH --fraction 1.0 --extra-options "$X" >> "$G/prepare.out" 2>&1 \
    && $LF "$PY" -X utf8 "$WT/scripts/check_replay_label_windows.py" --decay "$DEC" --trim-unlabelled-tail >/dev/null \
    && $LF "$PY" -X utf8 "$WT/scripts/check_replay_label_windows.py" --decay "$DEC" --out "$G/check-replay-label-windows-$(basename "$DEC").json" >/dev/null; } \
    || { log "STOP: cool-down prepare / label windows failed"; return 1; }
  CUDA_VISIBLE_DEVICES= $LF "$PY" -X utf8 "$WT/scripts/check_muon_fresh_decay.py" --decay "$DEC" --out "$G/check-muon-fresh-$(basename "$DEC").json" \
    | grep -c PASS | sed "s/^/[sm96-muon] $(basename "$DEC") PASS lines: /" >> "$LOG"
  [ "${PIPESTATUS[0]}" = 0 ] || { log "STOP: Muon fresh-state CPU check failed"; return 1; }
  return 0
}

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1) (DRYRUN=$DRYRUN; epoch 31 on show-time inputs from SM57 stage 4's instant-only epoch 30, then the release's cool-down on show-time inputs from epoch 31, curve loss weight 0)"
rc=0
cpu_step 5 "preparation" || rc=1
[ $rc -eq 0 ] && { preconditions || rc=1; }
if [ $rc -eq 0 ] && [ ! -e "$G/smoke-e31-ok" ] && ! done_in "$E31"; then
  ram_gate "smoke" strict || rc=1
  [ $rc -eq 0 ] && { q $MIN_SMOKE check "trainer smoke of epoch 31 (instant-only, constant LR, show-time inputs, from stage 4's epoch 30, 2 users x 512 reviews)" -- \
    $LF "$PY" -X utf8 "$DEC_PY" smoke --out "$E31" >> "$G/check.out" 2>&1 || { log "STOP: epoch 31 smoke failed (see $G/check.out)"; rc=1; }; }
  [ $rc -eq 0 ] && [ "$DRYRUN" != 1 ] && { touch "$G/smoke-e31-ok"; log "smoke: $(grep '"smoke"' "$G/check.out" | tail -1 | cut -c1-200)"; }
fi
EP="(w64, stage 4's epoch recipe: 8k final tier, the run's order rule, 2048 x 16 lanes carry, leak-free replay KD, AdamW 7e-4 constant, curve loss weight 0, show-time inputs)"
[ $rc -eq 0 ] && { run_folder "$E31" "instant-only epoch 31 on show-time inputs from SM57 stage 4's epoch 30 $EP" $MIN_EPOCH || rc=1; }
if [ $rc -eq 0 ] && [ ! -f "$DEC/decay.json" ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: the cool-down is prepared after epoch 31: $WARM cooldown --epoch $K --start $E31 --out ${DEC#$WT/} --arch $ARCH --fraction 1.0 --extra-options extra-options.json + label windows + Muon fresh"
  else
    cpu_step 20 "cool-down prepare" || rc=1
    [ $rc -eq 0 ] && { prepare_decay || { rm -rf "$DEC"; rc=1; }; }
    [ $rc -eq 0 ] && log "cool-down prepared: $(basename "$DEC")"
  fi
fi
if [ $rc -eq 0 ] && [ "$DRYRUN" != 1 ]; then
  "$PY" -X utf8 "$CHECK" cooldown >> "$LOG" 2>&1 \
    || { log "STOP: the cool-down is not stage 4's except the start (epoch 31), the switch and paths (see $S/check-cpu-sm96-cooldown.json)"; rc=1; }
fi
[ "$DRYRUN" = 1 ] && log "DRYRUN: then $PY -X utf8 $CHECK cooldown (stage 4's cool-down except the start, the switch, paths; start == epoch 31)"
if [ $rc -eq 0 ] && [ ! -e "$G/smoke-dec-ok" ]; then
  ram_gate "smoke" strict || rc=1
  [ $rc -eq 0 ] && { q $MIN_SMOKE check "trainer smoke of the show-time instant-only cool-down from epoch 31 (fresh Muon, curve loss weight 0, show-time inputs, 2 users x 512 reviews)" -- \
    $LF "$PY" -X utf8 "$DEC_PY" smoke --out "$DEC" >> "$G/check.out" 2>&1 || { log "STOP: cool-down smoke failed (see $G/check.out)"; rc=1; }; }
  [ $rc -eq 0 ] && [ "$DRYRUN" != 1 ] && { touch "$G/smoke-dec-ok"; log "smoke: $(grep '"smoke"' "$G/check.out" | tail -1 | cut -c1-200)"; }
fi
[ $rc -eq 0 ] && { run_folder "$DEC" "the release's one-epoch 32k Muon cool-down on show-time inputs from epoch 31 (stage 4's users, order, seeds, KD targets), curve loss weight 0" $MIN_DECAY || rc=1; }
[ $rc -eq 0 ] && { score732 e2s-query || rc=1; }
[ $rc -eq 0 ] && { rule "show-time 732" || rc=1; }
[ $rc -eq 0 ] && { score732 normal || rc=1; }
[ $rc -eq 0 ] && { score e2s-query "the new model" "$ROWS1500" "1500 the 1,500 validation users outside the 732 set + 6 check users" "$USERS1500" 771 0 $MIN_POOL $TLPOOL \
  || log "the 1,500-pool scoring failed; going on (secondary only)"; }
[ $rc -eq 0 ] && { rule "final" || rc=1; }
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; else log "done (verdict $("$PY" -X utf8 -c "import json; print(json.load(open(r'$S/sm96-decision.json')).get('verdict'))" 2>/dev/null))"; fi
else
  log "STOP: ended with exit $rc"
fi
exit $rc
