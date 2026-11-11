#!/usr/bin/env bash
# The show-time instant-only big model (SM99) (artifacts/small-model/sm99/DESIGN.md, HANDOFF.md), designed 2026-11-11:
#   the show-time big model (SM91 step 2) (wt/sm91s2 .../sm91s2-big-showtime, read only)
#   -> epoch 1: the big model's own epoch recipe (scripts/sm72_big_epoch.py: users 5000-10000, fresh AdamW, warmup 200,
#      3.5e-4 -> 0 one_minus_sin over the epoch, 2 strided lanes x 8,192 carried state, cap 0.8, no teacher, label-window
#      trims, snapshots every 50 users) on show-time inputs, with curve loss weight 0 and shuffle seed 2 (sm99_epoch.py)
#   -> scoring of epoch 1 on show-time inputs (723 users) -> EARLY RULE (sm99_rule.py early): continue if epoch 1 - the
#      show-time instant-only model, stage 2 (SM98) < 0 (instant, show-time); stop otherwise
#   -> epoch 2 (continue only): the same recipe from epoch 1, shuffle seed 3 (prepared after epoch 1, pair-checked)
#   -> final scoring of the final model (epoch 2, or epoch 1 after a stop): show-time + normal on the 732 set (+3 long
#      users), sm99_rule.py decide; the 1,500 pool on show-time inputs for the final model and for the show-time big
#      model (SM91 step 2); sm99_rule.py decide again.
#   verdict: "ceiling above SM98" if final - SM98 <= -0.05% with the 95% CI below 0 (show-time instant), else "at the ceiling"
# CPU proofs (sm99_check.py -> check-cpu-sm99.json) and the GPU pair check (sm99_gpu_check.py -> gpu/check-gpu-sm99.json)
#   must be ALL OK (the GPU check runs as the first ticket if its file is missing).
# Breaks: sm99_block.py = quiet.json (scope all/gpu) + the daily 20:00-01:00 (Sunday from 16:00) unless quiet.json
#   overlaps that night; re-read at every check; CPU steps too. An epoch segment starts when 30 min fit, snapshot-stops
#   <= 2 min before a break (snapshots every 50 users, ~1 min) and resumes after it. Scoring tickets must end before it.
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm99/artifacts/small-model/sm99/launch-sm99.sh
#   launch   WMI hidden (HANDOFF.md); log artifacts/small-model/sm99/gpu/launch-sm99.log
#   resume   the launch command again (finished steps are skipped; an interrupted epoch resumes from its snapshot)
#   stop     touch C:/Users/jonat/workspace/wt/sm99/artifacts/small-model/sm99/gpu/STOP (between tickets; a running
#            epoch gets a snapshot stop at the next user boundary, ~1 min; a hard kill loses <= 50 users, ~5 min)
# Do not edit scripts/ expertium/ rwkv/ of the sm99 worktree while it runs (SourceWatch stops the training).
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/sm99
SM=$WT/artifacts/small-model
S=$SM/sm99
G=$S/gpu
mkdir -p "$G"
LOG=$G/launch-sm99.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm99-dryrun.log
EPOCH="$PY -X utf8 $WT/scripts/sm72_big_epoch.py"
SC=$WT/scripts/small_model_heldout.py
RULE=$S/sm99_rule.py
EPY=$S/sm99_epoch.py
CPUCHECK=$S/check-cpu-sm99.json
GPUCHECK=$G/check-gpu-sm99.json
TRACK=big-inst
REF=C:/Users/jonat/workspace/wt/sm91s2/artifacts/small-model/sm91s2-big-showtime                       # read only
REFCK=$REF/expertium/big-showtime/master.pth
S72=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s                                      # read only
USERS732=$S72/eval-users-732-fit.json; ULONG=$S72/users-732-long.json
USERS1500=C:/Users/jonat/workspace/wt/sm57/artifacts/small-model/sm57/stage4/users-sm93-score.json
efold() { echo "$SM/sm99-big-inst-e$1"; }
elabel() { echo "sm99-big-inst-e$1"; }
OUTROWS=$G/sm99-rows.jsonl
ROWS1500=$G/sm99-rows-1500.jsonl
SEGS=$G/segments.jsonl
TICKETS=$G/tickets.jsonl
STOPF=$G/STOP
MODE=neighbour
LF="env RWKV_DELETED_CARDS=$MODE"
RATE=${RATE:-11000}   # training reviews/s for segment lengths: the show-time big model (SM91 step 2) 11,387; SM72 11,458
MIN_START=30          # an epoch segment starts when 30 min fit (snapshot stop before the break, resume after)
MIN_SMOKE=10; MIN_GPUCHECK=15; MIN_SCORE732=25; MIN_SCORE_RETRY=15; MIN_LONG=20; MIN_POOL=35
TL732=2400; TLLONG=1380; TLPOOL=2400
NAME="the show-time instant-only big model (SM99)"
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START   # scorer cap 0.8; trainer cap 0.8 (options)
log() { echo "[sm99 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
stopped() { [ -e "$STOPF" ] && { log "STOP: STOP file present"; return 0; }; return 1; }
nap() { [ "$DRYRUN" = 1 ] && return 0; local s=$1; [ "$s" -gt 600 ] && s=600; sleep "$s"; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'; }
free_mib() { awk '/MemFree/ {print int($2 / 1024)}' /proc/meminfo 2>/dev/null || echo 99999; }
block() { "$PY" -X utf8 "$S/sm99_block.py"; }
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
    "$PY" "$QUEUE" run --owner sm99 --purpose "$NAME: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm99/launch-sm99.sh" -- "$@"
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
sys.exit(0 if r.get('status') == 'complete' and not r.get('smoke_only') and r.get('curve_loss_weight') == 0.0 else 1)" 2>/dev/null; }
complete_in() { [ -f "$1/expertium/$TRACK/result.json" ] && grep -q '"status": "complete"' "$1/expertium/$TRACK/result.json"; }
started_in() { [ -d "$1/expertium" ] || [ -d "$1/attempts" ] || [ -f "$1/protocol.prepared.json" ]; }
train_left() {  # train_left DIR: reviews left in the epoch (protocol.json shrinks on every resume; the latest snapshot of this attempt)
  O="$1" T="$TRACK" "$PY" -X utf8 - <<'PYEOF'
import json, os
from pathlib import Path
out = Path(os.environ["O"])
d = json.loads((out / "decay.json").read_text())
total = d.get("training_reviews_after_trims", d["training_reviews"])
p = json.loads((out / "protocol.json").read_text())
done = (p["full_run"].get("lr_resume") or {}).get("reviews_offset", 0)
latest = out / "expertium" / os.environ["T"] / "snapshots" / "latest.json"
if latest.exists():
    done += json.loads(latest.read_text())["review_presentations"]
print(max(total - done, 0))
PYEOF
}

segment_line() {  # segment_line FOLDER STAGE START_EPOCH END_EPOCH EXIT BASE_MIB MAX_MIB STOP_SENT
  "$PY" -X utf8 -c "
import json, sys
f, stage, a, b, rc, base, mx, stop = sys.argv[1:9]
print(json.dumps({'folder': f, 'stage': stage, 'start_epoch_s': int(a), 'end_epoch_s': int(b), 'minutes': round((int(b) - int(a)) / 60, 1),
                  'exit': int(rc), 'nvidia_smi_baseline_mib': int(base or 0), 'nvidia_smi_max_used_mib': int(mx or 0), 'stop_requested': stop == '1'}))" \
    "$@" >> "$SEGS"
}

run_epoch() {  # run_epoch DIR N: sm72_big_epoch.py run in segments to the next break or the end; resumable from snapshots
  local D=$1 n=$2 rc qpid stop_sent b left need seg what t0 base mx u
  while ! done_in "$D"; do
    stopped && return 1
    complete_in "$D" && [ "$DRYRUN" != 1 ] && { log "STOP: epoch $n is complete but its result.json does not record curve_loss_weight 0.0 (or is a smoke)"; return 1; }
    wait_free $MIN_START "epoch $n" || return 1
    ram_gate "epoch $n" || return 1
    wait_free $MIN_START "epoch $n" || return 1
    if [ "$DRYRUN" = 1 ]; then left=370292439; else left=$(train_left "$D"); fi
    need=$(( left / RATE / 60 + 12 ))
    seg=$(( FREE - 5 )); [ "$seg" -gt "$need" ] && seg=$need
    what="big-model epoch $n on show-time inputs, instant-only (curve loss weight 0) from $([ $n = 1 ] && echo "the show-time big model (SM91 step 2)" || echo "epoch 1") (users 5000-10000, shuffle seed $((n + 1)), 2 lanes x 8,192 carried state, fresh AdamW 3.5e-4 -> 0, no teacher), $(( left / 1000000 ))M of 370M reviews left"
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $NAME: $what (segment to the next break or the end, expected $seg min of ~$need)"
      echo "    cmd: $LF $EPOCH run --out $D" >> "$LOG"; return 0; fi
    rm -f "$D/control/STOP"
    log "enqueue: $what (expected $seg min of ~$need, $FREE min to the next break, $(free_mib) MiB RAM free)"
    t0=$(date +%s); base=$(gpu_used); mx=0
    "$PY" "$QUEUE" run --owner sm99 --purpose "$NAME: $what (segment to the next break or the end)" --claim gpu \
      --expected-minutes "$seg" --max-wait-minutes $(( FREE - 20 > 1 ? FREE - 20 : 1 )) --launched-by "detached launcher sm99/launch-sm99.sh" -- \
      $LF $EPOCH run --out "$D" >> "$G/train.out" 2>&1 &
    qpid=$!
    stop_sent=0
    while kill -0 "$qpid" 2>/dev/null; do
      u=$(gpu_used); [ -n "$u" ] && [ "$u" -gt "$mx" ] && mx=$u
      b=$(block)
      if [ -e "$STOPF" ] || [[ "$b" == NOW* ]] || { [[ "$b" == NEXT* ]] && [ "${b#NEXT }" -le 2 ]; }; then
        # re-requested every 30 s: `run` clears control/STOP when it starts, so a request made while queued is repeated
        [ "$stop_sent" = 0 ] && log "requesting a snapshot stop ($b$([ -e "$STOPF" ] && echo ', gpu/STOP'))"
        $EPOCH stop --out "$D" > /dev/null 2>&1; stop_sent=1
      fi
      sleep 30
    done
    wait "$qpid"; rc=$?; log "exit $rc: epoch $n segment ($(train_left "$D") reviews left; nvidia-smi max $mx MiB, baseline $base MiB)"
    segment_line "$D" "epoch $n" "$t0" "$(date +%s)" "$rc" "$base" "$mx" "$stop_sent"
    [ "$rc" = 75 ] && { sleep 60; continue; }
    done_in "$D" && break
    [ "$stop_sent" = 0 ] && { log "STOP: epoch $n segment ended without completing and without a stop request (exit $rc); see $G/train.out"; return 1; }
  done
  [ "$DRYRUN" = 1 ] || log "epoch $n complete: $("$PY" -X utf8 -c "import json; r=json.load(open(r'$D/expertium/$TRACK/result.json')); print(f\"{r.get('review_presentations', 0):,} reviews (last attempt), {r.get('training_seconds', 0)/60:.1f} min training, {r.get('reviews_per_second', 0):,.0f} reviews/s, peak allocated {r.get('peak_allocated_mib', 0):,.0f} MiB, reserved {r.get('peak_reserved_mib', 0):,.0f} MiB, process tree RSS max {r.get('peak_process_tree_rss_mib', 0):,.0f} MiB, curve_loss_weight {r.get('curve_loss_weight')}, init {str(r.get('initial_parameter_sha256'))[:12]}, trained {str(r.get('trained_parameter_sha256'))[:12]}, update offset end {r.get('update_offset_end')}\")")"
  return 0
}

score() {  # score INPUTS WHAT ROWS POOL USERS_FILE EXPECT STRICT MIN1 TL CAP0 LABEL CKPT: passes at CAP0 (a pass without
           # progress -> cap 0.92; no progress at 0.92: strict -> STOP, else go on with what fits)
  local v=$1 what=$2 rows=$3 pool=$4 uf=$5 expect=$6 strict=$7 min1=$8 tl=$9 cap=${10} slbl=${11} ck=${12} pass have prev=-1 minutes lb
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
      --model "$slbl=fork::$ck" --only "$slbl/w8+s8" --inputs "$v" --time-limit $tl --split-existing \
      --deleted-cards $MODE --out "$rows" >> "$G/score.out" 2>&1 || { log "STOP: scoring ticket failed ($lb)"; return 1; }
    [ "$DRYRUN" = 1 ] && break
  done
  [ "$DRYRUN" = 1 ] || [ "$strict" != 1 ] || [ "$(label_users "$rows" "$lb")" -ge "$expect" ] || { log "STOP: $lb scored on $(label_users "$rows" "$lb") of $expect users"; return 1; }
  [ "$DRYRUN" = 1 ] || log "$lb: $(label_users "$rows" "$lb") users scored in $(basename "$rows")"
  return 0
}
P723="732 the 732 set minus 9 never-fit (723 users)"
P1500="1500 the 1,500 validation users outside the 732 set + 6 check users"
score723() {  # score723 INPUTS N: epoch N on 723 users of the 732 set (strict)
  score "$1" "epoch $2" "$OUTROWS" "$P723" "$USERS732" 723 1 $MIN_SCORE732 $TL732 0.8 "$(elabel $2)" "$(efold $2)/expertium/$TRACK/master.pth"
}
score_long() {  # score_long INPUTS N: the 3 long users (not needed by the rule)
  score "$1" "epoch $2" "$OUTROWS" "732 the 3 long users of the 732 set" "$ULONG" 726 0 $MIN_LONG $TLLONG 0.92 "$(elabel $2)" \
    "$(efold $2)/expertium/$TRACK/master.pth" || log "long users scoring failed ($1, epoch $2); going on (the rule pool does not need them)"
  return 0
}

rule() {  # rule CMD WHEN
  [ "$DRYRUN" = 1 ] && { log "DRYRUN: then $PY -X utf8 $RULE $1 ($2)"; return 0; }
  cpu_step 3 "rule" || return 1
  local out r; out=$("$PY" -X utf8 "$RULE" "$1" 2>&1); r=$?
  log "rule $1 ($2) exit $r: $(echo "$out" | tail -1 | cut -c1-900)"
  [ "$r" = 0 ]
}
early_decision() { "$PY" -X utf8 -c "import json; print(json.load(open(r'$S/sm99-early.json')).get('decision'))" 2>/dev/null; }
json_ok() { [ -f "$1" ] && "$PY" -X utf8 -c "import json, sys; d = json.load(open(sys.argv[1])); sys.exit(0 if d.get('$2') is True else 1)" "$1"; }

cd "$WT" || exit 1

preconditions() {
  for f in "$CPUCHECK" "$USERS732" "$ULONG" "$USERS1500" "$REFCK" "$REF/decay.json" "$(efold 1)/decay.json" "$(efold 1)/protocol.json" \
           "$RULE" "$EPY" "$S/sm99_block.py" "$S/sm99_gpu_check.py" "$SM/sm98/gpu/sm98-rows.jsonl" "$SM/sm96/gpu/sm96-rows.jsonl" \
           "$SM/sm91s2/gpu/sm91s2-inputs.jsonl"; do
    [ -f "$f" ] || { log "STOP: missing $f"; return 1; }
  done
  json_ok "$CPUCHECK" all_ok || { log "STOP: the CPU proofs are not ALL OK ($CPUCHECK)"; return 1; }
  if [ "$DRYRUN" != 1 ] && [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    log "STOP: uncommitted source changes in scripts/ expertium/ rwkv/ (SourceWatch refuses them)"; return 1; fi
  "$PY" -X utf8 "$RULE" selftest > /dev/null 2>&1 || { log "STOP: the rule selftest fails (reference rows or the report changed?)"; return 1; }
  return 0
}

log "started: $(git -C "$WT" log --oneline -1 | cut -c1-160) (DRYRUN=$DRYRUN; the show-time big model (SM91 step 2) + instant-only big-model epochs on show-time inputs, early rule after epoch 1)"
rc=0
cpu_step 5 "preparation" || rc=1
[ $rc -eq 0 ] && { preconditions || rc=1; }
if [ $rc -eq 0 ] && ! json_ok "$GPUCHECK" ok; then
  ram_gate "GPU check" strict || rc=1
  [ $rc -eq 0 ] && { q $MIN_GPUCHECK check "GPU pair check (trainer smokes of the show-time big model (SM91 step 2)'s epoch on its tree and this tree, epoch 1 with and without curve loss weight 0, scorer rows vs committed rows; 2 users each)" -- \
    "$PY" -X utf8 "$S/sm99_gpu_check.py" --out "$GPUCHECK" >> "$G/check-gpu.out" 2>&1 || { log "STOP: GPU pair check failed (see $G/check-gpu.out)"; rc=1; }; }
  [ $rc -eq 0 ] && [ "$DRYRUN" != 1 ] && log "GPU pair check: $(grep '\[sm99-gpu\] ' "$G/check-gpu.out" | tail -1 | cut -c1-400)"
fi
STOPPED_EARLY=0; LAST=0
for n in 1 2; do
  [ $rc -eq 0 ] || break
  D=$(efold $n)
  if [ $n = 2 ]; then
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN: epoch 2 runs only if sm99-early.json says continue"
    elif [ "$(early_decision)" != continue ]; then
      log "EARLY STOP: $(early_decision) (epoch 1 - SM98 on show-time instant is not below 0; see $S/sm99-early.md); no epoch 2, the final read-out runs on epoch 1"
      STOPPED_EARLY=1; break
    fi
  fi
  if [ ! -f "$D/decay.json" ]; then
    if [ "$DRYRUN" = 1 ]; then
      log "DRYRUN: epoch $n is prepared after epoch $((n - 1)) completes: $PY -X utf8 $EPY prepare --epoch $n, then pair --epoch $n"
      log "DRYRUN enqueue: $NAME: trainer smoke of epoch $n (expected $MIN_SMOKE min), then the epoch in segments (~$(( 370292439 / RATE / 60 + 12 )) min)"
      LAST=$n; continue
    fi
    cpu_step 5 "epoch $n prepare" || { rc=1; break; }
    "$PY" -X utf8 "$EPY" prepare --epoch $n >> "$G/prepare.out" 2>&1 || { log "STOP: epoch $n prepare failed (see $G/prepare.out)"; rc=1; break; }
    log "epoch $n prepared: $(tail -1 "$G/prepare.out" | cut -c1-300)"
  fi
  "$PY" -X utf8 "$EPY" pair --epoch $n >> "$LOG" 2>&1 || { log "STOP: epoch $n is not the show-time big model's epoch recipe + curve loss weight 0 + seed $((n + 1)) (see $G/pair-e$n.json)"; rc=1; break; }
  if [ ! -e "$G/smoke-e$n-ok" ] && ! started_in "$D"; then
    ram_gate "smoke" strict || { rc=1; break; }
    q $MIN_SMOKE check "trainer smoke of big-model epoch $n (instant-only, show-time inputs, 2 users x 512 reviews, scratch copy)" -- \
      $LF $EPOCH smoke --out "$D" >> "$G/check.out" 2>&1 || { log "STOP: epoch $n smoke failed (see $G/check.out)"; rc=1; break; }
    [ "$DRYRUN" != 1 ] && { touch "$G/smoke-e$n-ok"; log "smoke: $(grep '"smoke"' "$G/check.out" | tail -1 | cut -c1-200)"; }
  fi
  run_epoch "$D" $n || { rc=1; break; }
  LAST=$n
  [ "$DRYRUN" = 1 ] && [ $n = 2 ] && break
  score723 e2s-query $n || { rc=1; break; }
  if [ $n = 1 ]; then rule early "after epoch 1" || { rc=1; break; }; fi
done
FINAL=$LAST
[ $STOPPED_EARLY = 1 ] && FINAL=1
if [ $rc -eq 0 ]; then
  [ "$DRYRUN" = 1 ] && log "DRYRUN: the final model is epoch 2 (continue) or epoch 1 (early stop); shown for epoch $FINAL"
  FL=$(elabel $FINAL); FCK=$(efold $FINAL)/expertium/$TRACK/master.pth
  [ "$DRYRUN" = 1 ] && [ $FINAL = 2 ] && score723 e2s-query 2
  score_long e2s-query $FINAL
  score723 normal $FINAL || rc=1
  [ $rc -eq 0 ] && score_long normal $FINAL
  [ $rc -eq 0 ] && { rule decide "732 set" || rc=1; }
  [ $rc -eq 0 ] && { score e2s-query "epoch $FINAL (the final model)" "$ROWS1500" "$P1500" "$USERS1500" 771 0 $MIN_POOL $TLPOOL 0.8 "$FL" "$FCK" \
    || log "the 1,500-pool scoring of the final model failed; going on (read-out only)"; }
  [ $rc -eq 0 ] && { score e2s-query "the show-time big model (SM91 step 2) (reference)" "$ROWS1500" "$P1500" "$USERS1500" 771 0 $MIN_POOL $TLPOOL 0.8 \
    sm91s2-big-showtime "$REFCK" || log "the 1,500-pool scoring of the show-time big model (SM91 step 2) failed; going on (read-out only)"; }
  [ $rc -eq 0 ] && { rule decide "final" || rc=1; }
fi
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; else log "done (verdict $("$PY" -X utf8 -c "import json; d = json.load(open(r'$S/sm99-decision.json')); print(d.get('verdict'), '/', d.get('final_label'))" 2>/dev/null))"; fi
else
  log "STOP: ended with exit $rc"
fi
exit $rc
