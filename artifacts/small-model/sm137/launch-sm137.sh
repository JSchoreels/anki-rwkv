#!/usr/bin/env bash
# Trained smaller state for Tiny (SM137) (artifacts/small-model/sm137/design.md), designed 2026-11-21: the PILOT only.
#   The card / note WKV state of the six-layer Tiny (SM130) cut to V value x K key lanes per head (LANES=v16k8 default:
#   803 B per card and per note at the release-candidate format instead of 1,315 B), one constant-LR epoch:
#     the lane cut of SM130's Instant-Only epoch 41 (sm137_lanes.py mask, prepared on the CPU, sha256 pinned in its
#     extend.json) -> epoch 42 = exactly the six-layer Tiny with longer recovery (SM135)'s epoch 42 except the architecture
#     (sm137_epoch.py pair) -> w8+s8 + w8+s8hx8 on the 732 set -> sm137_rule.py pilot (go / pilot_v16k12 / stop).
#   Steps: checks (rule selftest, the CPU proofs report of this lane shape), memory guard, prepare + pair, GPU scorer
#   smoke (2 users vs the CPU rows, |difference| <= 1e-5), trainer smoke, zero-shot score of the cut (w8+s8, report only),
#   the epoch (segments to the next quiet window, resumable), dropped-row check of the trained master, scoring, rule.
#   The full plan (cool-down, two-head chain, long-card check) is NOT in this launcher; see design.md and HANDOFF.md.
# Breaks: the SM135 block helper (quiet.json windows of scope all / gpu only), re-read at every check.
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm137/artifacts/small-model/sm137/launch-sm137.sh
#   launch   WMI hidden (HANDOFF.md); log artifacts/small-model/sm137/gpu/launch-sm137.log
#   resume   the launch command again (finished steps are skipped; the epoch resumes from its last snapshot)
#   stop     touch C:/Users/jonat/workspace/wt/sm137/artifacts/small-model/sm137/gpu/STOP
#   second pilot (only if the first says pilot_v16k12): LANES=v16k12 after its cut + CPU proofs (HANDOFF.md)
# Do not edit scripts/ expertium/ rwkv/ of the sm137 worktree while it runs (SourceWatch stops the training).
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/sm137
SM=$WT/artifacts/small-model
S=$SM/sm137
G=$S/gpu
mkdir -p "$G" "$S/cpu"
LOG=$G/launch-sm137.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm137-dryrun.log
DEC_PY=$WT/scripts/expertium_decay.py
SC=$WT/scripts/small_model_heldout.py
RULE=$S/sm137_rule.py
EPY=$S/sm137_epoch.py
LPY=$S/sm137_lanes.py
BLOCK=$SM/sm135/sm135_block.py
TRACK=E89-F65e-F77a
BASE=w64-c1-d2p1u1
LANES=${LANES:-v16k8}
VL=$(echo "$LANES" | sed -E 's/^v([0-9]+)k([0-9]+)$/\1/'); KL=$(echo "$LANES" | sed -E 's/^v([0-9]+)k([0-9]+)$/\2/')
ARCH=$BASE-$LANES
CUT=$SM/sm137-io-$LANES-e41
E42=$SM/sm137-io-$LANES-e42-st
CHECK=$S/check-cpu-sm137-$LANES.json
W=C:/Users/jonat/workspace/wt
S72=$W/sm72s/artifacts/small-model/sm72s
USERS732=$S72/eval-users-732-fit.json
USMOKE=$S/users-smoke.json
EROWS=$G/sm137-early-rows.jsonl
SEGS=$G/segments.jsonl
TICKETS=$G/tickets.jsonl
STOPF=$G/STOP
MODE=neighbour
LF="env RWKV_DELETED_CARDS=$MODE"
MIN_EPOCH=${MIN_EPOCH:-70}    # SM135's identical epoch 42: 54.1 min at 179,160 reviews/s
MIN_START=30; MIN_SMOKE=15; MIN_SCORE=25; MIN_SCORE_RETRY=20; TL=1800
NAME="Trained smaller state for Tiny (SM137)"
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START SM11_INPUTS   # caps 0.8
log() { echo "[sm137 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
stopped() { [ -e "$STOPF" ] && { log "STOP: STOP file present"; return 0; }; return 1; }
nap() { [ "$DRYRUN" = 1 ] && return 0; local s=$1; [ "$s" -gt 600 ] && s=600; sleep "$s"; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'; }
free_mib() { awk '/MemFree/ {print int($2 / 1024)}' /proc/meminfo 2>/dev/null || echo 99999; }
block() { "$PY" -X utf8 "$BLOCK"; }
jline() { "$PY" -X utf8 -c "import json, sys; print(json.dumps(dict(zip(sys.argv[1::2], [int(v) if v.lstrip('-').isdigit() else v for v in sys.argv[2::2]]))))" "$@"; }


wait_free() {  # wait_free MINUTES WHAT: returns once MINUTES fit before the next quiet window (sets FREE = minutes to it)
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
             if [ "$DRYRUN" = 1 ]; then log "DRYRUN: $2 ($1 min) would wait: $FREE min to the next quiet window"; FREE=100000; return 0; fi
             [ $((n % 6)) = 0 ] && log "$2 needs $1 min, $FREE min before the next quiet window; waiting"; n=$((n + 1)); nap $(( FREE * 60 + 60 ));;
      *) log "window check failed: $b"; nap 600;;
    esac
  done
}

ram_gate() {  # ram_gate WHAT [strict]: hold while < 3 GB RAM is free, at most 30 min
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

q() {  # q MINUTES KIND PURPOSE -- CMD...   one ticket that must end before the next quiet window
  local minutes=$1 kind=$2 purpose=$3 rc t0 mx base wpid; shift 4
  while true; do
    wait_free "$minutes" "$purpose" || return 1
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $NAME: $purpose (expected $minutes min)"; echo "    cmd: $*" >> "$LOG"; return 0; fi
    local maxwait=$(( FREE - minutes - 3 )); [ "$maxwait" -lt 1 ] && maxwait=1
    log "enqueue: $purpose (expected $minutes min, max wait $maxwait min, $FREE min to the next quiet window)"
    t0=$(date +%s); base=$(gpu_used); rm -f "$G/.gpu-max"
    ( m=0; while true; do u=$(gpu_used); [ -n "$u" ] && [ "$u" -gt "$m" ] && { m=$u; echo "$m" > "$G/.gpu-max"; }; sleep 30; done ) &
    wpid=$!
    "$PY" "$QUEUE" run --owner sm137 --purpose "$NAME: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm137/launch-sm137.sh" -- "$@"
    rc=$?; kill "$wpid" 2>/dev/null; wait "$wpid" 2>/dev/null; mx=$(cat "$G/.gpu-max" 2>/dev/null || echo 0)
    log "exit $rc: $purpose (nvidia-smi max $mx MiB, baseline $base MiB)"
    [ "$rc" = 75 ] && { log "queue wait would reach the next quiet window; re-planning"; sleep 60; continue; }
    jline kind "$kind" purpose "$purpose" exit "$rc" minutes "$(( ($(date +%s) - t0 + 30) / 60 ))" nvidia_smi_max_mib "${mx:-0}" nvidia_smi_baseline_mib "${base:-0}" >> "$TICKETS"
    return $rc
  done
}

min_users() {  # min_users ROWS LABEL...: users with a row (benchmark, or a 'nonfinite' row from --nonfinite-rows) per label
  F="$1" "$PY" -X utf8 -c "
import json, os, sys
from pathlib import Path
p = Path(os.environ['F'])
got = {}
if p.exists():
    for raw in open(p, 'rb'):
        try:
            r = json.loads(raw.decode('utf-8', errors='replace'))
        except ValueError:
            continue
        if 'benchmark' in r or ('nonfinite' in r and 'error' not in r):
            got.setdefault(r.get('label'), set()).add(r['user_id'])
print(min(len(got.get(l, ())) for l in sys.argv[1:]))" "${@:2}"; }

done_in() { [ -f "$1/expertium/$TRACK/master.pth" ] && R="$1/expertium/$TRACK/result.json" "$PY" -X utf8 -c "
import json, os, sys
r = json.load(open(os.environ['R']))
sys.exit(0 if r.get('status') == 'complete' and not r.get('smoke_only') else 1)" 2>/dev/null; }

segment_line() {
  "$PY" -X utf8 -c "
import json, sys
f, stage, a, b, rc, base, mx, stop = sys.argv[1:9]
print(json.dumps({'folder': f, 'stage': stage, 'start_epoch_s': int(a), 'end_epoch_s': int(b), 'minutes': round((int(b) - int(a)) / 60, 1),
                  'exit': int(rc), 'nvidia_smi_baseline_mib': int(base or 0), 'nvidia_smi_max_used_mib': int(mx or 0), 'stop_requested': stop == '1'}))" \
    "$@" >> "$SEGS"
}

run_folder() {  # run_folder DIR WHAT SEG_MAX: expertium_decay.py run in segments to the next quiet window; resumable from snapshots
  local D=$1 what=$2 segmax=$3 rc qpid stop_sent b need t0 base mx u why
  while ! done_in "$D"; do
    stopped && return 1
    need=15; [ ! -d "$D/expertium" ] && [ ! -d "$D/attempts" ] && need=$MIN_START
    wait_free "$need" "$what" || return 1
    ram_gate "$what" || return 1
    wait_free "$need" "$what" || return 1
    local seg=$(( FREE - 5 )); [ "$seg" -gt "$segmax" ] && seg=$segmax
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $NAME: $what (expected $seg min of ~$segmax; snapshot stop before a quiet window)"
      echo "    cmd: $LF $PY -X utf8 $DEC_PY run --out $D" >> "$LOG"; return 0; fi
    rm -f "$D/control/STOP"
    log "enqueue: $what (expected $seg min of ~$segmax, $FREE min to the next quiet window, $(free_mib) MiB RAM free)"
    t0=$(date +%s); base=$(gpu_used); mx=0
    "$PY" "$QUEUE" run --owner sm137 --purpose "$NAME: $what (segment to the next quiet window or the end)" --claim gpu \
      --expected-minutes "$seg" --max-wait-minutes $(( FREE - 20 > 1 ? FREE - 20 : 1 )) --launched-by "detached launcher sm137/launch-sm137.sh" -- \
      $LF "$PY" -X utf8 "$DEC_PY" run --out "$D" >> "$G/train.out" 2>&1 &
    qpid=$!
    stop_sent=0
    while kill -0 "$qpid" 2>/dev/null; do
      u=$(gpu_used); [ -n "$u" ] && [ "$u" -gt "$mx" ] && mx=$u
      b=$(block); why=""
      [ -e "$STOPF" ] && why="STOP file"
      { [[ "$b" == NOW* ]] || { [[ "$b" == NEXT* ]] && [ "${b#NEXT }" -le 2 ]; }; } && why="quiet window ($b)"
      if [ -n "$why" ]; then
        [ "$stop_sent" = 0 ] && log "requesting a snapshot stop ($why)"
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
  [ "$DRYRUN" = 1 ] || log "$what complete: $("$PY" -X utf8 -c "import json; r=json.load(open(r'$D/expertium/$TRACK/result.json')); s=(r.get('expertium_catch_up') or {}).get('lr_schedule', {}); print(f\"{r.get('review_presentations', 0):,} reviews, {r.get('training_seconds', 0)/60:.1f} min training, {r.get('reviews_per_second', 0):,.0f} reviews/s, peak allocated {r.get('peak_allocated_mib', 0):,.0f} MiB, reserved {r.get('peak_reserved_mib', 0):,.0f} MiB, process tree RSS {r.get('peak_process_tree_rss_mib', 0):,.0f} MiB, curve_loss_weight {r.get('curve_loss_weight')}, schedule total {s.get('total_reviews')}, init {str(r.get('initial_parameter_sha256'))[:12]}, trained {str(r.get('trained_parameter_sha256'))[:12]}, update offset end {r.get('update_offset_end')}\")")"
  return 0
}

score() {  # score TAG ARCH CKPT ROWS STRICT NONFINITE VARIANT...: 723 users of the 732 set, show-time inputs, all variants in one pass
  local tag=$1 arch=$2 ck=$3 rows=$4 strict=$5 nf=$6 cap=0.8 pass have prev=-1 minutes; shift 6
  local only=() lbls=() v extra=()
  [ "$nf" = 1 ] && extra=(--nonfinite-rows)
  for v in "$@"; do only+=(--only "$tag/$v"); lbls+=("$tag/$v@e2s-query"); done
  for pass in 1 2 3 4 5; do
    have=$(min_users "$rows" "${lbls[@]}")
    [ "$have" -ge 723 ] && break
    if [ "$have" = "$prev" ]; then
      if [ "$cap" = 0.8 ]; then cap=0.92; log "no progress on $tag ($have of 723 users): next pass at cap 0.92 (memory only, same numerics)"
      elif [ "$strict" = 1 ]; then log "STOP: no progress on $tag ($have of 723 users)"; return 1
      else log "no progress at cap 0.92 on $tag ($have of 723 users); going on"; break; fi
    fi
    prev=$have
    minutes=$(( MIN_SCORE + 10 * ($# - 1) )); [ "$pass" -gt 1 ] && minutes=$MIN_SCORE_RETRY
    ram_gate "scoring $tag"
    q $minutes score "scoring $tag at $* on e2s-query inputs, the 732 set minus 9 never-fit (723 users) (pass $pass, $have done, cap $cap)" -- \
      env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START HELDOUT_GPU_MEMORY_FRACTION=$cap "$PY" -X utf8 "$SC" --pool 732 --users-file "$USERS732" \
      --model "$tag=expertium:$arch:$ck" "${only[@]}" --inputs e2s-query --time-limit $TL --split-existing \
      --deleted-cards $MODE "${extra[@]}" --out "$rows" >> "$G/score.out" 2>&1 || { log "STOP: scoring ticket failed ($tag)"; return 1; }
    [ "$DRYRUN" = 1 ] && break
  done
  [ "$DRYRUN" = 1 ] && return 0
  have=$(min_users "$rows" "${lbls[@]}")
  log "$tag: $have users scored for ${lbls[*]}"
  [ "$strict" != 1 ] || [ "$have" -ge 723 ] || { log "STOP: $tag scored on $have of 723 users"; return 1; }
  return 0
}

smoke() {  # smoke DIR TAG WHAT
  [ -e "$G/smoke-$2-ok" ] && return 0
  done_in "$1" && return 0
  ram_gate "smoke $2" strict || return 1
  q $MIN_SMOKE check "trainer smoke of $3 (2 users x 512 reviews)" -- $LF "$PY" -X utf8 "$DEC_PY" smoke --out "$1" >> "$G/check.out" 2>&1 \
    || { log "STOP: smoke $2 failed (see $G/check.out)"; return 1; }
  [ "$DRYRUN" = 1 ] || { touch "$G/smoke-$2-ok"; log "smoke $2: $(grep '"smoke"' "$G/check.out" | tail -1 | cut -c1-200)"; }
  return 0
}



memguard_on() {  # for the launcher's whole life: CPU-only commands of this run, kill above 5.9 GB (writes STOP)
  [ "$DRYRUN" = 1 ] && return 0
  rm -f "$G/memguard.off"
  local wp; wp=$(cat /proc/$$/winpid 2>/dev/null || echo 0)
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w "$S/memguard.ps1")" -Dir "$(cygpath -w "$G")" -ParentPid "$wp" \
    -Match "sm137_rule.py;sm137_epoch.py;sm137_lanes.py;sm135_block.py&sm137" > /dev/null 2>&1 &
  MG=$!
  log "memory guard on (launcher Windows PID $wp, limit 5.9 GB per CPU process)"
}
memguard_off() { [ "$DRYRUN" = 1 ] && return 0; touch "$G/memguard.off"; sleep 12; kill "$MG" 2>/dev/null
  [ -e "$G/memguard-kill" ] && { log "memory guard killed a CPU process ($(tail -1 "$G/memguard-kill"))"; return 1; }; return 0; }

gpu_smoke() {  # the lane cut on the GPU scorer vs its CPU rows (2 users, w8+s8 + w8+s8hx8), strict 1e-5
  [ -e "$G/smoke-gpu-$LANES-ok" ] && return 0
  local rows=$G/smoke-gpu-$LANES-rows.jsonl cpu=$S/cpu/smoke-cpu-$LANES-rows.jsonl
  [ -f "$cpu" ] || { log "STOP: CPU smoke rows $cpu missing (HANDOFF.md, CPU proofs)"; return 1; }
  rm -f "$rows"
  q $MIN_SMOKE check "GPU scorer smoke of the $LANES lane cut (2 users, w8+s8 + w8+s8hx8, vs CPU rows)" -- \
    env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START HELDOUT_GPU_MEMORY_FRACTION=0.8 "$PY" -X utf8 "$SC" --pool 732 --users-file "$USMOKE" \
    --count 2 --model "smoke=expertium:$ARCH:$CUT/master.pth" --variants w8+s8,w8+s8hx8 --inputs e2s-query --time-limit 600 \
    --split-existing --deleted-cards $MODE --out "$rows" >> "$G/check.out" 2>&1 || { log "STOP: GPU scorer smoke ticket failed"; return 1; }
  [ "$DRYRUN" = 1 ] && return 0
  local out r; out=$("$PY" -X utf8 "$RULE" smoke --gpu "$rows" --cpu "$cpu" 2>&1); r=$?
  log "GPU smoke: $(echo "$out" | tail -1)"
  [ "$r" = 0 ] || { log "STOP: the GPU scorer disagrees with the CPU on the lane cut"; return 1; }
  touch "$G/smoke-gpu-$LANES-ok"
}

main() {
  log "start (pilot $LANES = $ARCH, $(git -C "$WT" rev-parse --short HEAD), dirty: $(git -C "$WT" status --porcelain -- scripts expertium rwkv | wc -l) source files)"
  stopped && return 1
  for f in "$BLOCK" "$RULE" "$EPY" "$LPY" "$S/memguard.ps1" "$USERS732" "$USMOKE" "$CUT/master.pth" "$CUT/optimizer.pth" "$CUT/extend.json" "$CHECK"; do
    [ -f "$f" ] || { log "STOP: missing $f"; return 1; }
  done
  "$PY" -X utf8 "$RULE" selftest >> "$LOG" 2>&1 || { log "STOP: rule selftest failed"; return 1; }
  CK="$CHECK" CM="$CUT/master.pth" "$PY" -X utf8 -c "
import hashlib, json, os, sys
r = json.load(open(os.environ['CK']))
sha = hashlib.sha256(open(os.environ['CM'], 'rb').read()).hexdigest()
sys.exit(0 if r.get('ok') and r.get('cut_master_sha256') == sha else 1)" || { log "STOP: the CPU proofs report $CHECK is not ALL OK for this cut"; return 1; }
  log "checks OK (rule selftest, CPU proofs $(basename "$CHECK"))"
  memguard_on
  if [ ! -f "$E42/decay.json" ] && [ "$DRYRUN" != 1 ]; then
    "$PY" -X utf8 "$EPY" prepare --chain io --lanes $LANES >> "$LOG" 2>&1 || { log "STOP: prepare $E42 failed"; return 1; }
  fi
  "$PY" -X utf8 "$EPY" pair --chain io --lanes $LANES >> "$LOG" 2>&1 || { log "STOP: $E42 is not SM135's epoch 42 with the lane architecture"; return 1; }
  gpu_smoke || return 1
  smoke "$E42" "io-$LANES" "Instant-Only lane epoch 42 ($ARCH, from the lane cut of SM130's epoch 41)" || return 1
  score "sm137-io-$LANES-e41-cut" "$ARCH" "$CUT/master.pth" "$EROWS" 1 1 w8+s8 || return 1
  run_folder "$E42" "Instant-Only lane epoch 42 ($ARCH = SM135's epoch 42 with the card / note state cut to $VL x $KL lanes per head: 8k final tier, the run's order rule, 2048 x 16 lanes, leak-free replay KD, AdamW 7e-4, instant head only, show-time inputs)" $MIN_EPOCH || return 1
  if [ "$DRYRUN" != 1 ]; then
    "$PY" -X utf8 "$LPY" verify "$E42" --value-lanes $VL --key-lanes $KL >> "$LOG" 2>&1 || { log "STOP: the trained master has nonzero dropped rows"; return 1; }
  fi
  score "sm137-io-$LANES-e42-st-const" "$ARCH" "$E42/expertium/$TRACK/master.pth" "$EROWS" 1 1 w8+s8 w8+s8hx8 || return 1
  memguard_off || return 1
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: $PY -X utf8 $RULE pilot --lanes $LANES"; log "DRYRUN OK"; return 0; fi
  local out r; out=$("$PY" -X utf8 "$RULE" pilot --lanes $LANES 2>&1); r=$?
  log "rule pilot ($LANES) exit $r: $(echo "$out" | tail -1 | cut -c1-600)"
  [ "$r" = 0 ] || { log "STOP: rule pilot failed"; return 1; }
  local d; d=$("$PY" -X utf8 -c "import json; print(json.load(open(r'$S/sm137-pilot-io-$LANES.json'))['decision'])")
  echo "$d" > "$G/verdict-pilot-$LANES"
  log "done (pilot $LANES: $d)"
}

main
