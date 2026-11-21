#!/usr/bin/env bash
# Trained smaller state for Tiny (SM137) (artifacts/small-model/sm137/design.md "Full plan"), prepared 2026-11-21 after the
#   pilot's go (gpu/verdict-pilot-v16k8 = go: instant vs SM135 e42 +0.065% [+0.054, +0.077]). LANES=v16k8 only
#   (w64-c1-d2p1u1-v16k8, 803 B per card and per note at the release-candidate format instead of 1,315 B).
#   io  Instant-Only: the two-pass cool-down of the six-layer Tiny (SM130) / the longer cool-down for Tiny (SM115) (one
#       one_minus_sin decay 7e-4 -> 0 over 2 x 581.9M reviews, fresh Muon, 32k windows, the show-time Instant-Only big model
#       (SM99)'s targets, alpha 0.5) from the pilot's lane epoch 42 (sm137-io-v16k8-e42-st) -> dropped-row check after each
#       pass -> final scoring (student w8+s8 + w8+s8hx8, the SM130 final w8+s8hx8) -> long-card CPU rows (50 users of
#       users-longcard.json, --prior-buckets 0,32,128,300,1000, student + SM130 final at w8+s8hx8, one fresh CPU process
#       per user, 2 at a time, memory guard 5.9 GB) -> sm137_rule.py longcard --chain io -> decide --chain io
#   th  two-head, only after the Instant-Only chain's go (or TH_FORCE=1): the lane cut of SM130's epoch 32
#       (sm137-th-v16k8-e32, made and CPU-proved before launch: check-cpu-sm137-th-v16k8.json) -> GPU scorer smoke vs its
#       CPU rows -> sm137_epoch.py prepare --chain th + pair -> trainer smoke -> lane epoch 33 (SM135's two-head epoch-33
#       step) -> dropped-row check -> w8+s8 -> sm137_rule.py early-th (stop if any head > +1.00% vs SM130's epoch 32 or a
#       non-finite user) -> the same two-pass cool-down with the show-time big model (SM91s2)'s targets (SM124 / SM130)
#       -> final scoring -> long-card -> longcard --chain th -> decide --chain th
#   Rules (pre-registered in design.md, sm137_rule.py, unchanged): main rule = every trained head's 95% CI upper bound of
#   100 x (student w8+s8hx8 / Small w8+s8 - 1) <= +1.00% (io vs SM113 0.275933; th vs SM118 0.300016 / 0.276599), every
#   rule user finite, long-card pass (excess CI upper bound <= +0.25% per head). STOP (the chain and the launcher) on a
#   failed smoke, a GPU vs CPU disagreement above 1e-5, a non-finite loss (a segment that ends without completing), a
#   trained master whose dropped rows are not exactly 0, a pass that ends at the wrong learning rate, a memory-guard kill,
#   or the STOP file.
# Breaks: GPU steps = sm135_block.py (quiet.json scope all / gpu); the long-card CPU step = sm136_block.py --scopes
#   all,cpu,gpu. Training folders run in segments that snapshot-stop <= 2 min before a window and resume after it.
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm137/artifacts/small-model/sm137/launch-sm137-full.sh
#   launch   WMI hidden (HANDOFF.md); log artifacts/small-model/sm137/gpu/launch-sm137-full.log
#   resume   remove gpu/STOP, then the launch command again (finished steps are skipped; a folder resumes from its snapshot)
#   stop     touch C:/Users/jonat/workspace/wt/sm137/artifacts/small-model/sm137/gpu/STOP
#   chains   CHAINS="io th" (default); TH_FORCE=1 runs th without the io gate (needs the main session's OK)
# Do not edit scripts/ expertium/ rwkv/ of the sm137 worktree while it runs (SourceWatch stops the training).
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/sm137
SM=$WT/artifacts/small-model
S=$SM/sm137
G=$S/gpu
mkdir -p "$G" "$S/cpu/longcard"
LOG=$G/launch-sm137-full.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm137-full-dryrun.log
DEC_PY=$WT/scripts/expertium_decay.py
SC=$WT/scripts/small_model_heldout.py
WARM="$PY -X utf8 $WT/scripts/sm77_warm.py"
RULE=$S/sm137_rule.py
EPY=$S/sm137_epoch.py
LPY=$S/sm137_lanes.py
PREP=$S/sm137_prepare.py
PAIR=$S/sm137_pair.py
MZERO=$S/sm137_muon_zero.py
BLOCK=$SM/sm135/sm135_block.py
CBLOCK=$SM/sm136/sm136_block.py
TRACK=E89-F65e-F77a
BASE=w64-c1-d2p1u1
LANES=v16k8; VL=16; KL=8
ARCH=$BASE-$LANES
K=25
W=C:/Users/jonat/workspace/wt
S72=$W/sm72s/artifacts/small-model/sm72s
USERS732=$S72/eval-users-732-fit.json
USMOKE=$S/users-smoke.json
ULONG=$S/users-longcard.json
EROWS=$G/sm137-early-rows.jsonl
OUTROWS=$G/sm137-rows.jsonl
LONG_ROWS=$S/cpu/sm137-longcard-rows.jsonl
SEGS=$G/segments.jsonl
TICKETS=$G/tickets.jsonl
STOPF=$G/STOP
MODE=neighbour
LF="env RWKV_DELETED_CARDS=$MODE"
MIN_EPOCH=${MIN_EPOCH:-70}    # the pilot's lane epoch 42: 59.3 min at 163,493 reviews/s
MIN_DECAY=${MIN_DECAY:-180}   # SM130's six-layer cool-down passes: 147-151 min at 64-66k reviews/s (lane epoch ~9% slower)
MIN_START=30; MIN_SMOKE=15; MIN_SCORE=25; MIN_SCORE_RETRY=20; TL=1800
LC_WORKERS=2
CHAINS=${CHAINS:-"io th"}
TH_FORCE=${TH_FORCE:-0}
RC=w8+s8hx8
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
cblock() { "$PY" -X utf8 "$CBLOCK" --scopes all,cpu,gpu; }
jline() { "$PY" -X utf8 -c "import json, sys; print(json.dumps(dict(zip(sys.argv[1::2], [int(v) if v.lstrip('-').isdigit() else v for v in sys.argv[2::2]]))))" "$@"; }

chain_vars() {  # chain_vars io|th: the per-chain names
  C=$1
  if [ "$C" = th ]; then
    CNAME="two-head"; HEADS="both heads"; XOPT=$S/extra-options-th.json
    START=$SM/sm137-th-$LANES-e33-st; FINAL=sm137-th-$LANES-dec2ep32k-t91s2
    TEACHER=$W/sm91s2/artifacts/small-model/sm91s2-big-showtime/expertium/big-showtime/master.pth
    TEACHER_SHA=67ee8e9b0d0169733242c40646e0d2e5637845a25775762002ffc4e6396f9590
    S130TAG=sm130-th-dec2ep32k-t91s2
  else
    CNAME="Instant-Only"; HEADS="instant head only"; XOPT=$S/extra-options-io.json
    START=$SM/sm137-io-$LANES-e42-st; FINAL=sm137-io-$LANES-dec2ep32k-t99
    TEACHER=$W/sm99/artifacts/small-model/sm99-big-inst-e2/expertium/big-inst/master.pth
    TEACHER_SHA=c49e45d698194da033caa1231f598f875922a4a2a3e921714a3fd433d37e5768
    S130TAG=sm130-io-dec2ep32k-t99
  fi
  S130M=$W/sm130/artifacts/small-model/sm130-$C-dec2ep32k-p2/expertium/$TRACK/master.pth
  P1=$SM/sm137-$C-$LANES-dec2ep32k-p1; P2=$SM/sm137-$C-$LANES-dec2ep32k-p2
  LCD=$S/cpu/longcard/$C
  VFILE=$G/verdict-full-$C
}

wait_free() {  # wait_free MINUTES WHAT: returns once MINUTES fit before the next quiet window (sets FREE = minutes to it)
  local n=0 b
  while true; do
    stopped && return 1
    b=$(${BLOCKFN:-block})
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
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm137/launch-sm137-full.sh" -- "$@"
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
      --expected-minutes "$seg" --max-wait-minutes $(( FREE - 20 > 1 ? FREE - 20 : 1 )) --launched-by "detached launcher sm137/launch-sm137-full.sh" -- \
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

lr_end() {  # lr_end DIR LO HI WHAT
  [ "$DRYRUN" = 1 ] && { log "DRYRUN: lr check of $4: last update lr in [$2, $3] x 7e-4"; return 0; }
  local v; v=$(U="$1/expertium/$TRACK/updates.jsonl" "$PY" -X utf8 -c "
import json, os
last = None
for line in open(os.environ['U'], encoding='utf-8'):
    if line.strip():
        last = line
print(json.loads(last)['lr'] / 7e-4)")
  "$PY" -c "import sys; v = float(sys.argv[1]); sys.exit(0 if float(sys.argv[2]) <= v <= float(sys.argv[3]) else 1)" "$v" "$2" "$3" \
    || { log "STOP: $4 ended at lr $v x peak, not in [$2, $3] (the two-epoch schedule is wrong)"; return 1; }
  log "lr check: $4 ended at $v x peak (expected [$2, $3])"
}

verify() {  # verify DIR WHAT: the dropped W_k / W_v rows of the trained master are exactly 0
  [ "$DRYRUN" = 1 ] && { log "DRYRUN: $PY -X utf8 $LPY verify $1 --value-lanes $VL --key-lanes $KL"; return 0; }
  "$PY" -X utf8 "$LPY" verify "$1" --value-lanes $VL --key-lanes $KL >> "$LOG" 2>&1 || { log "STOP: $2: the trained master has nonzero dropped rows"; return 1; }
}

memguard_on() {  # for the launcher's whole life: CPU-only commands of this run, kill above 5.9 GB (writes STOP)
  [ "$DRYRUN" = 1 ] && return 0
  rm -f "$G/memguard.off"
  local wp; wp=$(cat /proc/$$/winpid 2>/dev/null || echo 0)
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w "$S/memguard.ps1")" -Dir "$(cygpath -w "$G")" -ParentPid "$wp" \
    -Match "sm137_rule.py;sm137_epoch.py;sm137_lanes.py;sm137_prepare.py;sm137_pair.py;sm137_muon_zero.py;sm135_block.py&sm137;sm136_block.py&sm137;check_muon_fresh_decay.py&sm137;check_replay_label_windows.py&sm137;sm77_warm.py&sm137;small_model_heldout.py&--device cpu&longcard" > /dev/null 2>&1 &
  MG=$!
  log "memory guard on (launcher Windows PID $wp, limit 5.9 GB per CPU process)"
}
memguard_off() { [ "$DRYRUN" = 1 ] && return 0; touch "$G/memguard.off"; sleep 12; kill "$MG" 2>/dev/null
  [ -e "$G/memguard-kill" ] && { log "memory guard killed a CPU process ($(tail -1 "$G/memguard-kill"))"; return 1; }; return 0; }

gpu_smoke() {  # gpu_smoke TAG CUT_MASTER CPU_ROWS: the cut on the GPU scorer vs its CPU rows (2 users, w8+s8 + w8+s8hx8), strict 1e-5
  local tag=$1 ck=$2 cpu=$3
  [ -e "$G/smoke-gpu-$tag-ok" ] && return 0
  local rows=$G/smoke-gpu-$tag-rows.jsonl
  [ -f "$cpu" ] || { log "STOP: CPU smoke rows $cpu missing"; return 1; }
  rm -f "$rows"
  q $MIN_SMOKE check "GPU scorer smoke of the $tag lane cut (2 users, w8+s8 + w8+s8hx8, vs CPU rows)" -- \
    env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START HELDOUT_GPU_MEMORY_FRACTION=0.8 "$PY" -X utf8 "$SC" --pool 732 --users-file "$USMOKE" \
    --count 2 --model "smoke=expertium:$ARCH:$ck" --variants w8+s8,w8+s8hx8 --inputs e2s-query --time-limit 600 \
    --split-existing --deleted-cards $MODE --out "$rows" >> "$G/check.out" 2>&1 || { log "STOP: GPU scorer smoke ticket failed ($tag)"; return 1; }
  [ "$DRYRUN" = 1 ] && return 0
  local out r; out=$("$PY" -X utf8 "$RULE" smoke --gpu "$rows" --cpu "$cpu" 2>&1); r=$?
  log "GPU smoke ($tag): $(echo "$out" | tail -1)"
  [ "$r" = 0 ] || { log "STOP: the GPU scorer disagrees with the CPU on the $tag lane cut"; return 1; }
  touch "$G/smoke-gpu-$tag-ok"
}

rule() {  # rule ARGS...: sm137_rule.py, exit 0 required (2 = rows missing)
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: $PY -X utf8 $RULE $*"; return 0; fi
  cpu_step 3 "rule $1" || return 1
  local out r; out=$("$PY" -X utf8 "$RULE" "$@" 2>&1); r=$?
  log "rule $* exit $r: $(echo "$out" | tail -1 | cut -c1-900)"
  [ "$r" = 0 ] || { log "STOP: rule $* exit $r"; return 1; }
}

prepare_p1() {  # the cool-down pass 1 from $START (sm77_warm cooldown + label-window trim + check + 2R schedule), CPU checks
  [ -e "$G/p1-$C-ok" ] && return 0
  if [ "$DRYRUN" = 1 ]; then
    log "DRYRUN: cooldown prepare $P1 ($([ -f "$P1/decay.json" ] && echo 'exists; ')sm77_warm cooldown --epoch $K --start $START --arch $ARCH --fraction 1.0 --extra-options $(basename "$XOPT")) + label-window trim + check + $(basename "$PREP") p1 / check_muon_fresh_decay / $(basename "$MZERO") / $(basename "$PAIR") p1 --chain $C"
    return 0
  fi
  cpu_step 10 "cool-down pass-1 prepare ($C)" || return 1
  local sha; sha=$(sha256sum "$TEACHER" | cut -d' ' -f1); [ "$sha" = "$TEACHER_SHA" ] || { log "STOP: the teacher changed (sha256 $sha)"; return 1; }
  if [ ! -f "$P1/decay.json" ]; then
    { $LF $WARM cooldown --epoch $K --start "$START" --out "${P1#$WT/}" --arch $ARCH --fraction 1.0 --extra-options "$(cat "$XOPT")" >> "$G/prepare.out" 2>&1 \
      && $LF "$PY" -X utf8 "$WT/scripts/check_replay_label_windows.py" --decay "$P1" --trim-unlabelled-tail >> "$G/trim.out" 2>&1 \
      && $LF "$PY" -X utf8 "$WT/scripts/check_replay_label_windows.py" --decay "$P1" --out "$G/check-replay-label-windows-$C-p1.json" > /dev/null \
      && "$PY" -X utf8 "$PREP" p1 --dec "$P1" >> "$LOG" 2>&1; } \
      || { log "STOP: pass-1 prepare / label windows failed ($C)"; rm -rf "$P1"; return 1; }
  fi
  CUDA_VISIBLE_DEVICES= $LF "$PY" -X utf8 "$WT/scripts/check_muon_fresh_decay.py" --decay "$P1" --out "$G/check-muon-fresh-$C-p1.json" \
    | grep -c PASS | sed "s/^/[sm137-muon] $(basename "$P1") PASS lines: /" >> "$LOG"
  [ "${PIPESTATUS[0]}" = 0 ] || { log "STOP: Muon fresh-state CPU check failed ($C)"; return 1; }
  CUDA_VISIBLE_DEVICES= "$PY" -X utf8 "$MZERO" --decay "$P1" --out "$S/cpu/check-muon-zero-$C-p1.json" 2>&1 | tail -1 >> "$LOG"
  [ "${PIPESTATUS[0]}" = 0 ] || { log "STOP: Muon steps do not keep the dropped rows at 0 ($C)"; return 1; }
  "$PY" -X utf8 "$PAIR" p1 --chain $C --lanes $LANES --arm "$P1" --start "$START" >> "$LOG" 2>&1 \
    || { log "STOP: pass 1 ($C) is not the reference pass 1 except the start and the architecture (sm137_pair.py p1)"; return 1; }
  touch "$G/p1-$C-ok"
}

prepare_p2() {
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: $PY -X utf8 $PREP p2 --p1 $P1 --out $P2; $PAIR p2 --arm $P2 --p1 $P1"; return 0; fi
  if [ ! -f "$P2/decay.json" ]; then
    cpu_step 5 "pass 2 prepare ($C)" || return 1
    "$PY" -X utf8 "$PREP" p2 --p1 "$P1" --out "$P2" >> "$LOG" 2>&1 || { log "STOP: pass 2 prepare failed ($C)"; rm -rf "$P2"; return 1; }
  fi
  "$PY" -X utf8 "$PAIR" p2 --arm "$P2" --p1 "$P1" >> "$LOG" 2>&1 || { log "STOP: pass 2 ($C) is not pass 1's continuation"; return 1; }
}

lc_users() {  # "user reviews" of users-longcard.json without complete rows in $LCD, smallest first
  U="$ULONG" D="$LCD" WT="$WT" L1="$FINAL/$RC@e2s-query" L2="$S130TAG/$RC@e2s-query" "$PY" -X utf8 -c "
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.environ['WT'])
from scripts import compress_common as cc
counts = cc.run_info()['review_counts']
want = {os.environ['L1'], os.environ['L2']}
for u in sorted(json.load(open(os.environ['U']))['users'], key=lambda u: (counts[str(u)], u)):
    p, got = Path(os.environ['D']) / f'u-{u}.jsonl', set()
    if p.exists():
        for raw in open(p, 'rb'):
            try:
                r = json.loads(raw.decode('utf-8', errors='replace'))
            except ValueError:
                continue
            if 'error' not in r and ('benchmark' in r or 'nonfinite' in r) and isinstance(r.get('prior_buckets', {}), dict):
                got.add(r.get('label'))
    if not want <= got:
        print(u, counts[str(u)])"
}

lc_one() {  # lc_one USER: the student and the SM130 final at w8+s8hx8 with --prior-buckets, CPU, one fresh process
  local u=$1
  echo "{\"users\": [$u]}" > "$LCD/users-$u.json"
  rm -f "$LCD/u-$u.jsonl" "$LCD/u-$u.part.jsonl"
  env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START CUDA_VISIBLE_DEVICES= "$PY" -X utf8 -W ignore "$SC" --device cpu --inputs e2s-query \
    --split-existing --deleted-cards $MODE --nonfinite-rows --prior-buckets 0,32,128,300,1000 --pool 732 --users-file "$LCD/users-$u.json" \
    --model "$FINAL=expertium:$ARCH:$P2/expertium/$TRACK/master.pth" --model "$S130TAG=expertium:$BASE:$S130M" \
    --only "$FINAL/$RC" --only "$S130TAG/$RC" --time-limit 1000000 --out "$LCD/u-$u.part.jsonl" >> "$LCD/score-$u.out" 2>&1
  local rc=$?
  if [ $rc = 0 ]; then mv "$LCD/u-$u.part.jsonl" "$LCD/u-$u.jsonl"; log "long-card $C user $u done: $(grep -a '^\[small-model\] 1/1' "$LCD/score-$u.out" | tail -1 | cut -c1-120)"
  else echo "$C $u exit $rc" >> "$S/cpu/longcard/failed.txt"; log "long-card $C user $u failed (exit $rc, see $LCD/score-$u.out)"; fi
  return $rc
}

lc_slots() {  # lc_slots N: wait until fewer than N long-card processes run (N = 1: all done); never waits on the memory guard
  local p alive
  while true; do
    alive=()
    for p in "${LC_PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive+=("$p"); done
    LC_PIDS=("${alive[@]}")
    [ ${#LC_PIDS[@]} -lt "$1" ] && return 0
    sleep 10
  done
}

longcard() {  # the long-card CPU rows of this chain (2 processes at a time, < 5.9 GB each), merged into $LONG_ROWS
  mkdir -p "$LCD"
  if [ "$DRYRUN" = 1 ]; then
    log "DRYRUN: long-card CPU rows ($C): $(lc_users | wc -l) of 50 users to score, $LC_WORKERS at a time, e.g."
    echo "    cmd: $SC --device cpu --inputs e2s-query --split-existing --deleted-cards $MODE --nonfinite-rows --prior-buckets 0,32,128,300,1000 --pool 732 --users-file $LCD/users-U.json --model $FINAL=expertium:$ARCH:$P2/expertium/$TRACK/master.pth --model $S130TAG=expertium:$BASE:$S130M --only $FINAL/$RC --only $S130TAG/$RC --out $LCD/u-U.part.jsonl" >> "$LOG"
    return 0
  fi
  [ -f "$S130M" ] || { log "STOP: SM130 final $S130M missing"; return 1; }
  local u n need todo
  LC_PIDS=()
  todo=$(lc_users | tr -d '\r')   # Windows Python prints CRLF; a CR in $n broke the arithmetic below (11-21 08:48)
  log "long-card CPU rows ($CNAME): $(echo "$todo" | grep -c .) of 50 users to score ($LC_WORKERS processes at a time, $(free_mib) MiB RAM free)"
  while read -r u n; do
    [ -z "$u" ] && continue
    lc_slots $LC_WORKERS
    need=$(( n * 3 / 60000 + 5 ))   # ~3 ms per review for the two models on 4 threads, with margin
    BLOCKFN=cblock wait_free "$need" "long-card user $u ($C)" || { lc_slots 1; return 1; }
    ram_gate "long-card user $u ($C)"
    stopped && { lc_slots 1; return 1; }
    lc_one "$u" &
    LC_PIDS+=($!)
  done <<< "$todo"
  lc_slots 1
  [ -e "$G/memguard-kill" ] && { log "STOP: the memory guard killed a long-card process ($(tail -1 "$G/memguard-kill"))"; return 1; }
  todo=$(lc_users | grep -c .)
  [ "$todo" = 0 ] || { log "STOP: long-card rows ($C) incomplete: $todo users left (see $S/cpu/longcard/failed.txt)"; return 1; }
  cat "$S"/cpu/longcard/*/u-*.jsonl > "$LONG_ROWS.tmp" && mv "$LONG_ROWS.tmp" "$LONG_ROWS"
  log "long-card rows merged: $(wc -l < "$LONG_ROWS") rows in $LONG_ROWS"
}

cooldown_chain() {  # cooldown_chain: the two passes from $START, final scoring, long-card, rules -> VERDICT
  prepare_p1 || return 1
  smoke "$P1" "$C-p1" "$CNAME cool-down pass 1 (fresh Muon, the teacher's show-time targets, $ARCH, from $(basename "$START"))" || return 1
  run_folder "$P1" "$CNAME cool-down pass 1 of 2 (32k Muon, 7e-4 -> 0.29 x peak, $ARCH, from $(basename "$START"), $HEADS)" $MIN_DECAY || return 1
  lr_end "$P1" 0.2925 0.2935 "$CNAME pass 1" || return 1
  verify "$P1" "$CNAME pass 1" || return 1
  prepare_p2 || return 1
  smoke "$P2" "$C-p2" "$CNAME cool-down pass 2 (pass 1's Muon state loaded)" || return 1
  run_folder "$P2" "$CNAME cool-down pass 2 of 2 (0.29 x peak -> 0, new order, $ARCH, $HEADS)" $MIN_DECAY || return 1
  lr_end "$P2" 0 0.0005 "$CNAME pass 2" || return 1
  verify "$P2" "$CNAME pass 2" || return 1
  log "final scoring ($CNAME): student w8+s8 + $RC (--nonfinite-rows), then the SM130 final at $RC"
  score "$FINAL" $ARCH "$P2/expertium/$TRACK/master.pth" "$OUTROWS" 1 1 w8+s8 $RC || return 1
  score "$S130TAG" $BASE "$S130M" "$OUTROWS" 0 1 $RC || return 1
  longcard || return 1
  rule longcard --chain $C --lanes $LANES || return 1
  rule decide --chain $C --lanes $LANES || return 1
  [ "$DRYRUN" = 1 ] && { VERDICT=dry; return 0; }
  VERDICT=$("$PY" -X utf8 -c "
import json
d = json.load(open(r'$S/sm137-decision-$C-$LANES.json'))
lc = json.load(open(r'$S/sm137-longcard-$C-$LANES.json'))
v = d['vs_small_release_format']
print(d['verdict'] + ' (vs Small at $RC: ' + (', '.join(f\"{h} {x['value']:+.3f}% [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]\" for h, x in v.items()) or 'non-finite')
      + '; long-card ' + lc['verdict'] + ' ' + ', '.join(f\"{h} excess {x['excess']:+.3f}% [.., {x['excess_ci'][1]:+.3f}]\" for h, x in lc['heads'].items()) + ')')")
  return 0
}

io_chain() {
  VERDICT=""
  cooldown_chain || return 1
}

th_chain() {
  local cut=$SM/sm137-th-$LANES-e32 E=$SM/sm137-th-$LANES-e33-st
  local check=$S/check-cpu-sm137-th-$LANES.json cpu=$S/cpu/smoke-cpu-th-$LANES-rows.jsonl d
  VERDICT=""
  for f in "$cut/master.pth" "$cut/optimizer.pth" "$cut/extend.json" "$check" "$cpu"; do
    [ -f "$f" ] || { log "STOP: missing $f (the two-head cut and its CPU proofs, HANDOFF.md)"; return 1; }
  done
  CK="$check" CM="$cut/master.pth" "$PY" -X utf8 -c "
import hashlib, json, os, sys
r = json.load(open(os.environ['CK']))
sha = hashlib.sha256(open(os.environ['CM'], 'rb').read()).hexdigest()
sys.exit(0 if r.get('ok') and r.get('cut_master_sha256') == sha else 1)" || { log "STOP: the CPU proofs report $check is not ALL OK for this cut"; return 1; }
  if [ ! -e "$S/sm137-early-th-$LANES.json" ]; then
    gpu_smoke "th-$LANES" "$cut/master.pth" "$cpu" || return 1
    if [ ! -f "$E/decay.json" ]; then
      if [ "$DRYRUN" = 1 ]; then log "DRYRUN: $PY -X utf8 $EPY prepare --chain th --lanes $LANES"
      else cpu_step 3 "two-head epoch 33 prepare" || return 1
        "$PY" -X utf8 "$EPY" prepare --chain th --lanes $LANES >> "$LOG" 2>&1 || { log "STOP: prepare $E failed"; return 1; }
      fi
    fi
    "$PY" -X utf8 "$EPY" pair --chain th --lanes $LANES >> "$LOG" 2>&1 || { log "STOP: $E is not SM135's two-head epoch-33 step with the lane architecture"; return 1; }
    smoke "$E" "th-$LANES" "two-head lane epoch 33 ($ARCH, from the lane cut of SM130's epoch 32)" || return 1
    run_folder "$E" "two-head lane epoch 33 ($ARCH = SM135's two-head epoch-33 step with the card / note state cut to $VL x $KL lanes per head: 8k final tier, the run's order rule, 2048 x 16 lanes, leak-free replay KD, AdamW 7e-4, both heads, show-time inputs)" $MIN_EPOCH || return 1
    verify "$E" "two-head lane epoch 33" || return 1
    score "sm137-th-$LANES-e33-st-const" $ARCH "$E/expertium/$TRACK/master.pth" "$EROWS" 1 1 w8+s8 || return 1
    rule early-th --lanes $LANES || return 1
  fi
  [ "$DRYRUN" = 1 ] && d=cooldown || d=$("$PY" -X utf8 -c "import json; print(json.load(open(r'$S/sm137-early-th-$LANES.json'))['decision'])")
  if [ "$d" != cooldown ]; then
    VERDICT="no go (early-th stop after the lane epoch 33: $("$PY" -X utf8 -c "import json; d = json.load(open(r'$S/sm137-early-th-$LANES.json')); print('non-finite' if not d['finite'] else ', '.join(f\"{h} {x['value']:+.3f}% vs SM130 e32\" for h, x in d['vs_sm130_e32'].items()))"))"
    log "EARLY STOP (two-head): $VERDICT; no cool-down"
    return 0
  fi
  log "early check (two-head): cooldown"
  cooldown_chain || return 1
}

run_chain() {  # run_chain io|th -> VERDICT_<C>
  chain_vars "$1"
  [ -e "$VFILE" ] && { eval "VERDICT_$C=\"$(cat "$VFILE")\""; log "$CNAME chain already decided: $(cat "$VFILE")"; return 0; }
  if [ "$C" = th ] && [ "$TH_FORCE" != 1 ]; then
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN: the two-head chain runs only if $G/verdict-full-io starts with 'go' (or TH_FORCE=1)"
    elif ! grep -q '^go' "$G/verdict-full-io" 2>/dev/null; then
      VERDICT="not run (the Instant-Only chain did not go: $(cat "$G/verdict-full-io" 2>/dev/null || echo 'no verdict'))"
      log "two-head chain skipped: $VERDICT"; echo "$VERDICT" > "$VFILE"; VERDICT_th=$VERDICT; return 0
    fi
  fi
  log "$CNAME chain: start ($ARCH, from $(basename "$START"))"
  if [ "$C" = th ]; then th_chain || return 1; else io_chain || return 1; fi
  [ "$DRYRUN" = 1 ] || echo "$VERDICT" > "$VFILE"
  log "$CNAME chain verdict: $VERDICT"
  eval "VERDICT_$C=\"$VERDICT\""
  return 0
}

preconditions() {
  [ "$(cat "$G/verdict-pilot-$LANES" 2>/dev/null)" = go ] || { log "STOP: the pilot verdict $G/verdict-pilot-$LANES is not go"; return 1; }
  for f in "$BLOCK" "$CBLOCK" "$RULE" "$EPY" "$LPY" "$PREP" "$PAIR" "$MZERO" "$S/extra-options-th.json" "$S/extra-options-io.json" \
           "$S/memguard.ps1" "$USERS732" "$USMOKE" "$ULONG" "$SM/sm137-io-$LANES-e42-st/expertium/$TRACK/master.pth"; do
    [ -f "$f" ] || { log "STOP: missing $f"; return 1; }
  done
  if [ "$DRYRUN" != 1 ] && [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    log "STOP: uncommitted source changes in scripts/ expertium/ rwkv/ (SourceWatch refuses them)"; return 1; fi
  "$PY" -X utf8 "$BLOCK" selftest >> "$LOG" 2>&1 || { log "STOP: block selftest failed"; return 1; }
  "$PY" -X utf8 "$RULE" selftest >> "$LOG" 2>&1 || { log "STOP: rule selftest failed"; return 1; }
  "$PY" -X utf8 "$PAIR" selfcheck >> "$LOG" 2>&1 || { log "STOP: pair selfcheck failed"; return 1; }
  return 0
}

cd "$WT" || exit 1
log "started full plan: $(git -C "$WT" log --oneline -1 | cut -c1-90) (DRYRUN=$DRYRUN; chains: $CHAINS; TH_FORCE=$TH_FORCE; arch $ARCH; dirty source files: $(git -C "$WT" status --porcelain -- scripts expertium rwkv | wc -l))"
rc=0
stopped && rc=1
[ $rc -eq 0 ] && memguard_on
[ $rc -eq 0 ] && { cpu_step 5 "preparation" || rc=1; }
[ $rc -eq 0 ] && { preconditions || rc=1; }
VERDICT_io="not run"; VERDICT_th="not run"
for c in $CHAINS; do
  [ $rc -eq 0 ] || break
  run_chain "$c" || rc=1
done
[ -n "${MG:-}" ] && { memguard_off || rc=1; }
if [ $rc -eq 0 ] && [ "$DRYRUN" != 1 ]; then   # a shell error can abort the chain loop without a STOP (11-21 08:48): every chain needs its verdict file
  for c in $CHAINS; do [ -s "$G/verdict-full-$c" ] || { log "STOP: no verdict file for the $c chain (the chain was aborted; see $G/launch-sm137-full.nohup)"; rc=1; }; done
fi
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; else log "done (full: io $VERDICT_io, th $VERDICT_th)"; fi
else
  log "STOP: ended with exit $rc"
fi
exit $rc
