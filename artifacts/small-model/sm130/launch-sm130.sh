#!/usr/bin/env bash
# Six-layer Tiny (SM130) (artifacts/small-model/sm130/design.md), designed 2026-11-18:
#   the context-stream pruning (SM15)'s depth cut applied to Tiny: w64-c1 -> w64-c1-d2p1u1 (card 1, deck 2, note 1,
#   preset 1, user 1 layers; 185,057 parameters, 6 layer steps instead of 12), both chains, one after the other:
#     th  two-head: cut of the Tiny two-head model (SM106)'s epoch 29 -> recovery epochs 30-32 (SM106's epoch step) ->
#         the show-time two-head cool-down for Tiny (SM124)'s two-pass cool-down (the show-time big model (SM91s2)'s targets)
#     io  Instant-Only: cut of the Tiny Instant-Only model (SM101)'s epoch 38 -> epochs 39-41 (SM101's step) -> the longer
#         cool-down for Tiny (SM115)'s two-pass cool-down (the show-time Instant-Only big model (SM99)'s targets)
#   early stops (sm130_rule.py early): after the first recovery epoch, stop the chain if a trained head is > +1.50% vs the
#   uncut start epoch; after the last, > +1.20%. Final rule (sm130_rule.py decide): every trained head's 95% CI upper bound
#   of 100 x (six-layer / Small - 1) <= +1.00% (published w8+s8; compact w8+s4c8nhx8 too if the compact state for Tiny
#   (SM129) went go for that model).
# Waits for SM129's final log line (done / STOP:) before its first GPU ticket; the CPU proofs run at 01:00 beside SM129.
# Breaks: sm130_block.py = quiet.json (scope all/gpu) + the daily 20:00-01:00 (Sunday from 16:00); re-read at every check.
#   Tickets must end before the next break; a training folder runs in segments that snapshot-stop <= 2 min before a break
#   and resume after it (snapshots every 400 users).
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm130/artifacts/small-model/sm130/launch-sm130.sh
#   launch   WMI hidden (HANDOFF.md); log artifacts/small-model/sm130/gpu/launch-sm130.log
#   resume   the launch command again (finished steps are skipped; a folder resumes from its last snapshot)
#   stop     touch C:/Users/jonat/workspace/wt/sm130/artifacts/small-model/sm130/gpu/STOP
#   one chain only: CHAINS="io" (default "th io")
# Do not edit scripts/ expertium/ rwkv/ of the sm130 worktree while it runs (SourceWatch stops the training).
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/sm130
SM=$WT/artifacts/small-model
S=$SM/sm130
G=$S/gpu
mkdir -p "$G"
LOG=$G/launch-sm130.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm130-dryrun.log
DEC_PY=$WT/scripts/expertium_decay.py
SC=$WT/scripts/small_model_heldout.py
WARM="$PY -X utf8 $WT/scripts/sm77_warm.py"
RULE=$S/sm130_rule.py
PAIR=$S/sm130_pair.py
PREP=$S/sm130_prepare.py
EPY=$S/sm130_epoch.py
CUT=$S/sm130_cut.py
TRACK=E89-F65e-F77a
ARCH=w64-c1-d2p1u1
K=25
W=C:/Users/jonat/workspace/wt
S72=$W/sm72s/artifacts/small-model/sm72s
USERS732=$S72/eval-users-732-fit.json
EROWS=$G/sm130-early-rows.jsonl
OUTROWS=$G/sm130-rows.jsonl
SEGS=$G/segments.jsonl
TICKETS=$G/tickets.jsonl
STOPF=$G/STOP
SM129LOG=$W/sm129/artifacts/small-model/sm129/gpu/launch-sm129.log
MODE=neighbour
LF="env RWKV_DELETED_CARDS=$MODE"
MIN_EPOCH=${MIN_EPOCH:-110}   # SM106's w64-c1 epochs: 106-107 min at ~91k reviews/s; six layers should be faster
MIN_DECAY=${MIN_DECAY:-480}   # one cool-down pass: SM124's w64-c1 passes 336-463 min; six layers should be faster
MIN_START=30; MIN_SMOKE=15; MIN_SCORE=25; MIN_SCORE_RETRY=20; TL=1800
CHAINS=${CHAINS:-"th io"}
NAME="Six-layer Tiny (SM130)"
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START SM11_INPUTS   # caps 0.8
log() { echo "[sm130 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
stopped() { [ -e "$STOPF" ] && { log "STOP: STOP file present"; return 0; }; return 1; }
nap() { [ "$DRYRUN" = 1 ] && return 0; local s=$1; [ "$s" -gt 600 ] && s=600; sleep "$s"; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'; }
free_mib() { awk '/MemFree/ {print int($2 / 1024)}' /proc/meminfo 2>/dev/null || echo 99999; }
block() { "$PY" -X utf8 "$S/sm130_block.py"; }
jline() { "$PY" -X utf8 -c "import json, sys; print(json.dumps(dict(zip(sys.argv[1::2], [int(v) if v.lstrip('-').isdigit() else v for v in sys.argv[2::2]]))))" "$@"; }

chain_vars() {  # chain_vars th|io: the per-chain names
  C=$1
  if [ "$C" = th ]; then
    FIRST=30; LAST=32; SRC="$W/sm106/artifacts/small-model/sm106-c1-e29-st"; SRC_LBL=sm106-c1-e29-st-const
    SRC_SHA=b3a694ca14cbd7a2291c7400203d5f9774bbdabc5d3e572a8408a196e5cb7be9
    XOPT=$S/extra-options-th.json; FINAL=sm130-th-dec2ep32k-t91s2; CNAME="two-head"; SM129M=two-head
    TEACHER=$W/sm91s2/artifacts/small-model/sm91s2-big-showtime/expertium/big-showtime/master.pth
    TEACHER_SHA=67ee8e9b0d0169733242c40646e0d2e5637845a25775762002ffc4e6396f9590
    REPLAY=C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/sm91s2-big-showtime-final8k-neighbour-e2s
    RUN8K=$W/sm55/artifacts/small-model/sm55-final8k-run.json
    HEADS="both heads"
  else
    FIRST=39; LAST=41; SRC="$W/sm101/artifacts/small-model/sm101-c1-e38-st"; SRC_LBL=sm101-c1-e38-st-const
    SRC_SHA=e6ea69124af00a3dcf2325047ae34580aaf96051d963ff6130dd06bde02d396c
    XOPT=$S/extra-options-io.json; FINAL=sm130-io-dec2ep32k-t99; CNAME="Instant-Only"; SM129M=instant-only
    TEACHER=$W/sm99/artifacts/small-model/sm99-big-inst-e2/expertium/big-inst/master.pth
    TEACHER_SHA=c49e45d698194da033caa1231f598f875922a4a2a3e921714a3fd433d37e5768
    REPLAY=C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/sm99-big-inst-final8k-neighbour-e2s
    RUN8K=$W/sm55/artifacts/small-model/sm55-final8k-run.json
    HEADS="instant head only"
  fi
  P1=$SM/sm130-$C-dec2ep32k-p1; P2=$SM/sm130-$C-dec2ep32k-p2
}
efold() { echo "$SM/sm130-$C-e$1-st"; }

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

q() {  # q MINUTES KIND PURPOSE -- CMD...   one ticket that must end before the next break
  local minutes=$1 kind=$2 purpose=$3 rc t0 mx base wpid; shift 4
  while true; do
    wait_free "$minutes" "$purpose" || return 1
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $NAME: $purpose (expected $minutes min)"; echo "    cmd: $*" >> "$LOG"; return 0; fi
    local maxwait=$(( FREE - minutes - 3 )); [ "$maxwait" -lt 1 ] && maxwait=1
    log "enqueue: $purpose (expected $minutes min, max wait $maxwait min, $FREE min to the next break)"
    t0=$(date +%s); base=$(gpu_used); rm -f "$G/.gpu-max"
    ( m=0; while true; do u=$(gpu_used); [ -n "$u" ] && [ "$u" -gt "$m" ] && { m=$u; echo "$m" > "$G/.gpu-max"; }; sleep 30; done ) &
    wpid=$!
    "$PY" "$QUEUE" run --owner sm130 --purpose "$NAME: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm130/launch-sm130.sh" -- "$@"
    rc=$?; kill "$wpid" 2>/dev/null; wait "$wpid" 2>/dev/null; mx=$(cat "$G/.gpu-max" 2>/dev/null || echo 0)
    log "exit $rc: $purpose (nvidia-smi max $mx MiB, baseline $base MiB)"
    [ "$rc" = 75 ] && { log "queue wait would reach the next break; re-planning"; sleep 60; continue; }
    jline kind "$kind" purpose "$purpose" exit "$rc" minutes "$(( ($(date +%s) - t0 + 30) / 60 ))" nvidia_smi_max_mib "${mx:-0}" nvidia_smi_baseline_mib "${base:-0}" >> "$TICKETS"
    return $rc
  done
}

min_users() {  # min_users ROWS LABEL...
  F="$1" "$PY" -X utf8 -c "
import json, os, sys
from pathlib import Path
p = Path(os.environ['F'])
got = {}
if p.exists():
    for r in map(json.loads, open(p, encoding='utf-8')):
        if 'benchmark' in r:
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

run_folder() {  # run_folder DIR WHAT SEG_MAX: expertium_decay.py run in segments to the next break; resumable from snapshots
  local D=$1 what=$2 segmax=$3 rc qpid stop_sent b need t0 base mx u why
  while ! done_in "$D"; do
    stopped && return 1
    need=15; [ ! -d "$D/expertium" ] && [ ! -d "$D/attempts" ] && need=$MIN_START
    wait_free "$need" "$what" || return 1
    ram_gate "$what" || return 1
    wait_free "$need" "$what" || return 1
    local seg=$(( FREE - 5 )); [ "$seg" -gt "$segmax" ] && seg=$segmax
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $NAME: $what (expected $seg min of ~$segmax; snapshot stop before a break)"
      echo "    cmd: $LF $PY -X utf8 $DEC_PY run --out $D" >> "$LOG"; return 0; fi
    rm -f "$D/control/STOP"
    log "enqueue: $what (expected $seg min of ~$segmax, $FREE min to the next break, $(free_mib) MiB RAM free)"
    t0=$(date +%s); base=$(gpu_used); mx=0
    "$PY" "$QUEUE" run --owner sm130 --purpose "$NAME: $what (segment to the next break or the end)" --claim gpu \
      --expected-minutes "$seg" --max-wait-minutes $(( FREE - 20 > 1 ? FREE - 20 : 1 )) --launched-by "detached launcher sm130/launch-sm130.sh" -- \
      $LF "$PY" -X utf8 "$DEC_PY" run --out "$D" >> "$G/train.out" 2>&1 &
    qpid=$!
    stop_sent=0
    while kill -0 "$qpid" 2>/dev/null; do
      u=$(gpu_used); [ -n "$u" ] && [ "$u" -gt "$mx" ] && mx=$u
      b=$(block); why=""
      [ -e "$STOPF" ] && why="STOP file"
      { [[ "$b" == NOW* ]] || { [[ "$b" == NEXT* ]] && [ "${b#NEXT }" -le 2 ]; }; } && why="break ($b)"
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

score() {  # score TAG ARCH CKPT ROWS STRICT VARIANT...: 723 users of the 732 set, show-time inputs, all variants in one pass
  local tag=$1 arch=$2 ck=$3 rows=$4 strict=$5 cap=0.8 pass have prev=-1 minutes; shift 5
  local only=() lbls=() v
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
      --deleted-cards $MODE --out "$rows" >> "$G/score.out" 2>&1 || { log "STOP: scoring ticket failed ($tag)"; return 1; }
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

memguard_on() {
  rm -f "$G/memguard.off"
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w "$S/memguard.ps1")" -Dir "$(cygpath -w "$G")" \
    -Match "sm130_cut.py;sm130_timing.py;sm77_probe.py;--device cpu&sm130" > /dev/null 2>&1 &
  MG=$!
}
memguard_off() { touch "$G/memguard.off"; sleep 12; kill "$MG" 2>/dev/null
  [ -e "$G/memguard-kill" ] && { log "STOP: memory guard killed a CPU process ($(tail -1 "$G/memguard-kill"))"; return 1; }; return 0; }

cpu_checks() {  # sm130_cut.py cpu-check (C0-C2, T1, S1, M1) under the memory guard; runs at 01:00 beside SM129
  [ -e "$G/cpu-ok" ] && return 0
  [ "$DRYRUN" = 1 ] && { log "DRYRUN: CUDA_VISIBLE_DEVICES= $PY -X utf8 $CUT cpu-check (under memguard.ps1)"; return 0; }
  cpu_step 30 "CPU proofs" || return 1
  memguard_on
  CUDA_VISIBLE_DEVICES= "$PY" -X utf8 "$CUT" cpu-check >> "$G/check-cpu.out" 2>&1; local rc=$?
  memguard_off || return 1
  [ "$rc" = 0 ] || { log "STOP: CPU proofs failed (see $G/check-cpu.out, $S/check-cpu-sm130.json)"; return 1; }
  log "CPU proofs ALL OK ($(grep -c '^PASS' "$G/check-cpu.out") PASS lines, $("$PY" -c "import json; print(json.load(open(r'$S/check-cpu-sm130.json'))['seconds'])") s)"
  touch "$G/cpu-ok"
}

wait_sm129() {  # SM129 goes first on the GPU: wait for its final line (done / STOP:), at most 6 h
  [ "$DRYRUN" = 1 ] && { log "DRYRUN: wait for SM129's done / STOP: line in $SM129LOG"; return 0; }
  local n=0
  while [ $n -lt 72 ]; do
    stopped && return 1
    if [ -f "$SM129LOG" ] && grep -qE '\] (done \(|STOP:)' "$SM129LOG"; then
      log "SM129 finished: $(grep -E '\] (done \(|STOP:)' "$SM129LOG" | tail -1 | cut -c1-200)"; return 0; fi
    [ $((n % 6)) = 0 ] && log "waiting for SM129's final line"; n=$((n + 1)); sleep 300
  done
  log "SM129 has no final line after 6 h; going on (the queue serialises the GPU)"
  return 0
}

score_ref() {  # the uncut start epoch of chain C (constant LR, w64-c1), the early checks' reference
  [ -e "$G/ref-$C-done" ] && return 0
  score "$SRC_LBL" w64-c1 "$SRC/expertium/$TRACK/master.pth" "$EROWS" 1 w8+s8 || return 1
  [ "$DRYRUN" = 1 ] || touch "$G/ref-$C-done"
}

rule_early() {  # rule_early N -> sets EARLY=continue|stop
  EARLY=continue
  [ "$DRYRUN" = 1 ] && { log "DRYRUN: $PY -X utf8 $RULE early --chain $C --epoch $1"; return 0; }
  cpu_step 3 "rule" || return 1
  local out r; out=$("$PY" -X utf8 "$RULE" early --chain $C --epoch $1 2>&1); r=$?
  log "rule early ($C, after epoch $1) exit $r: $(echo "$out" | tail -1 | cut -c1-600)"
  [ "$r" = 0 ] || return 1
  EARLY=$("$PY" -X utf8 -c "import json; print(json.load(open(r'$S/sm130-early-$C-e$1.json'))['decision'])")
}

prepare_p1() {
  [ -e "$G/p1-$C-ok" ] && return 0
  local E=$(efold $LAST)
  if [ "$DRYRUN" = 1 ]; then
    log "DRYRUN: cooldown prepare $P1 (sm77_warm cooldown --epoch $K --start $E --arch $ARCH --fraction 1.0 --extra-options $(basename "$XOPT")) + label-window trim + check + $PREP p1 / Muon fresh / $PAIR p1 --chain $C"
    return 0
  fi
  cpu_step 10 "cool-down pass-1 prepare ($C)" || return 1
  local sha; sha=$(sha256sum "$TEACHER" | cut -d' ' -f1); [ "$sha" = "$TEACHER_SHA" ] || { log "STOP: the teacher changed (sha256 $sha)"; return 1; }
  if [ ! -f "$P1/decay.json" ]; then
    { $LF $WARM cooldown --epoch $K --start "$E" --out "${P1#$WT/}" --arch $ARCH --fraction 1.0 --extra-options "$(cat "$XOPT")" >> "$G/prepare.out" 2>&1 \
      && $LF "$PY" -X utf8 "$WT/scripts/check_replay_label_windows.py" --decay "$P1" --trim-unlabelled-tail >> "$G/trim.out" 2>&1 \
      && $LF "$PY" -X utf8 "$WT/scripts/check_replay_label_windows.py" --decay "$P1" --out "$G/check-replay-label-windows-$C-p1.json" > /dev/null \
      && "$PY" -X utf8 "$PREP" p1 --dec "$P1" >> "$LOG" 2>&1; } \
      || { log "STOP: pass-1 prepare / label windows failed ($C)"; rm -rf "$P1"; return 1; }
  fi
  CUDA_VISIBLE_DEVICES= $LF "$PY" -X utf8 "$WT/scripts/check_muon_fresh_decay.py" --decay "$P1" --out "$G/check-muon-fresh-$C-p1.json" \
    | grep -c PASS | sed "s/^/[sm130-muon] $(basename "$P1") PASS lines: /" >> "$LOG"
  [ "${PIPESTATUS[0]}" = 0 ] || { log "STOP: Muon fresh-state CPU check failed ($C)"; return 1; }
  "$PY" -X utf8 "$PAIR" p1 --chain $C --arm "$P1" --start "$E" >> "$LOG" 2>&1 \
    || { log "STOP: pass 1 ($C) is not the reference pass 1 except the start and the architecture (sm130_pair.py p1)"; return 1; }
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

compact_go() {  # 0 if SM129's compact arm passed for this chain's model
  "$PY" -X utf8 -c "
import json, sys
try:
    d = json.load(open(r'$W/sm129/artifacts/small-model/sm129/sm129-decision.json'))
    sys.exit(0 if d['arms']['compact']['pass']['$SM129M'] else 1)
except Exception:
    sys.exit(1)"; }

run_chain() {  # run_chain th|io -> sets VERDICT_<C>
  chain_vars "$1"
  local v="" n D E
  [ -e "$G/verdict-$C" ] && { eval "VERDICT_$C=\"$(cat "$G/verdict-$C")\""; log "$CNAME chain already decided: $(cat "$G/verdict-$C")"; return 0; }
  score_ref || return 1
  EP="(w64-c1-d2p1u1, the chain's own constant-LR epoch step: 8k final tier, the run's order rule, 2048 x 16 lanes, leak-free replay KD, AdamW 7e-4, $HEADS, show-time inputs)"
  for n in $(seq $FIRST $LAST); do
    D=$(efold $n)
    if [ ! -f "$D/decay.json" ]; then
      if [ "$DRYRUN" = 1 ]; then
        log "DRYRUN: epoch $n prepared after epoch $((n - 1)): $PY -X utf8 $EPY prepare --chain $C --epoch $n, then pair"
        log "DRYRUN enqueue: $NAME: $CNAME epoch $n $EP (expected ~$MIN_EPOCH min)"
        [ $n = $LAST ] && { rule_early $n; }
        continue
      fi
      cpu_step 3 "epoch $n prepare ($C)" || return 1
      "$PY" -X utf8 "$EPY" prepare --chain $C --epoch $n >> "$LOG" 2>&1 || { log "STOP: epoch $n prepare failed ($C)"; return 1; }
    fi
    "$PY" -X utf8 "$EPY" pair --chain $C --epoch $n >> "$LOG" 2>&1 || { log "STOP: $CNAME epoch $n is not the chain's step"; return 1; }
    [ $n = $FIRST ] && { smoke "$D" "$C-e$n" "$CNAME epoch $n (six-layer from the cut, constant LR)" || return 1; }
    run_folder "$D" "$CNAME epoch $n $EP" $MIN_EPOCH || return 1
    if [ $n = $FIRST ] || [ $n = $LAST ]; then
      if [ ! -e "$S/sm130-early-$C-e$n.json" ]; then
        score "sm130-$C-e$n-st-const" $ARCH "$D/expertium/$TRACK/master.pth" "$EROWS" 1 w8+s8 || return 1
      fi
      rule_early $n || return 1
      if [ "$EARLY" != continue ]; then
        v="no go (early stop after epoch $n: $(grep -m1 Decision -A0 "$S/sm130-early-$C-e$n.md" | cut -c1-120))"
        log "EARLY STOP ($CNAME, after epoch $n): see $S/sm130-early-$C-e$n.md; this chain ends"
        echo "$v" > "$G/verdict-$C"; eval "VERDICT_$C=\"$v\""; return 0
      fi
      log "early check after epoch $n ($CNAME): continue"
    fi
  done
  prepare_p1 || return 1
  smoke "$P1" "$C-p1" "$CNAME cool-down pass 1 (fresh Muon, the teacher's show-time targets, six-layer)" || return 1
  run_folder "$P1" "$CNAME cool-down pass 1 of 2 (32k Muon, 7e-4 -> 0.29 x peak, six-layer)" $MIN_DECAY || return 1
  lr_end "$P1" 0.2925 0.2935 "$CNAME pass 1" || return 1
  prepare_p2 || return 1
  smoke "$P2" "$C-p2" "$CNAME cool-down pass 2 (pass 1's Muon state loaded)" || return 1
  run_folder "$P2" "$CNAME cool-down pass 2 of 2 (0.29 x peak -> 0, new order, six-layer)" $MIN_DECAY || return 1
  lr_end "$P2" 0 0.0005 "$CNAME pass 2" || return 1
  local variants="w8+s8"
  if [ "$DRYRUN" = 1 ] || compact_go; then variants="w8+s8 w8+s4c8nhx8"; fi
  log "final scoring ($CNAME): $variants (compact only if SM129's compact arm passed for $SM129M)"
  score "$FINAL" $ARCH "$P2/expertium/$TRACK/master.pth" "$OUTROWS" 1 $variants || return 1
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: $PY -X utf8 $RULE decide --chain $C"; eval "VERDICT_$C=dry"; return 0; fi
  cpu_step 3 "rule" || return 1
  local out r; out=$("$PY" -X utf8 "$RULE" decide --chain $C 2>&1); r=$?
  log "rule decide ($CNAME) exit $r: $(echo "$out" | tail -1 | cut -c1-700)"
  [ "$r" = 0 ] || return 1
  v=$("$PY" -X utf8 -c "import json; d=json.load(open(r'$S/sm130-decision-$C.json')); print(d['verdict'] + ((', compact ' + ('go' if d['compact']['pass'] else 'no go')) if 'compact' in d else ''))")
  echo "$v" > "$G/verdict-$C"; eval "VERDICT_$C=\"$v\""
  return 0
}

timing() {  # CPU latency, once, after the chains (GPU idle from this launcher); 20 min before a break
  [ -e "$S/sm130-timing.json" ] && return 0
  local six=$SM/sm130-th-dec2ep32k-p2/expertium/$TRACK/master.pth
  [ -f "$six" ] || six=$SM/sm130-io-dec2ep32k-p2/expertium/$TRACK/master.pth
  [ -f "$six" ] || six=$SM/sm130-th-cut-e29/master.pth
  [ "$DRYRUN" = 1 ] && { log "DRYRUN: CPU timing $PY -X utf8 $S/sm130_timing.py --tiny <SM124> --six $six"; return 0; }
  cpu_step 20 "CPU timing" || return 1
  memguard_on
  CUDA_VISIBLE_DEVICES= "$PY" -X utf8 "$S/sm130_timing.py" --tiny "$W/sm124/artifacts/small-model/sm124-c1-e29st-dec2ep32k-t91s2-p2/expertium/$TRACK/master.pth" \
    --six "$six" >> "$G/timing.out" 2>&1 || log "CPU timing failed (report only; see $G/timing.out)"
  memguard_off || return 1
  log "CPU timing: $(tail -1 "$G/timing.out" | cut -c1-400)"
}

preconditions() {
  for f in "$S/sm130_block.py" "$RULE" "$PAIR" "$PREP" "$EPY" "$CUT" "$S/extra-options-th.json" "$S/extra-options-io.json" \
           "$SM/sm130-th-cut-e29/extend.json" "$SM/sm130-io-cut-e38/extend.json" "$USERS732"; do
    [ -f "$f" ] || { log "STOP: missing $f"; return 1; }
  done
  if [ "$DRYRUN" != 1 ] && [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    log "STOP: uncommitted source changes in scripts/ expertium/ rwkv/ (SourceWatch refuses them)"; return 1; fi
  "$PY" -X utf8 "$CUT" quick > "$G/quick.out" 2>&1 || { log "STOP: sm130_cut.py quick failed (see $G/quick.out)"; return 1; }
  log "$(tail -1 "$G/quick.out")"
  "$PY" -X utf8 "$RULE" selftest >> "$LOG" 2>&1 || { log "STOP: rule selftest failed"; return 1; }
  "$PY" -X utf8 "$EPY" selfcheck >> "$LOG" 2>&1 || { log "STOP: epoch selfcheck failed"; return 1; }
  "$PY" -X utf8 "$PAIR" selfcheck >> "$LOG" 2>&1 || { log "STOP: pair selfcheck failed"; return 1; }
  return 0
}

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1) (DRYRUN=$DRYRUN; chains: $CHAINS; arch $ARCH)"
rc=0
cpu_step 5 "preparation" || rc=1
[ $rc -eq 0 ] && { preconditions || rc=1; }
[ $rc -eq 0 ] && { cpu_checks || rc=1; }
[ $rc -eq 0 ] && { wait_sm129 || rc=1; }
VERDICT_th="not run"; VERDICT_io="not run"
for c in $CHAINS; do
  [ $rc -eq 0 ] || break
  run_chain "$c" || rc=1
done
[ $rc -eq 0 ] && { timing || rc=1; }
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; else log "done (verdict two-head: $VERDICT_th; Instant-Only: $VERDICT_io)"; fi
else
  log "STOP: ended with exit $rc"
fi
exit $rc
