#!/usr/bin/env bash
# The longer cool-down for Tiny (SM115) (artifacts/small-model/sm115/design.md), designed 2026-11-14:
#   the longer big instant-only teacher (SM113)'s recipe applied to Tiny: one one_minus_sin decay 7e-4 -> 0 over
#   2 x 581,942,724 reviews from the Tiny Instant-Only model (SM101)'s constant-LR epoch 38 (w64-c1, read only; where
#   SM101's own cool-down started), fresh Muon, show-time inputs, curve loss weight 0, the show-time instant-only big
#   model (SM99)'s show-time KD replay targets (decay alpha 0.5), as two chained folders (sm115_prepare.py):
#     pass 1  == SM101's cool-down except lr_resume {2R, 0, segment} and the targets; == SM113's pass 1 except the start
#             and the architecture (sm115_pair.py p1): 7e-4 -> 0.293 x 7e-4
#     pass 2  pass 1's end (master + Muon state), the same rows in a new order, lr_resume {2R, R} (sm115_pair.py p2)
#   early stop after pass 1: its final model vs SM101's own cool-down snapshot after 4,094 users (attempt-2 users-01200,
#            learning rate 0.2931 x peak: the matching point), show-time instant; sm115_rule.py mid: continue if < 0.
#            The reference snapshot is scored first (the first GPU ticket).
# Final rule (sm115_rule.py decide, pre-registered): SM115 - SM101, instant, show-time inputs, 722 users of the 732 set:
#   go if the 95% CI is below 0 and the gain >= 0.05% (% of the big model (1 leak-free epoch) normal instant LogLoss).
# CPU proofs (sm115_check.py -> check-cpu-sm115.json): ALL OK required.
# Breaks: sm115_block.py = quiet.json (scope all/gpu) + the daily 20:00-01:00 (Sunday from 16:00) unless quiet.json overlaps
#   that night; re-read at every check, CPU steps too. Tickets must end before the next break; a pass starts when 30 min
#   fit, snapshot-stops <= 2 min before a break and resumes after it.
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm115/artifacts/small-model/sm115/launch-sm115.sh
#   launch   WMI hidden (HANDOFF.md); log artifacts/small-model/sm115/gpu/launch-sm115.log
#   resume   the launch command again (finished steps are skipped; a pass resumes from its last snapshot)
#   stop     touch C:/Users/jonat/workspace/wt/sm115/artifacts/small-model/sm115/gpu/STOP (between tickets; a running pass
#            gets a snapshot stop at the next user boundary)
# Do not edit scripts/ expertium/ rwkv/ of the sm115 worktree while it runs (SourceWatch stops the training).
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/sm115
SM=$WT/artifacts/small-model
S=$SM/sm115
G=$S/gpu
mkdir -p "$G"
LOG=$G/launch-sm115.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm115-dryrun.log
DEC_PY=$WT/scripts/expertium_decay.py
SC=$WT/scripts/small_model_heldout.py
RULE=$S/sm115_rule.py
PAIR=$S/sm115_pair.py
PREP=$S/sm115_prepare.py
TRACK=E89-F65e-F77a
ARCH=w64-c1
TEACHER=C:/Users/jonat/workspace/wt/sm99/artifacts/small-model/sm99-big-inst-e2/expertium/big-inst/master.pth   # read only
TEACHER_SHA=c49e45d698194da033caa1231f598f875922a4a2a3e921714a3fd433d37e5768
S101=C:/Users/jonat/workspace/wt/sm101/artifacts/small-model                                                 # read only
E38=$S101/sm101-c1-e38-st
E38_SHA=e6ea69124af00a3dcf2325047ae34580aaf96051d963ff6130dd06bde02d396c
DEC101=$S101/sm101-c1-e38st-st-dec-1ep32k-muon
SNAP101=$DEC101/attempts/attempt-2/expertium/$TRACK/snapshots/users-01200/master.pth   # SM101's cool-down after 4,094 users
SNAP101_SHA=3ab9e629e7d740cbd4b3f8c8f40f680e028dd6fbd7790c25e83e083f780a9c35
P113=C:/Users/jonat/workspace/wt/sm113/artifacts/small-model/sm113-inst-e35st-st-dec2ep32k-t99-p1             # read only
S72=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s                                             # read only
USERS732=$S72/eval-users-732-fit.json; ULONG=$S72/users-732-long.json
USERS1500=C:/Users/jonat/workspace/wt/sm57/artifacts/small-model/sm57/stage4/users-sm93-score.json
P1=$SM/sm115-c1-e38st-st-dec2ep32k-t99-p1        # pass 1 (prepared before launch)
P2=$SM/sm115-c1-e38st-st-dec2ep32k-t99-p2        # pass 2 (prepared after pass 1)
CLBL=sm115-c1-e38st-st-dec2ep32k-t99
MID_NEW=sm115-dec2ep-p1
MID_REF=sm101-dec-u4094
OUTROWS=$G/sm115-rows.jsonl
ROWS1500=$G/sm115-rows-1500.jsonl
MIDROWS=$G/sm115-mid-rows.jsonl
SEGS=$G/segments.jsonl
TICKETS=$G/tickets.jsonl
STOPF=$G/STOP
MODE=neighbour
LF="env RWKV_DELETED_CARDS=$MODE"
MIN_DECAY=${MIN_DECAY:-560}   # one pass: SM101's cool-down ~458 min wall at ~20.8-21.2k reviews/s (SM113's passes 20.2-20.3k) + setup
MIN_START=30                  # a pass starts when 30 min fit (snapshot stop before the break, resume after)
MIN_SMOKE=15; MIN_SCORE732=25; MIN_SCORE_RETRY=20; MIN_LONG=15; MIN_POOL=35
TL732=1800; TLLONG=1380; TLPOOL=2400
NAME="the longer cool-down for Tiny (SM115)"
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START SM11_INPUTS   # caps 0.8
log() { echo "[sm115 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
stopped() { [ -e "$STOPF" ] && { log "STOP: STOP file present"; return 0; }; return 1; }
nap() { [ "$DRYRUN" = 1 ] && return 0; local s=$1; [ "$s" -gt 600 ] && s=600; sleep "$s"; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'; }
free_mib() { awk '/MemFree/ {print int($2 / 1024)}' /proc/meminfo 2>/dev/null || echo 99999; }
block() { "$PY" -X utf8 "$S/sm115_block.py"; }
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

q() {  # q MINUTES KIND PURPOSE -- CMD...   one ticket that must end before the next break; timed in $TICKETS (+ nvidia-smi max)
  local minutes=$1 kind=$2 purpose=$3 rc t0 mx base wpid; shift 4
  while true; do
    wait_free "$minutes" "$purpose" || return 1
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $NAME: $purpose (expected $minutes min)"; echo "    cmd: $*" >> "$LOG"; return 0; fi
    local maxwait=$(( FREE - minutes - 3 )); [ "$maxwait" -lt 1 ] && maxwait=1
    log "enqueue: $purpose (expected $minutes min, max wait $maxwait min, $FREE min to the next break)"
    t0=$(date +%s); base=$(gpu_used); rm -f "$G/.gpu-max"
    ( m=0; while true; do u=$(gpu_used); [ -n "$u" ] && [ "$u" -gt "$m" ] && { m=$u; echo "$m" > "$G/.gpu-max"; }; sleep 30; done ) &
    wpid=$!
    "$PY" "$QUEUE" run --owner sm115 --purpose "$NAME: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm115/launch-sm115.sh" -- "$@"
    rc=$?; kill "$wpid" 2>/dev/null; wait "$wpid" 2>/dev/null; mx=$(cat "$G/.gpu-max" 2>/dev/null || echo 0)
    log "exit $rc: $purpose (nvidia-smi max $mx MiB, baseline $base MiB)"
    [ "$rc" = 75 ] && { log "queue wait would reach the next break; re-planning"; sleep 60; continue; }
    jline kind "$kind" purpose "$purpose" exit "$rc" minutes "$(( ($(date +%s) - t0 + 30) / 60 ))" nvidia_smi_max_mib "${mx:-0}" nvidia_smi_baseline_mib "${base:-0}" >> "$TICKETS"
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
    "$PY" "$QUEUE" run --owner sm115 --purpose "$NAME: $what (segment to the next break or the end)" --claim gpu \
      --expected-minutes "$seg" --max-wait-minutes $(( FREE - 20 > 1 ? FREE - 20 : 1 )) --launched-by "detached launcher sm115/launch-sm115.sh" -- \
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

score() {  # score INPUTS LABEL CKPT ROWS POOL USERS_FILE EXPECT STRICT MIN1 TL [CAP0]: passes at CAP0 (default 0.8; a pass
           # without progress -> cap 0.92; no progress at 0.92: strict -> STOP, else go on with what fits)
  local v=$1 name=$2 ck=$3 rows=$4 pool=$5 uf=$6 expect=$7 strict=$8 min1=$9 tl=${10} cap=${11:-0.8} pass have prev=-1 minutes lb
  lb="$name/w8+s8"; [ "$v" != normal ] && lb="$lb@$v"
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
    q $minutes score "scoring $name on $v inputs, ${pool#* } (pass $pass, $have done, cap $cap)" -- \
      env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START HELDOUT_GPU_MEMORY_FRACTION=$cap "$PY" -X utf8 "$SC" --pool "${pool%% *}" --users-file "$uf" \
      --model "$name=expertium:$ARCH:$ck" --only "$name/w8+s8" --inputs "$v" --time-limit $tl --split-existing \
      --deleted-cards $MODE --out "$rows" >> "$G/score.out" 2>&1 || { log "STOP: scoring ticket failed ($lb)"; return 1; }
    [ "$DRYRUN" = 1 ] && break
  done
  [ "$DRYRUN" = 1 ] || [ "$strict" != 1 ] || [ "$(label_users "$rows" "$lb")" -ge "$expect" ] || { log "STOP: $lb scored on $(label_users "$rows" "$lb") of $expect users"; return 1; }
  [ "$DRYRUN" = 1 ] || log "$lb: $(label_users "$rows" "$lb") users scored in $(basename "$rows")"
  return 0
}
score732() {  # score732 INPUTS: the final model, 723 users of the 732 set (strict), then the 3 long users (cap 0.92, not needed by the rule)
  local ck=$P2/expertium/$TRACK/master.pth
  score "$1" "$CLBL" "$ck" "$OUTROWS" "732 the 732 set minus 9 never-fit (723 users)" "$USERS732" 723 1 $MIN_SCORE732 $TL732 || return 1
  score "$1" "$CLBL" "$ck" "$OUTROWS" "732 the 3 long users of the 732 set" "$ULONG" 726 0 $MIN_LONG $TLLONG 0.92 \
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

smoke() {  # smoke DIR TAG WHAT
  [ -e "$G/smoke-$2-ok" ] && return 0
  done_in "$1" && return 0
  ram_gate "smoke $2" strict || return 1
  q $MIN_SMOKE check "trainer smoke of $3 (2 users x 512 reviews)" -- $LF "$PY" -X utf8 "$DEC_PY" smoke --out "$1" >> "$G/check.out" 2>&1 \
    || { log "STOP: smoke $2 failed (see $G/check.out)"; return 1; }
  [ "$DRYRUN" = 1 ] || { touch "$G/smoke-$2-ok"; log "smoke $2: $(grep '"smoke"' "$G/check.out" | tail -1 | cut -c1-200)"; }
  return 0
}

lr_end() {  # lr_end DIR LO HI WHAT: the pass's last update ran at LO..HI x 7e-4 (the 2R schedule's point), else STOP
  [ "$DRYRUN" = 1 ] && { log "DRYRUN: lr check of $4: last update lr in [$2, $3] x 7e-4"; return 0; }
  local v; v=$(U="$1/expertium/$TRACK/updates.jsonl" "$PY" -X utf8 -c "
import json, os
last = None
for line in open(os.environ['U'], encoding='utf-8'):
    if line.strip():
        last = line
print(json.loads(last)['lr'] / 7e-4)")
  "$PY" -c "import sys; v = float(sys.argv[1]); sys.exit(0 if float(sys.argv[2]) <= v <= float(sys.argv[3]) else 1)" "$v" "$2" "$3"     || { log "STOP: $4 ended at lr $v x peak, not in [$2, $3] (the two-epoch schedule is wrong)"; return 1; }
  log "lr check: $4 ended at $v x peak (expected [$2, $3])"
}

score_ref() {  # the early-stop reference: SM101's own cool-down snapshot after 4,094 users (show-time, 723 users), first ticket
  [ -e "$G/ref-done" ] && return 0
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: score $MID_REF = $SNAP101 on e2s-query (723 users)"; return 0; fi
  score e2s-query "$MID_REF" "$SNAP101" "$MIDROWS" "732 the 732 set minus 9 never-fit (723 users)" "$USERS732" 723 1 $MIN_SCORE732 $TL732 || return 1
  touch "$G/ref-done"
  return 0
}

early_stop() {  # score pass 1's final model (show-time, 723 users), then the mid rule
  [ -e "$G/mid-done" ] && return 0
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: early stop: score $MID_NEW = $P1/expertium/$TRACK/master.pth on e2s-query (723 users), then $PY -X utf8 $RULE mid"; return 0; fi
  score e2s-query "$MID_NEW" "$P1/expertium/$TRACK/master.pth" "$MIDROWS" "732 the 732 set minus 9 never-fit (723 users)" "$USERS732" 723 1 $MIN_SCORE732 $TL732 || return 1
  rule mid "after pass 1 of 2" || return 1
  local d; d=$("$PY" -X utf8 -c "import json; print(json.load(open(r'$S/sm115-mid.json'))['decision'])")
  touch "$G/mid-done"
  [ "$d" = stop ] && { touch "$G/mid-stop"; log "early stop says stop: pass 2 does not run; verdict no go (stopped after pass 1)"; }
  [ "$d" = continue ] && log "early stop says continue: pass 2 next"
  return 0
}

prepare_p2() {  # CPU: pass 2 from pass 1's end, then its pair check
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: $PY -X utf8 $PREP p2 --p1 $P1 --out $P2; $PAIR p2 --arm $P2 --p1 $P1"; return 0; fi
  if [ ! -f "$P2/decay.json" ]; then
    cpu_step 5 "pass 2 prepare" || return 1
    "$PY" -X utf8 "$PREP" p2 --p1 "$P1" --out "$P2" >> "$LOG" 2>&1 || { log "STOP: pass 2 prepare failed"; rm -rf "$P2"; return 1; }
  fi
  "$PY" -X utf8 "$PAIR" p2 --arm "$P2" --p1 "$P1" >> "$LOG" 2>&1 || { log "STOP: pass 2 is not pass 1's continuation (sm115_pair.py p2)"; return 1; }
  return 0
}

cd "$WT" || exit 1

preconditions() {
  for f in "$S/check-cpu-sm115.json" "$S/extra-options.json" "$S/sm115_block.py" "$RULE" "$PAIR" "$PREP" "$TEACHER" \
           "$E38/expertium/$TRACK/master.pth" "$SNAP101" "$DEC101/decay.json" "$P113/decay.json" "$P1/decay.json" "$P1/protocol.json" \
           "$USERS732" "$ULONG" "$USERS1500"; do
    [ -f "$f" ] || { log "STOP: missing $f"; return 1; }
  done
  grep -q '"all_ok": true' "$S/check-cpu-sm115.json" || { log "STOP: the CPU proofs are not ALL OK ($S/check-cpu-sm115.json)"; return 1; }
  done_in "$DEC101" || { log "STOP: SM101's cool-down is not complete ($DEC101)"; return 1; }
  if [ "$DRYRUN" != 1 ] && [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    log "STOP: uncommitted source changes in scripts/ expertium/ rwkv/ (SourceWatch refuses them)"; return 1; fi
  "$PY" -X utf8 "$PAIR" p1 --arm "$P1" --ref101 "$DEC101" --ref113 "$P113" >> "$LOG" 2>&1 || { log "STOP: pass 1 is not SM101's cool-down except lr_resume and the targets (or not SM113's pass 1 except the start and the architecture)"; return 1; }
  "$PY" -X utf8 "$RULE" selftest >> "$LOG" 2>&1 || { log "STOP: rule selftest failed"; return 1; }
  local sha
  sha=$(sha256sum "$TEACHER" | cut -d' ' -f1); [ "$sha" = "$TEACHER_SHA" ] || { log "STOP: the teacher changed (sha256 $sha)"; return 1; }
  sha=$(sha256sum "$E38/expertium/$TRACK/master.pth" | cut -d' ' -f1); [ "$sha" = "$E38_SHA" ] || { log "STOP: SM101's epoch 38 changed (sha256 $sha)"; return 1; }
  sha=$(sha256sum "$SNAP101" | cut -d' ' -f1); [ "$sha" = "$SNAP101_SHA" ] || { log "STOP: SM101's cool-down snapshot changed (sha256 $sha)"; return 1; }
  return 0
}

log "started: $(git -C "$WT" log --oneline -1) (DRYRUN=$DRYRUN; SM113's two-pass cool-down with SM99's show-time targets, from SM101's epoch 38 (w64-c1) -> $P1, $P2; early stop after pass 1)"
rc=0
cpu_step 5 "preparation" || rc=1
[ $rc -eq 0 ] && { preconditions || rc=1; }
W1="pass 1 of the two-epoch 32k Muon cool-down on show-time inputs from SM101's epoch 38 (w64-c1) with SM99's show-time targets (7e-4 -> 0.29 x peak), curve loss weight 0"
W2="pass 2 of the two-epoch 32k Muon cool-down (0.29 x peak -> 0, from pass 1's end, new order), curve loss weight 0"
[ $rc -eq 0 ] && { score_ref || rc=1; }
[ $rc -eq 0 ] && { smoke "$P1" p1 "pass 1 (fresh Muon, SM99's show-time targets, curve loss weight 0, show-time inputs)" || rc=1; }
[ $rc -eq 0 ] && { run_folder "$P1" "$W1" $MIN_DECAY || rc=1; }
[ $rc -eq 0 ] && { lr_end "$P1" 0.2925 0.2935 "pass 1" || rc=1; }
[ $rc -eq 0 ] && { early_stop || rc=1; }
if [ $rc -eq 0 ] && [ -e "$G/mid-stop" ]; then
  log "done (verdict no go: stopped after pass 1, $(grep -m1 Decision "$S/sm115-mid.md" 2>/dev/null | cut -c1-200))"
  exit 0
fi
[ $rc -eq 0 ] && { prepare_p2 || rc=1; }
[ $rc -eq 0 ] && { smoke "$P2" p2 "pass 2 (pass 1's Muon state loaded, show-time inputs)" || rc=1; }
[ $rc -eq 0 ] && { run_folder "$P2" "$W2" $MIN_DECAY || rc=1; }
[ $rc -eq 0 ] && { lr_end "$P2" 0 0.0005 "pass 2" || rc=1; }
# final scoring and the rule
[ $rc -eq 0 ] && { score732 e2s-query || rc=1; }
[ $rc -eq 0 ] && { rule decide "show-time 732" || rc=1; }
[ $rc -eq 0 ] && { score732 normal || rc=1; }
[ $rc -eq 0 ] && { score e2s-query "$CLBL" "$P2/expertium/$TRACK/master.pth" "$ROWS1500" "1500 the 1,500 validation users outside the 732 set + 6 check users" "$USERS1500" 771 0 $MIN_POOL $TLPOOL \
  || log "the 1,500-pool scoring failed; going on (secondary only)"; }
[ $rc -eq 0 ] && { rule decide "final" || rc=1; }
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; else log "done (verdict $("$PY" -X utf8 -c "import json; print(json.load(open(r'$S/sm115-decision.json')).get('verdict'))" 2>/dev/null))"; fi
else
  log "STOP: ended with exit $rc"
fi
exit $rc
