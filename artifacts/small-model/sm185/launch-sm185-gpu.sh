#!/usr/bin/env bash
# Small cool-down with session inputs (SM185): every GPU step, then the CPU read-out (design.md). One detached launcher;
# every GPU step is one scripts/r01_queue.py ticket (owner sm185, claim gpu, cap 0.8; 0.92 only for memory retry passes
# of the scorer), purpose prefix "Small cool-down with session inputs (SM185): ". SM183's launcher with the Small sources.
#
#   DRYRUN=1 bash launch-sm185-gpu.sh     (prints the tickets, runs nothing)
#   bash launch-sm185-gpu.sh              (detached: see HANDOFF.md for the WMI line; rerun = resume)
#   stop: touch C:/Users/jonat/workspace/wt/sm185/artifacts/small-model/sm185/gpu/STOP  (a running training snapshots at
#         the next user boundary and exits; no new ticket starts; running this launcher again resumes)
# Log patterns: two-head read-out "SM185 TH", done "SM185 DONE", stopped or failed "SM185 STOP" (cause on an earlier
# line starting "STOP:").
# Overrides (only with the main session's OK): BUDGET_TOTAL=<GPU minutes> (default 960 = 16 h; the Instant-Only chain
# starts only if the minutes used so far + IO_EST fit), CONTROL_ARM=1 (after a failed two-head G0: train the session
# arm and the control arm sm185-th-ctl instead of stopping).
# Breaks: quiet.json windows of scope all / gpu (sm135_block.py): no ticket starts inside or within its expected length
# of a window; a running training is snapshot-stopped <= 2 min before one and resumed after it.
# Do not edit scripts/ expertium/ rwkv/ of the run tree while this runs.
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
R=C:/Users/jonat/workspace/wt/sm179-run
RUN_COMMIT=cd220759
S=C:/Users/jonat/workspace/wt/sm185/artifacts/small-model/sm185
G=$S/gpu
DUMP=C:/Users/jonat/workspace/wt/sm185/work/sm185-dump
RUN=$S/sm185_run.py
PREP=$S/sm185_prepare.py
BLOCK=$R/artifacts/small-model/sm135/sm135_block.py
SC=$R/scripts/small_model_heldout.py
RS=$R/artifacts/small-model
USERS732=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json
USERSDEC=$S/users-decisive.json
TRACK=E89-F65e-F77a
ARCH=w64
REL_TH=C:/Users/jonat/workspace/wt/sm118/artifacts/small-model/sm118-e26-dec2ep32k-t91s2-p2/expertium/$TRACK/master.pth
REL_TH_SHA=b4fe1175d25a7e6abff70e87ac0af1f4d76db9622d54f7cb53196fbf35117eb6
REL_IO=C:/Users/jonat/workspace/wt/sm113/artifacts/small-model/sm113-inst-e35st-st-dec2ep32k-t99-p2/expertium/$TRACK/master.pth
REL_IO_SHA=e4574e2aa8b2cfe6360dbd86f2a75b71d23ea81677b00ffcc9c3abce6b4cb736
EDGES=0,1,3,5,9,16,32,128,300,1000
LOG=$G/launch-sm185-gpu.log
MINF=$G/gpu-minutes.jsonl
P="Small cool-down with session inputs (SM185): "
DRYRUN=${DRYRUN:-0}
BUDGET_TOTAL=${BUDGET_TOTAL:-960}   # GPU minutes in all (16 h)
IO_EST=${IO_EST:-740}            # the Instant-Only chain: G0 25 + smoke 1 + 2 passes x 345 + 732-set scoring 2 x 12
IO_DEC_EST=70                    # its decisive scoring, 2 models
CONTROL_ARM=${CONTROL_ARM:-0}
MIN_PASS=${MIN_PASS:-360}        # one cool-down pass: SM118's passes took 336 / 339 min (28.9k / 28.7k reviews/s)
mkdir -p "$G"
[ "$DRYRUN" = 1 ] && { LOG=$G/launch-sm185-gpu-dryrun.log; MINF=$G/gpu-minutes-dryrun.jsonl; : > "$LOG"; : > "$MINF"; }
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
die() { log "STOP: $*"; log "SM185 STOP"; exit 1; }
block() { "$PY" -X utf8 "$BLOCK" | tr -d '\r'; }
stopped() { [ -e "$G/STOP" ]; }
result_status() { R2="$RS/$1/expertium/$TRACK/result.json" "$PY" -X utf8 -c "
import json, os
p = os.environ['R2']
print(json.load(open(p)).get('status') if os.path.exists(p) else 'none')" | tr -d '\r'; }
done_in() { [ "$(result_status "$1")" = complete ]; }
used_minutes() { "$PY" -X utf8 -c "
import json, sys
try:
    print(round(sum(json.loads(l)['minutes'] for l in open(sys.argv[1]))))
except FileNotFoundError:
    print(0)" "$MINF" | tr -d '\r'; }
note_minutes() {  # note_minutes WHAT T0_FILE: GPU minutes from the ticket's own start (written by the command) to now
  local t0; t0=$(cat "$2" 2>/dev/null | tr -d '\r'); [ -n "$t0" ] || return 0
  "$PY" -X utf8 -c "import json, sys, time; print(json.dumps({'what': sys.argv[1], 'start': int(sys.argv[2]), 'minutes': round((time.time() - int(sys.argv[2])) / 60, 1)}))" "$1" "$t0" >> "$MINF"
  rm -f "$2"
}
wait_free() {  # wait_free MINUTES: until no quiet window is on or starts within MINUTES
  local b
  while true; do
    stopped && return 1
    b=$(block)
    case "$b" in NOW*) log "quiet window on ($b); waiting"; sleep 300; continue;; esac
    [ "${b#NEXT }" -ge $(( $1 + 5 )) ] && return 0
    log "next quiet window in ${b#NEXT } min (< $1 + 5 min needed); waiting"; sleep 300
  done
}
q() {  # q MINUTES PURPOSE -- CMD...: one queue ticket; the command records its own start time for the GPU minutes
  local minutes=$1 purpose=$2; shift 3
  wait_free "$minutes" || return 1
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN ticket ($minutes min): $purpose"; echo "    cmd: $*" >> "$LOG"; return 0; fi
  log "ticket: $purpose (expected $minutes min)"
  "$PY" "$QUEUE" run --owner sm185 --purpose "$P$purpose" --claim gpu --expected-minutes "$minutes" \
    --launched-by "detached launcher sm185/launch-sm185-gpu.sh" -- bash -c 'date +%s > "$0"; exec "$@"' "$G/.t0" "$@"
  local rc=$?; note_minutes "$purpose" "$G/.t0"; log "exit $rc: $purpose"; return $rc
}
envof() { [ -n "$1" ] && echo "SM179_INPUTS=$1" || echo "-u SM179_INPUTS"; }
train() {  # train FOLDER INPUTS MINUTES [STOP_AT_FILE]: run (resumes from a snapshot) until complete (or, with
           # STOP_AT_FILE, until that file exists: then a snapshot stop); snapshot stop on STOP / a quiet window
  local f=$1 m=$2 minutes=$3 stop_at=${4:-} pid b why sent e
  e=$(envof "$m")
  while ! done_in "$f"; do
    stopped && return 1
    [ -n "$stop_at" ] && [ -e "$stop_at" ] && return 0
    wait_free 30 || return 1
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN train $f (SM179_INPUTS=${m:-unset}, ~$minutes min${stop_at:+, until $(basename "$(dirname "$stop_at")")})"
      echo "    cmd: env $e RWKV_DELETED_CARDS=neighbour SM179_CODE=$R $PY -X utf8 $RUN run --out $RS/$f" >> "$LOG"; return 0; fi
    log "train: $f (SM179_INPUTS=${m:-unset}, expected ~$minutes min${stop_at:+, until $(basename "$(dirname "$stop_at")")})"
    ( cd "$R" && "$PY" "$QUEUE" run --owner sm185 --purpose "${P}train $f" --claim gpu --expected-minutes "$minutes" \
      --launched-by "detached launcher sm185/launch-sm185-gpu.sh" -- bash -c 'date +%s > "$0"; exec "$@"' "$G/.t0-$f" \
      env $e RWKV_DELETED_CARDS=neighbour SM179_CODE=$R "$PY" -X utf8 "$RUN" run --out "$RS/$f" ) >> "$G/train-$f.out" 2>&1 &
    pid=$!
    sent=0
    while kill -0 "$pid" 2>/dev/null; do
      b=$(block); why=""
      stopped && why="STOP file"
      { [[ "$b" == NOW* ]] || { [[ "$b" == NEXT* ]] && [ "${b#NEXT }" -le 2 ]; }; } && why="quiet window ($b)"
      [ -n "$stop_at" ] && [ -e "$stop_at" ] && why="reached $(basename "$(dirname "$stop_at")")"
      if [ -n "$why" ] && [ $sent = 0 ]; then
        log "snapshot stop of $f ($why)"; ( cd "$R" && env $e SM179_CODE=$R "$PY" -X utf8 "$RUN" stop --out "$RS/$f" > /dev/null 2>&1 ); sent=1
      fi
      sleep 30
    done
    wait "$pid"; local rc=$?; note_minutes "train $f" "$G/.t0-$f"; log "exit $rc: train $f"
    [ -n "$stop_at" ] && [ -e "$stop_at" ] && return 0
    done_in "$f" && break
    [ $sent = 0 ] && { log "STOP: $f ended without completing and without a stop request (see $G/train-$f.out)"; return 1; }
  done
  [ "$DRYRUN" = 1 ] || log "$f complete: $("$PY" -X utf8 -c "import json; r=json.load(open(r'$RS/$f/expertium/$TRACK/result.json')); print(f\"{r.get('review_presentations',0):,} reviews, {r.get('training_seconds',0)/60:.1f} min, {r.get('reviews_per_second',0):,.0f} reviews/s, peak allocated {r.get('peak_allocated_mib',0):,.0f} MiB, reserved {r.get('peak_reserved_mib',0):,.0f} MiB, process tree RSS {r.get('peak_process_tree_rss_mib',0):,.0f} MiB, update offset end {r.get('update_offset_end')}, trained {str(r.get('trained_parameter_sha256'))[:12]}\")" | tr -d '\r')"
  return 0
}
smoke() {  # smoke FOLDER INPUTS
  [ -e "$G/smoke-$1-ok" ] && return 0
  q 8 "trainer smoke of $1 (2 users x 512 reviews, scratch copy)" -- bash -c "cd $R && env $(envof "$2") RWKV_DELETED_CARDS=neighbour SM179_CODE=$R $PY -X utf8 $RUN smoke --out $RS/$1" >> "$G/smoke.out" 2>&1 \
    || { log "STOP: smoke of $1 failed (see $G/smoke.out)"; return 1; }
  [ "$DRYRUN" = 1 ] || { touch "$G/smoke-$1-ok"; log "smoke $1: $(grep '"smoke"' "$G/smoke.out" | tail -1 | cut -c1-200 | tr -d '\r')"; }
}
pass2() {  # pass2 CHAIN ARM INPUTS: prepare p2 from the finished p1 (once), pair check, train
  local c=$1 a=$2 m=$3
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: $PREP p2 --chain $c --arm $a; pair"; train "sm185-$c-$a-p2" "$m" $MIN_PASS; return 0; fi
  if [ ! -f "$RS/sm185-$c-$a-p2/decay.json" ]; then
    "$PY" -X utf8 "$PREP" p2 --chain $c --arm $a --root "$R" >> "$LOG" 2>&1 || { log "STOP: pass-2 prepare of sm185-$c-$a failed"; return 1; }
  fi
  "$PY" -X utf8 "$PREP" pair --root "$R" >> "$LOG" 2>&1 || { log "STOP: pair check failed after the pass-2 prepare of sm185-$c-$a (see the json line above)"; return 1; }
  train "sm185-$c-$a-p2" "$m" $MIN_PASS
}
users_for() {  # users_for ROWS LABEL: users with a benchmark row of that label
  "$PY" -X utf8 -c "
import json, sys
n = set()
try:
    for r in map(json.loads, open(sys.argv[1], encoding='utf-8')):
        if 'benchmark' in r and r['label'] == sys.argv[2]: n.add(r['user_id'])
except FileNotFoundError:
    pass
print(len(n))" "$1" "$2" | tr -d '\r'
}
score() {  # score SET TAG PTH INPUTS MINUTES: REF, show-time inputs, with the dump; retry passes at cap 0.92
  local set=$1 tag=$2 pth=$3 m=$4 minutes=$5 cap=0.8 pass n prev=-1 pool users rows target dump
  case "$set" in
    decisive) pool=1500; users=$USERSDEC; rows=$G/sm185-rows-decisive.jsonl; target=765;;
    732) pool=732; users=$USERS732; rows=$G/sm185-rows-732.jsonl; target=723;;
  esac
  dump=$DUMP/$set
  [ "$DRYRUN" = 1 ] || [ -s "$pth" ] || { log "STOP: missing checkpoint $pth"; return 1; }
  for pass in 1 2 3; do
    n=$([ "$DRYRUN" = 1 ] && echo 0 || users_for "$rows" "$tag/w8+s8hx8@e2s-query")
    [ "$n" -ge "$target" ] && break
    [ "$n" = "$prev" ] && { log "$tag ($set): no progress in the last pass, $n users; moving on"; break; }
    prev=$n
    q "$minutes" "scoring $tag (REF, inputs ${m:-none}, dump) on the $set set, pass $pass, cap $cap" -- bash -c "cd $R && env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS $(envof "$m") PYTORCH_JIT=0 HELDOUT_GPU_MEMORY_FRACTION=$cap $PY -X utf8 $SC --pool $pool --users-file $users --model $tag=expertium:$ARCH:$pth --only $tag/w8+s8hx8 $SCORE_FLAGS --dump $dump --time-limit 4200 --out $rows" >> "$G/score.out" 2>&1 || return 1
    [ "$DRYRUN" = 1 ] && break
    log "$tag ($set): $(users_for "$rows" "$tag/w8+s8hx8@e2s-query") users scored (target $target)"
    cap=0.92
  done
  return 0
}
read_out() {  # read_out TAG: the CPU report (about a minute)
  [ "$DRYRUN" = 1 ] && { log "DRYRUN report"; return 0; }
  ( cd "$S" && OMP_NUM_THREADS=2 "$PY" -X utf8 sm185_report.py report ) > "$G/report.out" 2>&1
  local rc=$?
  log "report exit $rc (0 decided, 2 pending): $(grep -m3 '^\*\*' "$S/sm185-result.md" 2>/dev/null | tr -d '\r' | tr '\n' ' ')"
}
arm_end() { echo "$RS/sm185-$1-$2-p2/expertium/$TRACK/master.pth"; }
g0() {  # g0 CHAIN: the control reproduction check (once; result in gpu/g0-CHAIN-done)
  local c=$1 snap=$RS/sm185-$1-ctlcheck/expertium/$TRACK/snapshots/users-00400/snapshot.json
  [ -e "$G/g0-$c-done" ] && return 0
  ls "$RS"/sm185-$c-ctlcheck/attempts/*/expertium/$TRACK/snapshots/users-00400/snapshot.json > /dev/null 2>&1 \
    || train sm185-$c-ctlcheck "" 25 "$snap" || return 1
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: sm185_check.py ctl --chain $c"; return 0; fi
  "$PY" -X utf8 "$S/sm185_check.py" ctl --chain $c >> "$LOG" 2>&1 && echo pass > "$G/g0-$c-done" || echo fail > "$G/g0-$c-done"
}
g0_result() { local r; r=$(cat "$G/g0-$1-done" 2>/dev/null | tr -d '\r'); echo "${r:-pass}"; }

log "launch (DRYRUN=$DRYRUN)"
# preconditions: CPU proofs, the pair check, the run tree clean at the committed feature code, the SM185 scripts
# committed, the release models (controls) unchanged
"$PY" -X utf8 -c "import json, sys; sys.exit(0 if json.load(open(r'$S/check-cpu-sm185.json'))['all_ok'] else 1)" || die "CPU proofs not ALL OK"
"$PY" -X utf8 "$PREP" pair --root "$R" >> "$LOG" 2>&1 || die "pair check failed"
[ -z "$(git -C "$R" status --porcelain rwkv scripts expertium)" ] || die "run tree has source edits"
[ "$(git -C "$R" rev-parse --short=8 HEAD | tr -d '\r')" = "$RUN_COMMIT" ] || die "run tree HEAD is not $RUN_COMMIT"
[ -z "$(git -C "$S" status --porcelain -- sm185_run.py sm185_prepare.py sm185_report.py sm185_check.py sm185_updates.py launch-sm185-gpu.sh users-decisive.json design.md)" ] || die "sm185 files have uncommitted edits"
[ "$(sha256sum "$REL_TH" | cut -c1-64)" = "$REL_TH_SHA" ] || die "the Small two-head (SM118) release master changed"
[ "$(sha256sum "$REL_IO" | cut -c1-64)" = "$REL_IO_SHA" ] || die "the Small Instant-Only (SM113) release master changed"
log "commit $(git -C "$S" rev-parse --short HEAD | tr -d '\r'), run tree $RUN_COMMIT, GPU minutes so far $(used_minutes), budget $BUDGET_TOTAL"
SCORE_FLAGS="--inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows --prior-buckets $EDGES"

# 1. trainer smoke of the two-head session arm
smoke sm185-th-ses-p1 session || die "smoke"
# 2. G0: SM118's pass 1 reproduced in the run tree (flag unset) to its 400-user snapshot, compared bit for bit
g0 th || die "G0 training (two-head)"
G0=$(g0_result th)
log "G0 control reproduction: $G0 ($( [ "$G0" = pass ] && echo 'the release models are the controls' || echo 'FAILED'))"
if [ "$G0" != pass ] && [ "$CONTROL_ARM" != 1 ]; then
  die "G0 failed (check-ctl-sm185.json): the release model is not a valid control; a control arm sm185-th-ctl would add ~11.5 GPU h. Relaunch with CONTROL_ARM=1 only with the main session's OK"
fi
# 3. two-head session arm: pass 1, pass 2
train sm185-th-ses-p1 session $MIN_PASS || die "train sm185-th-ses-p1"
pass2 th ses session || die "pass 2 of sm185-th-ses"
# 3b. only if G0 failed and CONTROL_ARM=1: the two-head control arm (same two passes, flag unset)
if [ "$G0" != pass ]; then
  [ -f "$RS/sm185-th-ctl-p1/decay.json" ] || "$PY" -X utf8 "$PREP" p1 --chain th --arm ctl --root "$R" >> "$LOG" 2>&1 || die "prepare sm185-th-ctl-p1"
  train sm185-th-ctl-p1 "" $MIN_PASS || die "train sm185-th-ctl-p1"
  pass2 th ctl "" || die "pass 2 of sm185-th-ctl"
  CTH=sm185-th-ctl; CTHP=$(arm_end th ctl)
else
  CTH=sm118-th-release; CTHP=$REL_TH
fi
# 4. two-head scoring: 732 set, then the decisive users (control first, then the arm)
score 732 $CTH "$CTHP" "" 25 || die "scoring $CTH on the 732 set"
score 732 sm185-th-ses "$(arm_end th ses)" session 25 || die "scoring sm185-th-ses on the 732 set"
score decisive $CTH "$CTHP" "" 35 || die "scoring $CTH on the decisive users"
score decisive sm185-th-ses "$(arm_end th ses)" session 35 || die "scoring sm185-th-ses on the decisive users"
read_out
log "SM185 TH read-out written ($(used_minutes) GPU minutes so far)"
# 5. Instant-Only chain, only if the budget allows (minutes used + IO_EST <= BUDGET_TOTAL) or it already started
U=$(used_minutes)
if [ "$DRYRUN" != 1 ] && [ $(( U + IO_EST )) -gt "$BUDGET_TOTAL" ] && [ ! -e "$G/g0-io-done" ]; then
  log "Instant-Only chain not run (budget: $U GPU minutes used + $IO_EST estimated > $BUDGET_TOTAL; follow-up with BUDGET_TOTAL=<minutes>, main session's OK)"
else
  smoke sm185-io-ses-p1 session || die "smoke io"
  g0 io || die "G0 training (Instant-Only)"
  G0IO=$(g0_result io)
  log "G0 control reproduction (Instant-Only): $G0IO"
  if [ "$G0IO" != pass ]; then
    log "Instant-Only chain not run (its G0 failed: check-ctl-io-sm185.json)"
  else
    train sm185-io-ses-p1 session $MIN_PASS || die "train sm185-io-ses-p1"
    pass2 io ses session || die "pass 2 of sm185-io-ses"
    score 732 sm113-io-release "$REL_IO" "" 25 || die "scoring sm113-io-release on the 732 set"
    score 732 sm185-io-ses "$(arm_end io ses)" session 25 || die "scoring sm185-io-ses on the 732 set"
    U=$(used_minutes)
    if [ "$DRYRUN" = 1 ] || [ $(( U + IO_DEC_EST )) -le "$BUDGET_TOTAL" ]; then
      score decisive sm113-io-release "$REL_IO" "" 35 || die "scoring sm113-io-release on the decisive users"
      score decisive sm185-io-ses "$(arm_end io ses)" session 35 || die "scoring sm185-io-ses on the decisive users"
    else
      log "Instant-Only decisive scoring skipped (budget: $U GPU minutes used + $IO_DEC_EST > $BUDGET_TOTAL)"
    fi
    read_out
  fi
fi
log "GPU minutes used: $(used_minutes)"
log "SM185 DONE"
