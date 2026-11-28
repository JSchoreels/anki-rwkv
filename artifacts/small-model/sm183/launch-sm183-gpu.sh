#!/usr/bin/env bash
# Tiny cool-down with session inputs (SM183): every GPU step, then the CPU read-out (design.md). One detached launcher;
# every GPU step is one scripts/r01_queue.py ticket (owner sm183, claim gpu, cap 0.8; 0.92 only for memory retry passes
# of the scorer), purpose prefix "Tiny cool-down with session inputs (SM183): ". SM181's launcher pattern.
#
#   DRYRUN=1 bash launch-sm183-gpu.sh     (prints the tickets, runs nothing)
#   bash launch-sm183-gpu.sh              (detached: see HANDOFF.md for the WMI line; rerun = resume)
#   stop: touch C:/Users/jonat/workspace/wt/sm183/artifacts/small-model/sm183/gpu/STOP  (a running training snapshots at
#         the next user boundary and exits; no new ticket starts; running this launcher again resumes)
# Log patterns: two-head read-out "SM183 TH", done "SM183 DONE", stopped or failed "SM183 STOP" (cause on an earlier
# line starting "STOP:").
# Breaks: quiet.json windows of scope all / gpu (sm135_block.py): no ticket starts inside or within its expected length
# of a window; a running training is snapshot-stopped <= 2 min before one and resumed after it.
# Do not edit scripts/ expertium/ rwkv/ of the run tree while this runs.
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
R=C:/Users/jonat/workspace/wt/sm179-run
RUN_COMMIT=cd220759
S=C:/Users/jonat/workspace/wt/sm183/artifacts/small-model/sm183
G=$S/gpu
DUMP=C:/Users/jonat/workspace/wt/sm183/work/sm183-dump
RUN=$S/sm183_run.py
PREP=$S/sm183_prepare.py
BLOCK=$R/artifacts/small-model/sm135/sm135_block.py
SC=$R/scripts/small_model_heldout.py
RS=$R/artifacts/small-model
USERS732=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json
USERSDEC=$S/users-decisive.json
TRACK=E89-F65e-F77a
ARCH=w64-c1-d2p1u1
SM130=C:/Users/jonat/workspace/wt/sm130/artifacts/small-model
EDGES=0,1,3,5,9,16,32,128,300,1000
LOG=$G/launch-sm183-gpu.log
MINF=$G/gpu-minutes.jsonl
P="Tiny cool-down with session inputs (SM183): "
DRYRUN=${DRYRUN:-0}
BUDGET_IO=${BUDGET_IO:-400}      # the Instant-Only chain starts only if the two-head steps used <= this many GPU minutes
BUDGET_IO_DEC=${BUDGET_IO_DEC:-680}   # Instant-Only decisive scoring only if all steps so far used <= this many
MIN_PASS=${MIN_PASS:-165}        # one cool-down pass: SM130's passes took 147-151 min (64-66k reviews/s)
mkdir -p "$G"
[ "$DRYRUN" = 1 ] && { LOG=$G/launch-sm183-gpu-dryrun.log; MINF=$G/gpu-minutes-dryrun.jsonl; : > "$LOG"; : > "$MINF"; }
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
die() { log "STOP: $*"; log "SM183 STOP"; exit 1; }
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
  "$PY" "$QUEUE" run --owner sm183 --purpose "$P$purpose" --claim gpu --expected-minutes "$minutes" \
    --launched-by "detached launcher sm183/launch-sm183-gpu.sh" -- bash -c 'date +%s > "$0"; exec "$@"' "$G/.t0" "$@"
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
    ( cd "$R" && "$PY" "$QUEUE" run --owner sm183 --purpose "${P}train $f" --claim gpu --expected-minutes "$minutes" \
      --launched-by "detached launcher sm183/launch-sm183-gpu.sh" -- bash -c 'date +%s > "$0"; exec "$@"' "$G/.t0-$f" \
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
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: $PREP p2 --chain $c --arm $a; pair"; train "sm183-$c-$a-p2" "$m" $MIN_PASS; return 0; fi
  if [ ! -f "$RS/sm183-$c-$a-p2/decay.json" ]; then
    "$PY" -X utf8 "$PREP" p2 --chain $c --arm $a --root "$R" >> "$LOG" 2>&1 || { log "STOP: pass-2 prepare of sm183-$c-$a failed"; return 1; }
  fi
  "$PY" -X utf8 "$PREP" pair --root "$R" >> "$LOG" 2>&1 || { log "STOP: pair check failed after the pass-2 prepare of sm183-$c-$a (see the json line above)"; return 1; }
  train "sm183-$c-$a-p2" "$m" $MIN_PASS
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
    decisive) pool=1500; users=$USERSDEC; rows=$G/sm183-rows-decisive.jsonl; target=765;;
    732) pool=732; users=$USERS732; rows=$G/sm183-rows-732.jsonl; target=723;;
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
  ( cd "$S" && OMP_NUM_THREADS=2 "$PY" -X utf8 sm183_report.py report ) > "$G/report.out" 2>&1
  local rc=$?
  log "report exit $rc (0 decided, 2 pending): $(grep -m3 '^\*\*' "$S/sm183-result.md" 2>/dev/null | tr -d '\r' | tr '\n' ' ')"
}
rel() { echo "$SM130/sm130-$1-dec2ep32k-p2/expertium/$TRACK/master.pth"; }
arm_end() { echo "$RS/sm183-$1-$2-p2/expertium/$TRACK/master.pth"; }

log "launch (DRYRUN=$DRYRUN)"
# preconditions: CPU proofs, the pair check, the run tree clean at the committed feature code, the SM183 scripts committed
"$PY" -X utf8 -c "import json, sys; sys.exit(0 if json.load(open(r'$S/check-cpu-sm183.json'))['all_ok'] else 1)" || die "CPU proofs not ALL OK"
"$PY" -X utf8 "$PREP" pair --root "$R" >> "$LOG" 2>&1 || die "pair check failed"
[ -z "$(git -C "$R" status --porcelain rwkv scripts expertium)" ] || die "run tree has source edits"
[ "$(git -C "$R" rev-parse --short=8 HEAD | tr -d '\r')" = "$RUN_COMMIT" ] || die "run tree HEAD is not $RUN_COMMIT"
[ -z "$(git -C "$S" status --porcelain -- sm183_run.py sm183_prepare.py sm183_report.py sm183_check.py sm183_updates.py launch-sm183-gpu.sh users-decisive.json design.md)" ] || die "sm183 files have uncommitted edits"
log "commit $(git -C "$S" rev-parse --short HEAD | tr -d '\r'), run tree $RUN_COMMIT, GPU minutes so far $(used_minutes)"
SCORE_FLAGS="--inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows --prior-buckets $EDGES"

# 1. trainer smoke of the two-head session arm
smoke sm183-th-ses-p1 session || die "smoke"
# 2. G0: SM130's two-head pass 1 reproduced in the run tree (flag unset) to its 400-user snapshot, compared bit for bit
G0SNAP=$RS/sm183-th-ctlcheck/expertium/$TRACK/snapshots/users-00400/snapshot.json
g0_in_attempt() { ls "$RS"/sm183-th-ctlcheck/attempts/*/expertium/$TRACK/snapshots/users-00400/snapshot.json > /dev/null 2>&1; }
if [ ! -e "$G/g0-done" ]; then
  g0_in_attempt || train sm183-th-ctlcheck "" 15 "$G0SNAP" || die "G0 training"
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: sm183_check.py ctl"; G0=pass; else
    "$PY" -X utf8 "$S/sm183_check.py" ctl >> "$LOG" 2>&1 && echo pass > "$G/g0-done" || echo fail > "$G/g0-done"
  fi
fi
G0=$(cat "$G/g0-done" 2>/dev/null | tr -d '\r'); G0=${G0:-pass}
log "G0 control reproduction: $G0 ($( [ "$G0" = pass ] && echo 'the release models are the controls' || echo 'control arm sm183-th-ctl will be trained; the Instant-Only chain is dropped'))"
# 3. two-head session arm: pass 1, pass 2
train sm183-th-ses-p1 session $MIN_PASS || die "train sm183-th-ses-p1"
pass2 th ses session || die "pass 2 of sm183-th-ses"
# 3b. only if G0 failed: the two-head control arm (same two passes, flag unset)
if [ "$G0" != pass ]; then
  [ -f "$RS/sm183-th-ctl-p1/decay.json" ] || "$PY" -X utf8 "$PREP" p1 --chain th --arm ctl --root "$R" >> "$LOG" 2>&1 || die "prepare sm183-th-ctl-p1"
  train sm183-th-ctl-p1 "" $MIN_PASS || die "train sm183-th-ctl-p1"
  pass2 th ctl "" || die "pass 2 of sm183-th-ctl"
  CTH=sm183-th-ctl; CTHP=$(arm_end th ctl)
else
  CTH=sm130-th-release; CTHP=$(rel th)
fi
# 4. two-head scoring: 732 set, then the decisive users (control first, then the arm)
score 732 $CTH "$CTHP" "" 20 || die "scoring $CTH on the 732 set"
score 732 sm183-th-ses "$(arm_end th ses)" session 20 || die "scoring sm183-th-ses on the 732 set"
score decisive $CTH "$CTHP" "" 30 || die "scoring $CTH on the decisive users"
score decisive sm183-th-ses "$(arm_end th ses)" session 30 || die "scoring sm183-th-ses on the decisive users"
read_out
log "SM183 TH read-out written ($(used_minutes) GPU minutes so far)"
# 5. Instant-Only chain, only if G0 passed and the budget allows
U=$(used_minutes)
if [ "$G0" != pass ]; then
  log "Instant-Only chain not run (G0 failed: the two-head control arm used its budget)"
elif [ "$DRYRUN" != 1 ] && [ "$U" -gt "$BUDGET_IO" ] && [ ! -e "$RS/sm183-io-ses-p1/expertium" ]; then
  log "Instant-Only chain not run (budget: $U GPU minutes used > $BUDGET_IO)"
else
  smoke sm183-io-ses-p1 session || die "smoke io"
  train sm183-io-ses-p1 session $MIN_PASS || die "train sm183-io-ses-p1"
  pass2 io ses session || die "pass 2 of sm183-io-ses"
  score 732 sm130-io-release "$(rel io)" "" 20 || die "scoring sm130-io-release on the 732 set"
  score 732 sm183-io-ses "$(arm_end io ses)" session 20 || die "scoring sm183-io-ses on the 732 set"
  U=$(used_minutes)
  if [ "$DRYRUN" = 1 ] || [ "$U" -le "$BUDGET_IO_DEC" ]; then
    score decisive sm130-io-release "$(rel io)" "" 30 || die "scoring sm130-io-release on the decisive users"
    score decisive sm183-io-ses "$(arm_end io ses)" session 30 || die "scoring sm183-io-ses on the decisive users"
  else
    log "Instant-Only decisive scoring skipped (budget: $U GPU minutes used > $BUDGET_IO_DEC)"
  fi
  read_out
fi
log "GPU minutes used: $(used_minutes)"
log "SM183 DONE"
