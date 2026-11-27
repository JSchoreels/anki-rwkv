#!/usr/bin/env bash
# Session inputs (SM179): the whole GPU screen on six-layer Tiny two-head (SM130), then the CPU read-out (design.md
# "Recipe", "Scoring", "Rules"). One detached launcher; every GPU step is one scripts/r01_queue.py ticket (owner sm179,
# claim gpu, cap 0.8 from the release options), purpose prefix "Session inputs (SM179): ".
#
#   DRYRUN=1 bash launch-sm179-gpu.sh     (prints the tickets, runs nothing)
#   bash launch-sm179-gpu.sh              (detached: see HANDOFF.md for the WMI line; rerun = resume)
#   stop: touch C:/Users/jonat/workspace/wt/sm179/artifacts/small-model/sm179/gpu/STOP  (a running arm snapshots at the
#         next user boundary and exits; running this launcher again resumes it from the snapshot)
# Log patterns: done "SM179 DONE", stopped or failed "SM179 STOP" (cause on an earlier line starting "STOP:").
# Breaks: quiet.json windows of scope all / gpu (sm135_block.py): no ticket starts inside or within its expected length
# of a window; a running arm is snapshot-stopped <= 2 min before one and resumed after it.
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
R=C:/Users/jonat/workspace/wt/sm179-run
RUN_COMMIT=cd220759
S=C:/Users/jonat/workspace/wt/sm179/artifacts/small-model/sm179
G=$S/gpu
RUN=$S/sm179_run.py
BLOCK=$R/artifacts/small-model/sm135/sm135_block.py
SC=$R/scripts/small_model_heldout.py
USERS732=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json
ARCH=w64-c1-d2p1u1
TRACK=E89-F65e-F77a
CTL=C:/Users/jonat/workspace/wt/sm171-run/artifacts/small-model/sm171-tiny-ctl/expertium/$TRACK/master.pth
EDGES=0,1,3,5,9,16,32,128,300,1000
ROWS=$G/sm179-rows-732.jsonl
LOG=$G/launch-sm179-gpu.log
P="Session inputs (SM179): "
DRYRUN=${DRYRUN:-0}
mkdir -p "$G"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
die() { log "STOP: $*"; log "SM179 STOP"; exit 1; }
block() { "$PY" -X utf8 "$BLOCK" | tr -d '\r'; }
stopped() { [ -e "$G/STOP" ]; }
inputs_of() { case "$1" in sm179-tiny-ses) echo session;; sm179-tiny-st) echo state;; *) echo "";; esac; }
done_in() { R2="$R/artifacts/small-model/$1/expertium/$TRACK/result.json" "$PY" -X utf8 -c "
import json, os, sys
p = os.environ['R2']
sys.exit(0 if os.path.exists(p) and json.load(open(p)).get('status') == 'complete' else 1)"; }
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
q() {  # q MINUTES PURPOSE -- CMD...
  local minutes=$1 purpose=$2; shift 3
  wait_free "$minutes" || return 1
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN ticket ($minutes min): $purpose"; echo "    cmd: $*" >> "$LOG"; return 0; fi
  log "ticket: $purpose (expected $minutes min)"
  "$PY" "$QUEUE" run --owner sm179 --purpose "$P$purpose" --claim gpu --expected-minutes "$minutes" \
    --launched-by "detached launcher sm179/launch-sm179-gpu.sh" -- "$@"
  local rc=$?; log "exit $rc: $purpose"; return $rc
}
train() {  # train FOLDER MINUTES: run (resumes from a snapshot) until complete; snapshot stop on STOP / a quiet window
  local f=$1 minutes=$2 pid b why sent m
  m=$(inputs_of "$f")
  while ! done_in "$f"; do
    stopped && return 1
    wait_free 30 || return 1
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN train $f (~$minutes min)"; echo "    cmd: env RWKV_DELETED_CARDS=neighbour SM179_INPUTS=$m $PY -X utf8 $RUN run --out $R/artifacts/small-model/$f" >> "$LOG"; return 0; fi
    log "train: $f (SM179_INPUTS=$m, expected ~$minutes min)"
    ( cd "$R" && "$PY" "$QUEUE" run --owner sm179 --purpose "${P}train $f" --claim gpu --expected-minutes "$minutes" \
      --launched-by "detached launcher sm179/launch-sm179-gpu.sh" -- \
      env RWKV_DELETED_CARDS=neighbour SM179_INPUTS=$m SM179_CODE=$R "$PY" -X utf8 "$RUN" run --out "$R/artifacts/small-model/$f" ) >> "$G/train-$f.out" 2>&1 &
    pid=$!
    sent=0
    while kill -0 "$pid" 2>/dev/null; do
      b=$(block); why=""
      stopped && why="STOP file"
      { [[ "$b" == NOW* ]] || { [[ "$b" == NEXT* ]] && [ "${b#NEXT }" -le 2 ]; }; } && why="quiet window ($b)"
      if [ -n "$why" ] && [ $sent = 0 ]; then
        log "snapshot stop of $f ($why)"; ( cd "$R" && env SM179_INPUTS=$m SM179_CODE=$R "$PY" -X utf8 "$RUN" stop --out "$R/artifacts/small-model/$f" > /dev/null 2>&1 ); sent=1
      fi
      sleep 30
    done
    wait "$pid"; log "exit $?: train $f"
    done_in "$f" && break
    [ $sent = 0 ] && { log "STOP: $f ended without completing and without a stop request (see $G/train-$f.out)"; return 1; }
  done
  [ "$DRYRUN" = 1 ] || log "$f complete: $("$PY" -X utf8 -c "import json; r=json.load(open(r'$R/artifacts/small-model/$f/expertium/$TRACK/result.json')); print(f\"{r.get('review_presentations',0):,} reviews, {r.get('training_seconds',0)/60:.1f} min, {r.get('reviews_per_second',0):,.0f} reviews/s, peak allocated {r.get('peak_allocated_mib',0):,.0f} MiB, reserved {r.get('peak_reserved_mib',0):,.0f} MiB\")" | tr -d '\r')"
  return 0
}
users_for() {  # users_for LABEL: users with a benchmark row of that label in ROWS
  "$PY" -X utf8 -c "
import json, sys
n = set()
try:
    for r in map(json.loads, open(r'$ROWS', encoding='utf-8')):
        if 'benchmark' in r and r['label'] == sys.argv[1]: n.add(r['user_id'])
except FileNotFoundError:
    pass
print(len(n))" "$1" | tr -d '\r'
}
score() {  # score TAG PTH INPUTS: REF on the 732 set, show-time inputs; a retry pass at cap 0.92 for missing users
  local tag=$1 pth=$2 m=$3 cap=0.8 pass n setenv
  [ -n "$m" ] && setenv="SM179_INPUTS=$m" || setenv="-u SM179_INPUTS"
  for pass in 1 2; do
    [ "$DRYRUN" = 1 ] || [ "$(users_for "$tag/w8+s8hx8@e2s-query")" -lt 723 ] || break
    q 15 "scoring $tag (REF, inputs ${m:-none}) on the 732 set, pass $pass, cap $cap" -- bash -c "cd $R && env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS $setenv PYTORCH_JIT=0 HELDOUT_GPU_MEMORY_FRACTION=$cap $PY -X utf8 $SC --pool 732 --users-file $USERS732 --model $tag=expertium:$ARCH:$pth --only $tag/w8+s8hx8 $SCORE_FLAGS --time-limit 2400 --out $ROWS" >> "$G/score.out" 2>&1 || return 1
    [ "$DRYRUN" = 1 ] && break
    n=$(users_for "$tag/w8+s8hx8@e2s-query"); log "$tag: $n users scored"
    cap=0.92
  done
  return 0
}

log "launch (DRYRUN=$DRYRUN)"
# preconditions: CPU proofs, the pair check, the run tree clean at the committed feature code, the SM179 scripts committed
"$PY" -X utf8 -c "import json, sys; sys.exit(0 if json.load(open(r'$S/check-cpu-sm179.json'))['all_ok'] else 1)" || die "CPU proofs not ALL OK"
"$PY" -X utf8 "$S/sm179_prepare.py" pair --arms ses,st --root "$R" >> "$LOG" 2>&1 || die "pair check failed"
[ -z "$(git -C "$R" status --porcelain rwkv scripts expertium)" ] || die "run tree has source edits"
[ "$(git -C "$R" rev-parse --short=8 HEAD | tr -d '\r')" = "$RUN_COMMIT" ] || die "run tree HEAD is not $RUN_COMMIT"
[ -z "$(git -C "$S" status --porcelain -- sm179_run.py sm179_prepare.py sm179_report.py sm179_check.py sm179_updates.py launch-sm179-gpu.sh)" ] || die "sm179 scripts have uncommitted edits"
log "commit $(git -C "$S" rev-parse --short HEAD | tr -d '\r'), run tree $RUN_COMMIT"

# 1. trainer smoke of the two arms (2 users x 512 reviews each, scratch copies)
[ -e "$G/smoke-ok" ] || {
  q 5 "trainer smoke, two arms" -- bash -c "cd $R && env RWKV_DELETED_CARDS=neighbour SM179_INPUTS=session SM179_CODE=$R $PY -X utf8 $RUN smoke --out $R/artifacts/small-model/sm179-tiny-ses && env RWKV_DELETED_CARDS=neighbour SM179_INPUTS=state SM179_CODE=$R $PY -X utf8 $RUN smoke --out $R/artifacts/small-model/sm179-tiny-st" >> "$G/smoke.out" 2>&1 \
    || die "smoke failed (see $G/smoke.out)"
  [ "$DRYRUN" = 1 ] || touch "$G/smoke-ok"
}
# 2. the two arms (session first: it is the main question)
train sm179-tiny-ses 30 || die "train sm179-tiny-ses"
train sm179-tiny-st 30 || die "train sm179-tiny-st"
# 3. scoring on the 732 set: the control rescored here with the flag off (integrity vs SM171's rows), then the arms
SCORE_FLAGS="--inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows --prior-buckets $EDGES"
score sm171-tiny-ctl "$CTL" "" || die "scoring the control failed"
score sm179-tiny-ses "$R/artifacts/small-model/sm179-tiny-ses/expertium/$TRACK/master.pth" session || die "scoring sm179-tiny-ses failed"
score sm179-tiny-st "$R/artifacts/small-model/sm179-tiny-st/expertium/$TRACK/master.pth" state || die "scoring sm179-tiny-st failed"
# 4. the read-out (CPU, seconds)
if [ "$DRYRUN" = 1 ]; then log "DRYRUN report"; log "SM179 DONE (dry run)"; exit 0; fi
( cd "$S" && OMP_NUM_THREADS=2 "$PY" -X utf8 sm179_report.py report ) >> "$LOG" 2>&1
rc=$?
log "report exit $rc (0 decided, 2 pending): $(grep -m1 '^\*\*Verdict' "$S/sm179-result.md" 2>/dev/null | tr -d '\r')"
log "SM179 DONE"
