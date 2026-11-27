#!/usr/bin/env bash
# Session inputs confirmation (SM181): every GPU step, then the CPU read-out (design.md). One detached launcher; every
# GPU step is one scripts/r01_queue.py ticket (owner sm181, claim gpu, cap 0.8; 0.92 only for memory retry passes),
# purpose prefix "Session inputs confirmation (SM181): ". Copy of SM179's launcher pattern.
#
#   DRYRUN=1 bash launch-sm181-gpu.sh     (prints the tickets, runs nothing)
#   bash launch-sm181-gpu.sh              (detached: see HANDOFF.md for the WMI line; rerun = resume)
#   stop: touch C:/Users/jonat/workspace/wt/sm181/artifacts/small-model/sm181/gpu/STOP  (a running training snapshots at
#         the next user boundary and exits; no new ticket starts; running this launcher again resumes)
# Log patterns: done "SM181 DONE", stopped or failed "SM181 STOP" (cause on an earlier line starting "STOP:").
# Breaks: quiet.json windows of scope all / gpu (sm135_block.py): no ticket starts inside or within its expected length
# of a window; a running training is snapshot-stopped <= 2 min before one and resumed after it.
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
R=C:/Users/jonat/workspace/wt/sm179-run
RUN_COMMIT=cd220759
S=C:/Users/jonat/workspace/wt/sm181/artifacts/small-model/sm181
G=$S/gpu
DUMP=C:/Users/jonat/workspace/wt/sm181/work/sm181-dump
RUN=$S/sm181_run.py
BLOCK=$R/artifacts/small-model/sm135/sm135_block.py
SC=$R/scripts/small_model_heldout.py
USERS732=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json
USERSDEC=$S/users-decisive.json
TRACK=E89-F65e-F77a
TINY=w64-c1-d2p1u1
SMALL=w64
TINY_CTL=C:/Users/jonat/workspace/wt/sm171-run/artifacts/small-model/sm171-tiny-ctl/expertium/$TRACK/master.pth
TINY_SES=$R/artifacts/small-model/sm179-tiny-ses/expertium/$TRACK/master.pth
SMALL_CTL=C:/Users/jonat/workspace/wt/sm149-run/artifacts/small-model/sm149-sm118-ctl/expertium/$TRACK/master.pth
SMALL_SES=$R/artifacts/small-model/sm181-small-ses/expertium/$TRACK/master.pth
EDGES=0,1,3,5,9,16,32,128,300,1000
LOG=$G/launch-sm181-gpu.log
P="Session inputs confirmation (SM181): "
DRYRUN=${DRYRUN:-0}
mkdir -p "$G"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
die() { log "STOP: $*"; log "SM181 STOP"; exit 1; }
block() { "$PY" -X utf8 "$BLOCK" | tr -d '\r'; }
stopped() { [ -e "$G/STOP" ]; }
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
  "$PY" "$QUEUE" run --owner sm181 --purpose "$P$purpose" --claim gpu --expected-minutes "$minutes" \
    --launched-by "detached launcher sm181/launch-sm181-gpu.sh" -- "$@"
  local rc=$?; log "exit $rc: $purpose"; return $rc
}
train() {  # train FOLDER MINUTES: run (resumes from a snapshot) until complete; snapshot stop on STOP / a quiet window
  local f=$1 minutes=$2 pid b why sent m=session
  while ! done_in "$f"; do
    stopped && return 1
    wait_free 30 || return 1
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN train $f (~$minutes min)"; echo "    cmd: env RWKV_DELETED_CARDS=neighbour SM179_INPUTS=$m $PY -X utf8 $RUN run --out $R/artifacts/small-model/$f" >> "$LOG"; return 0; fi
    log "train: $f (SM179_INPUTS=$m, expected ~$minutes min)"
    ( cd "$R" && "$PY" "$QUEUE" run --owner sm181 --purpose "${P}train $f" --claim gpu --expected-minutes "$minutes" \
      --launched-by "detached launcher sm181/launch-sm181-gpu.sh" -- \
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
score() {  # score SET TAG ARCH PTH INPUTS MINUTES: REF, show-time inputs, with the dump; retry passes at cap 0.92
  local set=$1 tag=$2 arch=$3 pth=$4 m=$5 minutes=$6 cap=0.8 pass n prev=-1 setenv pool users rows target dump
  case "$set" in
    decisive) pool=1500; users=$USERSDEC; rows=$G/sm181-rows-decisive.jsonl; target=765;;
    small732) pool=732; users=$USERS732; rows=$G/sm181-rows-732.jsonl; target=723;;
    tiny732) pool=732; users=$USERS732; rows=$G/sm181-rows-tiny732.jsonl; target=723;;
  esac
  dump=$DUMP/$set
  [ -n "$m" ] && setenv="SM179_INPUTS=$m" || setenv="-u SM179_INPUTS"
  [ "$DRYRUN" = 1 ] || [ -s "$pth" ] || { log "STOP: missing checkpoint $pth"; return 1; }
  for pass in 1 2 3; do
    n=$([ "$DRYRUN" = 1 ] && echo 0 || users_for "$rows" "$tag/w8+s8hx8@e2s-query")
    [ "$n" -ge "$target" ] && break
    [ "$n" = "$prev" ] && { log "$tag ($set): no progress in the last pass, $n users; moving on"; break; }
    prev=$n
    q "$minutes" "scoring $tag (REF, inputs ${m:-none}, dump) on $set, pass $pass, cap $cap" -- bash -c "cd $R && env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS $setenv PYTORCH_JIT=0 HELDOUT_GPU_MEMORY_FRACTION=$cap $PY -X utf8 $SC --pool $pool --users-file $users --model $tag=expertium:$arch:$pth --only $tag/w8+s8hx8 $SCORE_FLAGS --dump $dump --time-limit 4200 --out $rows" >> "$G/score.out" 2>&1 || return 1
    [ "$DRYRUN" = 1 ] && break
    log "$tag ($set): $(users_for "$rows" "$tag/w8+s8hx8@e2s-query") users scored (target $target)"
    cap=0.92
  done
  return 0
}

log "launch (DRYRUN=$DRYRUN)"
# preconditions: CPU proofs, the pair check, the run tree clean at the committed feature code, the SM181 scripts committed
"$PY" -X utf8 -c "import json, sys; sys.exit(0 if json.load(open(r'$S/check-cpu-sm181.json'))['all_ok'] else 1)" || die "CPU proofs not ALL OK"
"$PY" -X utf8 "$S/sm181_prepare.py" pair --root "$R" >> "$LOG" 2>&1 || die "pair check failed"
[ -z "$(git -C "$R" status --porcelain rwkv scripts expertium)" ] || die "run tree has source edits"
[ "$(git -C "$R" rev-parse --short=8 HEAD | tr -d '\r')" = "$RUN_COMMIT" ] || die "run tree HEAD is not $RUN_COMMIT"
[ -z "$(git -C "$S" status --porcelain -- sm181_run.py sm181_prepare.py sm181_report.py sm181_check.py sm181_updates.py launch-sm181-gpu.sh users-decisive.json design.md)" ] || die "sm181 files have uncommitted edits"
log "commit $(git -C "$S" rev-parse --short HEAD | tr -d '\r'), run tree $RUN_COMMIT"
SCORE_FLAGS="--inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows --prior-buckets $EDGES"

# 1. trainer smoke of the Small arm (2 users x 512 reviews, scratch copy)
[ -e "$G/smoke-ok" ] || {
  q 5 "trainer smoke, Small session arm" -- bash -c "cd $R && env RWKV_DELETED_CARDS=neighbour SM179_INPUTS=session SM179_CODE=$R $PY -X utf8 $RUN smoke --out $R/artifacts/small-model/sm181-small-ses" >> "$G/smoke.out" 2>&1 \
    || die "smoke failed (see $G/smoke.out)"
  [ "$DRYRUN" = 1 ] || touch "$G/smoke-ok"
}
# 2. axis 1: six-layer Tiny two-head (SM130), SM171's control and SM179's session arm on the decisive users
score decisive sm171-tiny-ctl $TINY "$TINY_CTL" "" 25 || die "scoring sm171-tiny-ctl on the decisive users failed"
score decisive sm179-tiny-ses $TINY "$TINY_SES" session 25 || die "scoring sm179-tiny-ses on the decisive users failed"
# 3. axis 2: Small two-head (SM118) session arm, then both Small models on the 732 set
train sm181-small-ses 45 || die "train sm181-small-ses"
score small732 sm149-sm118-ctl $SMALL "$SMALL_CTL" "" 20 || die "scoring sm149-sm118-ctl failed"
score small732 sm181-small-ses $SMALL "$SMALL_SES" session 20 || die "scoring sm181-small-ses failed"
# 4. report only: SM179's two Tiny models rescored on the 732 set with the dump (overall AUC)
score tiny732 sm171-tiny-ctl $TINY "$TINY_CTL" "" 15 || die "scoring sm171-tiny-ctl on the 732 set failed"
score tiny732 sm179-tiny-ses $TINY "$TINY_SES" session 15 || die "scoring sm179-tiny-ses on the 732 set failed"
# 5. the read-out (CPU, about a minute)
if [ "$DRYRUN" = 1 ]; then log "DRYRUN report"; log "SM181 DONE (dry run)"; exit 0; fi
( cd "$S" && OMP_NUM_THREADS=2 "$PY" -X utf8 sm181_report.py report ) >> "$LOG" 2>&1
rc=$?
log "report exit $rc (0 decided, 2 pending): $(grep -m1 '^\*\*Verdict' "$S/sm181-result.md" 2>/dev/null | tr -d '\r')"
log "SM181 DONE"
