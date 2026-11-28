#!/usr/bin/env bash
# AUC ceiling (SM182): the whole GPU screen on six-layer Tiny two-head (SM130), then the CPU read-out (design.md
# "Recipe", "Scoring", "Metrics", "Rules"). One detached launcher; every GPU step is one scripts/r01_queue.py ticket
# (owner sm182, claim gpu, cap 0.8 from the release options), purpose prefix "AUC ceiling (SM182): ".
#
#   DRYRUN=1 bash launch-sm182-gpu.sh     (prints the tickets, runs nothing)
#   bash launch-sm182-gpu.sh              (detached: see HANDOFF.md for the WMI line; rerun = resume)
#   stop: touch C:/Users/jonat/workspace/wt/sm182/artifacts/small-model/sm182/gpu/STOP  (a running arm snapshots at the
#         next user boundary and exits; running this launcher again resumes it from the snapshot)
# Log patterns: done "SM182 DONE", stopped or failed "SM182 STOP" (cause on an earlier line starting "STOP:").
# Breaks: quiet.json windows of scope all / gpu (sm135_block.py): no ticket starts inside or within its expected length
# of a window; a running arm is snapshot-stopped <= 2 min before one and resumed after it.
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
R=C:/Users/jonat/workspace/wt/sm182-run
RUN_COMMIT=060878d3
S=C:/Users/jonat/workspace/wt/sm182/artifacts/small-model/sm182
G=$S/gpu
WORK=$S/work
RUN=$S/sm182_run.py
BLOCK=$R/artifacts/small-model/sm135/sm135_block.py
SC=$R/scripts/small_model_heldout.py
USERS732=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json
USMOKE=$R/artifacts/small-model/sm148/users-smoke.json
ARCH=w64-c1-d2p1u1
TRACK=E89-F65e-F77a
CTL=C:/Users/jonat/workspace/wt/sm171-run/artifacts/small-model/sm171-tiny-ctl/expertium/$TRACK/master.pth
EDGES=0,1,3,5,9,16,32,128,300,1000
ROWS=$G/sm182-rows-732.jsonl
LOG=$G/launch-sm182-gpu.log
P="AUC ceiling (SM182): "
DRYRUN=${DRYRUN:-0}
mkdir -p "$G" "$WORK"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
die() { log "STOP: $*"; log "SM182 STOP"; exit 1; }
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
  "$PY" "$QUEUE" run --owner sm182 --purpose "$P$purpose" --claim gpu --expected-minutes "$minutes" \
    --launched-by "detached launcher sm182/launch-sm182-gpu.sh" -- "$@"
  local rc=$?; log "exit $rc: $purpose"; return $rc
}
train() {  # train FOLDER MINUTES: run (resumes from a snapshot) until complete; snapshot stop on STOP / a quiet window
  local f=$1 minutes=$2 pid b why sent
  while ! done_in "$f"; do
    stopped && return 1
    wait_free 30 || return 1
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN train $f (~$minutes min)"; echo "    cmd: env RWKV_DELETED_CARDS=neighbour SM182_CODE=$R $PY -X utf8 $RUN run --out $R/artifacts/small-model/$f" >> "$LOG"; return 0; fi
    log "train: $f (expected ~$minutes min)"
    ( cd "$R" && "$PY" "$QUEUE" run --owner sm182 --purpose "${P}train $f" --claim gpu --expected-minutes "$minutes" \
      --launched-by "detached launcher sm182/launch-sm182-gpu.sh" -- \
      env -u SM179_INPUTS RWKV_DELETED_CARDS=neighbour SM182_CODE=$R "$PY" -X utf8 "$RUN" run --out "$R/artifacts/small-model/$f" ) >> "$G/train-$f.out" 2>&1 &
    pid=$!
    sent=0
    while kill -0 "$pid" 2>/dev/null; do
      b=$(block); why=""
      stopped && why="STOP file"
      { [[ "$b" == NOW* ]] || { [[ "$b" == NEXT* ]] && [ "${b#NEXT }" -le 2 ]; }; } && why="quiet window ($b)"
      if [ -n "$why" ] && [ $sent = 0 ]; then
        log "snapshot stop of $f ($why)"; ( cd "$R" && env SM182_CODE=$R "$PY" -X utf8 "$RUN" stop --out "$R/artifacts/small-model/$f" > /dev/null 2>&1 ); sent=1
      fi
      sleep 30
    done
    wait "$pid"; log "exit $?: train $f"
    done_in "$f" && break
    [ $sent = 0 ] && { log "STOP: $f ended without completing and without a stop request (see $G/train-$f.out)"; return 1; }
  done
  [ "$DRYRUN" = 1 ] || log "$f complete: $("$PY" -X utf8 -c "import json; r=json.load(open(r'$R/artifacts/small-model/$f/expertium/$TRACK/result.json')); print(f\"{r.get('review_presentations',0):,} reviews, {r.get('training_seconds',0)/60:.1f} min, {r.get('reviews_per_second',0):,.0f} reviews/s, peak allocated {r.get('peak_allocated_mib',0):,.0f} MiB, reserved {r.get('peak_reserved_mib',0):,.0f} MiB, pair_rank {r.get('pair_rank')}\")" | tr -d '\r')"
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
ENVS="env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS -u SM179_INPUTS PYTORCH_JIT=0"
SCORE_FLAGS="--inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows --prior-buckets $EDGES"
score() {  # score TAG PTH: REF on the 732 set, show-time inputs, per-review dump; a retry pass at cap 0.92
  local tag=$1 pth=$2 cap=0.8 pass n
  for pass in 1 2; do
    [ "$DRYRUN" = 1 ] || [ "$(users_for "$tag/w8+s8hx8@e2s-query")" -lt 723 ] || break
    q 20 "scoring $tag (REF, per-review dump) on the 732 set, pass $pass, cap $cap" -- bash -c "cd $R && $ENVS HELDOUT_GPU_MEMORY_FRACTION=$cap $PY -X utf8 $SC --pool 732 --users-file $USERS732 --model $tag=expertium:$ARCH:$pth --only $tag/w8+s8hx8 $SCORE_FLAGS --dump $WORK/dump --time-limit 3000 --out $ROWS" >> "$G/score.out" 2>&1 || return 1
    [ "$DRYRUN" = 1 ] && break
    n=$(users_for "$tag/w8+s8hx8@e2s-query"); log "$tag: $n users scored"
    cap=0.92
  done
  return 0
}

log "launch (DRYRUN=$DRYRUN)"
# preconditions: CPU proofs, report selftest, the pair check, the run tree clean at the committed code, scripts committed
"$PY" -X utf8 -c "import json, sys; sys.exit(0 if json.load(open(r'$S/check-cpu-sm182.json'))['all_ok'] else 1)" || die "CPU proofs not ALL OK"
"$PY" -X utf8 -c "import json, sys; sys.exit(0 if json.load(open(r'$G/selftest-report.json'))['ok'] else 1)" || die "report selftest not OK"
"$PY" -X utf8 "$S/sm182_prepare.py" pair --arms r,m,ra --root "$R" >> "$LOG" 2>&1 || die "pair check failed"
[ -z "$(git -C "$R" status --porcelain rwkv scripts expertium)" ] || die "run tree has source edits"
[ "$(git -C "$R" rev-parse --short=8 HEAD | tr -d '\r')" = "$RUN_COMMIT" ] || die "run tree HEAD is not $RUN_COMMIT"
[ -z "$(git -C "$S" status --porcelain -- sm182_run.py sm182_prepare.py sm182_report.py sm182_check.py sm182_updates.py launch-sm182-gpu.sh)" ] || die "sm182 scripts have uncommitted edits"
log "commit $(git -C "$S" rev-parse --short HEAD | tr -d '\r'), run tree $RUN_COMMIT"

# 1. smoke: trainer smoke of arms R and M (2 users x 512 reviews each, scratch copies), then the control's REF with
#    the per-review dump on the 3 smoke users and the CPU pipeline check of the dump (vs the rows and SM171's rows)
[ -e "$G/smoke-ok" ] || {
  q 8 "smoke: trainer arms R and M, control REF with dump on 3 users" -- bash -c "cd $R && env -u SM179_INPUTS RWKV_DELETED_CARDS=neighbour SM182_CODE=$R $PY -X utf8 $RUN smoke --out $R/artifacts/small-model/sm182-tiny-r && env -u SM179_INPUTS RWKV_DELETED_CARDS=neighbour SM182_CODE=$R $PY -X utf8 $RUN smoke --out $R/artifacts/small-model/sm182-tiny-m && $ENVS HELDOUT_GPU_MEMORY_FRACTION=0.8 $PY -X utf8 $SC --pool 732 --users-file $USMOKE --model sm171-tiny-ctl=expertium:$ARCH:$CTL --only sm171-tiny-ctl/w8+s8hx8 $SCORE_FLAGS --dump $WORK/dump-smoke --time-limit 600 --out $G/smoke-rows.jsonl" >> "$G/smoke.out" 2>&1 \
    || die "smoke failed (see $G/smoke.out)"
  if [ "$DRYRUN" != 1 ]; then
    ( cd "$S" && OMP_NUM_THREADS=2 "$PY" -X utf8 sm182_report.py dumpcheck --rows "$G/smoke-rows.jsonl" --dump "$WORK/dump-smoke" ) >> "$G/smoke.out" 2>&1 \
      || die "dump pipeline check failed (see $G/smoke.out)"
    log "dumpcheck: $(tail -1 "$G/smoke.out" | cut -c1-300 | tr -d '\r')"
    touch "$G/smoke-ok"
  fi
}
# 2. the arms: R (pure ranking) first; if R stops on a non-finite loss, the pre-registered fallback RA (LogLoss 0.02)
if ! train sm182-tiny-r 30; then
  stopped && die "STOP file during sm182-tiny-r"
  grep -q "Non-finite" "$G/train-sm182-tiny-r.out" 2>/dev/null || die "train sm182-tiny-r (not a non-finite loss: see $G/train-sm182-tiny-r.out)"
  log "sm182-tiny-r stopped on a non-finite loss: training the fallback sm182-tiny-ra (design.md)"
  train sm182-tiny-ra 30 || die "train sm182-tiny-ra"
  [ "$DRYRUN" = 1 ] || touch "$R/artifacts/small-model/sm182-tiny-ra/USED"
fi
RTAG=sm182-tiny-r
[ -e "$R/artifacts/small-model/sm182-tiny-ra/USED" ] && RTAG=sm182-tiny-ra
train sm182-tiny-m 30 || die "train sm182-tiny-m"
# 3. scoring on the 732 set with the per-review dump: the control (integrity vs SM171's rows), then R, then M
score sm171-tiny-ctl "$CTL" || die "scoring the control failed"
score $RTAG "$R/artifacts/small-model/$RTAG/expertium/$TRACK/master.pth" || die "scoring $RTAG failed"
score sm182-tiny-m "$R/artifacts/small-model/sm182-tiny-m/expertium/$TRACK/master.pth" || die "scoring sm182-tiny-m failed"
# 4. the read-out (CPU, about 20 to 40 minutes: per-user Platt recalibration of 3 models x 2 heads)
if [ "$DRYRUN" = 1 ]; then log "DRYRUN report"; log "SM182 DONE (dry run)"; exit 0; fi
( cd "$S" && OMP_NUM_THREADS=2 "$PY" -X utf8 sm182_report.py report ) >> "$LOG" 2>&1
rc=$?
log "report exit $rc (0 decided, 2 pending): $(grep -m1 '^\*\*Verdict' "$S/sm182-result.md" 2>/dev/null | tr -d '\r')"
log "SM182 DONE"
