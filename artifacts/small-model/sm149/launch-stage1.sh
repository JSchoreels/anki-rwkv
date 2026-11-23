#!/usr/bin/env bash
# Rank-2 low-rank state cool-down (SM149), stage 1: Small two-head (SM118), QAT arm + control arm, then the 732-set
# scoring of both arms (design.md "Recipe", "Scoring"). One detached launcher; every GPU step is one scripts/r01_queue.py
# ticket (owner sm149, claim gpu, cap 0.8 from the release options). The long-card CPU scoring and the report are the
# next step (HANDOFF.md), not this launcher.
#
#   DRYRUN=1 bash launch-stage1.sh      (prints the tickets, runs nothing)
#   bash launch-stage1.sh               (detached: see HANDOFF.md for the WMI / nohup line)
#   stop: touch C:/Users/jonat/workspace/wt/sm149/artifacts/small-model/sm149/gpu/STOP  (a running arm snapshots at the
#         next user boundary and exits; running this launcher again resumes it from the snapshot)
# Breaks: quiet.json windows of scope all / gpu (sm135_block.py, no daily break since 11-19): no ticket starts inside or
# within its expected length of a window; a running arm is snapshot-stopped <= 2 min before one and resumed after it.
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
R=C:/Users/jonat/workspace/wt/sm149-run
S=C:/Users/jonat/workspace/wt/sm149/artifacts/small-model/sm149
G=$S/gpu
RUN=$S/sm149_run.py
BLOCK=$R/artifacts/small-model/sm135/sm135_block.py
SC=$R/scripts/small_model_heldout.py
USERS732=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json
USMOKE=C:/Users/jonat/workspace/wt/sm148/artifacts/small-model/sm148/users-smoke.json
ORIG=C:/Users/jonat/workspace/wt/sm118/artifacts/small-model/sm118-e26-dec2ep32k-t91s2-p2/expertium/E89-F65e-F77a/master.pth
TRACK=E89-F65e-F77a
EDGES=0,1,3,5,9,16,32,128,300,1000
ROWS=$G/sm149-rows-732.jsonl
LOG=$G/launch-stage1.log
DRYRUN=${DRYRUN:-0}
mkdir -p "$G"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
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
  "$PY" "$QUEUE" run --owner sm149 --purpose "SM149 stage 1: $purpose" --claim gpu --expected-minutes "$minutes" \
    --launched-by "detached launcher sm149/launch-stage1.sh" -- "$@"
  local rc=$?; log "exit $rc: $purpose"; return $rc
}
train() {  # train FOLDER MINUTES: run (resumes from a snapshot) until complete; snapshot stop on STOP / a quiet window
  local f=$1 minutes=$2 pid b why
  while ! done_in "$f"; do
    stopped && return 1
    wait_free 30 || return 1
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN train $f (~$minutes min)"; echo "    cmd: env RWKV_DELETED_CARDS=neighbour SM149_CODE=$R $PY -X utf8 $RUN run --out artifacts/small-model/$f" >> "$LOG"; return 0; fi
    log "train: $f (expected ~$minutes min)"
    "$PY" "$QUEUE" run --owner sm149 --purpose "SM149 stage 1: train $f" --claim gpu --expected-minutes "$minutes" \
      --launched-by "detached launcher sm149/launch-stage1.sh" -- \
      env RWKV_DELETED_CARDS=neighbour SM149_CODE=$R "$PY" -X utf8 "$RUN" run --out "artifacts/small-model/$f" >> "$G/train-$f.out" 2>&1 &
    pid=$!
    local sent=0
    while kill -0 "$pid" 2>/dev/null; do
      b=$(block); why=""
      stopped && why="STOP file"
      { [[ "$b" == NOW* ]] || { [[ "$b" == NEXT* ]] && [ "${b#NEXT }" -le 2 ]; }; } && why="quiet window ($b)"
      if [ -n "$why" ] && [ $sent = 0 ]; then
        log "snapshot stop of $f ($why)"; env SM149_CODE=$R "$PY" -X utf8 "$RUN" stop --out "artifacts/small-model/$f" > /dev/null 2>&1; sent=1
      fi
      sleep 30
    done
    wait "$pid"; log "exit $?: train $f"
    done_in "$f" && break
    [ $sent = 0 ] && { log "STOP: $f ended without completing and without a stop request (see $G/train-$f.out)"; return 1; }
  done
  [ "$DRYRUN" = 1 ] || log "$f complete: $("$PY" -X utf8 -c "import json; r=json.load(open(r'$R/artifacts/small-model/$f/expertium/$TRACK/result.json')); print(f\"{r.get('review_presentations',0):,} reviews, {r.get('training_seconds',0)/60:.1f} min, {r.get('reviews_per_second',0):,.0f} reviews/s, peak allocated {r.get('peak_allocated_mib',0):,.0f} MiB, reserved {r.get('peak_reserved_mib',0):,.0f} MiB, state_vq {r.get('state_vq',{}).get('recipe')}\")")"
  return 0
}

log "launch (DRYRUN=$DRYRUN)"
# preconditions: CPU proofs, the pair check, the run tree unedited at origin/sm148
"$PY" -X utf8 -c "import json, sys; sys.exit(0 if json.load(open(r'$S/check-cpu-sm149.json'))['all_ok'] else 1)" || { log "STOP: CPU proofs not ALL OK"; exit 1; }
"$PY" -X utf8 "$S/sm149_prepare.py" pair --model sm118 --root "$R" >> "$LOG" 2>&1 || { log "STOP: pair check failed"; exit 1; }
[ -z "$(git -C "$R" status --porcelain scripts rwkv expertium)" ] || { log "STOP: run tree has source edits"; exit 1; }
[ "$(git -C "$R" rev-parse HEAD)" = "$(git -C "$R" rev-parse origin/sm148)" ] || log "note: run tree HEAD is not origin/sm148"

# 1. smoke of both arms (2 users x 512 reviews each, scratch copies, ~1 min)
[ -e "$G/stage1-smoke-ok" ] || {
  q 5 "trainer smoke, both arms" -- bash -c "cd $R && env RWKV_DELETED_CARDS=neighbour SM149_CODE=$R $PY -X utf8 $RUN smoke --out artifacts/small-model/sm149-sm118-qat && env RWKV_DELETED_CARDS=neighbour SM149_CODE=$R $PY -X utf8 $RUN smoke --out artifacts/small-model/sm149-sm118-ctl" >> "$G/stage1-smoke.out" 2>&1 \
    || { log "STOP: smoke failed (see $G/stage1-smoke.out)"; exit 1; }
  [ "$DRYRUN" = 1 ] || touch "$G/stage1-smoke-ok"
}
# 2. the two arms (QAT first: the longer one, and the decisive one)
train sm149-sm118-qat 125 || exit 1   # measured: 3.08 x the control (gpu/timing-sm149.json)
train sm149-sm118-ctl 45 || exit 1
# 3. the 732 set, both arms, with the original model's REF on the 3 smoke users first (integrity vs SM148's rows)
QAT=$R/artifacts/small-model/sm149-sm118-qat/expertium/$TRACK/master.pth
CTL=$R/artifacts/small-model/sm149-sm118-ctl/expertium/$TRACK/master.pth
SCORE_FLAGS="--inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows --prior-buckets $EDGES"
ENVS="env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS PYTORCH_JIT=0"
q 3 "integrity: original Small two-head REF on the 3 smoke users" -- bash -c "$ENVS HELDOUT_GPU_MEMORY_FRACTION=0.8 $PY -X utf8 $SC --pool 732 --users-file $USMOKE --model sm118-e26-dec2ep32k-t91s2=expertium:w64:$ORIG --only sm118-e26-dec2ep32k-t91s2/w8+s8hx8 $SCORE_FLAGS --time-limit 600 --out $G/integrity-orig-smoke.jsonl" >> "$G/score.out" 2>&1 || { log "STOP: integrity ticket failed"; exit 1; }
cap=0.8
for pass in 1 2 3; do
  q 60 "scoring both arms on the 732 set (QAT: REF, Q2, T24; control: REF, Q2), pass $pass, cap $cap" -- bash -c "$ENVS HELDOUT_GPU_MEMORY_FRACTION=$cap $PY -X utf8 $SC --pool 732 --users-file $USERS732 --model sm149-sm118-qat=expertium:w64:$QAT --only sm149-sm118-qat/w8+s8hx8 --only sm149-sm118-qat/w8+lr2q8x8 --only sm149-sm118-qat/w8+lr2.16.4q8x8 --model sm149-sm118-ctl=expertium:w64:$CTL --only sm149-sm118-ctl/w8+s8hx8 --only sm149-sm118-ctl/w8+lr2q8x8 $SCORE_FLAGS --time-limit 4500 --out $ROWS" >> "$G/score.out" 2>&1 || { log "STOP: scoring ticket failed"; exit 1; }
  [ "$DRYRUN" = 1 ] && break
  n=$("$PY" -X utf8 -c "
import json
got = {}
for r in map(json.loads, open(r'$ROWS', encoding='utf-8')):
    if 'benchmark' in r: got.setdefault(r['label'], set()).add(r['user_id'])
print(min((len(v) for v in got.values()), default=0) if len(got) == 5 else 0)")
  log "732 set: $n users for every label"
  [ "$n" -ge 723 ] && break
  cap=0.92
done
[ "$DRYRUN" = 1 ] || touch "$G/stage1-gpu-DONE"
log "stage 1 GPU part done; next: HANDOFF.md step 3 (long-card CPU scoring, report, rules)"
