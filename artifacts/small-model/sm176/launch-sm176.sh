#!/usr/bin/env bash
# How many users for an MIT Small (SM176): every GPU step of the data-scaling test on the Small architecture (w64,
# the Small two-head model (SM118)); SM174's launcher with the Small model, reference and arm order (the 8,000-user
# teacher control first, then 1,000 / 2,000 / 4,000 users without a teacher), then the CPU read-out after each arm
# (design.md). One detached launcher; every GPU step is one scripts/r01_queue.py ticket (owner
# sm176, claim gpu, cap 0.8), purpose prefix "How many users for an MIT Small (SM176): ". The state lives on disk
# (folders of the run tree, gpu/<arm>-state.json, the row files): running this launcher again resumes where it stopped.
#
#   DRYRUN=1 bash launch-sm176.sh   (checks + prints the next action of every arm, runs no ticket)
#   bash launch-sm176.sh            (detached: see HANDOFF.md for the WMI line; rerun = resume)
#   stop: touch C:/Users/jonat/workspace/wt/sm176/artifacts/small-model/sm176/gpu/STOP  (a running training folder
#         snapshots at the next user boundary and exits; a running scoring ticket finishes; rerun resumes)
# Log patterns: per arm "SM176 <arm> DONE"; all done "SM176 DONE"; stopped or failed "SM176 STOP" (cause on an earlier
# line starting "STOP:").
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
R=C:/Users/jonat/workspace/wt/sm174-run   # SM174's run tree (detached at cea0f931, unedited), reused
S=C:/Users/jonat/workspace/wt/sm176/artifacts/small-model/sm176
G=$S/gpu
CH=$S/sm176_chain.py
REP=$S/sm176_report.py
DEC=$R/scripts/expertium_decay.py
BLOCK=$R/artifacts/small-model/sm135/sm135_block.py
SC=$R/scripts/small_model_heldout.py
USERS732=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json
USLICE=$S/users.json
USMOKE=$R/artifacts/small-model/sm148/users-smoke.json
ORIG=C:/Users/jonat/workspace/wt/sm118/artifacts/small-model/sm118-e26-dec2ep32k-t91s2-p2/expertium/E89-F65e-F77a/master.pth
OTAG=sm118-e26-dec2ep32k-t91s2
ARCH=w64
TRACK=E89-F65e-F77a
VAR=w8+s8hx8
EDGES=0,1,3,5,9,16,32,128,300,1000
ROWS=$G/sm176-rows-732.jsonl
SLICE=$G/sm176-slice-rows.jsonl
LOG=$G/launch-sm176.log; [ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm176-dryrun.log
P="How many users for an MIT Small (SM176): "
ARMS="n8000-kd n1000 n2000 n4000"
DRYRUN=${DRYRUN:-0}
ENVS="env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS PYTORCH_JIT=0"
SCORE_FLAGS="--inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows --prior-buckets $EDGES"
mkdir -p "$G"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
die() { log "STOP: $*"; log "SM176 STOP"; exit 1; }
block() { "$PY" -X utf8 "$BLOCK" | tr -d '\r'; }
stopped() { [ -e "$G/STOP" ]; }
done_in() { [ -f "$1/expertium/$TRACK/master.pth" ] && F="$1/expertium/$TRACK/result.json" "$PY" -X utf8 -c "
import json, os, sys
r = json.load(open(os.environ['F']))
sys.exit(0 if r.get('status') == 'complete' and not r.get('smoke_only') else 1)" 2>/dev/null; }
wait_free() {  # wait_free MINUTES: until no quiet window is on or starts within MINUTES (+5)
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
  "$PY" "$QUEUE" run --owner sm176 --purpose "$P$purpose" --claim gpu --expected-minutes "$minutes" \
    --launched-by "detached launcher sm176/launch-sm176.sh" -- "$@"
  local rc=$?; log "exit $rc: $purpose"; return $rc
}
summary() {  # one line from a finished folder's result.json
  F="$1/expertium/$TRACK/result.json" "$PY" -X utf8 -c "
import json, os
r = json.load(open(os.environ['F']))
c = r.get('expertium_catch_up') or {}
print(f\"{r.get('review_presentations', 0):,} reviews, {r.get('training_seconds', 0) / 60:.1f} min, {r.get('reviews_per_second', 0):,.0f} reviews/s, \"
      f\"peak allocated {r.get('peak_allocated_mib', 0):,.0f} MiB, reserved {r.get('peak_reserved_mib', 0):,.0f} MiB, process tree RSS {r.get('peak_process_tree_rss_mib', 0):,.0f} MiB, \"
      f\"teacher {'distillation' in c}, schedule {(c.get('lr_schedule') or {}).get('total_reviews')}, update offset end {r.get('update_offset_end')}\")" | tr -d '\r'
}
train() {  # train FOLDER MINUTES WHAT: run (resumes from a snapshot) until complete; snapshot stop on STOP / a quiet window
  local f=$1 minutes=$2 what=$3 pid b why sent rc
  while ! done_in "$f"; do
    stopped && return 1
    wait_free "$minutes" || return 1
    log "train: $what (expected ~$minutes min)"
    rm -f "$f/control/STOP"
    ( cd "$R" && "$PY" "$QUEUE" run --owner sm176 --purpose "${P}train $what" --claim gpu --expected-minutes "$minutes" \
      --launched-by "detached launcher sm176/launch-sm176.sh" -- \
      env RWKV_DELETED_CARDS=neighbour "$PY" -X utf8 "$DEC" run --out "$f" ) >> "$G/train.out" 2>&1 &
    pid=$!
    sent=0
    while kill -0 "$pid" 2>/dev/null; do
      b=$(block); why=""
      stopped && why="STOP file"
      { [[ "$b" == NOW* ]] || { [[ "$b" == NEXT* ]] && [ "${b#NEXT }" -le 2 ]; }; } && why="quiet window ($b)"
      if [ -n "$why" ] && [ $sent = 0 ]; then
        log "snapshot stop of $what ($why)"; ( cd "$R" && "$PY" -X utf8 "$DEC" stop --out "$f" > /dev/null 2>&1 ); sent=1
      fi
      sleep 30
    done
    wait "$pid"; rc=$?; log "exit $rc: train $what"
    done_in "$f" && break
    [ $sent = 0 ] && { log "STOP: $what ended without completing and without a stop request (see $G/train.out)"; return 1; }
  done
  log "$what complete: $(summary "$f")"
  return 0
}
slice_eval() {  # slice_eval TAG MASTER: the held-out slice (190 users), one ticket
  q 10 "slice eval $1 (190 held-out users)" -- bash -c "cd $R && $ENVS HELDOUT_GPU_MEMORY_FRACTION=$3 $PY -X utf8 $SC --pool 1500 --users-file $USLICE --model $1=expertium:$ARCH:$2 --only $1/$VAR $SCORE_FLAGS --time-limit 1500 --out $SLICE" >> "$G/score.out" 2>&1
}
score732() {  # score732 TAG MASTER CAP: the 732 set (723 users), one ticket
  q 30 "scoring $1 on the 732 set (723 users), cap $3" -- bash -c "cd $R && $ENVS HELDOUT_GPU_MEMORY_FRACTION=$3 $PY -X utf8 $SC --pool 732 --users-file $USERS732 --model $1=expertium:$ARCH:$2 --only $1/$VAR $SCORE_FLAGS --time-limit 5400 --out $ROWS" >> "$G/score.out" 2>&1
}

log "launch (DRYRUN=$DRYRUN)"
# preconditions: chain checks, report selftest, run tree unedited at cea0f931, scripts committed
( cd "$S" && OMP_NUM_THREADS=2 "$PY" -X utf8 "$CH" check ) > "$G/check.out" 2>&1 && grep -q "^ALL OK" "$G/check.out" || die "chain check failed (see $G/check.out)"
( cd "$S" && OMP_NUM_THREADS=2 "$PY" -X utf8 "$REP" selftest ) >> "$G/check.out" 2>&1 || die "report selftest failed (see $G/check.out)"
[ -z "$(git -C "$R" status --porcelain scripts rwkv expertium | tr -d '\r')" ] || die "run tree has source edits"
[ "$(git -C "$R" rev-parse HEAD | tr -d '\r')" = "cea0f931460996830736bdaa06e3a7d66418479b" ] || die "run tree HEAD is not cea0f931"
[ -z "$(git -C "$S" status --porcelain -- sm176_chain.py sm176_report.py launch-sm176.sh users.json template.json | tr -d '\r')" ] || die "sm176 scripts have uncommitted edits"
log "commit $(git -C "$S" rev-parse --short HEAD | tr -d '\r'); checks ALL OK"

if [ "$DRYRUN" = 1 ]; then
  for arm in $ARMS; do log "DRYRUN $arm next: $( cd "$S" && "$PY" -X utf8 "$CH" next --arm $arm | tr -d '\r')"; done
  log "SM176 DONE (dry run)"; exit 0
fi

# 0. the reference model is the pinned Small two-head (SM118) checkpoint
[ "$(sha256sum "$ORIG" | cut -d' ' -f1)" = "b4fe1175d25a7e6abff70e87ac0af1f4d76db9622d54f7cb53196fbf35117eb6" ] || die "SM118 checkpoint sha256 mismatch"
# 1. trainer smoke: the first folder of a no-teacher arm and of the teacher control (2 users x 512 reviews each, scratch)
[ -e "$G/smoke-ok" ] || {
  A1=$( cd "$S" && "$PY" -X utf8 "$CH" next --arm n1000 | tr -d '\r'); A2=$( cd "$S" && "$PY" -X utf8 "$CH" next --arm n8000-kd | tr -d '\r')
  F1=$(echo "$A1" | cut -d' ' -f2); F2=$(echo "$A2" | cut -d' ' -f2)
  [[ "$A1" == TRAIN*e01* && "$A2" == TRAIN*e01* ]] || die "smoke: first actions are not e01 trains ($A1 / $A2)"
  q 10 "trainer smoke, no-teacher and teacher arms" -- bash -c "cd $R && env RWKV_DELETED_CARDS=neighbour $PY -X utf8 $DEC smoke --out $F1 && env RWKV_DELETED_CARDS=neighbour $PY -X utf8 $DEC smoke --out $F2" >> "$G/smoke.out" 2>&1 \
    || die "smoke failed (see $G/smoke.out)"
  touch "$G/smoke-ok"; log "smoke: $(grep '"smoke"' "$G/smoke.out" | tail -2 | tr -d '\r' | tr '\n' ' ')"
}
# 2. integrity: the release model's REF on the 3 smoke users must equal SM148's rows (same scorer tree)
[ -s "$G/integrity-orig-smoke.jsonl" ] || {
  q 6 "integrity: Small two-head (SM118) REF on the 3 smoke users" -- bash -c "cd $R && $ENVS HELDOUT_GPU_MEMORY_FRACTION=0.8 $PY -X utf8 $SC --pool 732 --users-file $USMOKE --model $OTAG=expertium:$ARCH:$ORIG --only $OTAG/$VAR $SCORE_FLAGS --time-limit 600 --out $G/integrity-orig-smoke.jsonl" >> "$G/score.out" 2>&1 || die "integrity ticket failed"
  msg=$("$PY" -X utf8 -c "
import json
def rows(p):
    out = {}
    for line in open(p, encoding='utf-8'):
        r = json.loads(line)
        if 'benchmark' in r and r['label'].startswith('$OTAG/$VAR@'):
            out[r['user_id']] = r['benchmark']
    return out
a, b = rows(r'$G/integrity-orig-smoke.jsonl'), rows(r'$R/artifacts/small-model/sm148/gpu/sm148-rows-732.jsonl')
d = max(abs(a[u][h]['LogLoss'] - b[u][h]['LogLoss']) for u in a for h in ('ahead', 'instant'))
print(f'{len(a)} users, max |LogLoss difference| vs SM148 rows {d:.2e}', 'OK' if d <= 1e-6 and len(a) == 3 else 'MISMATCH')" | tr -d '\r')
  log "integrity: $msg"
  [[ "$msg" == *OK ]] || die "integrity mismatch: $msg"
}
# 3. the arms, the teacher control first, then smallest first: train / slice eval / decay / 732 scoring, as sm176_chain.py next says
for arm in $ARMS; do
  last=""; same=0
  while true; do
    stopped && die "STOP file"
    act=$( cd "$S" && "$PY" -X utf8 "$CH" next --arm $arm 2>> "$G/check.out" | tr -d '\r')
    [ -n "$act" ] || die "$arm: sm176_chain.py next failed (see $G/check.out)"
    if [ "$act" = "$last" ]; then same=$((same + 1)); else same=0; fi
    last=$act
    set -- $act
    case "$1" in
      TRAIN) train "$2" "$3" "$arm $(basename "$2")" || die "train $arm $(basename "$2")";;
      EVAL)  [ $same -ge 2 ] && die "$arm: slice eval of $2 incomplete after 2 retries"
             cap=0.8; [ $same -ge 1 ] && cap=0.92
             slice_eval "$2" "$3" $cap || die "slice eval ticket failed ($2)"
             log "$arm slice: $( cd "$S" && "$PY" -X utf8 "$CH" slice --arm $arm | tr -d '\r\n ' )";;
      SCORE) [ $same -ge 3 ] && die "$arm: 732 scoring of $2 below 720 users after 3 passes"
             cap=0.8; [ $same -ge 1 ] && cap=0.92
             score732 "$2" "$3" $cap || die "scoring ticket failed ($2)";;
      DONE)  [ -s "$G/$arm-state.json" ] && log "$arm decision: $("$PY" -X utf8 -c "import json; s=json.load(open(r'$G/$arm-state.json')); print(s['reason'], '| best epoch', s['best_epoch'], '| decay', s['decay_epochs'])" | tr -d '\r')"
             ( cd "$S" && OMP_NUM_THREADS=2 "$PY" -X utf8 "$REP" report ) > "$G/report.out" 2>&1
             log "report exit $? (0 all arms, 2 partial): $(grep -F "(SM176 $arm) |" "$S/sm176-result.md" | head -1 | tr -d '\r')"
             log "SM176 $arm DONE"
             break;;
      *) die "$arm: unknown action '$act'";;
    esac
  done
done
( cd "$S" && OMP_NUM_THREADS=2 "$PY" -X utf8 "$REP" report ) > "$G/report.out" 2>&1
log "final report exit $?: $(grep -A3 '^| threshold' "$S/sm176-result.md" | tail -2 | tr -d '\r' | tr '\n' ' ')"
log "SM176 DONE"
