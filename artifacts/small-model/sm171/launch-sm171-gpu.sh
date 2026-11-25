#!/usr/bin/env bash
# State-energy penalty (SM171): the whole GPU screen on six-layer Tiny two-head (SM130), then the CPU read-out
# (design.md "Recipe", "Scoring", "Rules"). One detached launcher; every GPU step is one scripts/r01_queue.py ticket
# (owner sm171, claim gpu, cap 0.8 from the release options), purpose prefix "State-energy penalty (SM171): ".
#
#   DRYRUN=1 bash launch-sm171-gpu.sh     (prints the tickets, runs nothing)
#   bash launch-sm171-gpu.sh              (detached: see HANDOFF.md for the WMI line; rerun = resume)
#   stop: touch C:/Users/jonat/workspace/wt/sm171/artifacts/small-model/sm171/gpu/STOP  (a running arm snapshots at the
#         next user boundary and exits; running this launcher again resumes it from the snapshot)
# Log patterns: done "SM171 DONE", stopped or failed "SM171 STOP" (cause on an earlier line starting "STOP:").
# Breaks: quiet.json windows of scope all / gpu (sm135_block.py): no ticket starts inside or within its expected length
# of a window; a running arm is snapshot-stopped <= 2 min before one and resumed after it.
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
R=C:/Users/jonat/workspace/wt/sm171-run
S=C:/Users/jonat/workspace/wt/sm171/artifacts/small-model/sm171
G=$S/gpu
RUN=$S/sm171_run.py
BLOCK=$R/artifacts/small-model/sm135/sm135_block.py
SC=$R/scripts/small_model_heldout.py
USERS732=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json
USMOKE=$R/artifacts/small-model/sm148/users-smoke.json
ORIG=C:/Users/jonat/workspace/wt/sm130/artifacts/small-model/sm130-th-dec2ep32k-p2/expertium/E89-F65e-F77a/master.pth
OTAG=sm130-th-dec2ep32k-t91s2
ARCH=w64-c1-d2p1u1
TRACK=E89-F65e-F77a
EDGES=0,1,3,5,9,16,32,128,300,1000
ROWS=$G/sm171-rows-732.jsonl
LOG=$G/launch-sm171-gpu.log
P="State-energy penalty (SM171): "
DRYRUN=${DRYRUN:-0}
mkdir -p "$G"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
die() { log "STOP: $*"; log "SM171 STOP"; exit 1; }
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
  "$PY" "$QUEUE" run --owner sm171 --purpose "$P$purpose" --claim gpu --expected-minutes "$minutes" \
    --launched-by "detached launcher sm171/launch-sm171-gpu.sh" -- "$@"
  local rc=$?; log "exit $rc: $purpose"; return $rc
}
train() {  # train FOLDER MINUTES: run (resumes from a snapshot) until complete; snapshot stop on STOP / a quiet window
  local f=$1 minutes=$2 pid b why sent
  while ! done_in "$f"; do
    stopped && return 1
    wait_free 30 || return 1
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN train $f (~$minutes min)"; echo "    cmd: env RWKV_DELETED_CARDS=neighbour SM171_CODE=$R $PY -X utf8 $RUN run --out $R/artifacts/small-model/$f" >> "$LOG"; return 0; fi
    log "train: $f (expected ~$minutes min)"
    ( cd "$R" && "$PY" "$QUEUE" run --owner sm171 --purpose "${P}train $f" --claim gpu --expected-minutes "$minutes" \
      --launched-by "detached launcher sm171/launch-sm171-gpu.sh" -- \
      env RWKV_DELETED_CARDS=neighbour SM171_CODE=$R "$PY" -X utf8 "$RUN" run --out "$R/artifacts/small-model/$f" ) >> "$G/train-$f.out" 2>&1 &
    pid=$!
    sent=0
    while kill -0 "$pid" 2>/dev/null; do
      b=$(block); why=""
      stopped && why="STOP file"
      { [[ "$b" == NOW* ]] || { [[ "$b" == NEXT* ]] && [ "${b#NEXT }" -le 2 ]; }; } && why="quiet window ($b)"
      if [ -n "$why" ] && [ $sent = 0 ]; then
        log "snapshot stop of $f ($why)"; ( cd "$R" && env SM171_CODE=$R "$PY" -X utf8 "$RUN" stop --out "$R/artifacts/small-model/$f" > /dev/null 2>&1 ); sent=1
      fi
      sleep 30
    done
    wait "$pid"; log "exit $?: train $f"
    done_in "$f" && break
    [ $sent = 0 ] && { log "STOP: $f ended without completing and without a stop request (see $G/train-$f.out)"; return 1; }
  done
  [ "$DRYRUN" = 1 ] || log "$f complete: $("$PY" -X utf8 -c "import json; r=json.load(open(r'$R/artifacts/small-model/$f/expertium/$TRACK/result.json')); print(f\"{r.get('review_presentations',0):,} reviews, {r.get('training_seconds',0)/60:.1f} min, {r.get('reviews_per_second',0):,.0f} reviews/s, peak allocated {r.get('peak_allocated_mib',0):,.0f} MiB, reserved {r.get('peak_reserved_mib',0):,.0f} MiB, lambda {r.get('state_vq',{}).get('lambda')}\")" | tr -d '\r')"
  return 0
}

log "launch (DRYRUN=$DRYRUN)"
# preconditions: CPU proofs, report selftest, the pair check, the run tree unedited at cea0f931
"$PY" -X utf8 -c "import json, sys; sys.exit(0 if json.load(open(r'$S/check-cpu-sm171.json'))['all_ok'] else 1)" || die "CPU proofs not ALL OK"
"$PY" -X utf8 "$S/sm171_prepare.py" pair --arms hi,lo,ctl --root "$R" >> "$LOG" 2>&1 || die "pair check failed"
[ -z "$(git -C "$R" status --porcelain scripts rwkv expertium)" ] || die "run tree has source edits"
[ "$(git -C "$R" rev-parse HEAD | tr -d '\r')" = "cea0f931460996830736bdaa06e3a7d66418479b" ] || die "run tree HEAD is not cea0f931"
[ -z "$(git -C "$S" status --porcelain -- sm171_energy.py sm171_run.py sm171_prepare.py sm171_report.py launch-sm171-gpu.sh)" ] || die "sm171 scripts have uncommitted edits"
log "commit $(git -C "$S" rev-parse --short HEAD | tr -d '\r'), hook sha256 $(sha256sum "$S/sm171_energy.py" | cut -c1-16)"

# 1. trainer smoke of the three arms (2 users x 512 reviews each, scratch copies, about 1 min)
[ -e "$G/smoke-ok" ] || {
  q 5 "trainer smoke, three arms" -- bash -c "cd $R && for a in hi lo ctl; do env RWKV_DELETED_CARDS=neighbour SM171_CODE=$R $PY -X utf8 $RUN smoke --out artifacts/small-model/sm171-tiny-\$a || exit 1; done" >> "$G/smoke.out" 2>&1 \
    || die "smoke failed (see $G/smoke.out)"
  [ "$DRYRUN" = 1 ] || touch "$G/smoke-ok"
}
# 2. the three arms (the strong penalty first: it decides whether the rest is worth reading)
train sm171-tiny-hi 65 || die "train sm171-tiny-hi"
train sm171-tiny-lo 65 || die "train sm171-tiny-lo"
train sm171-tiny-ctl 26 || die "train sm171-tiny-ctl"
# 3. integrity: the original model's REF on the 3 smoke users must equal SM148's rows (same scorer commit)
ENVS="env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS PYTORCH_JIT=0"
SCORE_FLAGS="--inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows --prior-buckets $EDGES"
[ -s "$G/integrity-orig-smoke.jsonl" ] || {
  q 3 "integrity: original Tiny two-head REF on the 3 smoke users" -- bash -c "cd $R && $ENVS HELDOUT_GPU_MEMORY_FRACTION=0.8 $PY -X utf8 $SC --pool 732 --users-file $USMOKE --model $OTAG=expertium:$ARCH:$ORIG --only $OTAG/w8+s8hx8 $SCORE_FLAGS --time-limit 600 --out $G/integrity-orig-smoke.jsonl" >> "$G/score.out" 2>&1 || die "integrity ticket failed"
}
[ "$DRYRUN" = 1 ] || log "integrity: $("$PY" -X utf8 -c "
import json
def rows(p):
    out = {}
    for line in open(p, encoding='utf-8'):
        r = json.loads(line)
        if 'benchmark' in r and r['label'].startswith('$OTAG/w8+s8hx8@'):
            out[r['user_id']] = r['benchmark']
    return out
a, b = rows(r'$G/integrity-orig-smoke.jsonl'), rows(r'$R/artifacts/small-model/sm148/gpu/sm148-rows-732.jsonl')
d = max(abs(a[u][h]['LogLoss'] - b[u][h]['LogLoss']) for u in a for h in ('ahead', 'instant'))
print(f'{len(a)} users, max |LogLoss difference| vs SM148 rows {d:.2e}', 'OK' if d <= 1e-6 and len(a) == 3 else 'MISMATCH')" | tr -d '\r')"
# 4. the 732 set: REF, T24 and Q2 of the three arms (9 labels), retry passes for out-of-memory users
HI=$R/artifacts/small-model/sm171-tiny-hi/expertium/$TRACK/master.pth
LO=$R/artifacts/small-model/sm171-tiny-lo/expertium/$TRACK/master.pth
CTL=$R/artifacts/small-model/sm171-tiny-ctl/expertium/$TRACK/master.pth
ONLY=""
for t in sm171-tiny-hi sm171-tiny-lo sm171-tiny-ctl; do
  for l in w8+s8hx8 w8+lr2.16.4q8x8 w8+lr2q8x8; do ONLY="$ONLY --only $t/$l"; done
done
cap=0.8
for pass in 1 2 3; do
  [ "$DRYRUN" = 1 ] || [ ! -e "$G/score-DONE" ] || break
  q 75 "scoring the three arms on the 732 set (REF, T24, Q2), pass $pass, cap $cap" -- bash -c "cd $R && $ENVS HELDOUT_GPU_MEMORY_FRACTION=$cap $PY -X utf8 $SC --pool 732 --users-file $USERS732 --model sm171-tiny-hi=expertium:$ARCH:$HI --model sm171-tiny-lo=expertium:$ARCH:$LO --model sm171-tiny-ctl=expertium:$ARCH:$CTL $ONLY $SCORE_FLAGS --time-limit 5400 --out $ROWS" >> "$G/score.out" 2>&1 || die "scoring ticket failed (pass $pass)"
  [ "$DRYRUN" = 1 ] && break
  n=$("$PY" -X utf8 -c "
import json
got = {}
for r in map(json.loads, open(r'$ROWS', encoding='utf-8')):
    if 'benchmark' in r: got.setdefault(r['label'], set()).add(r['user_id'])
print(min((len(v) for v in got.values()), default=0) if len(got) == 9 else 0)" | tr -d '\r')
  log "732 set: $n users for every label"
  [ "$n" -ge 723 ] && { touch "$G/score-DONE"; break; }
  cap=0.92
done
# 5. the read-out (CPU, seconds)
if [ "$DRYRUN" = 1 ]; then log "DRYRUN report"; log "SM171 DONE (dry run)"; exit 0; fi
( cd "$S" && OMP_NUM_THREADS=2 "$PY" -X utf8 sm171_report.py report ) >> "$LOG" 2>&1
rc=$?
log "report exit $rc (0 decided, 2 pending): $(grep -m1 '^\*\*Verdict' "$S/sm171-result.md" 2>/dev/null | tr -d '\r')"
log "SM171 DONE"
