#!/usr/bin/env bash
# 4-bit weights (SM180): the whole GPU screen, then the CPU read-outs (design.md "Step 1", "Step 2", "Run and budget").
# Adapted from launch-sm164-gpu.sh (scoring passes) and launch-sm171-gpu.sh (training arms, snapshot stops). One detached
# launcher; every GPU step is one scripts/r01_queue.py ticket, owner sm180, purpose prefix "4-bit weights (SM180): ".
#   0 preconditions   run tree at af7253de and clean, this folder committed, CPU proofs ALL OK, sha256 pins, selftest
#   1 GPU smoke       5 users x 6 models x {REF, T24, F6 at int8} + {w4g32, w4g16}+format -> sm180_report.py smoke
#   2 PTQ             per size (Tiny, Small, Big): 2 models x {w4g32, w4g16}+format on the 723-user list (cap 0.8, then
#                     0.92 after a pass without progress) -> gpu/sm180-rows-ptq.jsonl; sm180_report.py ptq -> targets
#   3 QAT (only the targets in gpu/qat-targets.txt, Tiny first): prepare + pair check, trainer smoke, train the arm (and
#                     the control unless reused), score arm / control at w4g32 and w8 -> gpu/sm180-rows-qat.jsonl;
#                     sm180_report.py qat
# Log: gpu/launch-sm180-gpu.log; last line "SM180 GPU DONE ..." or "SM180 GPU STOP ..." (cause on an earlier "STOP:" line).
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm180/artifacts/small-model/sm180/launch-sm180-gpu.sh
#   launch   WMI hidden (HANDOFF.md), stdout to gpu/launch-sm180-gpu.nohup; rerun = resume (rows, smoke-ok, snapshots kept)
#   stop     touch C:/Users/jonat/workspace/wt/sm180/artifacts/small-model/sm180/gpu/STOP (a running arm snapshots at the
#            next user boundary and exits; a scoring ticket finishes; nothing new starts)
# Do not edit scripts/ expertium/ rwkv/ of wt/sm180-run while it runs (the trainer's source watch, the clean-tree check).
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
R=C:/Users/jonat/workspace/wt/sm180-run
RUN_COMMIT=81c2e933
S=C:/Users/jonat/workspace/wt/sm180/artifacts/small-model/sm180
G=$S/gpu
mkdir -p "$G"
LOG=$G/launch-sm180-gpu.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm180-gpu-dryrun.log
SC=$R/scripts/small_model_heldout.py
DECAY=$R/scripts/expertium_decay.py
RULE=$S/sm180_report.py
PREP=$S/sm180_prepare.py
BLOCK=$R/artifacts/small-model/sm135/sm135_block.py
USMOKE=$R/artifacts/small-model/sm164/users-smoke.json
U732=$R/artifacts/small-model/sm72s/eval-users-732-fit.json
. "$S/models.sh"
EDGES=0,1,3,5,9,16,32,128,300,1000
SMOKEROWS=$G/smoke-gpu-rows.jsonl
PTQROWS=$G/sm180-rows-ptq.jsonl
QATROWS=$G/sm180-rows-qat.jsonl
TICKETS=$G/tickets.jsonl
STOPF=$G/STOP
P="4-bit weights (SM180): "
F6=lr2.16.4q6x8; T24=lr2.16.4q8x8
# PTQ jobs: size|expected minutes of pass 1 (SM164's per-(model, label) minutes scaled to 31.9M reviews, +30%)
PTQJOBS="
tiny|45
small|55
big|90
"
MIN_RETRY=40
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START SM11_INPUTS SM157_STATS
log() { echo "[sm180 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
stopped() { [ -e "$STOPF" ] && { log "STOP: STOP file present"; return 0; }; return 1; }
block() { "$PY" -X utf8 "$BLOCK" | tr -d '\r'; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'; }
jline() { "$PY" -X utf8 -c "import json, sys; print(json.dumps(dict(zip(sys.argv[1::2], [int(v) if v.lstrip('-').isdigit() else v for v in sys.argv[2::2]]))))" "$@" | tr -d '\r'; }

wait_free() {  # wait_free MINUTES: until no quiet window is on or starts within MINUTES (+5)
  local b n=0
  while true; do
    stopped && return 1
    b=$(block)
    if [ "$DRYRUN" = 1 ]; then return 0; fi
    case "$b" in
      NOW*) [ $((n % 6)) = 0 ] && log "quiet window on ($b); waiting"; n=$((n + 1)); sleep 300; continue;;
      NEXT*) [ "${b#NEXT }" -ge $(( $1 + 5 )) ] && return 0
             [ $((n % 6)) = 0 ] && log "next quiet window in ${b#NEXT } min (< $1 + 5 min needed); waiting"; n=$((n + 1)); sleep 300;;
      *) log "window check failed: $b"; sleep 300;;
    esac
  done
}

q() {  # q MINUTES KIND PURPOSE -- CMD...   one queue ticket
  local minutes=$1 kind=$2 purpose=$3 rc t0 mx base wpid; shift 4
  wait_free "$minutes" || return 1
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN ticket ($minutes min): $P$purpose"; echo "    cmd: $*" >> "$LOG"; return 0; fi
  log "ticket: $purpose (expected $minutes min)"
  t0=$(date +%s); base=$(gpu_used); rm -f "$G/.gpu-max"
  ( m=0; while true; do u=$(gpu_used); [ -n "$u" ] && [ "$u" -gt "$m" ] && { m=$u; echo "$m" > "$G/.gpu-max"; }; sleep 30; done ) &
  wpid=$!
  "$PY" "$QUEUE" run --owner sm180 --purpose "$P$purpose" --claim gpu --expected-minutes "$minutes" \
    --launched-by "detached launcher sm180/launch-sm180-gpu.sh" -- "$@"
  rc=$?; kill "$wpid" 2>/dev/null; wait "$wpid" 2>/dev/null; mx=$(cat "$G/.gpu-max" 2>/dev/null | tr -dc '0-9'); mx=${mx:-0}
  log "exit $rc: $purpose (nvidia-smi max $mx MiB, baseline ${base:-0} MiB)"
  jline kind "$kind" purpose "$purpose" exit "$rc" minutes "$(( ($(date +%s) - t0 + 30) / 60 ))" nvidia_smi_max_mib "$mx" nvidia_smi_baseline_mib "${base:-0}" >> "$TICKETS"
  return $rc
}

min_users() {  # min_users ROWS USERSFILE LABEL...: the minimum over labels of users with a benchmark / non-finite row
  F="$1" U="$2" "$PY" -X utf8 -c "
import json, os, sys
from pathlib import Path
p = Path(os.environ['F'])
want = set(json.loads(Path(os.environ['U']).read_text())['users'])
got = {}
if p.exists():
    for raw in open(p, 'rb'):
        try:
            r = json.loads(raw.decode('utf-8', errors='replace'))
        except ValueError:
            continue
        if r.get('user_id') in want and ('benchmark' in r or ('nonfinite' in r and 'error' not in r)):
            got.setdefault(r.get('label'), set()).add(r['user_id'])
print(min(len(got.get(l, ())) for l in sys.argv[1:]))" "${@:3}" | tr -d '\r'; }

SCORE_FLAGS=(--inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows --prior-buckets "$EDGES")
ENVS=(env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS -u SM157_STATS PYTORCH_JIT=0)

passes() {  # passes WHAT ROWS MINUTES: score GARGS / GLBLS on the 723-user list until covered (at most 4 passes)
  local what=$1 rows=$2 min1=$3 cap=0.8 pass have prev=-1 minutes want=715
  for pass in 1 2 3 4; do
    have=$(min_users "$rows" "$U732" "${GLBLS[@]}")
    [ "$DRYRUN" != 1 ] && [ "$have" -ge 723 ] && break
    if [ "$have" = "$prev" ]; then
      if [ "$cap" = 0.8 ]; then cap=0.92; log "no progress on $what ($have users): next pass at cap 0.92 (memory only, same numerics)"
      else log "no progress at cap 0.92 on $what ($have users); going on"; break; fi
    fi
    prev=$have
    minutes=$min1; [ "$pass" -gt 1 ] && minutes=$MIN_RETRY
    q $minutes score "$what, 723-user list of the 732 set, e2s-query (pass $pass, $have done, cap $cap)" -- \
      "${ENVS[@]}" HELDOUT_GPU_MEMORY_FRACTION=$cap "$PY" -X utf8 -W ignore "$SC" --pool 732 --users-file "$U732" \
      "${GARGS[@]}" "${SCORE_FLAGS[@]}" --time-limit $(( minutes * 72 )) --out "$rows" >> "$G/score.out" 2>&1 \
      || { log "STOP: scoring ticket failed ($what)"; return 1; }
    [ "$DRYRUN" = 1 ] && return 0
  done
  have=$(min_users "$rows" "$U732" "${GLBLS[@]}")
  log "$what: $have of 723 users for every label (${#GLBLS[@]} labels; the rule needs $want)"
  return 0
}

fmt_of() { case "$1" in tiny) echo "$T24";; *) echo "$F6";; esac; }

ptq_args() {  # ptq_args SIZE: GARGS / GLBLS for the 2 models of SIZE x {w4g32, w4g16}
  GARGS=(); GLBLS=()
  local tag ka path digest role size w f
  f=$(fmt_of "$1")
  while IFS='|' read -r tag ka path digest role size; do
    [ -z "$tag" ] || [ "$size" != "$1" ] && continue
    GARGS+=(--model "$tag=$ka:$path")
    for w in w4g32 w4g16; do GARGS+=(--only "$tag/$w+$f"); GLBLS+=("$tag/$w+$f@e2s-query"); done
  done <<< "$MODELS"
}

smoke() {
  [ -e "$G/smoke-ok" ] && { log "smoke kept: $G/smoke-ok"; return 0; }
  GARGS=()
  local tag ka path digest role size f
  while IFS='|' read -r tag ka path digest role size; do
    [ -z "$tag" ] && continue
    f=$(fmt_of "$size")
    GARGS+=(--model "$tag=$ka:$path" --only "$tag/w8+s8hx8" --only "$tag/w8+$T24" --only "$tag/w8+$F6"
            --only "$tag/w4g32+$f" --only "$tag/w4g16+$f")
  done <<< "$MODELS"
  q 15 smoke "GPU smoke: 5 users, 6 models, int8 REF / T24 / F6 (vs SM164's rows, bitwise) + int4 g32 / g16" -- \
    "${ENVS[@]}" HELDOUT_GPU_MEMORY_FRACTION=0.8 "$PY" -X utf8 -W ignore "$SC" --pool 1500 --users-file "$USMOKE" \
    "${GARGS[@]}" "${SCORE_FLAGS[@]}" --time-limit 900 --out "$SMOKEROWS" >> "$G/score.out" 2>&1 \
    || { log "STOP: smoke ticket failed (see $G/score.out)"; return 1; }
  [ "$DRYRUN" = 1 ] && return 0
  local out; out=$("$PY" -X utf8 "$RULE" smoke 2>&1 | tr -d '\r')
  echo "$out" > "$G/smoke-check.txt"; log "$out"
  case "$out" in *"smoke OK"*) touch "$G/smoke-ok"; return 0;; esac
  log "STOP: GPU smoke mismatch (see $G/smoke-check.txt, $G/smoke-check.json)"; return 1
}

done_in() { R2="$1/expertium/$TRACK/result.json" "$PY" -X utf8 -c "
import json, os, sys
p = os.environ['R2']
sys.exit(0 if os.path.exists(p) and json.load(open(p)).get('status') == 'complete' else 1)"; }

train() {  # train FOLDER MINUTES: run (resumes from a snapshot) until complete; snapshot stop on STOP / a quiet window
  local f=$1 minutes=$2 name pid b why sent
  name=$(basename "$f")
  while ! done_in "$f"; do
    [ -e "$STOPF" ] && { log "STOP: STOP file present"; return 1; }
    wait_free 30 || return 1
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN train $name (~$minutes min)"; echo "    cmd: env RWKV_DELETED_CARDS=neighbour $PY -X utf8 $DECAY run --out $f" >> "$LOG"; return 0; fi
    log "train: $name (expected ~$minutes min)"
    ( cd "$R" && "$PY" "$QUEUE" run --owner sm180 --purpose "${P}train $name" --claim gpu --expected-minutes "$minutes" \
      --launched-by "detached launcher sm180/launch-sm180-gpu.sh" -- \
      env RWKV_DELETED_CARDS=neighbour "$PY" -X utf8 "$DECAY" run --out "$f" ) >> "$G/train-$name.out" 2>&1 &
    pid=$!
    sent=0
    while kill -0 "$pid" 2>/dev/null; do
      b=$(block); why=""
      [ -e "$STOPF" ] && why="STOP file"
      { [[ "$b" == NOW* ]] || { [[ "$b" == NEXT* ]] && [ "${b#NEXT }" -le 2 ]; }; } && why="quiet window ($b)"
      if [ -n "$why" ] && [ $sent = 0 ]; then
        log "snapshot stop of $name ($why)"; ( cd "$R" && "$PY" -X utf8 "$DECAY" stop --out "$f" > /dev/null 2>&1 ); sent=1
      fi
      sleep 30
    done
    wait "$pid"; log "exit $?: train $name"
    done_in "$f" && break
    [ $sent = 0 ] && { log "STOP: $name ended without completing and without a stop request (see $G/train-$name.out)"; return 1; }
  done
  [ "$DRYRUN" = 1 ] || log "$name complete: $("$PY" -X utf8 -c "import json; r=json.load(open(r'$f/expertium/$TRACK/result.json')); print(f\"{r.get('review_presentations',0):,} reviews, {r.get('training_seconds',0)/60:.1f} min, {r.get('reviews_per_second',0):,.0f} reviews/s, peak allocated {r.get('peak_allocated_mib',0):,.0f} MiB, reserved {r.get('peak_reserved_mib',0):,.0f} MiB, state_vq {r.get('state_vq')}\")" | tr -d '\r')"
  echo "{\"kind\": \"train\", \"folder\": \"$name\"}" >> "$TICKETS"
  return 0
}

qat_one() {  # qat_one KEY: the QAT arm (and control) of one release model, then its 4 scoring labels
  local key=$1 arch qf cf ctag qtag f tmin cmin
  case "$key" in
    sm118) arch=w64; f=$F6; tmin=50; cmin=45;;
    sm113) arch=w64; f=$F6; tmin=70; cmin=65;;
    sm130th) arch=w64-c1-d2p1u1; f=$T24; tmin=30; cmin=0;;
    sm130io) arch=w64-c1-d2p1u1; f=$T24; tmin=30; cmin=25;;
    *) log "STOP: unknown QAT target $key"; return 1;;
  esac
  qtag=sm180-$key-qat; qf=$R/artifacts/small-model/$qtag
  if [ "$key" = sm130th ]; then ctag=sm171-tiny-ctl; cf=C:/Users/jonat/workspace/wt/sm171-run/artifacts/small-model/sm171-tiny-ctl
  else ctag=sm180-$key-ctl; cf=$R/artifacts/small-model/$ctag; fi
  log "QAT target $key: arm $qtag, control $ctag (estimate: train ~$tmin min + control ~$cmin min + scoring)"
  if [ "$DRYRUN" != 1 ]; then
    [ -d "$qf" ] || "$PY" -X utf8 "$PREP" arm --model "$key" --arm qat --root "$R" >> "$LOG" 2>&1 || { log "STOP: prepare $qtag"; return 1; }
    [ "$key" = sm130th ] || [ -d "$cf" ] || "$PY" -X utf8 "$PREP" arm --model "$key" --arm ctl --root "$R" >> "$LOG" 2>&1 || { log "STOP: prepare $ctag"; return 1; }
    "$PY" -X utf8 "$PREP" pair --model "$key" --root "$R" >> "$LOG" 2>&1 || { log "STOP: pair check failed for $key"; return 1; }
  fi
  if [ ! -e "$G/trainsmoke-$key-ok" ]; then
    local sm="cd $R && env RWKV_DELETED_CARDS=neighbour $PY -X utf8 $DECAY smoke --out $qf"
    [ "$key" = sm130th ] || sm="$sm && env RWKV_DELETED_CARDS=neighbour $PY -X utf8 $DECAY smoke --out $cf"
    local what="QAT arm and control"; [ "$key" = sm130th ] && what="QAT arm (control reused)"
    q 8 trainsmoke "trainer smoke, $what of $key" -- bash -c "$sm" >> "$G/trainsmoke-$key.out" 2>&1 \
      || { log "STOP: trainer smoke failed for $key (see $G/trainsmoke-$key.out)"; return 1; }
    if [ "$DRYRUN" != 1 ]; then
      grep -q '\[weight-qat\] first downcast' "$G/trainsmoke-$key.out" || { log "STOP: trainer smoke of $key did not apply the QAT downcast"; return 1; }
      touch "$G/trainsmoke-$key-ok"
    fi
  fi
  train "$qf" "$tmin" || return 1
  if [ "$DRYRUN" != 1 ]; then
    "$PY" -X utf8 -c "import json, sys; r = json.load(open(r'$qf/expertium/$TRACK/result.json')); s = r.get('state_vq') or {}; sys.exit(0 if s.get('recipe') == 'weight-qat' and s.get('downcasts', 0) > 0 else 1)"       || { log "STOP: $qtag result.json shows no QAT downcast"; return 1; }
  fi
  [ "$key" = sm130th ] || { train "$cf" "$cmin" || return 1; }
  GARGS=(--model "$qtag=expertium:$arch:$qf/expertium/$TRACK/master.pth" --model "$ctag=expertium:$arch:$cf/expertium/$TRACK/master.pth")
  GLBLS=()
  for t in "$qtag" "$ctag"; do for w in w4g32 w8; do GARGS+=(--only "$t/$w+$f"); GLBLS+=("$t/$w+$f@e2s-query"); done; done
  local smin=40; [ "$arch" = w64 ] && smin=50
  passes "QAT $key: arm and control at w4g32 / w8" "$QATROWS" "$smin" || return 1
  return 0
}

preconditions() {
  for f in "$BLOCK" "$RULE" "$PREP" "$S/models.sh" "$S/design.md" "$S/check-cpu-sm180.json" "$USMOKE" "$U732" "$SC" "$DECAY"; do
    [ -f "$f" ] || { log "STOP: missing $f"; return 1; }
  done
  [ "$(git -C "$R" rev-parse --short=8 HEAD | tr -d '\r')" = "$RUN_COMMIT" ] || { log "STOP: run tree HEAD is not $RUN_COMMIT"; return 1; }
  [ -z "$(git -C "$R" status --porcelain -- scripts expertium rwkv)" ] || { log "STOP: run tree has source edits"; return 1; }
  if [ "$DRYRUN" != 1 ] && [ -n "$(git -C "$S" status --porcelain -- '*.py' '*.sh' design.md)" ]; then
    log "STOP: uncommitted sm180 scripts"; return 1; fi
  "$PY" -X utf8 -c "import json, sys; d = json.load(open(r'$S/check-cpu-sm180.json')); sys.exit(0 if d['all_ok'] and d['commit'].startswith('$RUN_COMMIT') and not d['dirty'] else 1)" \
    || { log "STOP: CPU proofs not ALL OK on the run tree commit"; return 1; }
  local tag ka path digest role size got
  while IFS='|' read -r tag ka path digest role size; do
    [ -z "$tag" ] && continue
    got=$(sha256sum "$path" | cut -d' ' -f1)
    [ "$got" = "$digest" ] || { log "STOP: sha256 of $tag is $got, pinned $digest"; return 1; }
  done <<< "$MODELS"
  "$PY" -X utf8 "$RULE" selftest >> "$LOG" 2>&1 || { log "STOP: report selftest failed"; return 1; }
  return 0
}

cd "$R" || exit 1
log "started: sm180 $(git -C "$S" log --oneline -1 | cut -c1-80) | run tree $(git -C "$R" log --oneline -1 | cut -c1-60) (DRYRUN=$DRYRUN)"
rc=0
stopped && rc=1
[ $rc -eq 0 ] && { preconditions || rc=1; }
[ $rc -eq 0 ] && { smoke || rc=1; }
if [ $rc -eq 0 ]; then
  mapfile -t JOBL < <(printf '%s\n' "$PTQJOBS" | tr -d '\r' | grep -v '^$')
  for job in "${JOBL[@]}"; do
    IFS='|' read -r size minutes <<< "$job"
    stopped && { rc=1; break; }
    ptq_args "$size"
    passes "PTQ $size (2 models x w4g32 / w4g16)" "$PTQROWS" "$minutes" || { rc=1; break; }
  done
fi
if [ $rc -eq 0 ] && [ "$DRYRUN" != 1 ]; then
  out=$(cd "$S" && OMP_NUM_THREADS=2 "$PY" -X utf8 -W ignore "$RULE" ptq 2>&1 | tr -d '\r'); prc=$?
  log "PTQ report (exit $prc: 0 decided, 2 pending, 3 invalid):"; echo "$out" | sed 's/^/    /' >> "$LOG"
  [ $prc -eq 0 ] || { log "STOP: PTQ read-out not decided (exit $prc)"; rc=1; }
fi
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then TARGETS="sm130th sm118"; log "DRYRUN: QAT steps shown for sm130th and sm118"
  else TARGETS=$(tr -d '\r' < "$G/qat-targets.txt" | xargs); fi
  [ -z "$TARGETS" ] && log "no QAT target: PTQ decided every Small / Tiny size (no QAT step)"
  for key in $TARGETS; do
    stopped && { rc=1; break; }
    qat_one "$key" || { rc=1; break; }
  done
  if [ $rc -eq 0 ] && [ -n "$TARGETS" ] && [ "$DRYRUN" != 1 ]; then
    out=$(cd "$S" && OMP_NUM_THREADS=2 "$PY" -X utf8 -W ignore "$RULE" qat 2>&1 | tr -d '\r'); qrc=$?
    log "QAT report (exit $qrc: 0 decided, 2 pending, 3 invalid):"; echo "$out" | sed 's/^/    /' >> "$LOG"
  fi
fi
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; else log "SM180 GPU DONE: PTQ read-out sm180-ptq.md${TARGETS:+, QAT read-out sm180-qat.md}"; fi
else
  log "SM180 GPU STOP: ended with exit $rc"
fi
exit $rc
