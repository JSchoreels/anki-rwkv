#!/usr/bin/env bash
# Tiny architecture additions (SM77) stage 1 (artifacts/small-model/sm77/DESIGN.md on main): one two-arm warm start
# from the leak-free small model's (SM72s) constant-LR epoch K (START_EPOCH, default 20; 25 once it exists):
#   arm        w64-sm77 = key extras (A: separate removal / insertion key vectors + r / k biases, 3,328 values) and a
#              learned initial state for the card and note streams (B: S0 + first-review shift offsets, 3,392 values),
#              all zero at the start (function preserving, checked on CPU and in ticket 0 on GPU);
#   reference  the same recipe with the flags off (w64). Its constant-LR epoch is SM72s's own epoch K+1 when that
#              exists (K = 20..24: identical protocol, free), else it is trained here (K = 25).
# Recipe per arm: epoch K+1 = SM72s's epoch-(K+1) protocol rebuilt (same user order, seeds, options, AdamW state;
#   scripts/sm77_warm.py check-protocol reproduces epochs 2, 11, 21, 22 exactly), ~106 min; then a QUARTER-epoch 32k
#   Muon cool-down (SM72s's cool-down options, fraction 0.25 = ~145M reviews, the same users / order / schedule for both
#   arms, fresh Muon), ~2 h; then the 732 scoring (723 fit users + the 3 long users at the 0.92 cap), leak-free inputs.
# Rule (scripts/sm77_rule.py decide, CPU): 722 users, points of % of the big model (1 leak-free epoch); go if curve
#   <= -0.20 or instant <= -0.15 (CI clear of zero, other head <= +0.03); stop if both > -0.10; grey otherwise.
# Steps: wait for the better teacher's (SM65) launcher to end -> CPU preparation -> ticket 0 GPU check (new dS0
#   backward kernel, scorer kernel from S0, trainer-path function preservation on 2 users x 512 reviews) + trainer
#   smoke of the arm -> arm epoch -> [reference epoch] -> arm cool-down -> reference cool-down -> scoring -> rule.
# Breaks: quiet.json (scope all/gpu) plus the daily 20:00-01:00 break as a safeguard, re-read at every check (<= 10
#   min). An epoch starts only when it fits before the next window (snapshot-resumable if one is cut anyway);
#   cool-downs run in segments (snapshot stop <= 2 min before a window, resume after). CPU steps stay out of windows.
#   dry run  DRYRUN=1 [START_EPOCH=K] bash C:/Users/jonat/workspace/wt/sm77/artifacts/small-model/sm77/launch-sm77.sh
#   launch   WMI hidden (HANDOFF-sm77.md); log artifacts/small-model/sm77/gpu/launch-sm77.log
#   resume   the launch command again (finished steps are skipped; an interrupted epoch / cool-down resumes from its snapshot)
#   stop     touch C:/Users/jonat/workspace/wt/sm77/artifacts/small-model/sm77/gpu/STOP
#   no wait  touch .../sm77/gpu/NO-WAIT (skip the wait for SM65's launcher)
# Do not edit scripts/ expertium/ rwkv/ of the sm77 worktree while it runs (SourceWatch stops the training).
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
BASH_EXE="C:/Program Files/Git/usr/bin/bash.exe"
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
QUIET=C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json
WT=C:/Users/jonat/workspace/wt/sm77
SM=$WT/artifacts/small-model
S=$SM/sm77
G=$S/gpu
mkdir -p "$G"
LOG=$G/launch-sm77.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm77-dryrun.log
WARM="$PY -X utf8 $WT/scripts/sm77_warm.py"
DEC_PY=$WT/scripts/expertium_decay.py
SC=$WT/scripts/small_model_heldout.py
RULE=$WT/scripts/sm77_rule.py
TRACK=E89-F65e-F77a
ARCH=w64-sm77
K=${START_EPOCH:-20}; K1=$((K + 1))
RUN72S=C:/Users/jonat/workspace/wt/sm72s/artifacts/accuracy/experiments/scale-w64kd-final-sc-lanes-leakfree/run   # read only
S72=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s                                                  # read only
SM65LOG=C:/Users/jonat/workspace/wt/sm65/artifacts/small-model/sm65/gpu/launch-sm65-teacher.log                     # read only
BIGCK=C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72-big-leakfree-1ep/expertium/big-leakfree-1ep/master.pth
REPLAY=C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/big-leakfree-1ep-final8k-neighbour
USERS732=$S72/eval-users-732-fit.json; ULONG=$S72/users-732-long.json
P=sm77; [ "${DRYRUN:-0}" = 1 ] && P=sm77-dryrun   # the dry run prepares scratch folders and deletes them
EXT=$SM/$P-ext-e$K
ARM=$SM/$P-arm-e$K1
if [ -f "$RUN72S/epoch-$(printf %02d $K1)/expertium/$TRACK/master.pth" ]; then
  REF=$RUN72S/epoch-$(printf %02d $K1); REF_FREE=1   # SM72s's own epoch K+1 (identical protocol, flags off)
else
  REF=$SM/$P-ref-e$K1; REF_FREE=0
fi
DARM=$SM/$P-arm-e$K1-dec-q32k-muon
DREF=$SM/$P-ref-e$K1-dec-q32k-muon
ALBL=sm77-arm-e$K1-q32k; RLBL=sm77-ref-e$K1-q32k
OUTROWS=$G/sm77-neighbour.jsonl
MODE=neighbour
LF="env RWKV_DELETED_CARDS=$MODE"
SPEC='{"lanes": "strided"}'
DIST="{\"teacher\": \"$BIGCK\", \"alpha\": 0.9, \"decay_alpha\": 0.5, \"kd_source\": \"replay\", \"replay\": \"$REPLAY\"}"
EXTRA="{\"distillation\": $DIST, \"state_carry\": $SPEC, \"chunk_reviews\": 32768, \"batch_chunks\": 1, \"optimizer\": \"expertium-iter53\"}"  # = SM72s's cool-down
FRACTION=0.25
MIN_EPOCH=${MIN_EPOCH:-125}   # SM72s epochs: 108-112 min training, ~120 min tickets
MIN_DECAY=${MIN_DECAY:-150}   # quarter cool-down: ~145M reviews at ~20.2k reviews/s = 120 min + setup
MIN_CHECK=15; MIN_SCORE732=12; MIN_LONG=6
NAME="tiny architecture additions (SM77) stage 1"
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS   # scorer cap 0.8; trainer cap 0.8 (options)
log() { echo "[sm77 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
stopped() { [ -e "$G/STOP" ] && { log "STOP: STOP file present"; return 0; }; return 1; }
nap() { [ "$DRYRUN" = 1 ] && return 0; local s=$1; [ "$s" -gt 600 ] && s=600; sleep "$s"; }

block() {  # -> "NOW <seconds to its end>" inside a quiet window (quiet.json scope all/gpu, or the daily break), else "NEXT <minutes>"
  Q="$QUIET" "$PY" -X utf8 - <<'PYEOF'
import json, os, re
from datetime import datetime, timedelta
now = datetime.now()
try:
    text = open(os.environ["Q"], encoding="utf-8").read()
except OSError as error:
    print(f"ERR quiet.json unreadable: {error}"); raise SystemExit
try:
    wins = [(w["start"], w["end"]) for w in json.loads(text) if w.get("scope", "all") in ("all", "gpu")]
except (ValueError, KeyError, TypeError):  # malformed JSON: take every start/end pair textually, scope assumed "all"
    wins = re.findall(r'"start"\s*:\s*"([^"]+)"[^{}]*?"end"\s*:\s*"([^"]+)"', text, re.S)
blocks = [(datetime.fromisoformat(s), datetime.fromisoformat(e)) for s, e in wins]
day = now.replace(hour=0, minute=0, second=0, microsecond=0)
blocks += [(day + timedelta(days=d, hours=20), day + timedelta(days=d + 1, hours=1)) for d in range(-1, 9)]  # daily break
blocks.sort()
merged = []
for s, e in blocks:
    if merged and s <= merged[-1][1]:
        merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    else:
        merged.append((s, e))
for s, e in merged:
    if s <= now < e:
        print(f"NOW {int((e - now).total_seconds()) + 60}"); raise SystemExit
    if s > now:
        print(f"NEXT {int((s - now).total_seconds() // 60)}"); raise SystemExit
print("NEXT 100000")
PYEOF
}

wait_free() {  # wait_free MINUTES WHAT: returns once MINUTES fit before the next window (sets FREE = minutes to it)
  local n=0
  while true; do
    stopped && return 1
    local b; b=$(block)
    case "$b" in
      NOW*) local s=${b#NOW }
            if [ "$DRYRUN" = 1 ]; then log "DRYRUN: $2 would wait inside a quiet window (${s}s left)"; FREE=100000; return 0; fi
            [ $((n % 6)) = 0 ] && log "$2: quiet window, ${s}s left"; n=$((n + 1)); nap "$s";;
      NEXT*) FREE=${b#NEXT }
             [ "$FREE" -ge $(( $1 + 3 )) ] && return 0
             if [ "$DRYRUN" = 1 ]; then log "DRYRUN: $2 ($1 min) would wait: $FREE min to the next window"; FREE=100000; return 0; fi
             [ $((n % 6)) = 0 ] && log "$2 needs $1 min, $FREE min before the next window; waiting"; n=$((n + 1)); nap $(( FREE * 60 + 60 ));;
      *) log "window check failed: $b"; nap 600;;
    esac
  done
}

q() {  # q MINUTES PURPOSE -- CMD...   one ticket that must end before the next window
  local minutes=$1 purpose=$2 rc; shift 3
  while true; do
    wait_free "$minutes" "$purpose" || return 1
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $purpose (expected $minutes min)"; echo "    cmd: $*" >> "$LOG"; return 0; fi
    local maxwait=$(( FREE - minutes - 3 )); [ "$maxwait" -lt 1 ] && maxwait=1
    log "enqueue: $purpose (expected $minutes min, max wait $maxwait min, $FREE min to the next window)"
    "$PY" "$QUEUE" run --owner small-model --purpose "small-model: $NAME: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm77/launch-sm77.sh" -- "$@"
    rc=$?; log "exit $rc: $purpose"
    [ "$rc" = 75 ] && { log "queue wait would reach the next window; re-planning"; sleep 60; continue; }
    return $rc
  done
}
cpu_step() { wait_free "$1" "$2"; }   # CPU work also stays out of the windows
label_users() { F="$1" LB="$2" "$PY" -X utf8 -c "
import json, os
from pathlib import Path
p = Path(os.environ['F'])
print(len({r['user_id'] for r in map(json.loads, open(p, encoding='utf-8')) if r.get('label') == os.environ['LB'] and 'benchmark_existing' in r}) if p.exists() else 0)"; }
done_in() { [ -f "$1/expertium/$TRACK/master.pth" ] && R="$1/expertium/$TRACK/result.json" "$PY" -X utf8 -c "
import json, os, sys
r = json.load(open(os.environ['R']))
sys.exit(0 if r.get('status') == 'complete' and not r.get('smoke_only') else 1)" 2>/dev/null; }

# ---- wait for the better teacher's (SM65) launcher: its log's last line is "done" or a failure / STOP line
sm65_last() { tail -1 "$SM65LOG" 2>/dev/null; }
sm65_ended() { local l; l=$(sm65_last)
  [[ "$l" == *"] done" ]] || [[ "$l" == *"] STOP"* ]] || [[ "$l" == *"STOP file present"* ]] || [[ "$l" =~ (failed|stopped|no\ prepared\ run|no\ checkpoint|rule\ exit\ [1-9]) ]]; }
wait_sm65() {
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: would wait for the better teacher's (SM65) launcher; its last line now: $(sm65_last | cut -c1-200)"; return 0; fi
  local n=0
  until sm65_ended || [ -e "$G/NO-WAIT" ]; do
    stopped && return 1
    [ $(( n % 30 )) = 0 ] && log "waiting for the better teacher's (SM65) launcher to end; its last line: $(sm65_last | cut -c1-200)"
    n=$(( n + 1 )); sleep 120
  done
  log "SM65 launcher ended (or NO-WAIT): $(sm65_last | cut -c1-200)"
}

preconditions() {
  local e; e=$RUN72S/epoch-$(printf %02d $K)/expertium/$TRACK
  for f in "$e/master.pth" "$e/optimizer.pth" "$e/result.json" "$RUN72S/run.json" "$BIGCK" "$REPLAY/manifest.json" "$USERS732" "$ULONG" \
           "C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72/gpu-big/bigleak-neighbour.jsonl"; do
    [ -f "$f" ] || { log "STOP: missing $f"; return 1; }
  done
  R="$e/result.json" "$PY" -X utf8 -c "import json, os, sys; sys.exit(0 if json.load(open(os.environ['R'])).get('status') == 'complete' else 1)" \
    || { log "STOP: SM72s epoch $K is not complete"; return 1; }
  if [ "$DRYRUN" != 1 ] && [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    log "STOP: uncommitted source changes in scripts/ expertium/ rwkv/ (SourceWatch refuses them)"; return 1; fi
  return 0
}

pair_check() {  # pair_check KIND A B: two prepared folders identical except the start weights, the architecture and paths
  KIND="$1" A="$2" B="$3" "$PY" -X utf8 - <<'PYEOF'
import json, os
from pathlib import Path
a, b, kind = Path(os.environ["A"]), Path(os.environ["B"]), os.environ["KIND"]
def strip_protocol(p, folder):
    p = json.loads(json.dumps(p)); f = p["full_run"]
    for k in ("init_master", "init_optimizer", "control_dir", "benchmark_db", "update_offset" if kind == "decay" else "-"):
        f.pop(k, None)
    return p
pa, pb = json.loads((a / "protocol.json").read_text()), json.loads((b / "protocol.json").read_text())
fail = []
if strip_protocol(pa, a) != strip_protocol(pb, b):
    fail.append("protocol.json differs beyond the start weights and paths")
if (b / "decay.json").exists():
    da, db = json.loads((a / "decay.json").read_text()), json.loads((b / "decay.json").read_text())
    oa, ob = dict(da["options"]), dict(db["options"])
    if oa.pop("architecture", None) != "w64-sm77" or ob.pop("architecture", "w64") != "w64":
        fail.append("architectures are not w64-sm77 / w64")
    if oa != ob:
        fail.append(f"options differ: { {k: (oa.get(k), ob.get(k)) for k in set(oa) | set(ob) if oa.get(k) != ob.get(k)} }")
    for k in ("training_users", "training_reviews", "shuffle_seed", "lr", "fraction_of_epoch_reviews", "model", "track"):
        if da.get(k) != db.get(k):
            fail.append(f"{k}: {da.get(k)} vs {db.get(k)}")
else:   # the free reference: SM72s's own epoch folder
    oa = dict(json.loads((a / "decay.json").read_text())["options"]); oa.pop("architecture")
    run = dict(json.loads((b.parent / "run.json").read_text())["options"]); run.pop("architecture", None)
    if oa != run:
        fail.append("arm options differ from SM72s's run options beyond the architecture")
if kind == "decay" and pa["full_run"].get("update_offset") != pb["full_run"].get("update_offset"):
    fail.append(f"start update offsets differ: {pa['full_run'].get('update_offset')} vs {pb['full_run'].get('update_offset')}")
if fail:
    print(f"[sm77-pair] {kind} REFUSE: " + "; ".join(fail)); raise SystemExit(1)
print(f"[sm77-pair] {kind} OK: {a.name} == {b.name} except the start weights, the architecture (w64-sm77 vs w64) and paths "
      f"({len(pa['rows']):,} users)")
PYEOF
}

prepare() {  # CPU: extended start, epoch folders, pair check
  if [ ! -f "$EXT/extend.json" ]; then
    local e=$RUN72S/epoch-$(printf %02d $K)/expertium/$TRACK
    $WARM extend --master "$e/master.pth" --optimizer "$e/optimizer.pth" --arch $ARCH --out "$EXT" > "$G/prepare.out" 2>&1 \
      || { log "STOP: extend failed (see $G/prepare.out)"; return 1; }
    log "extended SM72s epoch $K to $ARCH: $("$PY" -c "import json; r=json.load(open(r'$EXT/extend.json')); print(r['added_values'], 'new values,', r['optimizer_states_moved'], 'AdamW states moved, equal after load:', r['optimizer_states_equal_after_load'])")"
  fi
  if [ ! -f "$ARM/decay.json" ]; then
    $LF $WARM epoch --epoch $K1 --arch $ARCH --init "$EXT" --out "$ARM" >> "$G/prepare.out" 2>&1 || { log "STOP: arm epoch prepare failed"; return 1; }
    log "arm epoch prepared: $ARM"
  fi
  if [ "$REF_FREE" = 0 ] && [ ! -f "$REF/decay.json" ]; then
    $LF $WARM epoch --epoch $K1 --out "$REF" >> "$G/prepare.out" 2>&1 || { log "STOP: reference epoch prepare failed"; return 1; }
    log "reference epoch prepared: $REF"
  fi
  pair_check epoch "$ARM" "$REF" >> "$LOG" 2>&1 || { log "STOP: the arm and reference epochs are not matched"; return 1; }
  return 0
}

run_folder() {  # run_folder DIR WHAT FIRST_MINUTES SEG_MAX: expertium_decay.py run in segments; resumable from snapshots
  local D=$1 what=$2 first=$3 segmax=$4 rc qpid stop_sent b need
  while ! done_in "$D"; do
    stopped && return 1
    need=30; [ ! -d "$D/expertium" ] && [ ! -d "$D/attempts" ] && need=$first   # a fresh start must fit as a whole
    wait_free "$need" "$what" || return 1
    local seg=$(( FREE - 5 )); [ "$seg" -gt "$segmax" ] && seg=$segmax
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $what (segment to the next window or the end, expected $seg min of ~$segmax)"
      echo "    cmd: $LF $PY -X utf8 $DEC_PY run --out $D" >> "$LOG"; return 0; fi
    rm -f "$D/control/STOP"
    log "enqueue: $what (expected $seg min of ~$segmax, $FREE min to the next window)"
    "$PY" "$QUEUE" run --owner small-model --purpose "small-model: $NAME: $what (segment to the next window or the end)" --claim gpu \
      --expected-minutes "$seg" --max-wait-minutes $(( FREE - 20 > 1 ? FREE - 20 : 1 )) --launched-by "detached launcher sm77/launch-sm77.sh" -- \
      $LF "$PY" -X utf8 "$DEC_PY" run --out "$D" >> "$G/train.out" 2>&1 &
    qpid=$!
    stop_sent=0
    while kill -0 "$qpid" 2>/dev/null; do
      b=$(block)
      if [ -e "$G/STOP" ] || [[ "$b" == NOW* ]] || { [[ "$b" == NEXT* ]] && [ "${b#NEXT }" -le 2 ]; }; then
        [ "$stop_sent" = 0 ] && log "requesting a snapshot stop ($b)"
        "$PY" -X utf8 "$DEC_PY" stop --out "$D" > /dev/null 2>&1; stop_sent=1
      fi
      sleep 30
    done
    wait "$qpid"; rc=$?; log "exit $rc: $what segment"
    [ "$rc" = 75 ] && { sleep 60; continue; }
    done_in "$D" && break
    [ "$stop_sent" = 0 ] && { log "STOP: $what segment ended without completing and without a stop request (exit $rc); see $G/train.out"; return 1; }
  done
  [ "$DRYRUN" = 1 ] || log "$what complete: $("$PY" -X utf8 -c "import json; r=json.load(open(r'$D/expertium/$TRACK/result.json')); print(f\"{r.get('review_presentations', 0):,} reviews, {r.get('training_seconds', 0)/60:.1f} min training, {r.get('reviews_per_second', 0):,.0f} reviews/s\")")"
  return 0
}

prepare_decay() {  # prepare_decay START OUT ARCH: the quarter cool-down from START (users / schedule from SM72s epoch K or K1)
  local src=$K1; [ "$REF_FREE" = 0 ] && src=$K
  $LF $WARM cooldown --epoch $src --start "$1" --out "${2#$WT/}" --arch "$3" --fraction $FRACTION --extra-options "$EXTRA" >> "$G/prepare.out" 2>&1 \
    || { log "cool-down prepare failed ($2)"; return 1; }
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: label-window trim/check and the Muon fresh-state CPU check skipped for $2"; return 0; fi
  $LF "$PY" -X utf8 "$WT/scripts/check_replay_label_windows.py" --decay "$2" --trim-unlabelled-tail >/dev/null || { log "label windows (trim pass) failed ($2)"; return 1; }
  $LF "$PY" -X utf8 "$WT/scripts/check_replay_label_windows.py" --decay "$2" --out "$G/check-replay-label-windows-$(basename "$2").json" >/dev/null \
    || { log "label windows check failed ($2)"; return 1; }
  CUDA_VISIBLE_DEVICES= $LF "$PY" -X utf8 "$WT/scripts/check_muon_fresh_decay.py" --decay "$2" --out "$G/check-muon-fresh-$(basename "$2").json" \
    | grep -c PASS | sed "s/^/[sm77-muon] $(basename "$2") PASS lines: /" >> "$LOG"
  [ "${PIPESTATUS[0]}" = 0 ] || { log "Muon fresh-state CPU check failed ($2)"; return 1; }
  return 0
}

score() {  # score USERS_FILE OUT LABEL MODEL_SPEC MIN1 MIN2 WANT WHAT [ENV]: leak-free scoring, 2 passes
  local pass
  for pass in 1 2; do
    if [ "$DRYRUN" = 1 ] || [ "$(label_users "$2" "$3/w8+s8")" -lt "$7" ]; then
      q $([ $pass = 1 ] && echo "$5" || echo "$6") "scoring $3 on $8 (pass $pass)" -- \
        env -u RWKV_DELETED_CARDS ${9:-} "$PY" -X utf8 "$SC" --pool 732 --users-file "$1" --model "$3=$4" --only "$3/w8+s8" \
        --time-limit 2400 --split-existing --deleted-cards $MODE --out "$2" >> "$G/score.out" 2>&1 || { log "STOP: scoring ticket failed ($3)"; return 1; }
      [ "$DRYRUN" = 1 ] && return 0
    fi
  done
  local got; got=$(label_users "$2" "$3/w8+s8")
  [ "$got" -ge "$7" ] || { log "STOP: $3 scored on $got of $7 users after 2 passes"; return 1; }
  return 0
}
score732() {  # score732 LABEL MODEL_SPEC: 723 users at the 0.8 cap, then the 3 long users at 0.92 (726 in all)
  score "$USERS732" "$OUTROWS" "$1" "$2" $MIN_SCORE732 8 723 "the 732 pool minus 9 never-fit (723)" || return 1
  if [ "$DRYRUN" = 1 ] || [ "$(label_users "$OUTROWS" "$1/w8+s8")" -lt 726 ]; then
    q $MIN_LONG "scoring $1 on the 3 long users of the 732 pool (cap 0.92)" -- \
      env -u RWKV_DELETED_CARDS HELDOUT_GPU_MEMORY_FRACTION=0.92 "$PY" -X utf8 "$SC" --pool 732 --users-file "$ULONG" --model "$1=$2" \
      --only "$1/w8+s8" --time-limit 2400 --split-existing --deleted-cards $MODE --out "$OUTROWS" >> "$G/score.out" 2>&1 \
      || log "long users ticket failed ($1); continuing with $(label_users "$OUTROWS" "$1/w8+s8") users"
  fi
  return 0
}

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1) (DRYRUN=$DRYRUN, START_EPOCH=$K, reference epoch $K1 $([ "$REF_FREE" = 1 ] && echo "= SM72s's own (free)" || echo "trained here"))"
rc=0
wait_sm65 || rc=1
[ $rc -eq 0 ] && { cpu_step 15 "preparation" || rc=1; }
[ $rc -eq 0 ] && { preconditions || rc=1; }
[ $rc -eq 0 ] && { prepare || rc=1; }
# ticket 0: GPU check + trainer smoke of the arm (2 users x 512 reviews), once
if [ $rc -eq 0 ] && [ ! -e "$G/gpu-check-ok" ]; then
  q $MIN_CHECK "GPU check (dS0 backward kernel, scorer kernel from S0, trainer-path function preservation, 2 users x 512 reviews) + trainer smoke of the arm" -- \
    $LF "$BASH_EXE" -c "'$PY' -X utf8 '$WT/scripts/sm77_warm.py' gpu-check --out '$G/check-gpu-sm77.json' --ext '$EXT' && '$PY' -X utf8 '$DEC_PY' smoke --out '$ARM'" >> "$G/check.out" 2>&1 \
    || { log "STOP: GPU check or smoke failed (see $G/check.out, $G/check-gpu-sm77.json)"; rc=1; }
  [ $rc -eq 0 ] && [ "$DRYRUN" != 1 ] && { touch "$G/gpu-check-ok"; log "GPU check: $(grep -E 'ALL OK|FAILED' "$G/check.out" | tail -1); smoke: $(grep '"smoke"' "$G/check.out" | tail -1 | cut -c1-200)"; }
fi
[ $rc -eq 0 ] && { run_folder "$ARM" "arm epoch $K1 (w64-sm77 from SM72s epoch $K, 8k final tier, 2048 x 16 lanes carry, leak-free replay KD, AdamW 7e-4 constant)" $MIN_EPOCH $MIN_EPOCH || rc=1; }
if [ $rc -eq 0 ] && [ "$REF_FREE" = 0 ]; then
  run_folder "$REF" "reference epoch $K1 (w64 from SM72s epoch $K, same protocol, flags off)" $MIN_EPOCH $MIN_EPOCH || rc=1
fi
# cool-downs: prepare both (CPU), check they are matched, then train
if [ $rc -eq 0 ]; then
  for pair in "$DARM|$ARM|$ARCH" "$DREF|$REF|w64"; do
    D=${pair%%|*}; rest=${pair#*|}; START=${rest%%|*}; A=${rest#*|}
    if [ ! -f "$D/decay.json" ]; then
      [ "$DRYRUN" = 1 ] || done_in "$START" || [ "$START" = "$REF" -a "$REF_FREE" = 1 ] || { log "STOP: $START is not trained"; rc=1; break; }
      if [ "$DRYRUN" = 1 ] && [ "$A" = "$ARCH" ]; then log "DRYRUN: the arm cool-down needs the trained arm epoch; prepared after it"; continue; fi
      cpu_step 15 "cool-down prepare $(basename "$D")" || { rc=1; break; }
      prepare_decay "$START" "$D" "$A" || { rm -rf "$D"; rc=1; break; }
      log "cool-down prepared: $(basename "$D")"
    fi
  done
  if [ $rc -eq 0 ] && [ "$DRYRUN" != 1 ]; then pair_check decay "$DARM" "$DREF" >> "$LOG" 2>&1 || { log "STOP: the two cool-downs are not matched"; rc=1; }; fi
fi
[ $rc -eq 0 ] && { run_folder "$DARM" "arm quarter-epoch 32k Muon cool-down from arm epoch $K1 (w64-sm77)" 30 $MIN_DECAY || rc=1; }
[ $rc -eq 0 ] && { run_folder "$DREF" "reference quarter-epoch 32k Muon cool-down from epoch $K1 (w64)" 30 $MIN_DECAY || rc=1; }
[ $rc -eq 0 ] && { score732 "$ALBL" "expertium:$ARCH:$DARM/expertium/$TRACK/master.pth" || rc=1; }
[ $rc -eq 0 ] && { score732 "$RLBL" "expertium:w64:$DREF/expertium/$TRACK/master.pth" || rc=1; }
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then
    log "DRYRUN: then $PY -X utf8 $RULE decide --rows $OUTROWS --arm-label $ALBL/w8+s8 --ref-label $RLBL/w8+s8"
  else
    cpu_step 3 "rule" || rc=1
    line=$("$PY" -X utf8 "$RULE" decide --rows "$OUTROWS" --arm-label "$ALBL/w8+s8" --ref-label "$RLBL/w8+s8" 2>&1); r=$?
    log "rule exit $r: $line"; [ "$r" = 0 ] || rc=1
  fi
fi
if [ "$DRYRUN" = 1 ]; then rm -rf "$EXT" "$ARM" "$DARM" "$DREF"; [ "$REF_FREE" = 0 ] && rm -rf "$REF"; fi
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; else log "done"; fi
else
  log "STOP: ended with exit $rc"
fi
exit $rc
