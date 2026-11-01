#!/usr/bin/env bash
# Response-time prediction (SM82) stage 1: the answer-time head. One arm (the reference is already trained and scored):
#   arm        the leak-free small model (SM72s) epoch 21 + the zero answer-time head (preset w64-sm82: per grade 16
#              log-time bins, Linear 128 -> 64 on the instant head's hidden layer, time loss weight 0.1, only reviews
#              with a recorded time on a later day than the card's previous one), then the quarter-epoch 32k Muon
#              cool-down with the reference's options, users, order, seeds and schedule (~2 h); then the 732 grade
#              scoring (723 + the 3 long users at 0.92, w8+s8, leak-free) and the time scoring (723 users, fp32).
#   reference  the tiny architecture additions (SM77) reference, read only: SM72s's own epoch 21 +
#              wt/sm77/artifacts/small-model/sm77-ref-e21-dec-q32k-muon, label sm77-ref-e21-q32k/w8+s8 in
#              wt/sm77/artifacts/small-model/sm77/gpu/sm77-neighbour.jsonl.
# Rule (scripts/sm82_time.py decide, CPU): 722 users; grade curve and instant each <= +0.03 points of % of the big model
#   (1 leak-free epoch) vs the reference (paired user bootstrap 2,000, seed 0); the time head beats the card's last time
#   and the user's running median per grade on the censored time LogLoss and on the mean |ln error| (CI clear of zero).
#   Writes artifacts/small-model/sm82/sm82-decision.{md,json}.
# Steps: CPU preparation (extend, cool-down prepare, label-window trims, Muon fresh check, pair check vs the reference,
#   CPU proof) -> ticket 0 (trainer smoke of the cool-down + time-scorer smoke) -> cool-down -> grade scoring ->
#   time scoring -> rule.
# Breaks: quiet.json (scope all/gpu) plus the daily 20:00-01:00 break, re-read at every check (<= 10 min); the cool-down
#   runs in segments (snapshot stop <= 2 min before a window, resume after); tickets start only when they fit.
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm82/artifacts/small-model/sm82/launch-sm82.sh
#   launch   WMI hidden (artifacts/small-model/sm82/HANDOFF.md); log artifacts/small-model/sm82/gpu/launch-sm82.log
#   resume   the launch command again (finished steps are skipped; an interrupted cool-down resumes from its snapshot)
#   stop     touch C:/Users/jonat/workspace/wt/sm82/artifacts/small-model/sm82/gpu/STOP
# Do not edit scripts/ expertium/ rwkv/ of the sm82 worktree while it runs (SourceWatch stops the training).
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
QUIET=C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json
WT=C:/Users/jonat/workspace/wt/sm82
SM=$WT/artifacts/small-model
S=$SM/sm82
G=$S/gpu
mkdir -p "$G" "$S/work"
LOG=$G/launch-sm82.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm82-dryrun.log
T82="$PY -X utf8 $WT/scripts/sm82_time.py"
DEC_PY=$WT/scripts/expertium_decay.py
SC=$WT/scripts/small_model_heldout.py
TRACK=E89-F65e-F77a
K1=21
RUN72S=C:/Users/jonat/workspace/wt/sm72s/artifacts/accuracy/experiments/scale-w64kd-final-sc-lanes-leakfree/run   # read only
E21=$RUN72S/epoch-$K1/expertium/$TRACK
S72=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s                                                  # read only
SM77=C:/Users/jonat/workspace/wt/sm77/artifacts/small-model                                                        # read only
DREF=$SM77/sm77-ref-e21-dec-q32k-muon
REFROWS=$SM77/sm77/gpu/sm77-neighbour.jsonl
RLBL=sm77-ref-e21-q32k
BIGCK=C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72-big-leakfree-1ep/expertium/big-leakfree-1ep/master.pth
REPLAY=C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/big-leakfree-1ep-final8k-neighbour
USERS732=$S72/eval-users-732-fit.json; ULONG=$S72/users-732-long.json
P=sm82; [ "${DRYRUN:-0}" = 1 ] && P=sm82-dryrun   # the dry run prepares scratch folders and deletes them
EXT=$SM/$P-ext-e$K1                 # SM72s epoch 21 master + the zero head
EXTP=$S/work/$P-ext-e$K1-perturbed  # the same with a random head (CPU leak test only, never trained)
DARM=$SM/$P-arm-e$K1-dec-q32k-muon
ALBL=sm82-arm-e$K1-q32k
ARCH=w64-sm82
OUTROWS=$G/sm82-neighbour.jsonl
TROWS=$G/sm82-time.jsonl
BASE=$S/time-baselines.jsonl
MODE=neighbour
LF="env RWKV_DELETED_CARDS=$MODE"
SPEC='{"lanes": "strided"}'
DIST="{\"teacher\": \"$BIGCK\", \"alpha\": 0.9, \"decay_alpha\": 0.5, \"kd_source\": \"replay\", \"replay\": \"$REPLAY\"}"
EXTRA="{\"distillation\": $DIST, \"state_carry\": $SPEC, \"chunk_reviews\": 32768, \"batch_chunks\": 1, \"optimizer\": \"expertium-iter53\"}"  # = SM72s's cool-down = SM77's
FRACTION=0.25
MIN_DECAY=${MIN_DECAY:-150}   # quarter cool-down: the reference trained 145.6M reviews in 124 min (19.5k reviews/s) + setup
MIN_CHECK=12; MIN_SCORE732=12; MIN_LONG=6; MIN_TIME=25
NAME="response-time prediction (SM82)"
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS   # scorer cap 0.8; trainer cap 0.8 (options)
log() { echo "[sm82 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
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
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm82/launch-sm82.sh" -- "$@"
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
time_users() { F="$1" "$PY" -X utf8 -c "
import json, os
from pathlib import Path
p = Path(os.environ['F'])
print(len({r['user_id'] for r in map(json.loads, open(p, encoding='utf-8')) if 'error' not in r}) if p.exists() else 0)"; }
done_in() { [ -f "$1/expertium/$TRACK/master.pth" ] && R="$1/expertium/$TRACK/result.json" "$PY" -X utf8 -c "
import json, os, sys
r = json.load(open(os.environ['R']))
sys.exit(0 if r.get('status') == 'complete' and not r.get('smoke_only') else 1)" 2>/dev/null; }

preconditions() {
  for f in "$E21/master.pth" "$E21/result.json" "$RUN72S/run.json" "$RUN72S/epoch-$K1/protocol.json" "$BIGCK" \
           "$REPLAY/manifest.json" "$USERS732" "$ULONG" "$DREF/decay.json" "$DREF/protocol.json" "$DREF/expertium/$TRACK/master.pth" "$REFROWS" \
           "C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72/gpu-big/bigleak-neighbour.jsonl" "$BASE"; do
    [ -f "$f" ] || { log "STOP: missing $f"; return 1; }
  done
  R="$E21/result.json" "$PY" -X utf8 -c "import json, os, sys; sys.exit(0 if json.load(open(os.environ['R'])).get('status') == 'complete' else 1)" \
    || { log "STOP: SM72s epoch $K1 is not complete"; return 1; }
  done_in "$DREF" || { log "STOP: the reference cool-down $DREF is not complete"; return 1; }
  local n; n=$(label_users "$REFROWS" "$RLBL/w8+s8")
  [ "$n" -ge 725 ] || { log "STOP: the reference label $RLBL/w8+s8 has $n users in $REFROWS (want >= 725)"; return 1; }
  if [ "$DRYRUN" != 1 ] && [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    log "STOP: uncommitted source changes in scripts/ expertium/ rwkv/ (SourceWatch refuses them)"; return 1; fi
  return 0
}

prepare() {  # CPU: the extended start, the cool-down folder, its checks, the pair check, the CPU proof
  if [ ! -f "$EXT/extend.json" ]; then
    CUDA_VISIBLE_DEVICES= $T82 extend --master "$E21/master.pth" --out "$EXT" >> "$G/prepare.out" 2>&1 || { log "STOP: extend failed"; return 1; }
    log "extended: $EXT (SM72s epoch $K1 + the zero answer-time head)"
  fi
  if [ ! -f "$EXTP/extend.json" ]; then
    CUDA_VISIBLE_DEVICES= $T82 extend --master "$E21/master.pth" --out "$EXTP" --perturb 0.1 --seed 1 >> "$G/prepare.out" 2>&1 \
      || { log "STOP: perturbed extend failed"; return 1; }
  fi
  if [ ! -f "$DARM/decay.json" ]; then
    $LF $T82 cooldown --epoch $K1 --start "$EXT" --out "${DARM#$WT/}" --fraction $FRACTION --extra-options "$EXTRA" >> "$G/prepare.out" 2>&1 \
      || { log "STOP: cool-down prepare failed ($DARM)"; rm -rf "$DARM"; return 1; }
    $LF "$PY" -X utf8 "$WT/scripts/check_replay_label_windows.py" --decay "$DARM" --trim-unlabelled-tail >/dev/null \
      || { log "STOP: label windows (trim pass) failed ($DARM)"; rm -rf "$DARM"; return 1; }
    $LF "$PY" -X utf8 "$WT/scripts/check_replay_label_windows.py" --decay "$DARM" --out "$G/check-replay-label-windows-$(basename "$DARM").json" >/dev/null \
      || { log "STOP: label windows check failed ($DARM)"; rm -rf "$DARM"; return 1; }
    log "cool-down prepared: $(basename "$DARM")"
  fi
  CUDA_VISIBLE_DEVICES= $LF "$PY" -X utf8 "$WT/scripts/check_muon_fresh_decay.py" --decay "$DARM" --out "$G/check-muon-fresh-$(basename "$DARM").json" \
    | grep -c PASS | sed "s/^/[sm82-muon] $(basename "$DARM") PASS lines: /" >> "$LOG"
  [ "${PIPESTATUS[0]}" = 0 ] || { log "STOP: Muon fresh-state CPU check failed ($DARM)"; return 1; }
  pair_decay >> "$LOG" 2>&1 || { log "STOP: the cool-down does not match the reference's"; return 1; }
  if [ ! -e "$G/check-cpu-ok-$P" ]; then
    local line
    line=$(CUDA_VISIBLE_DEVICES= $T82 check --ext "$EXT" --perturbed "$EXTP" --decay-options "$DREF/decay.json" --out "$G/check-cpu-$P.json" 2>&1 | tail -1)
    log "$line"
    [[ "$line" == "[sm82-check] PASS"* ]] || { log "STOP: CPU proof failed (see $G/check-cpu-$P.json)"; return 1; }
    touch "$G/check-cpu-ok-$P"
  fi
  return 0
}

pair_decay() {  # the arm's cool-down vs the reference's: identical except the start weights (+ the zero head), the architecture and paths
  A="$DARM" B="$DREF" X="$EXT" "$PY" -X utf8 - <<'PYEOF'
import json, os
from pathlib import Path
a, b, x = Path(os.environ["A"]), Path(os.environ["B"]), Path(os.environ["X"])
def strip(p):
    p = json.loads(json.dumps(p))
    for k in ("init_master", "init_optimizer", "control_dir", "benchmark_db"):
        p["full_run"].pop(k, None)
    return p
# a resumed cool-down rewrites protocol.json to the users left; protocol.prepared.json keeps the full one
proto = lambda d: json.loads(((d / "protocol.prepared.json") if (d / "protocol.prepared.json").exists() else (d / "protocol.json")).read_text())
pa, pb = proto(a), proto(b)
da, db = json.loads((a / "decay.json").read_text()), json.loads((b / "decay.json").read_text())
ext = json.loads((x / "extend.json").read_text())
fail = []
if strip(pa) != strip(pb):
    diff = sorted(k for k in set(pa) | set(pb) if strip(pa).get(k) != strip(pb).get(k))
    fail.append(f"protocol.json differs beyond the start weights and paths ({diff}; full_run "
                f"{ {k: (pa['full_run'].get(k), pb['full_run'].get(k)) for k in set(pa['full_run']) | set(pb['full_run']) if pa['full_run'].get(k) != pb['full_run'].get(k)} })")
oa, ob = dict(da["options"]), dict(db["options"])
if (oa.pop("architecture", None), ob.pop("architecture", None)) != ("w64-sm82", "w64"):
    fail.append(f"architecture options {da['options'].get('architecture')} vs {db['options'].get('architecture')} (want w64-sm82 vs w64)")
if oa != ob:
    fail.append(f"options differ: { {k: (oa.get(k), ob.get(k)) for k in set(oa) | set(ob) if oa.get(k) != ob.get(k)} }")
free = {"created_utc", "source_case", "start", "start_master_sha256", "sm77", "sm82", "architecture", "options", "out", "out_dir", "output"}  # options: compared above
other = sorted(k for k in set(da) | set(db) if k not in free and da.get(k) != db.get(k) and "path" not in k and "dir" not in k)
for k in other:
    fail.append(f"decay.json {k}: {str(da.get(k))[:120]} vs {str(db.get(k))[:120]}")
if Path(da["source_case"]).resolve() != Path(db["source_case"]).resolve():
    fail.append(f"source case {da['source_case']} vs {db['source_case']}")
if ext["source_master_sha256"] != db["start_master_sha256"] or ext["init"] != "zeros":
    fail.append(f"the start is not the reference's start + a zero head ({ext['source_master_sha256'][:12]} vs {db['start_master_sha256'][:12]}, {ext['init']})")
if Path(pa["full_run"]["init_master"]).resolve() != (x / "master.pth").resolve():
    fail.append(f"init_master {pa['full_run']['init_master']} is not {x / 'master.pth'}")
if fail:
    print("[sm82-pair] decay REFUSE: " + "; ".join(fail)); raise SystemExit(1)
print(f"[sm82-pair] decay OK: {a.name} == {b.name} except the start weights (the reference's start + the zero head), the "
      f"architecture (w64-sm82) and paths ({len(pa['rows']):,} users, same order, options, seeds, schedule; update offset "
      f"{pa['full_run'].get('update_offset')})")
PYEOF
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
      --expected-minutes "$seg" --max-wait-minutes $(( FREE - 20 > 1 ? FREE - 20 : 1 )) --launched-by "detached launcher sm82/launch-sm82.sh" -- \
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
  [ "$DRYRUN" = 1 ] || log "time loss: $(grep '\[sm82-time\]' "$G/train.out" | tail -1)"
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
score_time() {  # the answer-time head on the 723 users (fp32, leak-free inputs), resumable, 2 passes
  local pass
  for pass in 1 2; do
    if [ "$DRYRUN" = 1 ] || [ "$(time_users "$TROWS")" -lt 723 ]; then
      q $([ $pass = 1 ] && echo $MIN_TIME || echo 10) "time scoring of the answer-time head on the 732 pool minus 9 never-fit (723, pass $pass)" -- \
        env -u RWKV_DELETED_CARDS $T82 score --master "$DARM/expertium/$TRACK/master.pth" --baselines "$BASE" --users-file "$USERS732" \
        --out "$TROWS" >> "$G/time.out" 2>&1 || { log "STOP: time scoring ticket failed"; return 1; }
      [ "$DRYRUN" = 1 ] && return 0
    fi
  done
  local got; got=$(time_users "$TROWS")
  [ "$got" -ge 722 ] || { log "STOP: time scoring has $got of 723 users after 2 passes"; return 1; }
  return 0
}

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1) (DRYRUN=$DRYRUN, cool-down from SM72s epoch $K1 + the zero answer-time head; reference = SM77's: $DREF)"
rc=0
cpu_step 10 "preparation" || rc=1
[ $rc -eq 0 ] && { preconditions || rc=1; }
[ $rc -eq 0 ] && { prepare || rc=1; }
# ticket 0: trainer smoke of the cool-down (2 users x 512 reviews) + the time scorer on 3 users with the zero head, once
if [ $rc -eq 0 ] && [ ! -e "$G/smoke-ok-$P" ]; then
  echo '{"users": [2388, 4676, 1712]}' > "$S/work/smoke-users.json"
  rm -f "$S/work/smoke-time-$P.jsonl"
  q $MIN_CHECK "trainer smoke of the cool-down with the answer-time head (2 users x 512 reviews) + time-scorer smoke (3 users)" -- \
    bash -c "$LF $PY -X utf8 $DEC_PY smoke --out $DARM && env -u RWKV_DELETED_CARDS $T82 score --master $EXT/master.pth --baselines $BASE --users-file $S/work/smoke-users.json --out $S/work/smoke-time-$P.jsonl" \
    >> "$G/check.out" 2>&1 || { log "STOP: smoke failed (see $G/check.out)"; rc=1; }
  if [ $rc -eq 0 ] && [ "$DRYRUN" != 1 ]; then
    first=$(grep '\[sm82-time\] updates 1:' "$G/check.out" | tail -1)
    [ -n "$first" ] || { log "STOP: the smoke printed no [sm82-time] line (the time loss is not active)"; rc=1; }
    [ "$(time_users "$S/work/smoke-time-$P.jsonl")" = 3 ] || { log "STOP: the time-scorer smoke did not score 3 users"; rc=1; }
    [ $rc -eq 0 ] && { touch "$G/smoke-ok-$P"; log "smoke: $(grep '"smoke"' "$G/check.out" | tail -1 | cut -c1-200); $first"; }
  fi
fi
[ $rc -eq 0 ] && { run_folder "$DARM" "arm quarter-epoch 32k Muon cool-down with the answer-time head (w64-sm82 from SM72s epoch $K1, the reference's options and order)" $MIN_DECAY $MIN_DECAY || rc=1; }
[ $rc -eq 0 ] && { score732 "$ALBL" "expertium:$ARCH:$DARM/expertium/$TRACK/master.pth" || rc=1; }
[ $rc -eq 0 ] && { score_time || rc=1; }
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then
    log "DRYRUN: then $T82 decide --time-rows $TROWS --baselines $BASE --arm-rows $OUTROWS --arm-label $ALBL/w8+s8 --ref-rows $REFROWS --ref-label $RLBL/w8+s8"
  else
    cpu_step 3 "rule" || rc=1
    out=$($T82 decide --time-rows "$TROWS" --baselines "$BASE" --arm-rows "$OUTROWS" --arm-label "$ALBL/w8+s8" --ref-rows "$REFROWS" --ref-label "$RLBL/w8+s8" 2>&1); r=$?
    line=$(echo "$out" | tail -1)
    log "rule exit $r: $line"; [ "$r" = 0 ] || rc=1
  fi
fi
if [ "$DRYRUN" = 1 ]; then rm -rf "$EXT" "$EXTP" "$DARM"; rm -f "$G/check-cpu-ok-$P"; fi
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; else log "done"; fi
else
  log "STOP: ended with exit $rc"
fi
exit $rc
