#!/usr/bin/env bash
# Better teacher (SM65) stage 1 (artifacts/small-model/sm65/DESIGN.md): the big model (fork RWKV, 2,762,884 parameters)
# gets ONE more leak-free epoch, from the big model (1 leak-free epoch) (SM72 step 1, master.pth sha256 ac7f0d8a...), on
# the 3,969 final-tier training users it has never seen (train_users of the 8k run with id < 5000, 285.9M reviews;
# disjointness from the validation / final-test users and the 732 / 464 sets in decay.json "users_check"), then is
# scored on the 732 set and paired against the reference. A copy of sm72/launch-sm72-big.sh without the export.
#   0. smoke  scripts/sm72_big_epoch.py smoke (trainer smoke mode on a scratch copy: 2 users x 512 reviews), once
#   1. train  scripts/sm72_big_epoch.py run on the folder prepared with --users-run --users-below 5000 --init --track
#             big-moreusers (same recipe as SM72 step 1: 2 strided lanes x 8,192 carried state, fresh AdamW, warmup 200,
#             3.5e-4 -> 0 one_minus_sin, cap 0.8, budgeted carry table 768 MiB, shuffle seed 1,
#             RWKV_DELETED_CARDS=neighbour; label-window pre-flight applied, 14 tails trimmed), in segments: each ticket
#             ends at the next quiet window (the watcher requests a snapshot stop when a window starts or is 2 min away);
#             the next ticket resumes from the snapshot after the window (scripts/expertium_decay.py resume)
#   2. score  scripts/small_model_heldout.py --split-existing --deleted-cards neighbour, label big-moreusers/w8+s8, the
#             732 pool (723 fit users) then the 3 long users at the 0.92 cap; scorers never see RWKV_DELETED_CARDS
#   3. rule   scripts/sm65_rule.py decide (CPU): 722 users vs wt/sm72 gpu-big/bigleak-neighbour.jsonl; go / stop / grey
# No teacher export here: that is stage 2, after the decision (HANDOFF-sm65.md).
# Starts once the leak-free small model's (SM72s) launcher log ends with "done" (or a STOP line). Every GPU step goes
# through the queue.
#   start   hidden via WMI Win32_Process.Create (HANDOFF-sm65.md); DRYRUN=1 bash <this file> prints the plan
#   stop    touch $S/STOP (checked between steps; a running training segment is stopped at its next snapshot too)
# Do not edit scripts/ expertium/ rwkv/ of the sm65 worktree while it runs (SourceWatch stops the training).
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
QUIET=C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json
WT=C:/Users/jonat/workspace/wt/sm65
SM=$WT/artifacts/small-model
S=$SM/sm65/gpu
OUT=$SM/sm65-big-moreusers
EPOCH="$PY -X utf8 $WT/scripts/sm72_big_epoch.py"
SC=$WT/scripts/small_model_heldout.py
CK=   # set from decay.json below ($OUT/<model>/<track>/master.pth, the shipped .pth's state-dict keys)
TAG=big-moreusers; LABEL=$TAG/w8+s8
S56=C:/Users/jonat/workspace/wt/sm56/artifacts/small-model                                    # read only
U732=$S56/sm56/eval-users-732-fit.json; ULONG=$S56/sm56/smalljobs/users-732-retry3.json
SM72SLOG=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/launch-sm72s.log      # read only
RATE=${RATE:-11400}      # training reviews/s for ticket lengths: SM72 step 1 ran 370.3M reviews in 558 min (11,460/s)
NAME="better teacher (SM65) stage 1"
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION   # both default to a 0.8 cap
mkdir -p "$S"; LOG=$S/launch-sm65-teacher.log
[ "$DRYRUN" = 1 ] && LOG=$S/launch-sm65-teacher-dryrun.log
log() { echo "[sm65 $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
stopped() { [ -e "$S/STOP" ] && { log "STOP file present"; return 0; }; return 1; }

block() {  # -> "NOW <seconds to its end>" inside a scope-all quiet window, else "NEXT <minutes to the next start>"
  Q="$QUIET" "$PY" -X utf8 - <<'PYEOF'
import json, os, re
from datetime import datetime, timedelta
now = datetime.now()
text = open(os.environ["Q"], encoding="utf-8").read()
try:
    wins = [(w["start"], w["end"]) for w in json.loads(text) if w.get("scope", "all") in ("all", "gpu")]
except (ValueError, KeyError, TypeError):
    wins = re.findall(r'"start"\s*:\s*"([^"]+)"[^{}]*?"end"\s*:\s*"([^"]+)"', text, re.S)
blocks = [(datetime.fromisoformat(s), datetime.fromisoformat(e)) for s, e in wins]
day = now.replace(hour=0, minute=0, second=0, microsecond=0)
blocks += [(day + timedelta(days=d, hours=21), day + timedelta(days=d + 1, hours=2)) for d in range(-1, 9)]  # daily break
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

# minutes until the next window, after sleeping through a window we are in; sets FREE
free_minutes() {
  while true; do
    local b; b=$(block)
    case "$b" in
      NOW*) local s=${b#NOW }; if [ "$DRYRUN" = 1 ]; then log "DRYRUN: inside a quiet window ($s s left)"; FREE=600; return; fi
            log "quiet window: sleeping ${s}s"; sleep "$s";;
      NEXT*) FREE=${b#NEXT }; return;;
      *) log "window check failed: $b"; sleep 600;;
    esac
  done
}

q() {  # q MINUTES PURPOSE -- CMD...: one short ticket that must end before the next window
  local minutes=$1 purpose=$2; shift 3
  while true; do
    stopped && return 1
    free_minutes
    if [ "$FREE" -lt $(( minutes + 3 )) ]; then log "$purpose: $FREE min before the next window, needs $minutes; waiting"
      [ "$DRYRUN" = 1 ] || { sleep $(( FREE * 60 + 120 )); continue; }; fi
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $purpose (expected $minutes min)"; echo "    cmd: $*"; return 0; fi
    log "enqueue: $purpose (expected $minutes min)"
    "$PY" "$QUEUE" run --owner small-model --purpose "small-model: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes $(( FREE - minutes - 2 )) --launched-by "detached launcher sm65/launch-sm65-teacher.sh" -- "$@"
    local rc=$?; log "exit $rc: $purpose"
    [ "$rc" = 75 ] && { log "queue wait would reach the next window; re-planning"; sleep 60; continue; }
    return $rc
  done
}

train_state() {  # -> "complete" | "<reviews left>" (from the run folder; protocol.json shrinks on every resume)
  O="$OUT" "$PY" -X utf8 - <<'PYEOF'
import json, os
from pathlib import Path
out = Path(os.environ["O"])
decay = json.loads((out / "decay.json").read_text())
case = out / decay.get("model", "expertium") / decay["track"]
r = json.loads((case / "result.json").read_text()) if (case / "result.json").exists() else {}
if r.get("status") == "complete":
    print("complete"); raise SystemExit
total = decay.get("training_reviews_after_trims", decay["training_reviews"])
p = json.loads((out / "protocol.json").read_text())
done = (p["full_run"].get("lr_resume") or {}).get("reviews_offset", 0)
latest = case / "snapshots" / "latest.json"
if latest.exists():
    done += json.loads(latest.read_text())["review_presentations"]
print(max(total - done, 0))
PYEOF
}

# ---- wait for the leak-free small model's (SM72s) launcher: its log's last line is "done" or a STOP line
sm72s_last() { tail -1 "$SM72SLOG" 2>/dev/null; }
sm72s_ended() { local l; l=$(sm72s_last); [[ "$l" == *"] done" ]] || [[ "$l" == *"] STOP"* ]]; }
if [ "$DRYRUN" != 1 ]; then
  n=0
  until sm72s_ended; do
    stopped && exit 1
    [ $(( n % 30 )) = 0 ] && log "waiting for the leak-free small model's (SM72s) launcher to end; its last line: $(sm72s_last | cut -c1-200)"
    n=$(( n + 1 )); sleep 120
  done
else
  log "DRYRUN: would wait for SM72s; its last line now: $(sm72s_last | cut -c1-200)"
fi
log "started (commit $(git -C $WT rev-parse --short HEAD)); SM72s: $(sm72s_last | cut -c1-200)"

# ---- checks of the prepared folder (CPU)
[ -e "$OUT/decay.json" ] || { log "no prepared run in $OUT"; exit 1; }
"$PY" -X utf8 - "$OUT" <<'PYEOF' >> "$LOG" 2>&1 || { log "prepared folder check failed"; exit 1; }
import json, sys
d = json.load(open(sys.argv[1] + "/decay.json"))
u = d["users_check"]
ok = (d["track"] == "big-moreusers" and d["training_users"] == 3969 and u["users_below"] == 5000
      and d["source_sha256"].startswith("ac7f0d8a") and d["deleted_cards"] == "neighbour"
      and all(v["overlap"] == 0 for v in u["disjoint_from"].values()))
print(f"[sm65] decay.json: {d['training_users']} users, {d['training_reviews']:,} reviews, from {d['source_sha256'][:8]}, "
      f"disjoint from {', '.join(u['disjoint_from'])}: {'OK' if ok else 'FAILED'}")
sys.exit(0 if ok else 1)
PYEOF
# every update needs a labelled review (SM72's 11:45 stop): CPU pre-flight, trims tails (done at prepare time, 14 tails)
grep -q '"label_windows"' "$OUT/decay.json" || [ "$DRYRUN" = 1 ] || {
  env RWKV_DELETED_CARDS=neighbour "$PY" -X utf8 $WT/scripts/sm72_label_windows.py --decay "$OUT" --trim --out $S/check-label-windows-sm65.json >> "$LOG" 2>&1 || { log "label-window pre-flight failed"; exit 1; }
  log "label-window pre-flight: $("$PY" -c "import json; print(json.load(open(r'$OUT/decay.json'))['label_windows'])")"; }
CK=$OUT/$("$PY" -c "import json; d = json.load(open(r'$OUT/decay.json')); print(d.get('model', 'expertium') + '/' + d['track'])")/master.pth
log "prepared: $("$PY" -c "import json; d = json.load(open(r'$OUT/decay.json')); print(d['model'], '|', d['batches'], '|', d['lr'], '|', d.get('label_windows'))"); checkpoint $CK"

# ---- 0. GPU smoke of the prepared folder (scratch copy, 2 users x 512 reviews), once
if [ ! -e "$S/smoke-ok" ] && [ "$(train_state)" != complete ]; then
  q 10 "$NAME: GPU smoke of the big model's epoch on the new users (2 users x 512 reviews, scratch copy)" -- \
    env RWKV_DELETED_CARDS=neighbour $EPOCH smoke --out "$OUT" >> "$S/smoke.out" 2>&1 || { log "smoke failed (see $S/smoke.out)"; exit 1; }
  [ "$DRYRUN" = 1 ] || { touch "$S/smoke-ok"; log "smoke: $(grep '"smoke"' "$S/smoke.out" | tail -1)"; }
fi

# ---- 1. training, in segments that end at the next quiet window
while true; do
  stopped && exit 1
  st=$(train_state); [ "$st" = complete ] && { log "training complete"; break; }
  free_minutes
  need=$(( st / RATE / 60 + 12 ))
  if [ "$FREE" -lt 30 ]; then log "training: only $FREE min before the next window; waiting for it"
    [ "$DRYRUN" = 1 ] && { log "DRYRUN: would wait"; break; }; sleep $(( FREE * 60 + 120 )); continue; fi
  exp=$(( need < FREE - 5 ? need : FREE - 5 ))
  purpose="$NAME: big model (fork RWKV) one more leak-free epoch on the 3,969 new users (ids below 5000) from the big model (1 leak-free epoch), $(( st / 1000000 ))M reviews left (segment to the next window or the end)"
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $purpose (expected $exp min of ~$need, $FREE min to the next window)"; echo "    cmd: env RWKV_DELETED_CARDS=neighbour $EPOCH run --out $OUT"; break; fi
  log "enqueue: $purpose (expected $exp min of ~$need, $FREE min to the next window)"
  "$PY" "$QUEUE" run --owner small-model --purpose "small-model: $purpose" --claim gpu --expected-minutes "$exp" \
    --max-wait-minutes $(( FREE - 20 )) --launched-by "detached launcher sm65/launch-sm65-teacher.sh" -- \
    env RWKV_DELETED_CARDS=neighbour $EPOCH run --out "$OUT" >> "$S/train.out" 2>&1 &
  qpid=$!
  stop_sent=0
  while kill -0 "$qpid" 2>/dev/null; do
    b=$(block)
    if [ -e "$S/STOP" ] || [[ "$b" == NOW* ]] || { [[ "$b" == NEXT* ]] && [ "${b#NEXT }" -le 2 ]; }; then
      # re-requested every 30 s: `run` clears control/STOP when it starts, so a request made while queued is repeated
      [ "$stop_sent" = 0 ] && log "requesting a snapshot stop ($b)"
      $EPOCH stop --out "$OUT" > /dev/null 2>&1; stop_sent=1
    fi
    sleep 30
  done
  wait "$qpid"; rc=$?; log "exit $rc: training segment"
  [ "$rc" = 75 ] && { sleep 60; continue; }
  st=$(train_state)
  if [ "$st" != complete ] && [ "$stop_sent" = 0 ]; then
    log "training segment ended without completing and without a stop request (exit $rc); see $S/train.out"; exit 1
  fi
done

# ---- 2. scoring the new checkpoint (leak-free inputs, existing/deleted split): 732 pool (723) + the 3 long users
[ "$DRYRUN" = 1 ] || [ -e "$CK" ] || { log "no checkpoint $CK"; exit 1; }
score() {  # score POOLTAG MINUTES
  local p=$1 m=$2 users envs="" what
  case $p in 732) users=$U732; what="732 set (723 users)";; long) users=$ULONG; envs="HELDOUT_GPU_MEMORY_FRACTION=0.92"; what="3 long users of the 732 set at the 0.92 cap";; esac
  q "$m" "$NAME: score $LABEL, leak-free inputs, $what, split existing/deleted" -- \
    env -u RWKV_DELETED_CARDS $envs "$PY" -X utf8 "$SC" --pool 732 --users-file "$users" --model "$TAG=fork::$CK" \
    --only "$LABEL" --time-limit 3000 --split-existing --deleted-cards neighbour --out "$S/bigmore-neighbour.jsonl" >> "$S/score.out" 2>&1
}
score 732 12 && score long 6 || { log "scoring stopped"; exit 1; }
log "scoring done: $S/bigmore-neighbour.jsonl"

# ---- 3. the pre-registered rule (CPU)
if [ "$DRYRUN" = 1 ]; then log "DRYRUN: then $PY -X utf8 $WT/scripts/sm65_rule.py decide"; exit 0; fi
line=$("$PY" -X utf8 "$WT/scripts/sm65_rule.py" decide 2>&1); rc=$?
log "rule exit $rc: $line"
[ "$rc" = 0 ] || exit 1
log "done"
