#!/usr/bin/env bash
# SM72 steps 1 and 2 (user 10-24 ~20:40, "faisons comme ça"): the big model (fork RWKV, 2,762,884 parameters) gets ONE
# leak-free epoch from its shipped weights, is scored, then exports the leak-free teacher targets for the small model.
#   1. train  scripts/sm72_big_epoch.py run (users 5000-10000, 370.3M reviews, fresh AdamW, warmup 200 updates,
#             3.5e-4 -> 0 one_minus_sin (user 10-25), carried state in strided lanes through the variant class "fork"
#             (layout in decay.json; probe-big-carry-sm72.json), RWKV_DELETED_CARDS=neighbour), in segments: each ticket ends
#             at the next quiet window (the watcher below requests a snapshot stop when a window starts or is 2 min
#             away); the next ticket resumes from the snapshot after the window (scripts/expertium_decay.py resume)
#   2. score  scripts/small_model_heldout.py --split-existing, label big-leakfree-1ep/w8+s8, inputs keep and neighbour,
#             464 set (458) then the 732 pool (723) + the 3 long users at the 0.92 cap; scorers never see RWKV_DELETED_CARDS
#   3. export scripts/sm11_replay_export.py --teacher <new master.pth> into a NEW folder (never fork-fp32-final8k),
#             RWKV_DELETED_CARDS=neighbour, tickets cut at the next window (--time-limit), then status and a CPU verify
# Starts once SM71's GPU runbook logged "done" (or its process is gone). Every GPU step goes through the queue.
#   start   hidden via WMI Win32_Process.Create (HANDOFF-sm72.md); DRYRUN=1 bash <this file> prints the plan
#   stop    touch $S/STOP (checked between steps; a running training segment is stopped at its next snapshot too)
# Do not edit scripts/ expertium/ rwkv/ of the sm72 worktree while it runs (SourceWatch stops the training).
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
QUIET=C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json
WT=C:/Users/jonat/workspace/wt/sm72
SM=$WT/artifacts/small-model
S=$SM/sm72/gpu-big
OUT=$SM/sm72-big-leakfree-1ep
EPOCH="$PY -X utf8 $WT/scripts/sm72_big_epoch.py"
SC=$WT/scripts/small_model_heldout.py
EXPORT=$WT/scripts/sm11_replay_export.py
CK=   # set from decay.json below ($OUT/<model>/big-leakfree-1ep/master.pth, the shipped .pth's state-dict keys)
TAG=big-leakfree-1ep; LABEL=$TAG/w8+s8
R8N=C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/big-leakfree-1ep-final8k-neighbour
RUN8K=C:/Users/jonat/workspace/wt/sm55/artifacts/small-model/sm55-final8k-run.json            # read only
S56=C:/Users/jonat/workspace/wt/sm56/artifacts/small-model                                    # read only
U464=$S56/sm56/eval-users-464-fit.json; U732=$S56/sm56/eval-users-732-fit.json; ULONG=$S56/sm56/smalljobs/users-732-retry3.json
SM71LOG=C:/Users/jonat/workspace/wt/sm71/artifacts/small-model/sm71/gpu/run-sm71.log           # read only
RATE=${RATE:-10800}      # training reviews/s for ticket lengths: probe 10,821 at 8,192 x 2 carried (probe-big-carry-sm72.json)
EXPORT_RATE=34000        # fork-fp32-final8k export, measured
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION   # both default to a 0.8 cap
mkdir -p "$S"; LOG=$S/launch-sm72-big.log
log() { echo "[sm72-big $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
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
      --max-wait-minutes $(( FREE - minutes - 2 )) --launched-by "detached launcher sm72/launch-sm72-big.sh" -- "$@"
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
case = out / json.loads((out / "decay.json").read_text()).get("model", "expertium") / "big-leakfree-1ep"
r = json.loads((case / "result.json").read_text()) if (case / "result.json").exists() else {}
if r.get("status") == "complete":
    print("complete"); raise SystemExit
total = json.loads((out / "decay.json").read_text())["training_reviews"]
p = json.loads((out / "protocol.json").read_text())
done = (p["full_run"].get("lr_resume") or {}).get("reviews_offset", 0)
latest = case / "snapshots" / "latest.json"
if latest.exists():
    done += json.loads(latest.read_text())["review_presentations"]
print(max(total - done, 0))
PYEOF
}

# ---- wait for SM71's GPU runbook
if [ "$DRYRUN" != 1 ]; then
  until grep -q "\] done$" "$SM71LOG" 2>/dev/null; do
    stopped && exit 1
    pgrep -f run-sm71.sh >/dev/null 2>&1 || powershell -NoProfile -c "if (Get-CimInstance Win32_Process -Filter \"Name='bash.exe'\" | ? { \$_.CommandLine -match 'run-sm71.sh' }) { exit 0 } else { exit 1 }" \
      || { log "SM71 runbook process gone without 'done'; going ahead"; break; }
    sleep 120
  done
fi
log "started (commit $(git -C $WT rev-parse --short HEAD)); SM71: $(tail -1 "$SM71LOG" 2>/dev/null)"

# ---- 1. training, in segments that end at the next quiet window
[ -e "$OUT/decay.json" ] || { log "no prepared run in $OUT"; exit 1; }
# every update needs a labelled review (11:45 stop at user 8279's 3-review final window): CPU pre-flight, trims tails
grep -q '"label_windows"' "$OUT/decay.json" || [ "$DRYRUN" = 1 ] || {
  env RWKV_DELETED_CARDS=neighbour "$PY" -X utf8 $WT/scripts/sm72_label_windows.py --decay "$OUT" --trim     --out $S/check-label-windows-sm72-big.json >> "$LOG" 2>&1 || { log "label-window pre-flight failed"; exit 1; }
  log "label-window pre-flight: $("$PY" -c "import json; print(json.load(open(r'$OUT/decay.json'))['label_windows'])")"; }
# expertium (carried state, variant class "fork") or baseline (the --fresh-state fallback)
CK=$OUT/$("$PY" -c "import json; print(json.load(open(r'$OUT/decay.json')).get('model', 'expertium'))")/big-leakfree-1ep/master.pth
log "prepared: $("$PY" -c "import json; d = json.load(open(r'$OUT/decay.json')); print(d['model'], '|', d['batches'], '|', d['lr'])"); checkpoint $CK"
while true; do
  stopped && exit 1
  st=$(train_state); [ "$st" = complete ] && { log "training complete"; break; }
  free_minutes
  need=$(( st / RATE / 60 + 12 ))
  if [ "$FREE" -lt 30 ]; then log "training: only $FREE min before the next window; waiting for it"
    [ "$DRYRUN" = 1 ] && { log "DRYRUN: would wait"; break; }; sleep $(( FREE * 60 + 120 )); continue; fi
  exp=$(( need < FREE - 5 ? need : FREE - 5 ))
  purpose="SM72 big model (fork RWKV) leak-free epoch from the shipped weights, users 5000-10000, $(( st / 1000000 ))M reviews left (segment to the next window or the end)"
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $purpose (expected $exp min of ~$need)"; echo "    cmd: env RWKV_DELETED_CARDS=neighbour $EPOCH run --out $OUT"; break; fi
  log "enqueue: $purpose (expected $exp min of ~$need, $FREE min to the next window)"
  "$PY" "$QUEUE" run --owner small-model --purpose "small-model: $purpose" --claim gpu --expected-minutes "$exp" \
    --max-wait-minutes $(( FREE - 20 )) --launched-by "detached launcher sm72/launch-sm72-big.sh" -- \
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

# ---- 2. scoring the new checkpoint (keep and neighbour inputs, existing/deleted split)
[ "$DRYRUN" = 1 ] || [ -e "$CK" ] || { log "no checkpoint $CK"; exit 1; }
score() {  # score COND POOLTAG MINUTES
  local c=$1 p=$2 m=$3 users pool envs="" extra=""
  case $p in 464) users=$U464; pool=464;; 732) users=$U732; pool=732;; long) users=$ULONG; pool=732; envs="HELDOUT_GPU_MEMORY_FRACTION=0.92";; esac
  [ "$c" = neighbour ] && extra="--deleted-cards neighbour"
  q "$m" "SM72 big model leak-free epoch ($LABEL), inputs=$c, $p users, split existing/deleted" -- \
    env -u RWKV_DELETED_CARDS $envs "$PY" -X utf8 "$SC" --pool $pool --users-file "$users" --model "$TAG=fork::$CK" \
    --only "$LABEL" --time-limit 3000 --split-existing $extra --out "$S/bigleak-$c.jsonl" >> "$S/score.out" 2>&1
}
score keep 464 12 && score neighbour 464 12 && score keep 732 10 && score neighbour 732 10 && score keep long 6 && score neighbour long 6 \
  || { log "scoring stopped"; exit 1; }
log "scoring done: $S/bigleak-keep.jsonl, $S/bigleak-neighbour.jsonl"

# ---- 3. leak-free teacher targets from the new checkpoint (resumable per user)
EXPORT_ENV="env RWKV_DELETED_CARDS=neighbour"
while true; do
  stopped && exit 1
  if [ "$DRYRUN" != 1 ] && $EXPORT_ENV "$PY" -X utf8 "$EXPORT" status --out "$R8N" --run "$RUN8K" >> "$LOG" 2>&1; then log "export complete"; break; fi
  free_minutes
  if [ "$FREE" -lt 40 ]; then log "export: only $FREE min before the next window"; [ "$DRYRUN" = 1 ] && break; sleep $(( FREE * 60 + 120 )); continue; fi
  tl=$(( (FREE - 25) * 60 )); [ "$tl" -gt 21600 ] && tl=21600
  q $(( tl / 60 + 15 )) "SM72 leak-free teacher targets from the big model's leak-free epoch (8,000 final-tier users, ~4.8 h total, this ticket stops starting users after $(( tl / 60 )) min)" -- \
    $EXPORT_ENV "$PY" -X utf8 "$EXPORT" export --out "$R8N" --run "$RUN8K" --teacher "$CK" --device cuda --time-limit "$tl" >> "$S/export.out" 2>&1
  [ "$DRYRUN" = 1 ] && break
done
if [ "$DRYRUN" = 1 ]; then log "DRYRUN: then CPU verify: $EXPORT_ENV $PY -X utf8 $EXPORT verify --out $R8N --run $RUN8K --teacher $CK --sample 8"; exit 0; fi
free_minutes
log "verify (CPU)"; $EXPORT_ENV "$PY" -X utf8 "$EXPORT" verify --out "$R8N" --run "$RUN8K" --teacher "$CK" --sample 8 >> "$S/export.out" 2>&1
log "verify exit $?; manifest deleted_cards: $(grep -o '"deleted_cards": *"[a-z]*"' "$R8N/manifest.json")"
log "done"
