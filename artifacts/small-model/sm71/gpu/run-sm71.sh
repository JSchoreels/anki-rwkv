#!/usr/bin/env bash
# SM71 deleted-card leak check on GPU: the big model (fork w8+s8), the full retrain's decay from epoch 10 (SM56) and the
# 8k model (SM55), w8+s8, all with --split-existing (view 2 = existing-card subset of the keep rows), under
#   keep (today's inputs), neighbour (flags 0 + deck/preset of the nearest existing card), flags (flags 0 only),
#   reseed (keep with --seed-offset 1: noise floor of the random id/day codes, which neighbour re-draws).
# 464 set (458 fit users) first, in priority order; then the 732 pool (723 fit users, the 458 already done are skipped)
# + the 3 long users (650, 1601, 4501) at the 0.92 memory cap, for keep and neighbour (big + SM56).
# Waits for the SM56 launcher's epoch-14 scoring to finish (or 20:30), checks quiet.json itself before every enqueue
# (no new ticket into a break; the queue also holds a ticket whose expected end reaches a window). CANCEL file stops it.
export PATH=/usr/bin:/mingw64/bin:$PATH
WT=C:/Users/jonat/workspace/wt/sm71; O=$WT/artifacts/small-model/sm71/gpu; LOG=$O/run-sm71.log
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe; QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
QUIET=C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json
SC=$WT/scripts/small_model_heldout.py; S56=C:/Users/jonat/workspace/wt/sm56/artifacts/small-model
FK=C:/Users/jonat/workspace/anki/qt/aqt/rwkv_inference/RWKV_trained_on_5000_10000.pth
CK56=$S56/scale-w64kd-final-sc-lanes-dec10-1ep32k-muon/expertium/E89-F65e-F77a/master.pth
CK55=C:/Users/jonat/workspace/wt/sm55/artifacts/small-model/scale-w64kd-gate-sc-lanes-cont-decay1ep32k-muon-final8k/expertium/E89-F65e-F77a/master.pth
U464=$S56/sm56/eval-users-464-fit.json; U732=$S56/sm56/eval-users-732-fit.json; ULONG=$S56/sm56/smalljobs/users-732-retry3.json
L56=$S56/sm56/launch-sm56.log
log() { echo "[sm71 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
declare -A MODEL=( [big]="fork=fork::$FK" [small]="sm56=expertium:w64:$CK56" [small8k]="sm55=expertium:w64:$CK55" )
declare -A LABEL=( [big]="fork/w8+s8" [small]="sm56/w8+s8" [small8k]="sm55/w8+s8" )
declare -A MIN=( [big]=20 [small]=10 [small8k]=10 )
# seconds to wait so that a ticket of $1 minutes neither starts inside a scope-all quiet window nor runs into one
quiet_wait() { "$PY" - "$QUIET" "$1" <<'PYEOF'
import json, sys, datetime as dt
now = dt.datetime.now(); need = dt.timedelta(minutes=float(sys.argv[2]))
wait = 0
for w in json.load(open(sys.argv[1])):
    if w.get("scope", "all") != "all":
        continue
    a, b = dt.datetime.fromisoformat(w["start"]), dt.datetime.fromisoformat(w["end"])
    t = now + dt.timedelta(seconds=wait)
    if t < b and t + need > a:
        wait = max(wait, (b - now).total_seconds() + 60)
print(int(wait))
PYEOF
}
ticket() {  # ticket MODEL COND POOLTAG [extra env]
  local m=$1 c=$2 p=$3 mins=${MIN[$1]} users pool envs="" extra=""
  [ -e "$O/CANCEL" ] && { log "CANCEL"; exit 1; }
  case $p in 464) users=$U464; pool=464;; 732) users=$U732; pool=732; mins=$(( mins * 6 / 10 + 2 ));;
             long) users=$ULONG; pool=732; envs="HELDOUT_GPU_MEMORY_FRACTION=0.92"; mins=$(( mins / 2 + 3 ));; esac
  case $c in keep) ;; flags) extra="--deleted-cards flags";; neighbour) extra="--deleted-cards neighbour";;
             reseed) extra="--seed-offset 1";; esac
  local w; w=$(quiet_wait "$mins"); if [ "$w" -gt 0 ]; then log "quiet window: sleeping ${w}s before $m/$c/$p"; sleep "$w"; fi
  [ -e "$O/CANCEL" ] && { log "CANCEL"; exit 1; }
  local purpose="SM71 deleted-card leak: $m (${LABEL[$m]}) inputs=$c, $p users, split existing/deleted"
  log "enqueue: $purpose (expected $mins min)"
  "$PY" "$QUEUE" run --owner small-model --purpose "small-model: $purpose" --claim gpu --expected-minutes "$mins" \
    --max-wait-minutes 900 --launched-by "detached runbook sm71/gpu/run-sm71.sh" -- \
    env $envs "$PY" -X utf8 "$SC" --pool $pool --users-file "$users" --model "${MODEL[$m]}" --only "${LABEL[$m]}" \
    --time-limit 3000 --split-existing $extra --out "$O/$m-$c.jsonl" >> "$O/run-sm71.out" 2>&1
  local rc=$?; log "exit $rc: $purpose"
}
log "started (commit $(git -C $WT rev-parse --short HEAD))"
n0=$(wc -l < "$L56")
until tail -n +"$n0" "$L56" | grep -q -E "exit [0-9]+: SM56 scoring s-w64kd-final-sc-lanes-e14|STOP:"; do
  [ -e "$O/CANCEL" ] && { log "CANCEL"; exit 1; }
  [ "$(date +%H%M)" -ge 2030 ] && { log "SM56 e14 scoring not seen by 20:30; going ahead (queue order decides)"; break; }
  sleep 60; done
log "SM56 epoch-14 scoring done or launcher stopped: $(tail -n +"$n0" "$L56" | grep -E 'exit [0-9]+: SM56 scoring|STOP:' | head -1)"
for mc in big:keep small:keep big:neighbour small:neighbour big:flags small:flags big:reseed small:reseed \
          small8k:keep small8k:neighbour; do ticket ${mc%%:*} ${mc##*:} 464; done
log "464 set done"
for mc in big:keep small:keep big:neighbour small:neighbour; do ticket ${mc%%:*} ${mc##*:} 732; ticket ${mc%%:*} ${mc##*:} long; done
log "732 pool done"
log "done"
