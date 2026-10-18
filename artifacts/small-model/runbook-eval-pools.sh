#!/usr/bin/env bash
# Score the bigger evaluation pools (scripts/eval_pools.py; artifacts/small-model/eval-users-{732,2225}.json) with the
# full-history protocol (small_model_heldout.py --pool N, w8+s8), only for the users that have no row yet, for:
#   fork/w8+s8, s-w64kd-gate-ctx32768c-dec10, s-w64kd-gate-sc-carry-dec12, s-w64kd-gate-sc-zero-dec12,
#   s-w64kd-gate-sc-lanes-dec12 (skipped, with a message, while its checkpoint does not exist yet).
# New rows -> artifacts/small-model/heldout-pools.jsonl. A user counts as done for a label when ANY known rows file already
# has a benchmark row for (user, label): the 464-user rows stay in their old files and are not rescored.
#
#   POOL=732  bash runbook-eval-pools.sh        (732 first; 2225 is for future reference / decisive checks and is INVALID
#   POOL=2225 bash runbook-eval-pools.sh         for any model trained on the final tier)
#   DRYRUN=1 POOL=732 bash runbook-eval-pools.sh   prints the plan and the queue commands, enqueues nothing
# Resumable: rerun the same command. ONLY=<label substring> restricts to matching labels. GPU is capped at 80% by the scorer
# (HELDOUT_GPU_MEMORY_FRACTION); never run it next to a training job (training holds ~13.4 of 16.3 GB).
# Tickets: at most MAXT (60) minutes each; users left after a ticket go to the next one (up to 6 passes per label).
# Time model: 11 min per 31.87M reviews for our w64 checkpoints; the fork is 8x bigger: 29 min per 31.87M (measured on
# the 464 users: model 26.8 min + loading/prep); +2 min fixed per ticket.
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/small-model
SC=$WT/scripts/small_model_heldout.py
SM=$WT/artifacts/small-model
OUT=$SM/heldout-pools.jsonl
TODO=$SM/pool-todo
CKPT=expertium/E89-F65e-F77a/master.pth
TEACHER=C:/Users/jonat/workspace/anki/qt/aqt/rwkv_inference/RWKV_trained_on_5000_10000.pth
SCW=C:/Users/jonat/workspace/wt/state-carry/artifacts/small-model
V2W=C:/Users/jonat/workspace/wt/sm21v2/artifacts/small-model
POOL=${POOL:-732}
DRYRUN=${DRYRUN:-0}
MAXT=${MAXT:-60}
ROWS="$SM/heldout.jsonl $SCW/heldout-sm21.jsonl $V2W/heldout-sm21v2.jsonl $OUT"
case "$POOL" in 732|2225) ;; *) echo "POOL must be 732 or 2225"; exit 2;; esac
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$TODO"
log() { echo "[pools $(date +%H:%M:%S)] $*"; }

# label | tag | kind:arch | checkpoint | minutes per 31.87M reviews   (order: fork and dec10 first)
LABELS=(
  "fork/w8+s8|fork|fork:|$TEACHER|29"
  "s-w64kd-gate-ctx32768c-dec10/w8+s8|st|expertium:w64|$SM/scale-w64kd-gate-cached-ctx32768-cont-decay/$CKPT|11"
  "s-w64kd-gate-sc-carry-dec12/w8+s8|st|expertium:w64|$SCW/scale-w64kd-gate-sc-carry-cont-decay/$CKPT|11"
  "s-w64kd-gate-sc-zero-dec12/w8+s8|st|expertium:w64|$SCW/scale-w64kd-gate-sc-zero-cont-decay/$CKPT|11"
  "s-w64kd-gate-sc-lanes-dec12/w8+s8|st|expertium:w64|$V2W/scale-w64kd-gate-sc-lanes-cont-decay/$CKPT|11"
)

# missing LABEL RATE -> prints "<users> <reviews> <minutes>" and writes $TODO/<n>-<pool>.json
missing() {
  LABEL="$1" RATE="$2" POOL="$POOL" ROWS="$ROWS" TODOF="$3" "$PY" -X utf8 - <<'PYEOF'
import json, math, os
from pathlib import Path
label, rate, pool = os.environ["LABEL"], float(os.environ["RATE"]), int(os.environ["POOL"])
doc = json.loads(Path(f"C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/eval-users-{pool}.json").read_text())
counts = json.loads(Path(doc["split"]).read_text())["review_counts"]
have = set()
for f in os.environ["ROWS"].split():
    if Path(f).exists():
        for line in open(f, encoding="utf-8"):
            if label in line:
                r = json.loads(line)
                if r.get("label") == label and "benchmark" in r:
                    have.add(r["user_id"])
todo = [u for u in doc["users"] if u not in have]
reviews = sum(counts[str(u)] for u in todo)
Path(os.environ["TODOF"]).write_text(json.dumps({"label": label, "pool": pool, "users": todo}))
print(len(todo), reviews, math.ceil(reviews / 31.87e6 * rate) + 2 if todo else 0)
PYEOF
}

for entry in "${LABELS[@]}"; do
  IFS='|' read -r label tag kind ckpt rate <<< "$entry"
  [ -n "$ONLY" ] && [[ "$label" != *"$ONLY"* ]] && continue
  if [ "$tag" != fork ] && [ ! -f "$ckpt" ]; then
    log "SKIP $label: checkpoint missing ($ckpt)"; continue
  fi
  name=$(echo "${label%%/*}" | tr -c 'A-Za-z0-9\n' '_')
  todof="$TODO/$name-$POOL.json"
  for pass in 1 2 3 4 5 6; do
    read -r nusers nrev minutes <<< "$(missing "$label" "$rate" "$todof")"
    if [ "$nusers" = 0 ]; then log "$label: pool $POOL complete"; break; fi
    t=$(( minutes > MAXT ? MAXT : minutes ))
    log "$label pass $pass: $nusers users / $nrev reviews to score, $minutes min total, this ticket $t min"
    if [ "$tag" = fork ]; then spec="--model fork=fork::$ckpt"; else spec="--model ${label%%/*}=$kind:$ckpt"; fi
    purpose="small-model: pool $POOL full-history scoring of $label ($nusers users, pass $pass)"
    cmd=("$PY" -X utf8 "$SC" $spec --only "$label" --pool "$POOL" --users-file "$todof" --time-limit $(( (t - 2) * 60 )) --out "$OUT")
    if [ "$DRYRUN" = 1 ]; then
      log "DRYRUN enqueue: $purpose (expected $t min)"; echo "    cmd: ${cmd[*]}"; break
    fi
    "$PY" "$QUEUE" run --owner small-model --purpose "$purpose" --claim gpu --expected-minutes "$t" -- "${cmd[@]}"
    rc=$?; log "exit $rc: $label pass $pass"
    [ "$rc" != 0 ] && { log "stopping $label after a failed ticket"; break; }
  done
done
[ "$DRYRUN" = 1 ] && log "DRYRUN OK" || log "done (POOL=$POOL)"
