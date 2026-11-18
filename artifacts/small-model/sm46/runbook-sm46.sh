#!/usr/bin/env bash
# SM46 (2026-10-20): codebook state compression at the large budget asked by the outside researcher, recipe
# vq5902x3028 (5,902 bits per card, 3,028 per note; rank-6 WKV heads, see scripts/state_vq.py), codebooks fitted on SM36's
# own states after training (no retraining), scored on SM36 (the best small model) with the GPU held-out scorer.
# Both streams are rank 6, so both run the per-step torch loop on the GPU (no fused kernel); see HANDOFF-sm46.md.
#   0. Preconditions: committed source; the CPU-first check (sm46/check-sm46-cpu.json) passed; checkpoint and codebooks
#      present with the pinned sha256.
#   1. Wait for the SM44 runbook to end (its log line "[sm44 <date>] done" or "... STOP: ..."); then wait while
#      HOLD-chain exists before every ticket.
#   2. GPU check ticket: the CUDA path vs the stored CPU-scorer predictions on the 3 smallest pool users (benchmark
#      LogLoss within 1e-5 per user and head), base float-state numerics, reviews/s on 2 near-median users. STOP on a
#      mismatch.
#   3. Scoring tickets (<= MAXT min each, --time-limit leaves room for the slowest user), first the 464 pool, then the
#      rest of the 732 pool. If the 732 pool projects above BUDGET minutes at the check's measured rate (+20%), only the
#      464 pool is scored (logged). Rows -> sm46/heldout-sm46.jsonl. GPU memory: the scorer's default cap 0.8.
#   4. CPU: sm46_summary.py -> sm46-summary.md / .json (732 and 464).
# Launch: nohup bash C:/Users/jonat/workspace/wt/sm46/artifacts/small-model/sm46/runbook-sm46.sh \
#           > C:/Users/jonat/workspace/wt/sm46/artifacts/small-model/sm46/runbook-sm46.log 2>&1 & disown
# DRYRUN=1: preconditions, the plan and the queue commands only; no waiting, no queue, no summary.
# Resumable: rerun the same command (done users are skipped; a present check-sm46-gpu.json that passed is reused).
# WAIT_SM44=0 skips the SM44 wait.
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/sm46
SM=$WT/artifacts/small-model
O=$SM/sm46
SC=$WT/scripts/small_model_heldout.py
CHK=$WT/scripts/check_state_vq_score_gpu.py
HOLD=C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/HOLD-chain
SM44_LOG=C:/Users/jonat/workspace/wt/sm32c/artifacts/small-model/sm44/runbook-sm44.log
CKPT=C:/Users/jonat/workspace/wt/sm42/artifacts/small-model/scale-w64kd-gate-sc-lanes-cont-decay1ep32k-muon/expertium/E89-F65e-F77a/master.pth
CKPT_SHA=e00000c02e3c6445c4693868e10d64de5ea5b6e9b8637960920dbce42e7303b8
BOOKS=$O/books/codebooks-sm46.npz
BOOKS_SHA=af46e6a253c4f18e349792527c815860006924c301f70c3924a4872c38cf81f7
RECIPE=vq5902x3028
L=s-w64kd-gate-sc-lanes-dec12-1ep32k-muon
LABEL=$L/w8+$RECIPE
CASES=$O/check-cases.json
OUT=$O/heldout-sm46.jsonl
MAXT=${MAXT:-110}
BUDGET=${BUDGET:-240}
PASSES=${PASSES:-6}
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONWARNINGS=ignore
unset HELDOUT_GPU_MEMORY_FRACTION   # the scorer's default cap (0.8)
log() { echo "[sm46 $(date '+%m-%d %H:%M:%S')] $*"; }
wait_hold() { [ "$DRYRUN" = 1 ] && return 0; until [ ! -e "$HOLD" ]; do sleep 60; done; }
q() {  # q MINUTES PURPOSE -- CMD...
  local minutes=$1 purpose=$2; shift 3
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $purpose (expected $minutes min)"; echo "    cmd: $*"; return 0; fi
  wait_hold
  log "enqueue: $purpose (expected $minutes min)"
  "$PY" "$QUEUE" run --owner small-model --purpose "$purpose" --claim gpu --expected-minutes "$minutes" \
    --launched-by "detached launcher sm46" -- "$@"
  local rc=$?; log "exit $rc: $purpose"; return $rc
}
cd "$WT" || exit 1
mkdir -p "$O/logs"

# ---- 0. preconditions
if [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then log "STOP: uncommitted source changes"; exit 2; fi
"$PY" -c "import json,sys; r=json.load(open(r'$O/check-sm46-cpu.json')); sys.exit(0 if r.get('passed') and not r.get('fails') else 1)" \
  || { log "STOP: CPU check sm46/check-sm46-cpu.json has not passed"; exit 2; }
for f in "$CKPT" "$BOOKS" "$CASES"; do [ -f "$f" ] || { log "STOP: missing $f"; exit 2; }; done
[ "$(sha256sum "$CKPT" | cut -d' ' -f1)" = "$CKPT_SHA" ] || { log "STOP: SM36 checkpoint sha256 changed"; exit 2; }
[ "$(sha256sum "$BOOKS" | cut -d' ' -f1)" = "$BOOKS_SHA" ] || { log "STOP: SM46 codebooks sha256 changed"; exit 2; }
log "start ($(git -C "$WT" log --oneline -1))"

# ---- 1. wait for the SM44 runbook to end
if [ "$DRYRUN" != 1 ] && [ "${WAIT_SM44:-1}" = 1 ]; then
  log "waiting for the SM44 runbook to end ($SM44_LOG: 'done' or 'STOP')"
  until grep -aEq '^\[sm44 [^]]*\] (done|STOP)' "$SM44_LOG" 2>/dev/null; do sleep 120; done
  log "SM44: $(grep -aE '^\[sm44 [^]]*\] (done|STOP)' "$SM44_LOG" | tail -1)"
fi

# ---- 2. GPU check
GCHK=$O/check-sm46-gpu.json
passed() { "$PY" -c "import json,sys; r=json.load(open(r'$GCHK')); sys.exit(0 if r.get('passed') else 1)" 2>/dev/null; }
if [ "$DRYRUN" = 1 ] || ! passed; then
  q 25 "SM46 GPU check: GPU +$RECIPE scorer path (rank-6 torch loop) vs CPU scorer on 3 small users, reviews/s" -- \
    "$PY" -X utf8 "$CHK" gpu --cases "$CASES" --users 3 --ref-dir "$O/check-ref" --output "$GCHK" \
    || { log "STOP: GPU check failed (see $GCHK and the queue output above)"; exit 1; }
fi
if [ "$DRYRUN" != 1 ]; then
  passed || { log "STOP: GPU check report not passed"; exit 1; }
  "$PY" -c "import json; r=json.load(open(r'$GCHK')); print('[gpu-check] reviews/s', r['timing']['reviews_per_second'])"
fi

# ---- 3. scoring passes
# plan POOL -> "<users> <reviews> <minutes> <margin>" for the users of POOL without a row: minutes at the GPU check's
# rate for the recipe (+20%, +3); margin = minutes of the largest such user at that rate (+2), for --time-limit
plan() {
  POOL="$1" LABEL="$LABEL" KEY="w8+$RECIPE" OUT="$OUT" GCHK="$GCHK" ROOT="$WT" "$PY" -X utf8 - <<'PYEOF'
import json, math, os, sys
from pathlib import Path
sys.path.insert(0, os.environ["ROOT"])
from scripts.eval_pools import pool_users
from scripts import compress_common as cc
counts = cc.run_info()["review_counts"]
label, key = os.environ["LABEL"], os.environ["KEY"]
have = set()
out = Path(os.environ["OUT"])
if out.exists():
    for line in out.open(encoding="utf-8"):
        if label in line:
            r = json.loads(line)
            if r.get("label") == label and "benchmark" in r:
                have.add(r["user_id"])
todo = [u for u in pool_users(int(os.environ["POOL"])) if u not in have]
reviews = sum(counts[str(u)] for u in todo)
rate = 3000.0   # dry-run default only; the real plan uses the GPU check's measured rate
gchk = Path(os.environ["GCHK"])
if gchk.exists():
    rate = json.loads(gchk.read_text()).get("timing", {}).get("reviews_per_second", {}).get(key, rate)
big = max((counts[str(u)] for u in todo), default=0)
print(len(todo), reviews, math.ceil(reviews / rate / 60 * 1.2) + 3 if todo else 0, math.ceil(big / rate / 60 * 1.2) + 2)
PYEOF
}
read -r n732 r732 m732 _ <<< "$(plan 732)"
POOLS="464 732"
if [ "$m732" -gt "$BUDGET" ]; then
  POOLS="464"
  log "732 pool projects $m732 min (> $BUDGET): scoring only the 464 pool"
else
  log "732 pool projects $m732 min ($n732 users, $r732 reviews): 464 first, then the rest of 732"
fi
for pool in $POOLS; do
  for pass in $(seq 1 "$PASSES"); do
    read -r nusers nrev minutes margin <<< "$(plan "$pool")"
    if [ "$nusers" = 0 ]; then log "pool $pool complete"; break; fi
    t=$(( minutes > MAXT ? MAXT : minutes ))
    t=$(( t < margin + 5 ? margin + 5 : t ))
    limit=$(( t - margin - 1 )); [ "$limit" -lt 1 ] && limit=1
    log "pool $pool pass $pass: $nusers users / $nrev reviews to score, ~$minutes min at the check's rate; ticket $t min, user start limit $limit min (largest user ~$margin min)"
    q "$t" "SM46: pool $pool scoring of SM36 w8+$RECIPE ($nusers users, pass $pass)" -- "$PY" -X utf8 "$SC" \
      --model "$L=expertium:w64:$CKPT" --only "$LABEL" --pool "$pool" --vq-codebooks "$BOOKS" \
      --time-limit $(( limit * 60 )) --out "$OUT" || { log "ticket failed: pool $pool pass $pass"; break; }
    [ "$DRYRUN" = 1 ] && break
  done
done

# ---- 4. summary (CPU)
if [ "$DRYRUN" = 1 ]; then log "DRYRUN summary: $PY -X utf8 $O/sm46_summary.py"; log "DRYRUN OK"; exit 0; fi
"$PY" -X utf8 "$O/sm46_summary.py" || log "summary exit $?"
log "done"
