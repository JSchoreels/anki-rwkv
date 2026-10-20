#!/usr/bin/env bash
# SM44 (2026-10-20): GPU scoring of the compressed card/note state (~400 and ~800 bits) on the 732-user pool (the
# 464 pool is a subset of it, so its readout needs no extra GPU pass). Code: scripts/state_vq.py "GPU scoring" +
# scripts/small_model_heldout.py (+vq variants on CUDA), check scripts/check_state_vq_score_gpu.py.
#   0. Preconditions: committed source; the CPU-first check (sm44/check-sm44-cpu.json) passed; checkpoints and
#      codebook files present with the pinned sha256.
#   1. Wait for the SM43 full launcher to end (its log's last line "done" / "A failed" / "B failed"), so SM44 does not
#      slot in between SM43's tickets; then wait while HOLD-chain exists before every ticket.
#   2. GPU check ticket (~10 min): the GPU path vs the stored CPU-scorer predictions on the 5 smallest pool users
#      (SM42 vq378x412, vq770x828, SM32c vq378x412 with its learned books), plus reviews/s on 2 near-median users.
#      STOP on a mismatch.
#   3. Scoring tickets, one label at a time, at most MAXT minutes each (users left go to the next pass, up to PASSES):
#        SM42  w8+vq378x412 (k-means books), SM32c w8+vq378x412 (its learned books), SM32c w8+s8,
#        SM42  w8+vq770x828 (k-means books; rank-2 note on the torch step loop), SM42 w8+s8 (re-score: reproducibility
#        against the earlier SM42 rows).
#      Rows -> sm44/heldout-sm44.jsonl. GPU memory capped by the scorer at HELDOUT_GPU_MEMORY_FRACTION (0.8).
#   4. CPU: sm44_summary.py -> sm44-summary.md / .json (pools 732 and 464).
# Launch: nohup bash C:/Users/jonat/workspace/wt/sm32c/artifacts/small-model/sm44/runbook-sm44.sh \
#           > C:/Users/jonat/workspace/wt/sm32c/artifacts/small-model/sm44/runbook-sm44.log 2>&1 & disown
# DRYRUN=1: preconditions, the plan and the queue commands only; no waiting, no queue, no summary.
# Resumable: rerun the same command (done users are skipped; a present check-sm44-gpu.json that passed is reused).
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/sm32c
SM=$WT/artifacts/small-model
O=$SM/sm44
SC=$WT/scripts/small_model_heldout.py
CHK=$WT/scripts/check_state_vq_score_gpu.py
HOLD=C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/HOLD-chain
SM43_LOG=C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/sm43/launch-sm43-full.log
CKPT=expertium/E89-F65e-F77a/master.pth
SM42_CKPT=C:/Users/jonat/workspace/wt/sm42/artifacts/small-model/scale-w64kd-gate-cached-ctx32768-cont-decay1ep32k-lanes/$CKPT
SM32C_D=$SM/scale-w64kd-gate-cached-ctx32768-cont-decay1ep32k-lanes-vqlearn/expertium/E89-F65e-F77a
SM32C_CKPT=$SM32C_D/master.pth
KMEANS=$SM/state-vq/codebooks.npz
KMEANS_SHA=eab6db585da4f2944b0abf26788a93da49bb81a8b8a44ba2f62819e11a73aee0
LEARNED=$SM32C_D/state-vq-books.npz
LEARNED_SHA=a4d487b194c68cdf36e6ffb07eb92904e7d1cef01ae85fdd4ebcf8735af878fd
L42=s-w64kd-gate-ctx32768c-dec10-1ep32k-lanes
L32C=s-w64kd-gate-ctx32768c-dec10-1ep32k-lanes-vqlearn
OUT=$O/heldout-sm44.jsonl
POOL=732
MAXT=${MAXT:-90}
PASSES=${PASSES:-4}
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONWARNINGS=ignore
export HELDOUT_GPU_MEMORY_FRACTION=${HELDOUT_GPU_MEMORY_FRACTION:-0.8}
log() { echo "[sm44 $(date '+%m-%d %H:%M:%S')] $*"; }
wait_hold() { [ "$DRYRUN" = 1 ] && return 0; until [ ! -e "$HOLD" ]; do sleep 60; done; }
q() {  # q MINUTES PURPOSE -- CMD...
  local minutes=$1 purpose=$2; shift 3
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $purpose (expected $minutes min)"; echo "    cmd: $*"; return 0; fi
  wait_hold
  log "enqueue: $purpose (expected $minutes min)"
  "$PY" "$QUEUE" run --owner small-model --purpose "$purpose" --claim gpu --expected-minutes "$minutes" \
    --launched-by "detached runbook sm44" -- "$@"
  local rc=$?; log "exit $rc: $purpose"; return $rc
}
cd "$WT" || exit 1
mkdir -p "$O/logs"

# ---- 0. preconditions
if [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then log "STOP: uncommitted source changes"; exit 2; fi
"$PY" -c "import json,sys; r=json.load(open(r'$O/check-sm44-cpu.json')); sys.exit(0 if r.get('passed') and not r.get('fails') else 1)" \
  || { log "STOP: CPU check sm44/check-sm44-cpu.json has not passed"; exit 2; }
for f in "$SM42_CKPT" "$SM32C_CKPT" "$KMEANS" "$LEARNED"; do [ -f "$f" ] || { log "STOP: missing $f"; exit 2; }; done
[ "$(sha256sum "$KMEANS" | cut -d' ' -f1)" = "$KMEANS_SHA" ] || { log "STOP: k-means codebooks sha256 changed"; exit 2; }
[ "$(sha256sum "$LEARNED" | cut -d' ' -f1)" = "$LEARNED_SHA" ] || { log "STOP: SM32c learned codebooks sha256 changed"; exit 2; }
log "start ($(git -C "$WT" log --oneline -1))"

# ---- 1. wait for SM43 full to end
if [ "$DRYRUN" != 1 ] && [ "${WAIT_SM43:-1}" = 1 ]; then
  log "waiting for the SM43 full launcher to end ($SM43_LOG)"
  until grep -aEq '^\[sm43full [^]]*\] (done|A failed|B failed)' "$SM43_LOG" 2>/dev/null; do sleep 120; done
  log "SM43: $(grep -aE '^\[sm43full [^]]*\] (done|A failed|B failed)' "$SM43_LOG" | tail -1)"
fi

# ---- 2. GPU check
GCHK=$O/check-sm44-gpu.json
passed() { "$PY" -c "import json,sys; r=json.load(open(r'$GCHK')); sys.exit(0 if r.get('passed') else 1)" 2>/dev/null; }
if [ "$DRYRUN" = 1 ] || ! passed; then
  q 10 "SM44 GPU check: GPU +vq scorer path vs CPU scorer on 5 small users, reviews/s" -- "$PY" -X utf8 "$CHK" gpu \
    || { log "STOP: GPU check failed (see $GCHK and the queue output above)"; exit 1; }
fi
if [ "$DRYRUN" != 1 ]; then
  passed || { log "STOP: GPU check report not passed"; exit 1; }
  "$PY" -c "import json; r=json.load(open(r'$GCHK')); print('[gpu-check] reviews/s', r['timing']['reviews_per_second'])"
fi

# ---- 3. scoring passes
# missing LABEL RATEKEY -> "<users> <reviews> <minutes>" (minutes at the GPU check's rate for RATEKEY, +20%, +3)
missing() {
  LABEL="$1" KEY="$2" OUT="$OUT" GCHK="$GCHK" SMD="$SM" "$PY" -X utf8 - <<'PYEOF'
import json, math, os
from pathlib import Path
label, key = os.environ["LABEL"], os.environ["KEY"]
doc = json.loads((Path(os.environ["SMD"]) / "eval-users-732.json").read_text())
counts = json.loads(Path(doc["split"]).read_text())["review_counts"]
have = set()
out = Path(os.environ["OUT"])
if out.exists():
    for line in out.open(encoding="utf-8"):
        if label in line:
            r = json.loads(line)
            if r.get("label") == label and "benchmark" in r:
                have.add(r["user_id"])
todo = [u for u in doc["users"] if u not in have]
reviews = sum(counts[str(u)] for u in todo)
rate = 48300.0 if key == "w8+s8" else 15000.0   # dry-run defaults (48.3k/s = 11 min per 31.87M, the measured w64 rate)
gchk = Path(os.environ["GCHK"])
if gchk.exists():
    rate = json.loads(gchk.read_text()).get("timing", {}).get("reviews_per_second", {}).get(key, rate)
print(len(todo), reviews, math.ceil(reviews / rate / 60 * 1.2) + 3 if todo else 0)
PYEOF
}
# label | tag | checkpoint | codebooks | rate key
LABELS=(
  "$L42/w8+vq378x412|$L42|$SM42_CKPT|$KMEANS|w8+vq378x412"
  "$L32C/w8+vq378x412|$L32C|$SM32C_CKPT|$LEARNED|w8+vq378x412"
  "$L32C/w8+s8|$L32C|$SM32C_CKPT|$KMEANS|w8+s8"
  "$L42/w8+vq770x828|$L42|$SM42_CKPT|$KMEANS|w8+vq770x828"
  "$L42/w8+s8|$L42|$SM42_CKPT|$KMEANS|w8+s8"
)
for entry in "${LABELS[@]}"; do
  IFS='|' read -r label tag ckpt books key <<< "$entry"
  for pass in $(seq 1 "$PASSES"); do
    read -r nusers nrev minutes <<< "$(missing "$label" "$key")"
    if [ "$nusers" = 0 ]; then log "$label: pool $POOL complete"; break; fi
    t=$(( minutes > MAXT ? MAXT : minutes ))
    log "$label pass $pass: $nusers users / $nrev reviews to score, ~$minutes min at the check's rate, this ticket $t min"
    q "$t" "SM44: pool $POOL scoring of $label ($nusers users, pass $pass)" -- "$PY" -X utf8 "$SC" \
      --model "$tag=expertium:w64:$ckpt" --only "$label" --pool "$POOL" --vq-codebooks "$books" \
      --time-limit $(( (t - 3) * 60 )) --out "$OUT" || { log "ticket failed: $label pass $pass; next label"; break; }
    [ "$DRYRUN" = 1 ] && break
  done
done

# ---- 4. summary (CPU)
if [ "$DRYRUN" = 1 ]; then log "DRYRUN summary: $PY -X utf8 $O/sm44_summary.py"; log "DRYRUN OK"; exit 0; fi
"$PY" -X utf8 "$O/sm44_summary.py" || log "summary exit $?"
log "done"
