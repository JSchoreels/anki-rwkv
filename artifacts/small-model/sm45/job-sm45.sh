#!/usr/bin/env bash
# SM45 (one GPU queue ticket): int4 card state + int8 note state (+s4c8n) and int4 everywhere (+s4) for SM36 on the 732 pool.
#   1. GPU check: the 3 smallest pool users on the GPU vs the CPU rows of check-cpu.jsonl (|dLogLoss| <= 1e-5 per head) -> stop on mismatch
#   2. up to two passes over the 732 pool (rows resume per user), default GPU memory cap 0.8
#   3. CPU summary sm45-summary.md
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/sm45
O=$WT/artifacts/small-model/sm45
CK=C:/Users/jonat/workspace/wt/sm42/artifacts/small-model/scale-w64kd-gate-sc-lanes-cont-decay1ep32k-muon/expertium/E89-F65e-F77a/master.pth
L=s-w64kd-gate-sc-lanes-dec12-1ep32k-muon
log() { echo "[sm45 $(date '+%m-%d %H:%M:%S')] $*"; }
cd "$WT" || exit 1
unset HELDOUT_GPU_MEMORY_FRACTION
log "GPU check, 3 users"
rm -f "$O/check-gpu.jsonl"
"$PY" -X utf8 scripts/small_model_heldout.py --pool 732 --count 3 --model "$L=expertium:w64:$CK" \
  --only "$L/w8+s8" --only "$L/w8+s4" --only "$L/w8+s4c8n" --out artifacts/small-model/sm45/check-gpu.jsonl || { log "STOP: GPU check run failed"; exit 1; }
"$PY" -X utf8 - "$O" <<'PYEOF' || { log "STOP: GPU check mismatch"; exit 1; }
import json, sys
from pathlib import Path
o = Path(sys.argv[1])
key = lambda r: (r["user_id"], r["variant"])
cpu = {key(r): r for r in map(json.loads, open(o / "check-cpu.jsonl")) if "benchmark" in r}
gpu = {key(r): r for r in map(json.loads, open(o / "check-gpu.jsonl")) if "benchmark" in r}
assert set(cpu) == set(gpu) and len(cpu) == 9, (sorted(cpu), sorted(gpu))
worst = max(abs(cpu[k]["benchmark"][h]["LogLoss"] - gpu[k]["benchmark"][h]["LogLoss"]) for k in cpu for h in ("ahead", "instant"))
json.dump({"pairs": len(cpu), "max_abs_dlogloss": worst, "passed": worst <= 1e-5}, open(o / "check-gpu.json", "w"), indent=1)
print("GPU vs CPU max |dLogLoss|", worst)
sys.exit(0 if worst <= 1e-5 else 1)
PYEOF
for pass in 1 2; do
  log "scoring pass $pass"
  "$PY" -X utf8 scripts/small_model_heldout.py --pool 732 --model "$L=expertium:w64:$CK" \
    --only "$L/w8+s4c8n" --only "$L/w8+s4" --time-limit 2400 --out artifacts/small-model/sm45/heldout-sm45-pools.jsonl \
    || log "pass $pass exit $?"
done
"$PY" -X utf8 "$O/sm45_summary.py" || log "summary exit $?"
log "done"
