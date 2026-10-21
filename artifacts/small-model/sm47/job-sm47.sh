#!/usr/bin/env bash
# SM47 (one GPU queue ticket): float16 row scales (h) and int8 token-shift vectors (x8) on top of +s8 / +s4c8n for SM36,
# 732 pool, 6 variants.
#   1. GPU check: the 3 smallest pool users, all 8 variants (w8+s8, w8+s4c8n + the 6 new) on the GPU vs the CPU rows of
#      check-cpu.jsonl (|dLogLoss| <= 1e-5 per head), and w8+s8 / w8+s4c8n vs SM45's GPU rows (the kernel without the
#      F16_SCALE flag must be unchanged; also <= 1e-5, the exact max is logged) -> stop on mismatch
#   2. up to two passes over the 732 pool (rows resume per user; OOM rows are retried), default GPU memory cap 0.8
#   3. CPU summary sm47-summary.md / .json
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/sm45
O=$WT/artifacts/small-model/sm47
CK=C:/Users/jonat/workspace/wt/sm42/artifacts/small-model/scale-w64kd-gate-sc-lanes-cont-decay1ep32k-muon/expertium/E89-F65e-F77a/master.pth
L=s-w64kd-gate-sc-lanes-dec12-1ep32k-muon
NEW="--only $L/w8+s8h --only $L/w8+s8x8 --only $L/w8+s8hx8 --only $L/w8+s4c8nh --only $L/w8+s4c8nx8 --only $L/w8+s4c8nhx8"
log() { echo "[sm47 $(date '+%m-%d %H:%M:%S')] $*"; }
cd "$WT" || exit 1
unset HELDOUT_GPU_MEMORY_FRACTION
log "start ($(git -C "$WT" log --oneline -1))"
log "GPU check, 3 users"
rm -f "$O/check-gpu.jsonl"
"$PY" -X utf8 scripts/small_model_heldout.py --pool 732 --count 3 --model "$L=expertium:w64:$CK" \
  --only "$L/w8+s8" --only "$L/w8+s4c8n" $NEW --out artifacts/small-model/sm47/check-gpu.jsonl || { log "STOP: GPU check run failed"; exit 1; }
"$PY" -X utf8 - "$O" "$WT/artifacts/small-model/sm45/check-gpu.jsonl" <<'PYEOF' || { log "STOP: GPU check mismatch"; exit 1; }
import json, sys
from pathlib import Path
o, sm45 = Path(sys.argv[1]), Path(sys.argv[2])
key = lambda r: (r["user_id"], r["variant"])
rows = lambda p: {key(r): r for r in map(json.loads, open(p)) if "benchmark" in r}
cpu, gpu, old = rows(o / "check-cpu.jsonl"), rows(o / "check-gpu.jsonl"), rows(sm45)
assert set(cpu) == set(gpu) and len(cpu) == 24, (sorted(cpu), sorted(gpu))
d = lambda a, b, ks: max(abs(a[k]["benchmark"][h]["LogLoss"] - b[k]["benchmark"][h]["LogLoss"]) for k in ks for h in ("ahead", "instant"))
per = {v: d(cpu, gpu, [k for k in cpu if k[1] == v]) for v in sorted({k[1] for k in cpu})}
oldk = [k for k in gpu if k in old]
assert len(oldk) == 6, oldk
vs45 = d(gpu, old, oldk)
vs45_curve = max(abs(gpu[k]["curve_log_loss"] - old[k]["curve_log_loss"]) for k in oldk)
worst = max(per.values())
out = {"pairs": len(cpu), "max_abs_dlogloss": worst, "per_variant": per, "old_variants_vs_sm45_gpu_benchmark": vs45,
       "old_variants_vs_sm45_gpu_curve_log_loss": vs45_curve, "passed": worst <= 1e-5 and vs45 <= 1e-5}
json.dump(out, open(o / "check-gpu.json", "w"), indent=1)
print(json.dumps(out, indent=1))
sys.exit(0 if out["passed"] else 1)
PYEOF
log "GPU check passed"
for pass in 1 2; do
  log "scoring pass $pass"
  limit=$([ $pass = 1 ] && echo 7200 || echo 1500)
  "$PY" -X utf8 scripts/small_model_heldout.py --pool 732 --model "$L=expertium:w64:$CK" \
    $NEW --time-limit $limit --out artifacts/small-model/sm47/heldout-sm47-pools.jsonl \
    || log "pass $pass exit $?"
done
"$PY" -X utf8 "$O/sm47_summary.py" || log "summary exit $?"
log "done"
