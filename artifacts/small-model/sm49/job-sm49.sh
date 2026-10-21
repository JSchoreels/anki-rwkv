#!/usr/bin/env bash
# SM49 (one GPU queue ticket): smaller int states for SM36, no retraining, 732 pool (464 subset in the summary).
#   A w8+s4hx8        int4 card and note rows, f16 row scales, int8 shift vectors
#   B w8+s4hx4        A + int4 shift vectors
#   C w8+s4hgx4       B + one f16 scale per head (16 rows) instead of per row
#   C' w8+s4hgx4c8hx8n C on cards, SM47's w8+s4c8nhx8 notes (int8 rows, f16 row scales, int8 shift vectors)
#   1. GPU check: the 3 users of --count 3, 4 old (w8+s8, w8+s4c8n, w8+s8hx8, w8+s4c8nhx8) + 4 new variants on the GPU
#      vs check-cpu.jsonl (strict |dLogLoss| <= 1e-5 per head; int formats, not codebooks), and the 4 old ones vs SM47's
#      GPU rows (<= 1e-5; shows the kernel without the HEAD_SCALE flag is unchanged) -> STOP on mismatch
#   2. two passes over users-sm49.json (732 pool minus the 9 users that never fit the 0.8 memory cap; not retried),
#      --lanes 1, default GPU memory cap 0.8
#   3. CPU summary sm49-summary.md / .json
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/sm49
O=$WT/artifacts/small-model/sm49
SM47=C:/Users/jonat/workspace/wt/sm45/artifacts/small-model/sm47
CK=C:/Users/jonat/workspace/wt/sm42/artifacts/small-model/scale-w64kd-gate-sc-lanes-cont-decay1ep32k-muon/expertium/E89-F65e-F77a/master.pth
L=s-w64kd-gate-sc-lanes-dec12-1ep32k-muon
NEW="--only $L/w8+s4hx8 --only $L/w8+s4hx4 --only $L/w8+s4hgx4 --only $L/w8+s4hgx4c8hx8n"
OLD="--only $L/w8+s8 --only $L/w8+s4c8n --only $L/w8+s8hx8 --only $L/w8+s4c8nhx8"
log() { echo "[sm49 $(date '+%m-%d %H:%M:%S')] $*"; }
cd "$WT" || exit 1
unset HELDOUT_GPU_MEMORY_FRACTION
log "start ($(git -C "$WT" log --oneline -1))"
# the queue ignores quiet.json when it is not valid JSON: wait here (GPU idle) if the run could reach a quiet window
w=$("$PY" -X utf8 "$O/quiet_wait.py" 65)
if [ "${w:-0}" != 0 ]; then log "quiet window within 65 min: waiting ${w}s"; sleep "$w"; fi
log "GPU check, 3 users"
rm -f "$O/check-gpu.jsonl"
"$PY" -X utf8 "$WT/scripts/small_model_heldout.py" --pool 732 --count 3 --lanes 1 --model "$L=expertium:w64:$CK" \
  $OLD $NEW --out artifacts/small-model/sm49/check-gpu.jsonl || { log "STOP: GPU check run failed"; exit 1; }
"$PY" -X utf8 - "$O" "$SM47/check-gpu.jsonl" "${SM49_GPU_RULE:-strict}" <<'PYEOF' || { log "STOP: GPU check mismatch"; exit 1; }
import json, sys
from pathlib import Path
o, sm47, rule = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
NEW = {"w8+s4hx8", "w8+s4hx4", "w8+s4hgx4", "w8+s4hgx4c8hx8n"}
key = lambda r: (r["user_id"], r["variant"])
rows = lambda p: {key(r): r for r in map(json.loads, open(p)) if "benchmark" in r}
cpu, gpu, old = rows(o / "check-cpu.jsonl"), rows(o / "check-gpu.jsonl"), rows(sm47)
assert set(cpu) == set(gpu) and len(cpu) == 24, (sorted(cpu), sorted(gpu))
d = lambda a, b, ks: max(abs(a[k]["benchmark"][h]["LogLoss"] - b[k]["benchmark"][h]["LogLoss"]) for k in ks for h in ("ahead", "instant"))
per = {v: d(cpu, gpu, [k for k in cpu if k[1] == v]) for v in sorted({k[1] for k in cpu})}
oldk = [k for k in gpu if k in old]
assert len(oldk) == 12, oldk
vs47 = d(gpu, old, oldk)
vs47_curve = max(abs(gpu[k]["curve_log_loss"] - old[k]["curve_log_loss"]) for k in oldk)
worst = max(per.values())
old_worst = max(x for v, x in per.items() if v not in NEW)
# SM49_GPU_RULE=discrete (only with the user's approval): the new variants (int4 notes / int4 shift vectors) under the
# +vq rule (per user |d| <= 3e-4, review-weighted mean |d| <= 1e-4 per head); the old variants stay strict
wmean = {v: max(abs(sum((gpu[k]["benchmark"][h]["LogLoss"] - cpu[k]["benchmark"][h]["LogLoss"]) * gpu[k]["benchmark"][h]["size"]
                        for k in cpu if k[1] == v) / sum(gpu[k]["benchmark"][h]["size"] for k in cpu if k[1] == v))
                for h in ("ahead", "instant")) for v in per}
if rule == "discrete":
    ok = old_worst <= 1e-5 and all(per[v] <= 3e-4 and wmean[v] <= 1e-4 for v in NEW)
else:
    ok = worst <= 1e-5
out = {"pairs": len(cpu), "rule": rule + (": new variants per user |d| <= 3e-4 and |weighted mean d| <= 1e-4, old strict"
       if rule == "discrete" else ": per user and head |dLogLoss| <= 1e-5"), "max_abs_dlogloss": worst,
       "per_variant": per, "weighted_mean_abs": wmean, "old_variants_vs_sm47_gpu_benchmark": vs47,
       "old_variants_vs_sm47_gpu_curve_log_loss": vs47_curve, "passed": ok and vs47 <= 1e-5}
json.dump(out, open(o / "check-gpu.json", "w"), indent=1)
print(json.dumps(out, indent=1))
sys.exit(0 if out["passed"] else 1)
PYEOF
log "GPU check passed"
for pass in 1 2; do
  log "scoring pass $pass"
  limit=$([ $pass = 1 ] && echo 6000 || echo 1200)
  "$PY" -X utf8 "$WT/scripts/small_model_heldout.py" --pool 732 --users-file "$O/users-sm49.json" --lanes 1 \
    --model "$L=expertium:w64:$CK" $NEW --time-limit $limit --out artifacts/small-model/sm49/heldout-sm49-pools.jsonl \
    || log "pass $pass exit $?"
done
"$PY" -X utf8 "$O/sm49_summary.py" || log "summary exit $?"
log "done"
