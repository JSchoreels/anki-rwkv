"""Codebook state compression sweep (SM90) CPU proof (c, d): proof/cpu-sm90.jsonl vs proof/cpu-sm72s.jsonl (bitwise,
every benchmark field) and the sm90 CPU w8+s8 rows vs the recorded released GPU rows (strict 1e-5 rule).
writes proof/proof-int8.json; exit 1 on a failure."""
import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
REL = Path("C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/gpu/small-lf-neighbour.jsonl")
LABEL = "s-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon/w8+s8"
KEYS = ("benchmark", "benchmark_existing", "benchmark_deleted", "curve_log_loss", "instant_log_loss")


def load(p):
    return {(r["user_id"], r["label"]): r for r in map(json.loads, open(p))}


a, b = load(H / "proof" / "cpu-sm90.jsonl"), load(H / "proof" / "cpu-sm72s.jsonl")
rel = {r["user_id"]: r for r in map(json.loads, open(REL)) if r.get("label") == LABEL}
out = {"sm90_provenance": sorted({r.get("provenance_commit") for r in a.values()}),
       "bitwise_sm90_vs_sm72s": {f"{u} {l}": all(a[(u, l)].get(k) == b[(u, l)].get(k) for k in KEYS) for (u, l) in sorted(a)},
       "cpu_vs_recorded_gpu": {}}
for (u, l), r in a.items():
    if l == "rel/w8+s8":
        for h in ("ahead", "instant"):
            out["cpu_vs_recorded_gpu"][f"{u} {h}"] = abs(r["benchmark"][h]["LogLoss"] - rel[u]["benchmark"][h]["LogLoss"])
out["passes"] = {"bitwise": len(a) == 6 and all(out["bitwise_sm90_vs_sm72s"].values()),
                 "cpu_vs_gpu_1e-5": all(v <= 1e-5 for v in out["cpu_vs_recorded_gpu"].values())}
(H / "proof" / "proof-int8.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out["passes"]), "max cpu-gpu", max(out["cpu_vs_recorded_gpu"].values()))
sys.exit(0 if all(out["passes"].values()) else 1)
