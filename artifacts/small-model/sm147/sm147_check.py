"""Low-rank structure of the card and note states (SM147): checks C1-C4 of design.md -> check-sm147.json."""
import json
import math
import subprocess
import sys
from pathlib import Path

S = Path(__file__).resolve().parent
WT = S.parents[2]
SM145_SMOKE = WT.parent / "sm145" / "artifacts" / "small-model" / "sm145" / "cpu" / "smoke-cpu-rows.jsonl"
FIELDS = ("curve_log_loss", "instant_log_loss", "curve_labels", "instant_labels", "benchmark", "benchmark_existing",
          "benchmark_deleted", "prior_buckets")


def load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def finite(x):
    if isinstance(x, dict):
        return all(finite(v) for v in x.values())
    if isinstance(x, list):
        return all(finite(v) for v in x)
    return not isinstance(x, float) or math.isfinite(x)


def main():
    out = {"checks": {}}
    py = sys.executable
    r = subprocess.run([py, "-X", "utf8", "-m", "scripts.state_lowrank"], cwd=WT, capture_output=True, text=True)
    out["checks"]["C1_selftest"] = r.returncode == 0 and "OK" in r.stdout
    smoke = load(S / "cpu" / "smoke-rows.jsonl")
    by = {(x["user_id"], x["tag"], x["variant"]): x for x in smoke if "tag" in x}
    tags = sorted({k[1] for k in by})
    users = sorted({k[0] for k in by})
    c2 = [all(by[(u, t, "w8")][f] == by[(u, t, "w8+lrsv")][f] for f in FIELDS if f in by[(u, t, "w8")])
          for u in users for t in tags]
    out["checks"]["C2_f32_equals_w8"] = {"pass": all(c2) and len(c2) == 18, "pairs": len(c2), "equal": sum(c2)}
    old = {(x["user_id"], x["tag"]): x for x in load(SM145_SMOKE) if x.get("variant") == "w8+s8hx8"}
    c3 = [all(old[(u, t)][f] == by[(u, t, "w8+s8hx8")][f] for f in FIELDS if f in old[(u, t)])
          for u in users for t in tags if (u, t) in old]
    out["checks"]["C3_ref_equals_sm145"] = {"pass": all(c3) and len(c3) == 18, "pairs": len(c3), "equal": sum(c3)}
    rows = load(S / "cpu" / "sm147-rows.jsonl")
    bad = [x for x in rows if "benchmark" not in x or not finite({k: x.get(k) for k in FIELDS})]
    spec_ok = True
    for x in rows:
        for cell in x.get("lowrank_spectrum", []):
            tot = [sum(h) for h in cell["count"]]
            spec_ok &= len(set(tot)) == 1
        per_stream = {}
        for cell in x.get("lowrank_spectrum", []):
            per_stream.setdefault(cell["stream"], set()).add(sum(cell["count"][0]))
        spec_ok &= all(len(v) == 1 for v in per_stream.values())
    grid = {(x["user_id"], x["tag"], x["variant"]) for x in rows if "benchmark" in x}
    out["checks"]["C4_finite_complete"] = {"pass": not bad and spec_ok and len(grid) == 8 * 6 * 5,
                                           "rows": len(rows), "grid": len(grid), "bad_rows": len(bad),
                                           "spectrum_counts_consistent": spec_ok}
    out["all_ok"] = all(v if isinstance(v, bool) else v["pass"] for v in out["checks"].values())
    (S / "check-sm147.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
