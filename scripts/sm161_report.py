"""Aggregate the SM161 parity comparisons (work/cmp_*.json, scripts/sm161_parity.py compare) and the DayLearn impact
rows (work/impact/*.jsonl, scripts/small_model_heldout.py) into artifacts/small-model/sm161/aggregate.json.
Aggregates only (per column, per path; per-user totals of matched / differing rows)."""
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
OUT = ROOT / "artifacts" / "small-model" / "sm161"


def merge(results):
    paths = {}
    for user in results:
        for name, k in user["kinds"].items():
            agg = paths.setdefault(name, {"users": 0, "rows": 0, "mismatching_rows": 0, "unexplained_rows": 0,
                                          "rows_with_identical_raw_inputs": 0, "unmatched_rows": 0,
                                          "mismatch_causes": {}, "columns": {}, "grouping_bijective": {},
                                          "per_user": {}})
            agg["users"] += 1
            for key in ("rows", "mismatching_rows", "unexplained_rows", "rows_with_identical_raw_inputs",
                        "unmatched_rows"):
                agg[key] += k[key]
            for cause, n in k["mismatch_causes"].items():
                agg["mismatch_causes"][cause] = agg["mismatch_causes"].get(cause, 0) + n
            for col, d in k["columns"].items():
                c = agg["columns"].setdefault(col, {"max_abs_diff": 0.0, "mismatch_rows": 0,
                                                    "max_abs_diff_same_raw_inputs": 0.0})
                c["max_abs_diff"] = max(c["max_abs_diff"], d["max_abs_diff"])
                c["mismatch_rows"] += d["mismatch_rows"]
                c["max_abs_diff_same_raw_inputs"] = max(c["max_abs_diff_same_raw_inputs"],
                                                        d["max_abs_diff_same_raw_inputs"])
            for ent, ok in k["grouping_bijective_vs_frame_ids"].items():
                agg["grouping_bijective"].setdefault(ent, []).append(bool(ok))
            agg["per_user"][str(user["user"])] = {"rows": k["rows"], "mismatching_rows": k["mismatching_rows"],
                                                  "unexplained_rows": k["unexplained_rows"]}
    for agg in paths.values():
        agg["grouping_bijective"] = {ent: f"{sum(v)}/{len(v)} users" for ent, v in agg["grouping_bijective"].items()}
    return paths


def impact():
    rows = {}
    for path in sorted((WORK / "impact").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if "benchmark" in r:
                rows[(r["tag"], r["inputs"], r["user_id"])] = r
    users = json.loads((WORK / "impact-users.json").read_text())
    out = {"users": len(users["users"]), "top_affected_users": len(users["top_affected"]),
           "random_users": len(users["random"]), "models": {}}
    for tag in sorted({k[0] for k in rows}):
        per = {}
        for u in users["users"]:
            a, b = rows.get((tag, "e2s-query", u)), rows.get((tag, "e2s-query-daylearn", u))
            if a and b:
                per[u] = (a["benchmark"]["instant"]["LogLoss"], b["benchmark"]["instant"]["LogLoss"],
                          a["benchmark"]["instant"]["size"], a["benchmark"]["ahead"]["LogLoss"],
                          b["benchmark"]["ahead"]["LogLoss"])
        res = {}
        for group, members in (("all", users["users"]), ("top_affected", users["top_affected"]),
                               ("random", users["random"])):
            us = [u for u in members if u in per]
            if not us:
                continue
            ref = np.array([per[u][0] for u in us]); drift = np.array([per[u][1] for u in us])
            w = np.array([per[u][2] for u in us], float)
            delta = 100 * (np.sum(w * drift) / np.sum(w * ref) - 1)
            rng = random.Random(161)
            boot = []
            for _ in range(2000):
                pick = [rng.randrange(len(us)) for _ in us]
                boot.append(100 * (np.sum(w[pick] * drift[pick]) / np.sum(w[pick] * ref[pick]) - 1))
            res[group] = {"users": len(us), "benchmark_reviews": int(w.sum()),
                          "instant_logloss_e2s_query": float(np.sum(w * ref) / w.sum()),
                          "instant_logloss_fork_daylearn": float(np.sum(w * drift) / w.sum()),
                          "instant_delta_pct": float(delta),
                          "instant_delta_pct_ci95_user_bootstrap": [float(np.percentile(boot, 2.5)),
                                                                     float(np.percentile(boot, 97.5))],
                          "users_worse": int((drift > ref).sum()),
                          "curve_max_abs_change": float(max(abs(per[u][4] - per[u][3]) for u in us))}
        out["models"][tag] = res
    return out


def main():
    agg = {"present": merge(json.loads((WORK / "cmp_present.json").read_text())),
           "export": merge(json.loads((WORK / "cmp_export.json").read_text()))}
    builds = [json.loads(p.read_text()) for p in sorted(WORK.glob("u*-present/build.json"))]
    agg["construction"] = {"users": len(builds), "reviews": sum(b["reviews"] for b in builds),
                           "elapsed_seconds_mismatch": sum(b["construction_elapsed_mismatch"] for b in builds),
                           "day_offset_mismatch": sum(b["construction_day_mismatch"] for b in builds),
                           "deleted_cards": sum(b["deleted_cards"] for b in builds),
                           "deleted_reviews": sum(b["deleted_reviews"] for b in builds),
                           "users_list": sorted(b["user"] for b in builds)}
    if (WORK / "impact").is_dir():
        agg["daylearn_impact"] = impact()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "aggregate.json").write_text(json.dumps(agg, indent=1))
    print(json.dumps({k: v for k, v in agg.items() if k in ("construction", "daylearn_impact")}, indent=1))
    for variant in ("present", "export"):
        for name, a in agg[variant].items():
            print(variant, name, {k: a[k] for k in ("users", "rows", "mismatching_rows", "unexplained_rows",
                                                    "rows_with_identical_raw_inputs")}, a["mismatch_causes"],
                  a["grouping_bijective"])
            for col, d in a["columns"].items():
                if d["mismatch_rows"] or d["max_abs_diff"] > 2e-6:
                    print("   ", col, f"{d['max_abs_diff']:.3g}", d["mismatch_rows"],
                          f"{d['max_abs_diff_same_raw_inputs']:.2g}")
            print("    max diff, identical raw inputs:",
                  f"{max(d['max_abs_diff_same_raw_inputs'] for d in a['columns'].values()):.2g}")


if __name__ == "__main__":
    sys.exit(main())
