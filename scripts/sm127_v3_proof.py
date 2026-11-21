#!/usr/bin/env python
"""SM127 v3 CPU proof comparison (design.md 13.6). Reads aggregates only.
  old   <pilot row> <new-run-without-flags row>  : every pilot key equal (wall time / stamps excluded)
  new   <pilot row> <fix row> : tf lane accumulators equal except the fluent read-out; budget, nofb rows, S_u, tc,
                                fr-shared1#0 / tf seconds per later-day review
Usage: sm127_v3_proof.py U  (reads rows/U.json, proof-v3/old-U.json, proof-v3/new-U.json) -> proof-v3/proof-U.json
"""
import json
import sys
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parents[1] / "artifacts/small-model/sm127"
SKIP = {"wall_seconds", "lane_reviews_per_second", "finished", "warmup_seconds"}


def secs(r, nm):
    tl = np.asarray(r["acc"]["time_later"])[r["lanes"].index(nm)]
    return float(tl[:, 1].sum() / tl[:, 0].sum())


def main(u):
    pilot = json.loads((H / "rows" / f"{u}.json").read_text())
    out = {"user": u}
    p = H / "proof-v3" / f"old-{u}.json"
    if p.exists():
        old = json.loads(p.read_text())
        diff = sorted(k for k in pilot if k not in SKIP and json.dumps(pilot[k], sort_keys=True) != json.dumps(old.get(k), sort_keys=True))
        out["a_fix_off"] = {"pilot_keys": len(pilot), "differing_keys": diff, "identical": not diff,
                            "extra_keys_in_new_code": sorted(set(old) - set(pilot))}
    p = H / "proof-v3" / f"new-{u}.json"
    if p.exists():
        new = json.loads(p.read_text())
        acc_p, acc_n = pilot["acc"], new["acc"]
        k0 = new["lanes"].index("tf")
        # batch composition changes the float summation order (proof K7: batched == alone within 1e-4 relative), so the
        # tf lane is compared exactly on counts and within 1e-4 relative on probability sums
        def same(a, b):
            a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
            return bool(np.max(np.abs(a - b) / np.maximum(1.0, np.abs(a))) < 1e-4)
        tf_eq = {k: same(np.asarray(acc_p[k])[k0], np.asarray(acc_n[k])[k0]) for k in acc_p}
        pol = [k for k, nm in enumerate(new["lanes"]) if nm != "tf" and not nm.startswith("fr-")]
        out["b_fix_on"] = {
            "s_user": new["s_fast"], "time_scale_final": new["time_scale_final"],
            "time_scale_first_days": new["time_scale_by_day"][:3], "time_scale_day_30": new["time_scale_by_day"][30],
            "tf_acc_equal": tf_eq,
            "tf_acc_equal_except_fluent": all(v for k, v in tf_eq.items() if k != "fluent_later"),
            "budget_violations_policy_lanes": int(sum(new["budget_violations"][k] for k in pol)),
            "nofb_rows": new["nofb_rows"],
            "fr_over_tf_seconds_pilot": secs(pilot, "fr-shared1#0") / secs(pilot, "tf"),
            "fr_over_tf_seconds_fix": secs(new, "fr-shared1#0") / secs(new, "tf"),
            "lane_reviews_pilot": pilot["lane_reviews"], "lane_reviews_fix": new["lane_reviews"],
            "flu_floor_share_fix": {nm: new["flu_floor_decisions"][k] / max(new["flu_decisions"][k], 1)
                                    for k, nm in enumerate(new["lanes"]) if nm.startswith("flu-0.80#")},
            "bars": {}}
        b = out["b_fix_on"]
        b["bars"] = {"tf_equal": b["tf_acc_equal_except_fluent"], "budget": b["budget_violations_policy_lanes"] == 0,
                     "nofb": b["nofb_rows"] > 0, "S_range": 3.0 <= b["s_user"] <= 45.0,
                     "tc_range": 0.7 <= b["time_scale_final"] <= 1.3,
                     "level_within_5pct": abs(b["fr_over_tf_seconds_fix"] - 1.0) <= 0.05}
    (H / "proof-v3" / f"proof-{u}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]))
