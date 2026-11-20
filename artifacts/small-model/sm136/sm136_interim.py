"""SM136 interim readout (NOT the decision): the rule's computations on the main-pool users scored so far (complete users
per model and format), written to sm136-interim.json. Never writes sm136-decision.json."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sm136_rule as R  # noqa: E402

users = json.loads(R.USERS.read_text())["users"]
counts = {}
rows = R.by_label(R.load(R.ROWS, counts))
out = {"note": "interim, non-decisive: complete users only; the pre-registered rule needs >= 64 of 66 users per model and format",
       "pool_users": len(users), "models": {}}
for model, m in R.MODELS.items():
    heads = m["heads"]
    entry = {"size": m["size"], "name": m["name"], "heads": list(heads), "formats": {}}
    ref_users = [u for u in users if R.finite(rows.get(R.label(model, R.REF), {}).get(u), heads)]
    entry["users_with_w8s8"] = len(ref_users)
    for f in (R.COMPACT, R.DEFAULT, R.INT8C):
        got = rows.get(R.label(model, f), {})
        t, ok = R.table(rows, model, f, R.REF, users)
        entry["formats"][f] = {
            "rows": sum(1 for u in users if u in got), "complete_users": len(ok),
            "nonfinite_users": sorted(u for u in users if u in got and not R.finite(got[u], heads)),
            "vs_w8s8": {h: {g: {k: t[h][g].get(k) for k in ("users", "reviews", "pct", "ci")} for g in ("128-299", "128+")} for h in heads}}
    out["models"][model] = entry
(HERE / "sm136-interim.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
for model, e in out["models"].items():
    for f, d in e["formats"].items():
        for h, gs in d["vs_w8s8"].items():
            print(model, e["size"], f, h, f"rows={d['rows']} complete={d['complete_users']} nonfinite={d['nonfinite_users']}",
                  *[f"{g}: n_users={v['users']} n_rev={v['reviews']} pct={v['pct']} ci={v['ci']}" for g, v in gs.items()])
