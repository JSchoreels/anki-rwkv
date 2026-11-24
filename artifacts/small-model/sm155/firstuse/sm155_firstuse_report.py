#!/usr/bin/env python
"""First use of the bounded-memory stream scorer (SM155): Big long-card read-out of the low-rank int8 states test (SM148),
before (SM148 coverage) and after (plus the 18 first-use users). Same statistic and buckets as sm148_report.py (imported,
read-only): % = 100 x (sum LogLoss new / sum LogLoss REF - 1), paired user bootstrap 2,000 draws, seed 0, 95% CI, all-scope
buckets by answers of the card before the review. Aggregates only. Output: sm155-firstuse-result.json / .md."""
import json, re, sys
from pathlib import Path
sys.dont_write_bytecode = True
S148 = Path("C:/Users/jonat/workspace/wt/sm148/artifacts/small-model/sm148")
sys.path.insert(0, str(S148))
import sm148_report as R

HERE = Path(__file__).resolve().parent
NEW = HERE / "sm155-big-longcard-rows.jsonl"
longusers = set(json.loads((S148 / "users-longcard.json").read_text())["users"])
old, _, _ = R.read_rows(S148 / "cpu/sm148-longcard-rows.jsonl")
new, _, _ = R.read_rows(NEW)
new_users = {u for L in new for u in new[L]}
merged = {L: dict(old.get(L, {})) for L in set(old) | set(new)}
for L in new:
    for u, r in new[L].items():
        assert u not in old.get(L, {}), (L, u)
        merged[L][u] = r
BK = [(128, None)] + [b for b in R.BUCKETS[:-1]] + [(0, 1)]
BK = [(0, 1), (1, 3), (3, 5), (5, 9), (9, 16), (16, 32), (32, 128), (128, None)]
heads = {"sm91s2": ("Big two-head (SM91s2)", ("curve", "instant")), "sm99": ("Big Instant-Only (SM99)", ("instant",))}
res = {"first_use_users": sorted(new_users), "coverage": {}, "cost": {}}
for key, (name, hs) in heads.items():
    tag = R.MODELS[key][0]
    for lab in ("REF", "T24"):
        L = f"{tag}/{R.LABELS[lab]}{R.E2S}"
        res["coverage"][f"{name} {lab}"] = {"before": len(R.sub(old, L, longusers)), "after": len(R.sub(merged, L, longusers)),
                                            "of": len(longusers)}
    for when, rows in (("before", old), ("after", merged)):
        ref = R.sub(rows, f"{tag}/{R.LABELS['REF']}{R.E2S}", longusers)
        t24 = R.sub(rows, f"{tag}/{R.LABELS['T24']}{R.E2S}", longusers)
        for h in hs:
            d = {}
            for lo, hi in BK:
                nm = f"{lo}-{hi - 1 if hi else ''}"
                d[nm] = R.bucket_cmp(t24, ref, R.KEY[h], lo, hi)
            res["cost"].setdefault(f"{name} | {h}", {})[when] = d
            res["cost"][f"{name} | {h}"][when]["users"] = len(set(t24) & set(ref))
json.dump(res, open(HERE / "sm155-firstuse-result.json", "w"), indent=1, default=str)
print(json.dumps(res["coverage"], indent=1))
for k, v in res["cost"].items():
    for when in ("before", "after"):
        print(k, when, "users", v[when]["users"])
        for nm, c in v[when].items():
            if nm != "users":
                print("   ", nm, R.fmt(c), (c or {}).get("reviews"))
