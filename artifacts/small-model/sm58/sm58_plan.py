"""SM58 planner: which benchmark users still need a per-review dump, and the honest ticket minutes.

  python sm58_plan.py users                         -> writes users.json (732 pool minus the long users) and prints counts
  python sm58_plan.py todo TAG RATE TODOFILE       -> users without a dump (and not marked failed), prints "<users> <reviews> <minutes>"
TAG is sm36, sm55 or fork; RATE = minutes per 31.87M reviews (w64 11, fork 29), +2 min fixed per ticket.
A user whose last row for the label is an error (e.g. out of memory at the 0.8 cap) is listed in failed-<TAG>.json and not retried.
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SMW = Path("C:/Users/jonat/workspace/wt/small-model/artifacts/small-model")
LONG = (1534, 1601, 4625)
DUMPS = {"fork": [SMW / "sm24-dump", HERE / "dump"], "sm36": [HERE / "dump"], "sm55": [HERE / "dump"]}


def pool():
    doc = json.loads((SMW / "eval-users-732.json").read_text())
    return doc["users"], json.loads(Path(doc["split"]).read_text())["review_counts"]


def has_dump(tag, user):
    return any((d / f"{user}_{tag}_w8+s8.npz").exists() for d in DUMPS[tag])


def main():
    users, counts = pool()
    if sys.argv[1] == "users":
        keep = [u for u in users if u not in LONG]
        (HERE / "users.json").write_text(json.dumps({"users": keep, "excluded_long_users": list(LONG),
                                                      "pool": "eval-users-732.json"}))
        print(len(keep), sum(counts[str(u)] for u in keep))
        return
    tag, rate, todof = sys.argv[2], float(sys.argv[3]), Path(sys.argv[4])
    keep = json.loads((HERE / "users.json").read_text())["users"]
    failed = set()
    rows = HERE / "heldout-sm58.jsonl"
    label = f"{tag}/w8+s8"
    if rows.exists():
        ok = set()
        for line in rows.open(encoding="utf-8"):
            r = json.loads(line)
            if r.get("label") == label and "error" not in r:
                ok.add(r["user_id"])
            elif "error" in r and r.get("user_id") is not None and (r.get("label") in (label, "error")):
                failed.add(r["user_id"])
        failed -= ok
    (HERE / f"failed-{tag}.json").write_text(json.dumps(sorted(failed)))
    todo = [u for u in keep if not has_dump(tag, u) and u not in failed]
    reviews = sum(counts[str(u)] for u in todo)
    todof.write_text(json.dumps({"label": label, "users": todo}))
    print(len(todo), reviews, math.ceil(reviews / 31.87e6 * rate) + 2 if todo else 0)


main()
