#!/usr/bin/env python
"""Output-aware T24 on the 732 set (SM158): CPU proofs before the launch (design.md "Proofs before the launch"), run by
cpu-proof.sh on the rows of cpu/smoke-cpu-rows.jsonl (the 3 smoke users x the 6 models x REF, Q2, T24, Q2RK, T24RK).

  python sm158_check.py cpu [--out check-cpu-sm158.json]

P1 REF, Q2 and T24 == the low-rank int8 states (SM148)'s committed CPU smoke rows, every metric and descriptor field
   identical (bit for bit: equal JSON values, not a tolerance).
P2 Q2RK and T24RK == output-aware low-rank truncation (SM157)'s committed CPU probe rows of the same users, every metric
   and descriptor field identical (the SM157 statistics sha256 included).
Fields that legitimately differ (label, variant spelling, timings, provenance, scorer lanes) are skipped, as in SM157's K2.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABELS = {"REF": "w8+s8hx8", "Q2": "w8+lr2q8x8", "T24": "w8+lr2.16.4q8x8",
          "Q2RK": "w8+lr2q8x8wrk", "T24RK": "w8+lr2.16.4q8x8wrk"}
OLD = {"REF": "sm148", "Q2": "sm148", "T24": "sm148", "Q2RK": "sm157", "T24RK": "sm157"}
SOURCES = {"sm148": HERE.parent / "sm148" / "cpu" / "smoke-cpu-rows.jsonl",
           "sm157": HERE.parent / "sm157" / "cpu" / "sm157-rows.jsonl"}
SKIP_FIELDS = {"label", "variant", "model_seconds", "prepare_seconds", "provenance_commit", "provenance_dirty", "scorer"}
STATS_SHA = "d0394c73e44d442a276e6e86031cde939abb942319f3068598ac2d8cae7c54fe"


def read(path):
    out = {}
    for raw in open(path, "rb"):
        try:
            r = json.loads(raw.decode("utf-8", errors="replace"))
        except ValueError:
            continue
        if "benchmark" in r or "nonfinite" in r:
            out[(r["user_id"], r["tag"] if "tag" in r else r["label"].split("/")[0], r["variant"])] = r
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("cpu",))
    p.add_argument("--out", default=str(HERE / "check-cpu-sm158.json"))
    a = p.parse_args()
    rows = read(HERE / "cpu" / "smoke-cpu-rows.jsonl")
    old = {k: read(v) for k, v in SOURCES.items()}
    users = json.loads((HERE / "users-smoke.json").read_text())["users"]
    tags = sorted({t for _, t, _ in rows})
    res, info = {}, {}
    for part, names in (("P1", ("REF", "Q2", "T24")), ("P2", ("Q2RK", "T24RK"))):
        bad, compared, fields = [], 0, set()
        for u in users:
            for t in tags:
                for n in names:
                    x, y = rows.get((u, t, LABELS[n])), old[OLD[n]].get((u, t, LABELS[n]))
                    if x is None or y is None:
                        bad.append([u, t, n, "missing", x is None, y is None])
                        continue
                    keys = (set(x) | set(y)) - SKIP_FIELDS
                    diff = [k for k in sorted(keys) if x.get(k) != y.get(k)]
                    fields |= keys
                    compared += 1
                    if diff:
                        bad.append([u, t, n, diff[:6]])
                    if part == "P2" and (x.get("state_lowrank") or {}).get("weight_stats_sha256") != STATS_SHA:
                        bad.append([u, t, n, "statistics sha256"])
        want = len(users) * len(names) * 6
        name = ("P1 REF, Q2 and T24 == SM148's committed CPU smoke rows (every metric and descriptor field identical)"
                if part == "P1" else
                "P2 Q2RK and T24RK == SM157's committed CPU probe rows of the smoke users (every field identical, statistics sha256)")
        res[name] = not bad and compared == want and len(tags) == 6
        info[part] = {"compared_rows": compared, "expected_rows": want, "users": users, "models": len(tags),
                      "fields_compared": len(fields), "mismatches": bad[:10]}
    doc = {"all_ok": all(res.values()), "checks": res, "info": info}
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    for k, v in res.items():
        print(("OK  " if v else "FAIL") + " " + k)
    print("ALL OK" if doc["all_ok"] else "SOME CHECKS FAILED")
    sys.exit(0 if doc["all_ok"] else 1)


if __name__ == "__main__":
    main()
