#!/usr/bin/env python
"""Deleted-card note ids (SM173): phase-1 report, the pre-registered rules of design.md.

  python -X utf8 sm173_report.py report [--rows gpu/sm173-rows-732.jsonl]   -> sm173-result.json, sm173-result.md
Exit 0 = verdict written, 2 = pending (coverage below 720 rule users or a non-finite row).
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
MODELS = {"sm130-th": "six-layer Tiny two-head (SM130)", "sm118": "Small two-head (SM118)"}
ARMS = ("cur", "lanes", "fix", "reseed")
HEADS = {"ahead": "curve", "instant": "instant"}
SCOPES = {"benchmark": "overall", "benchmark_deleted": "affected", "benchmark_existing": "existing"}
DRAWS, MIN_USERS, MIN_EFFECT, AFFECTED_EFFECT, WORSE_GUARD = 2000, 720, -0.05, -1.0, 0.05


def load(path):
    rows = {}
    for raw in open(path, "rb"):
        try:
            r = json.loads(raw.decode("utf-8", errors="replace"))
        except ValueError:
            continue
        if "label" in r and "error" not in r:
            rows[(r["label"], r["user_id"])] = r   # the last row per label and user wins
    return rows


def label(model, arm):
    return f"{model}-{arm}/fp32@e2s-query"


def sums(row, scope, head):
    """(LogLoss sum, size) of one scope and head of a row; (0, 0) if the scope is empty."""
    part = row.get(scope) or {}
    h = part.get(head) or {}
    if not h or not h.get("size") or h.get("LogLoss") is None:
        return 0.0, 0
    return float(h["LogLoss"]) * int(h["size"]), int(h["size"])


def compare(rows, users, model, arm, ref, scope, head, rng_seed=0):
    a = np.array([sums(rows[(label(model, arm), u)], scope, head)[0] for u in users])
    b = np.array([sums(rows[(label(model, ref), u)], scope, head)[0] for u in users])
    n = np.array([sums(rows[(label(model, ref), u)], scope, head)[1] for u in users])
    keep = n > 0
    a, b, n = a[keep], b[keep], n[keep]
    if not len(a) or b.sum() == 0:
        return {"users": int(len(a)), "reviews": int(n.sum()), "pct": None}
    pct = 100 * (a.sum() / b.sum() - 1)
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, len(a), size=(DRAWS, len(a)))
    boot = 100 * (a[idx].sum(1) / b[idx].sum(1) - 1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"users": int(len(a)), "reviews": int(n.sum()), "pct": round(float(pct), 4), "ci": [round(float(lo), 4),
            round(float(hi), 4)], "logloss_ref": round(float(b.sum() / n.sum()), 6), "logloss_arm": round(float(a.sum() / n.sum()), 6),
            "users_better": int((a < b - 1e-12).sum()), "users_worse": int((a > b + 1e-12).sum())}


def fmt(c):
    if c.get("pct") is None:
        return "n/a"
    return f"{c['pct']:+.3f}% [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]"


def report(rows_path):
    rows = load(rows_path)
    labels = [label(m, a) for m in MODELS for a in ARMS]
    have = [{u for (l, u) in rows if l == lab and "benchmark" in rows[(l, u)]} for lab in labels]
    nonfinite = sorted({u for (l, u), r in rows.items() if l in labels and "nonfinite" in r})
    users = sorted(set.intersection(*have)) if have else []
    out = {"rows": str(rows_path), "rule_users": len(users), "nonfinite_users": nonfinite,
           "labels_users": dict(zip(labels, map(len, have)))}
    if len(users) < MIN_USERS or nonfinite:
        out["verdict"] = "pending"
        return out, 2
    # users with deleted-card reviews (affected scope non-empty), and the bitwise check of the others
    affected_users = [u for u in users if sums(rows[(label("sm130-th", "cur"), u)], "benchmark_deleted", "ahead")[1] > 0]
    out["users_with_deleted_reviews"] = len(affected_users)
    mismatch = {}
    for m in MODELS:
        for arm in ("fix", "lanes"):
            bad = 0
            for u in users:
                if u in set(affected_users):
                    continue
                for head in HEADS:
                    if sums(rows[(label(m, arm), u)], "benchmark", head) != sums(rows[(label(m, "cur"), u)], "benchmark", head):
                        bad += 1
                        break
            mismatch[f"{m}/{arm}"] = bad
    out["unaffected_users_not_equal_to_cur"] = mismatch
    res = {}
    for m in MODELS:
        res[m] = {}
        for arm, ref in (("fix", "cur"), ("lanes", "cur"), ("reseed", "cur"), ("fix", "reseed")):
            key = f"{arm}_vs_{ref}"
            res[m][key] = {}
            for scope in SCOPES:
                for head in HEADS:
                    pool = users if scope != "benchmark_deleted" else affected_users
                    res[m][key][f"{SCOPES[scope]}/{HEADS[head]}"] = compare(rows, pool, m, arm, ref, scope, head)
            for head in HEADS:   # overall, users with deleted cards only
                res[m][key][f"overall-deleted-users/{HEADS[head]}"] = compare(rows, affected_users, m, arm, ref, "benchmark", head)
    out["results"] = res
    # pre-registered verdict on fix vs cur
    a_hits, b_hits, worse = [], [], []
    for m in MODELS:
        for head in HEADS.values():
            o = res[m]["fix_vs_cur"][f"overall/{head}"]
            f = res[m]["fix_vs_cur"][f"affected/{head}"]
            if o["pct"] is not None and o["pct"] <= MIN_EFFECT and o["ci"][1] < 0:
                a_hits.append(f"{m}/{head}")
            if f["pct"] is not None and f["pct"] <= AFFECTED_EFFECT and f["ci"][1] < 0:
                b_hits.append(f"{m}/{head}")
            if o["pct"] is not None and o["ci"][0] > WORSE_GUARD:
                worse.append(f"{m}/{head}")
    hit = a_hits or b_hits
    out["rule_hits"] = {"A_overall": a_hits, "B_affected": b_hits, "worse_overall": worse}
    out["verdict"] = "MIXED" if hit and worse else "MATTERS" if hit else "HARMLESS"
    return out, 0


def markdown(out):
    lines = [f"# Deleted-card note ids (SM173): phase-1 read-out", "",
             f"VERDICT: {out['verdict']} (rule users {out['rule_users']}, with deleted-card reviews "
             f"{out.get('users_with_deleted_reviews')}; rule hits {json.dumps(out.get('rule_hits'))})", ""]
    if "results" not in out:
        return "\n".join(lines)
    lines.append(f"Unaffected users not bitwise equal to cur: {json.dumps(out['unaffected_users_not_equal_to_cur'])}")
    for m, name in MODELS.items():
        lines += ["", f"## {name}", "", "| comparison | scope | curve | instant | reviews |", "|---|---|---:|---:|---:|"]
        for key, r in out["results"][m].items():
            for scope in ("overall", "affected", "existing", "overall-deleted-users"):
                c, i = r[f"{scope}/curve"], r[f"{scope}/instant"]
                lines.append(f"| {key.replace('_vs_', ' vs ')} | {scope} | {fmt(c)} | {fmt(i)} | {c.get('reviews', 0):,} |")
        r = out["results"][m]["fix_vs_cur"]
        lines.append("")
        lines.append(f"LogLoss cur / fix, affected reviews: curve {r['affected/curve'].get('logloss_ref')} / "
                     f"{r['affected/curve'].get('logloss_arm')}, instant {r['affected/instant'].get('logloss_ref')} / "
                     f"{r['affected/instant'].get('logloss_arm')}; overall: curve {r['overall/curve'].get('logloss_ref')} / "
                     f"{r['overall/curve'].get('logloss_arm')}, instant {r['overall/instant'].get('logloss_ref')} / "
                     f"{r['overall/instant'].get('logloss_arm')}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("report",))
    parser.add_argument("--rows", default=str(HERE / "gpu" / "sm173-rows-732.jsonl"))
    parser.add_argument("--out", default=str(HERE / "sm173-result"))
    args = parser.parse_args()
    out, rc = report(Path(args.rows))
    Path(args.out + ".json").write_text(json.dumps(out, indent=1))
    md = markdown(out)
    Path(args.out + ".md").write_text(md, encoding="utf-8")
    print(md)
    return rc


if __name__ == "__main__":
    sys.exit(main())
