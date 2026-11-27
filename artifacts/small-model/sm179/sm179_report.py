#!/usr/bin/env python
"""Session inputs (SM179): read-out of the screen (design.md "Rules"). CPU only, numpy; a few seconds.

  python sm179_report.py report   -> sm179-result.json, sm179-result.md; exit 0 decided, 2 pending

Statistic (as SM148 / SM171): % = 100 x (sum over users of LogLoss x reviews of the arm / the same sum of the control
- 1), per head (curve = "ahead", instant), REF labels (w8+s8hx8) on show-time inputs, paired user bootstrap, 2,000
draws, numpy default_rng(0), 95% CI, on the rule users (users with a finite row for all three models, same benchmark
size). The control is SM171's control cool-down rescored by this screen's scorer with SM179_INPUTS unset; integrity:
those rows equal SM171's own rows of that model (max |LogLoss difference| <= 1e-6 on every common user).

Go for a follow-up (per arm): on one head the change is <= -0.05% with the CI upper bound below 0, and on the other
head the point estimate is <= +0.05%. Verdict GO if an arm passes; otherwise NO GO. Pending if fewer than 720 rule
users, a non-finite row, or the integrity check fails.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROWS = HERE / "gpu" / "sm179-rows-732.jsonl"
SM171_ROWS = Path("C:/Users/jonat/workspace/wt/sm171/artifacts/small-model/sm171/gpu/sm171-rows-732.jsonl")
RUN = Path("C:/Users/jonat/workspace/wt/sm179-run/artifacts/small-model")
CTL_RESULT = Path("C:/Users/jonat/workspace/wt/sm171-run/artifacts/small-model/sm171-tiny-ctl/expertium/E89-F65e-F77a/result.json")
TAGS = {"ctl": "sm171-tiny-ctl", "ses": "sm179-tiny-ses", "st": "sm179-tiny-st"}
NAMES = {"ctl": "control (SM171's control cool-down, no new input)", "ses": "session inputs (6 columns)",
         "st": "card-state inputs (4 one-hot columns)"}
LABEL = "w8+s8hx8@e2s-query"
HEADS = {"curve": "ahead", "instant": "instant"}
MIN_USERS = 720
GAIN = -0.05
GUARD = 0.05
DRAWS = 2000


def read_rows(path):
    rows, bad = {}, {}
    if Path(path).exists():
        for raw in open(path, "rb"):
            try:
                r = json.loads(raw.decode("utf-8", errors="replace"))
            except ValueError:
                continue
            if "error" in r:
                continue
            if "benchmark" in r:
                rows.setdefault(r["label"], {})[r["user_id"]] = r
            elif "nonfinite" in r:
                bad.setdefault(r["label"], set()).add(r["user_id"])
    return rows, bad


def finite(r):
    return all(math.isfinite(r["benchmark"][h]["LogLoss"]) for h in HEADS.values())


def sums(rows, users, head):
    h = HEADS[head]
    return np.array([rows[u]["benchmark"][h]["LogLoss"] * rows[u]["benchmark"][h]["size"] for u in users])


def change(a, b, idx):
    value = (a.sum() / b.sum() - 1) * 100
    boot = (a[idx].sum(1) / b[idx].sum(1) - 1) * 100
    return {"value": float(value), "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


def buckets(rows, users, head):
    """Per answer-count bucket (prior answers of the card), LogLoss sum over every predicted review (report only)."""
    out = {}
    for u in users:
        pb = rows[u].get("prior_buckets")
        if not pb or HEADS[head] not in pb["all"]:
            continue
        h = pb["all"][HEADS[head]]
        for i, e in enumerate(pb["edges"]):
            o = out.setdefault(e, [0, 0.0])
            o[0] += h["n"][i]
            o[1] += h["ll"][i]
    return out


def training(arm):
    p = CTL_RESULT if arm == "ctl" else RUN / TAGS[arm] / "expertium/E89-F65e-F77a/result.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    return {k: r.get(k) for k in ("review_presentations", "training_seconds", "reviews_per_second",
                                  "peak_allocated_mib", "peak_reserved_mib")}


def report(args):
    rows, bad = read_rows(args.rows)
    ref, _ = read_rows(SM171_ROWS)
    labels = {a: f"{t}/{LABEL}" for a, t in TAGS.items()}
    have = {a: rows.get(labels[a], {}) for a in TAGS}
    common = set.intersection(*(set(have[a]) for a in TAGS))
    nonfinite = sorted({u for a in TAGS for u in bad.get(labels[a], set())} |
                       {u for a in TAGS for u in common if not finite(have[a][u])})
    users = sorted(u for u in common if u not in nonfinite and all(
        have[a][u]["benchmark"][h]["size"] == have["ctl"][u]["benchmark"][h]["size"] for a in TAGS for h in HEADS.values()))
    sm171 = ref.get(labels["ctl"], {})
    both = [u for u in have["ctl"] if u in sm171]
    integrity = max((abs(have["ctl"][u]["benchmark"][h]["LogLoss"] - sm171[u]["benchmark"][h]["LogLoss"])
                     for u in both for h in HEADS.values()), default=None)
    integrity_ok = integrity is not None and integrity <= 1e-6 and len(both) >= MIN_USERS
    out = {"rule_users": len(users), "users_per_label": {a: len(have[a]) for a in TAGS}, "nonfinite_users": nonfinite,
           "integrity": {"common_users_with_sm171_rows": len(both), "max_abs_logloss_diff": integrity, "ok": integrity_ok},
           "label": LABEL, "arms": {}, "training": {a: training(a) for a in TAGS}}
    pending = len(users) < MIN_USERS or bool(nonfinite) or not integrity_ok
    idx = np.random.default_rng(0).integers(0, len(users), size=(DRAWS, len(users))) if users else None
    verdict = "NO GO"
    if users:
        for a in ("ses", "st"):
            res = {h: change(sums(have[a], users, h), sums(have["ctl"], users, h), idx) for h in HEADS}
            passes = []
            for h, other in (("curve", "instant"), ("instant", "curve")):
                if res[h]["value"] <= GAIN and res[h]["ci"][1] < 0 and res[other]["value"] <= GUARD:
                    passes.append(h)
            res["passes_on"] = passes
            res["go"] = bool(passes)
            bc = {}
            for h in HEADS:
                x, c = buckets(have[a], users, h), buckets(have["ctl"], users, h)
                bc[h] = {str(e): {"reviews": c[e][0], "change_pct": (x[e][1] / c[e][1] - 1) * 100 if c[e][1] else None}
                         for e in sorted(c) if e in x}
            res["buckets_all_predicted"] = bc
            out["arms"][a] = res
            if passes:
                verdict = "GO"
        out["control_logloss"] = {h: float(sums(have["ctl"], users, h).sum() /
                                           sum(have["ctl"][u]["benchmark"][HEADS[h]]["size"] for u in users)) for h in HEADS}
    out["status"] = "pending" if pending else "decided"
    out["verdict"] = verdict if not pending else f"pending ({verdict} on the users so far)"
    (HERE / "sm179-result.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = [f"# Session inputs (SM179): result tables", "",
          f"**Verdict: {out['verdict']}** ({out['status']}; {len(users)} rule users of the 732 set; label REF {LABEL}).", "",
          f"Integrity (control rescored here vs SM171's rows): {len(both)} users, max |LogLoss difference| {integrity}, "
          f"{'OK' if integrity_ok else 'FAIL'}.", "",
          "| arm | curve vs control | instant vs control | go |", "|---|---|---|---|"]
    for a, res in out["arms"].items():
        f = lambda r: f"{r['value']:+.3f}% [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]"  # noqa: E731
        md.append(f"| {NAMES[a]} | {f(res['curve'])} | {f(res['instant'])} | "
                  f"{'yes (' + ', '.join(res['passes_on']) + ')' if res['go'] else 'no'} |")
    md += ["", "| model | reviews/s | minutes | peak allocated / reserved MiB |", "|---|---|---|---|"]
    for a, t in out["training"].items():
        if t:
            md.append(f"| {NAMES[a]} | {t['reviews_per_second'] or 0:,.0f} | {(t['training_seconds'] or 0) / 60:.1f} | "
                      f"{t['peak_allocated_mib'] or 0:,.0f} / {t['peak_reserved_mib'] or 0:,.0f} |")
    (HERE / "sm179-result.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(md[2])
    return 2 if pending else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report")
    r.add_argument("--rows", default=str(ROWS))
    args = p.parse_args()
    return report(args)


if __name__ == "__main__":
    sys.exit(main())
