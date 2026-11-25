#!/usr/bin/env python
"""Sharpness-aware cool-down (SM172): read-out of the screen (design.md "Rules"). CPU only, numpy; a few seconds.

  python sm172_report.py report [--rows gpu/sm172-rows-732.jsonl]   -> sm172-result.json, sm172-result.md
       exit 0 decided, 2 pending (coverage < 720 rule users for a label, or a non-finite row)
  python sm172_report.py selftest   (the read-out on SM148's rows of the original six-layer Tiny two-head (SM130), used
                                     as every arm: every difference must be exactly 0, verdict "NO GO")

The control is the state-energy penalty screen's control (SM171 `sm171-tiny-ctl`, the same cool-down without SAM):
its rows are read from ../sm171/gpu/sm171-rows-732.jsonl (same scorer tree cea0f931, same flags), not rescored.

Statistic (as SM148 / SM149): % = 100 x (sum of the label's LogLoss / sum of the reference's LogLoss - 1), review-
weighted (LogLoss x reviews per user and head), paired user bootstrap, 2,000 draws, numpy default_rng(0), 95% CI.
Q2 cost of an arm = its Q2 vs its own REF. Effect of a SAM arm = its Q2 cost minus the control's Q2 cost, the two
ratios recomputed on the same bootstrap draw (users paired across all labels). REF guard of a SAM arm = its REF vs the
original model's REF minus the control's REF vs the original model's REF, on the same draw.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = Path("C:/Users/jonat/workspace/wt/sm172-run")
CTL_ROWS = HERE.parent / "sm171" / "gpu" / "sm171-rows-732.jsonl"
SM148_ROWS = RUN / "artifacts/small-model/sm148/gpu/sm148-rows-732.jsonl"
U732 = Path("C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json")
ORIG_TAG = "sm130-th-dec2ep32k-t91s2"
E2S = "@e2s-query"
LABELS = {"REF": "w8+s8hx8", "T24": "w8+lr2.16.4q8x8", "Q2": "w8+lr2q8x8"}
HEADS = {"curve": "ahead", "instant": "instant"}
ARMS = {"r20": "sm172-tiny-r20", "r05": "sm172-tiny-r05", "ctl": "sm171-tiny-ctl"}
PENALTY = ("r20", "r05")   # the SAM arms (name kept from the SM171 read-out code)
MIN_USERS = 720
EFFECT = -0.05        # go: the Q2 cost shrinks by at least 0.05% (value <= -0.05) with CI upper bound < 0, either head
REF_GUARD = 0.05      # go: (arm REF vs original REF) - (control REF vs original REF), CI upper bound <= +0.05%, both heads
BUCKETS = ((1, 3), (3, 5), (5, 9), (9, 16), (16, 32), (32, 128), (128, None))
DRAWS = 2000


def read_rows(path):
    rows, nonfinite = {}, {}
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
                nonfinite.setdefault(r["label"], set()).add(r["user_id"])
    return rows, nonfinite


def label(tag, scheme):
    return f"{tag}/{LABELS[scheme]}{E2S}"


def sums(rows, users, head):
    """Per user (LogLoss x reviews) of one head."""
    h = HEADS[head]
    return np.array([rows[u]["benchmark"][h]["LogLoss"] * rows[u]["benchmark"][h]["size"] for u in users])


def sizes_equal(a, b, users, head):
    h = HEADS[head]
    return all(a[u]["benchmark"][h]["size"] == b[u]["benchmark"][h]["size"] for u in users)


def pct(a, b, idx):
    value = (a.sum() / b.sum() - 1) * 100
    boot = (a[idx].sum(1) / b[idx].sum(1) - 1) * 100
    return value, boot


def summary(value, boot):
    return {"value": float(value), "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


def cmp(new, ref, users, head, idx):
    return summary(*pct(sums(new, users, head), sums(ref, users, head), idx))


def did(pen_q, pen_r, ctl_q, ctl_r, users, head, idx):
    v1, b1 = pct(sums(pen_q, users, head), sums(pen_r, users, head), idx)
    v0, b0 = pct(sums(ctl_q, users, head), sums(ctl_r, users, head), idx)
    return summary(v1 - v0, b1 - b0)


def bucket_sums(row, head, lo, hi):
    pb = row.get("prior_buckets")
    if not pb:
        return 0, 0.0
    edges = pb["edges"]
    h = pb["all"].get(HEADS[head])
    if h is None:
        return 0, 0.0
    idx = [i for i, e in enumerate(edges) if e >= lo and (hi is None or (edges[i + 1] if i + 1 < len(edges) else math.inf) <= hi)]
    return sum(h["n"][i] for i in idx), sum(h["ll"][i] for i in idx)


def bucket_did(rows, arm, users, head, lo, hi):
    """Q2 cost (vs own REF) per answer-count bucket of an arm, and the effect vs the control (report only)."""
    def series(tag, scheme):
        return np.array([bucket_sums(rows[label(tag, scheme)][u], head, lo, hi)[1] for u in users])
    n = sum(bucket_sums(rows[label(ARMS["ctl"], "REF")][u], head, lo, hi)[0] for u in users)
    if n == 0:
        return None
    idx = np.random.default_rng(0).integers(0, len(users), size=(DRAWS, len(users)))
    keep = np.array([bucket_sums(rows[label(ARMS["ctl"], "REF")][u], head, lo, hi)[0] > 0 for u in users])
    out = {"reviews": int(n)}
    v, b = pct(series(ARMS[arm], "Q2"), series(ARMS[arm], "REF"), idx)
    out["q2_cost"] = summary(v, b)
    if arm != "ctl":
        v0, b0 = pct(series(ARMS["ctl"], "Q2"), series(ARMS["ctl"], "REF"), idx)
        out["effect_vs_ctl"] = summary(v - v0, b - b0)
    del keep
    return out


def fmt(x):
    return "n/a" if x is None else f"{x['value']:+.3f}% [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]"


def decide(rows, nonfinite, orig_rows, arms=ARMS, orig_tag=ORIG_TAG):
    users732 = set(json.load(open(U732))["users"])
    needed = [label(arms[a], s) for a in arms for s in LABELS]
    have = {lab: set(rows.get(lab, {})) & users732 for lab in needed}
    common = set(users732)
    for lab in needed:
        common &= have[lab]
    orig = {s: orig_rows.get(label(orig_tag, s), {}) for s in LABELS}
    for s in LABELS:
        if not orig[s]:
            raise SystemExit(f"SM148 rows lack {label(orig_tag, s)}")
    common = sorted(common & set(orig["REF"]) & set(orig["Q2"]) & set(orig["T24"]))
    nf = {lab: sorted(nonfinite.get(lab, set())) for lab in needed if nonfinite.get(lab)}
    res = {"users": len(common), "coverage": {lab: len(v) for lab, v in have.items()}, "nonfinite": nf,
           "rule_users_needed": MIN_USERS, "arms": {a: arms[a] for a in arms}}
    if len(common) < MIN_USERS or nf or any(len(v) < MIN_USERS for v in have.values()):
        res["status"] = "pending"
        return res
    idx = np.random.default_rng(0).integers(0, len(common), size=(DRAWS, len(common)))
    R = {a: {s: rows[label(arms[a], s)] for s in LABELS} for a in arms}
    for a in arms:
        for s in ("T24", "Q2"):
            for h in HEADS:
                if not sizes_equal(R[a][s], R[a]["REF"], common, h):
                    raise SystemExit(f"{a} {s} {h}: review counts differ from REF")
    res["cost_vs_own_ref"] = {a: {s: {h: cmp(R[a][s], R[a]["REF"], common, h, idx) for h in HEADS}
                                  for s in ("T24", "Q2")} for a in arms}
    res["orig_cost_vs_own_ref"] = {s: {h: cmp(orig[s], orig["REF"], common, h, idx) for h in HEADS}
                                   for s in ("T24", "Q2")}
    res["effect_q2_vs_ctl"] = {a: {h: did(R[a]["Q2"], R[a]["REF"], R["ctl"]["Q2"], R["ctl"]["REF"], common, h, idx)
                                   for h in HEADS} for a in PENALTY}
    res["effect_t24_vs_ctl"] = {a: {h: did(R[a]["T24"], R[a]["REF"], R["ctl"]["T24"], R["ctl"]["REF"], common, h, idx)
                                    for h in HEADS} for a in PENALTY}
    res["ref_vs_ctl_ref"] = {a: {h: cmp(R[a]["REF"], R["ctl"]["REF"], common, h, idx) for h in HEADS} for a in PENALTY}
    res["ref_guard"] = {a: {h: did(R[a]["REF"], orig["REF"], R["ctl"]["REF"], orig["REF"], common, h, idx)
                            for h in HEADS} for a in PENALTY}
    res["vs_orig_ref"] = {a: {s: {h: cmp(R[a][s], orig["REF"], common, h, idx) for h in HEADS} for s in LABELS}
                          for a in arms}
    res["q2_vs_ctl_q2"] = {a: {h: cmp(R[a]["Q2"], R["ctl"]["Q2"], common, h, idx) for h in HEADS} for a in PENALTY}
    res["q2_vs_ctl_t24"] = {a: {h: cmp(R[a]["Q2"], R["ctl"]["T24"], common, h, idx) for h in HEADS} for a in PENALTY}
    res["buckets_instant"] = {a: {f"{lo}-{'' if hi is None else hi - 1}": bucket_did(rows, a, common, "instant", lo, hi)
                                  for lo, hi in BUCKETS} for a in arms}
    verdict = {}
    for a in PENALTY:
        per_head = {}
        for h in HEADS:
            e = res["effect_q2_vs_ctl"][a][h]
            g = res["ref_guard"][a][h]
            per_head[h] = {"effect_ok": e["value"] <= EFFECT and e["ci"][1] < 0, "ref_guard_ok": g["ci"][1] <= REF_GUARD}
        effect_heads = [h for h, v in per_head.items() if v["effect_ok"]]
        guard = all(v["ref_guard_ok"] for v in per_head.values())
        verdict[a] = {"heads": per_head, "effect_heads": effect_heads, "ref_guard_both_heads": guard,
                      "go": bool(effect_heads) and guard}
    res["verdict_per_arm"] = verdict
    goes = [a for a in PENALTY if verdict[a]["go"]]
    res["verdict"] = "GO" if goes else "NO GO"
    res["go_arms"] = goes
    res["status"] = "decided"
    return res


def write_md(res, path):
    L = ["# Sharpness-aware cool-down (SM172): read-out", "",
         f"Status: {res['status']}. Rule users: {res['users']} (732 set, users common to all nine labels and the "
         "original model's rows). Statistic: review-weighted LogLoss ratio, paired user bootstrap, 2,000 draws, "
         "seed 0, 95% CI. Model: six-layer Tiny two-head (SM130).", ""]
    if res["status"] != "decided":
        L += ["Coverage per label: " + json.dumps(res["coverage"]), "", "Non-finite rows: " + json.dumps(res["nonfinite"])]
        Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")
        return
    L += [f"**Verdict: {res['verdict']}**" + (f" (arms: {', '.join(res['go_arms'])})" if res["go_arms"] else ""), "",
          "## Go rule: Q2 cost (vs own REF) of the SAM arm minus the control's (either head); REF guard: (arm REF vs "
          "original REF) minus (control REF vs original REF) (both heads)", "",
          "| arm | head | Q2 cost change vs control | REF guard | REF vs control REF | effect ok | guard ok |",
          "|---|---|---|---|---|---|---|"]
    for a in PENALTY:
        for h in HEADS:
            v = res["verdict_per_arm"][a]["heads"][h]
            L.append(f"| {a} | {h} | {fmt(res['effect_q2_vs_ctl'][a][h])} | {fmt(res['ref_guard'][a][h])} | "
                     f"{fmt(res['ref_vs_ctl_ref'][a][h])} | "
                     f"{'yes' if v['effect_ok'] else 'no'} | {'yes' if v['ref_guard_ok'] else 'no'} |")
    L += ["", "## Cost vs the arm's own REF (report)", "", "| model | label | curve | instant |", "|---|---|---|---|"]
    for a in res["cost_vs_own_ref"]:
        for s in ("T24", "Q2"):
            c = res["cost_vs_own_ref"][a][s]
            L.append(f"| {a} | {s} | {fmt(c['curve'])} | {fmt(c['instant'])} |")
    for s in ("T24", "Q2"):
        c = res["orig_cost_vs_own_ref"][s]
        L.append(f"| original (SM148 rows) | {s} | {fmt(c['curve'])} | {fmt(c['instant'])} |")
    L += ["", "## Against the original model at REF (report)", "", "| arm | label | curve | instant |", "|---|---|---|---|"]
    for a in res["vs_orig_ref"]:
        for s in LABELS:
            c = res["vs_orig_ref"][a][s]
            L.append(f"| {a} | {s} | {fmt(c['curve'])} | {fmt(c['instant'])} |")
    L += ["", "## Other report-only lines", ""]
    for a in PENALTY:
        L.append(f"- {a}: T24 cost change vs control: curve {fmt(res['effect_t24_vs_ctl'][a]['curve'])}, instant "
                 f"{fmt(res['effect_t24_vs_ctl'][a]['instant'])}; Q2 vs control Q2: curve "
                 f"{fmt(res['q2_vs_ctl_q2'][a]['curve'])}, instant {fmt(res['q2_vs_ctl_q2'][a]['instant'])}; Q2 vs "
                 f"control T24: curve {fmt(res['q2_vs_ctl_t24'][a]['curve'])}, instant "
                 f"{fmt(res['q2_vs_ctl_t24'][a]['instant'])}.")
    L += ["", "## Instant Q2 cost by answer count (report)", "", "| bucket | " + " | ".join(
        f"{a} Q2 cost" + ("" if a == "ctl" else f" | {a} change vs control") for a in res["buckets_instant"]) + " |"]
    L.append("|---|" + "---|" * sum(1 if a == "ctl" else 2 for a in res["buckets_instant"]))
    for b in next(iter(res["buckets_instant"].values())):
        cells = []
        for a in res["buckets_instant"]:
            x = res["buckets_instant"][a][b]
            cells.append(fmt(x and x["q2_cost"]))
            if a != "ctl":
                cells.append(fmt(x and x["effect_vs_ctl"]))
        L.append(f"| {b} | " + " | ".join(cells) + " |")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def report(args):
    rows, nonfinite = read_rows(args.rows)
    ctl_rows, ctl_nonfinite = read_rows(CTL_ROWS)
    for s in LABELS:
        lab = label(ARMS["ctl"], s)
        rows[lab] = ctl_rows[lab]
        if lab in ctl_nonfinite:
            nonfinite[lab] = ctl_nonfinite[lab]
    orig_rows, _ = read_rows(SM148_ROWS)
    res = decide(rows, nonfinite, orig_rows)
    (HERE / "sm172-result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    write_md(res, HERE / "sm172-result.md")
    print(f"[sm172-report] {res['status']}: {res.get('verdict', '')} users {res['users']}")
    return 0 if res["status"] == "decided" else 2


def selftest(args):
    orig_rows, _ = read_rows(SM148_ROWS)
    rows = {}
    for a, tag in ARMS.items():
        for s in LABELS:
            rows[label(tag, s)] = orig_rows[label(ORIG_TAG, s)]
    res = decide(rows, {}, orig_rows)
    zero = all(abs(res["effect_q2_vs_ctl"][a][h]["value"]) < 1e-12 and abs(res["ref_guard"][a][h]["value"]) < 1e-12
               for a in PENALTY for h in HEADS)
    ok = res["status"] == "decided" and zero and res["verdict"] == "NO GO" and res["users"] >= MIN_USERS
    print(json.dumps({"selftest": "OK" if ok else "FAIL", "users": res["users"],
                      "orig_q2_cost": {h: fmt(res["orig_cost_vs_own_ref"]["Q2"][h]) for h in HEADS},
                      "orig_t24_cost": {h: fmt(res["orig_cost_vs_own_ref"]["T24"][h]) for h in HEADS}}))
    write_md(res, HERE / "gpu" / "selftest-result.md")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report")
    r.add_argument("--rows", default=str(HERE / "gpu" / "sm172-rows-732.jsonl"))
    sub.add_parser("selftest")
    args = p.parse_args()
    return report(args) if args.cmd == "report" else selftest(args)


if __name__ == "__main__":
    sys.exit(main())
