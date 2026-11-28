#!/usr/bin/env python
"""Tiny cool-down with session inputs (SM183): read-out (design.md "Rules"). CPU only, numpy; about a minute, < 2 GB.

  python sm183_report.py report [--no-auc]   -> sm183-result.json, sm183-result.md; exit 0 decided, 2 pending

Statistic (as SM179 / SM181): % = 100 x (sum over users of LogLoss x reviews of the arm / the same sum of the control
- 1), per head (curve = "ahead", instant), REF label w8+s8hx8 on show-time inputs, paired user bootstrap, 2,000 draws,
numpy default_rng(0), 95% CI, on the rule users of each set (a finite row for both models, equal benchmark sizes).

Per chain (th = six-layer Tiny two-head (SM130), io = six-layer Tiny Instant-Only (SM130)): the session arm (the end of
its two-pass cool-down) vs the control = the release model of the same chain (when the control reproduction check G0
passed; otherwise the th control arm sm183-th-ctl, and io is not run).
Passes on a set: instant change <= -0.05% with the CI upper bound below 0, and (th only) curve point estimate <= +0.05%.
The io chain's curve head is not trained (curve_loss_weight 0): reported only.
Verdict per chain: GO (passes on the 732 set and on the decisive users), PARTIAL (passes on one set; the main session
decides), NO GO (neither); for a chain whose decisive users were not scored (budget) the verdict is the 732 set's,
marked "732 set only".
Pending (exit 2): fewer than 720 rule users on the 732 set or 743 on the decisive users (97% of 765), a non-finite
row, or an integrity check fails (the control's benchmark sizes equal the reference rows of the same model: SM130's
own rows on the 732 set, SM164's REF rows on the decisive users, at least 600 common users).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
W = Path("C:/Users/jonat/workspace/wt")
G = HERE / "gpu"
ROWS = {"732": G / "sm183-rows-732.jsonl", "decisive": G / "sm183-rows-decisive.jsonl"}
REF_ROWS = {"732": W / "sm130/artifacts/small-model/sm130/gpu/sm130-rows.jsonl",
            "decisive": W / "sm164/artifacts/small-model/sm164/gpu/sm164-rows-new.jsonl"}
REF_LABEL = {("th", "732"): "sm130-th-dec2ep32k-t91s2/w8+s8@e2s-query", ("io", "732"): "sm130-io-dec2ep32k-t99/w8+s8@e2s-query",
             ("th", "decisive"): "sm130-th-dec2ep32k-t91s2/w8+s8hx8@e2s-query",
             ("io", "decisive"): "sm130-io-dec2ep32k-t99/w8+s8hx8@e2s-query"}
DECISIVE = HERE / "users-decisive.json"   # a copy of SM181's (= SM164's users-new-all.json, 765 users)
RUN = W / "sm179-run/artifacts/small-model"
SM130 = W / "sm130/artifacts/small-model"
CHECK_G0 = HERE / "check-ctl-sm183.json"
LABEL = "w8+s8hx8@e2s-query"
HEADS = {"curve": "ahead", "instant": "instant"}
GAIN = -0.05
GUARD = 0.05
DRAWS = 2000
MIN_USERS = {"732": 720, "decisive": 743}
CHAINS = {"th": "six-layer Tiny two-head (SM130)", "io": "six-layer Tiny Instant-Only (SM130)"}
TRACK = "E89-F65e-F77a"
DUMPS = W / "sm183/work/sm183-dump"
BINS = 16384
LOGIT_RANGE = 12.0
AUC_CHUNK = 250
ARGS = None


def control_tag(chain):
    g0 = json.loads(CHECK_G0.read_text()) if CHECK_G0.exists() else {}
    if chain == "th" and not g0.get("ok"):
        return "sm183-th-ctl"
    return f"sm130-{chain}-release"


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


def draws_for(n):
    return np.random.default_rng(0).integers(0, n, size=(DRAWS, n))


def auc_exact(y, p):
    from scipy.stats import rankdata
    pos = y > 0.5
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load_dump(folder, user, tag):
    path = folder / f"{user}_{tag}_{LABEL}.npz"
    if not path.exists():
        return None
    d = np.load(path)
    return d["y"] > 0.5, {"curve": d["p_ahead"].astype(np.float64), "instant": d["p_instant"].astype(np.float64)}


def hist(y, p):
    z = np.log(np.clip(p, 1e-12, 1 - 1e-12) / np.clip(1 - p, 1e-12, 1))
    b = np.clip(((z + LOGIT_RANGE) / (2 * LOGIT_RANGE) * BINS).astype(np.int64), 0, BINS - 1)
    return (np.bincount(b[y], minlength=BINS).astype(np.float32), np.bincount(b[~y], minlength=BINS).astype(np.float32))


def auc_from_hist(pos, neg):
    below = np.cumsum(neg, 1) - neg
    return (pos * (below + 0.5 * neg)).sum(1) / (pos.sum(1) * neg.sum(1))


def auc_metrics(set_name, ctl_tag, arm_tag, users, idx):
    """Secondary metric (report only): overall (pooled) AUC and within-user AUC per head, arm minus control (SM181's)."""
    folder = DUMPS / set_name
    tags = {"control": ctl_tag, "arm": arm_tag}
    keep, data = [], {}
    for i, u in enumerate(users):
        got = {k: load_dump(folder, u, t) for k, t in tags.items()}
        if any(v is None for v in got.values()):
            continue
        if not np.array_equal(got["control"][0], got["arm"][0]):
            raise SystemExit(f"{set_name}: user {u}: the two dumps have different labels")
        keep.append(i)
        data[u] = got
    if not keep:
        return {"users_with_dumps": 0}
    out = {"users_with_dumps": len(keep), "users_rule": len(users), "bins": BINS, "heads": {}}
    draws = idx if len(keep) == len(users) else draws_for(len(keep))
    us = [users[i] for i in keep]
    for head in HEADS:
        res, per_user, H = {}, {}, {}
        for k in tags:
            ys = [data[u][k][0] for u in us]
            ps = [data[u][k][1][head] for u in us]
            per_user[k] = np.array([auc_exact(y, p) for y, p in zip(ys, ps)])
            hs = [hist(y, p) for y, p in zip(ys, ps)]
            H[k] = (np.stack([h[0] for h in hs]), np.stack([h[1] for h in hs]))
            res[f"{k}_overall_exact"] = auc_exact(np.concatenate(ys), np.concatenate(ps))
            res[f"{k}_within_user"] = float(np.nanmean(per_user[k]))
        res["within_user_users"] = int(np.isfinite(per_user["control"]).sum())
        boot_o, boot_w = [], []
        for s in range(0, DRAWS, AUC_CHUNK):
            d = draws[s:s + AUC_CHUNK]
            counts = np.zeros((len(d), len(us)), dtype=np.float32)
            for j, row in enumerate(d):
                np.add.at(counts[j], row, 1)
            o = {k: auc_from_hist(counts @ H[k][0], counts @ H[k][1]) for k in tags}
            boot_o.append(o["arm"] - o["control"])
            boot_w.append(np.nanmean(per_user["arm"][d], 1) - np.nanmean(per_user["control"][d], 1))
        boot_o, boot_w = np.concatenate(boot_o), np.concatenate(boot_w)
        diff_o = res["arm_overall_exact"] - res["control_overall_exact"]
        diff_w = res["arm_within_user"] - res["control_within_user"]
        res["overall_change"] = {"value": float(diff_o), "ci": [float(np.percentile(boot_o, 2.5)), float(np.percentile(boot_o, 97.5))],
                                 "relative_pct": float(diff_o / res["control_overall_exact"] * 100)}
        res["within_user_change"] = {"value": float(diff_w), "ci": [float(np.percentile(boot_w, 2.5)), float(np.percentile(boot_w, 97.5))],
                                     "relative_pct": float(diff_w / res["control_within_user"] * 100)}
        out["heads"][head] = res
    return out


def one_set(chain, set_name):
    rows, bad = read_rows(ROWS[set_name])
    ctl_tag, arm_tag = control_tag(chain), f"sm183-{chain}-ses"
    lc, la = f"{ctl_tag}/{LABEL}", f"{arm_tag}/{LABEL}"
    ctl, arm = rows.get(lc, {}), rows.get(la, {})
    if not ctl and not arm:
        return None
    common = set(ctl) & set(arm)
    if set_name == "decisive":
        common &= set(json.loads(DECISIVE.read_text())["users"])
    nonfinite = sorted((bad.get(lc, set()) | bad.get(la, set())) | {u for u in common if not (finite(ctl[u]) and finite(arm[u]))})
    users = sorted(u for u in common if u not in nonfinite and all(
        arm[u]["benchmark"][h]["size"] == ctl[u]["benchmark"][h]["size"] for h in HEADS.values()))
    ref, _ = read_rows(REF_ROWS[set_name])
    ref = ref.get(REF_LABEL[(chain, set_name)], {})
    both = [u for u in ctl if u in ref]
    bad_size = [u for u in both if any(ctl[u]["benchmark"][h]["size"] != ref[u]["benchmark"][h]["size"] for h in HEADS.values())]
    integrity = {"what": f"control benchmark sizes equal {REF_ROWS[set_name].name} ({REF_LABEL[(chain, set_name)]})",
                 "common_users": len(both), "size_mismatch_users": bad_size[:20], "ok": len(both) >= 600 and not bad_size}
    if ctl_tag.endswith("-release") and REF_LABEL[(chain, set_name)].endswith(LABEL):   # same model and label: report
        integrity["max_abs_logloss_diff_same_label"] = max((abs(ctl[u]["benchmark"][h]["LogLoss"] - ref[u]["benchmark"][h]["LogLoss"])
                                                            for u in both for h in HEADS.values()), default=None)
    out = {"set": set_name, "control": ctl_tag, "arm": arm_tag, "rule_users": len(users),
           "users_per_label": {"control": len(ctl), "arm": len(arm)}, "nonfinite_users": nonfinite,
           "integrity": integrity, "min_users": MIN_USERS[set_name]}
    out["pending"] = len(users) < MIN_USERS[set_name] or bool(nonfinite) or not integrity["ok"]
    if users:
        idx = draws_for(len(users))
        for h in HEADS:
            out[h] = change(sums(arm, users, h), sums(ctl, users, h), idx)
        out["passes"] = bool(out["instant"]["value"] <= GAIN and out["instant"]["ci"][1] < 0
                             and (chain == "io" or out["curve"]["value"] <= GUARD))
        out["control_logloss"] = {h: float(sums(ctl, users, h).sum() / sum(ctl[u]["benchmark"][HEADS[h]]["size"] for u in users))
                                  for h in HEADS}
        out["arm_logloss"] = {h: float(sums(arm, users, h).sum() / sum(arm[u]["benchmark"][HEADS[h]]["size"] for u in users))
                              for h in HEADS}
        bc = {}
        for h in HEADS:
            x, c = buckets(arm, users, h), buckets(ctl, users, h)
            bc[h] = {str(e): {"reviews": c[e][0], "change_pct": (x[e][1] / c[e][1] - 1) * 100 if c[e][1] else None}
                     for e in sorted(c) if e in x}
        out["buckets_all_predicted"] = bc
        if not getattr(ARGS, "no_auc", False):
            out["auc_secondary"] = auc_metrics(set_name, ctl_tag, arm_tag, users, idx)
    return out


def training(folder):
    p = folder / "expertium" / TRACK / "result.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    return {k: r.get(k) for k in ("review_presentations", "training_seconds", "reviews_per_second", "peak_allocated_mib",
                                  "peak_reserved_mib", "update_offset_end")} | {"resumed": (folder / "resume.json").exists()}


def report(args):
    out = {"label": LABEL, "chains": {}, "g0": json.loads(CHECK_G0.read_text()) if CHECK_G0.exists() else None}
    for chain in CHAINS:
        sets = {s: one_set(chain, s) for s in ROWS}
        sets = {s: v for s, v in sets.items() if v is not None}
        if not sets:
            continue
        c = {"what": CHAINS[chain], "sets": sets}
        c["pending"] = "732" not in sets or any(v["pending"] for v in sets.values())
        passed = [s for s, v in sets.items() if v.get("passes")]
        if "decisive" in sets:
            v = "GO" if len(passed) == 2 else "PARTIAL" if passed else "NO GO"
        else:
            v = ("GO" if passed else "NO GO") + " (732 set only; decisive users not scored)"
        c["verdict"] = v if not c["pending"] else f"pending ({v} on the users so far)"
        tr = {}
        for p in (1, 2):
            for tag, folder in ((f"sm183-{chain}-ses-p{p}", RUN / f"sm183-{chain}-ses-p{p}"),
                                (f"sm183-{chain}-ctl-p{p}", RUN / f"sm183-{chain}-ctl-p{p}"),
                                (f"sm130-{chain}-dec2ep32k-p{p} (release)", SM130 / f"sm130-{chain}-dec2ep32k-p{p}")):
                t = training(folder)
                if t:
                    tr[tag] = t
        c["training"] = tr
        out["chains"][chain] = c
    pend = [k for k, c in out["chains"].items() if c["pending"]] or ([] if out["chains"] else ["all"])
    out["status"] = "pending" if pend else "decided"
    (HERE / "sm183-result.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    f = lambda r: f"{r['value']:+.3f}% [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]"  # noqa: E731
    md = ["# Tiny cool-down with session inputs (SM183): result tables", "",
          f"Status: {out['status']}; label REF {LABEL}; control reproduction check (G0): "
          f"{'PASS (the release models are the controls)' if (out['g0'] or {}).get('ok') else 'not passed (th control arm)'}.", ""]
    for chain, c in out["chains"].items():
        md.append(f"**{c['what']}: {c['verdict']}**")
    md += ["", "| model | set | control | rule users | curve vs control | instant vs control | passes | integrity |",
           "|---|---|---|---|---|---|---|---|"]
    for chain, c in out["chains"].items():
        for s, a in c["sets"].items():
            if "instant" in a:
                md.append(f"| {c['what']} | {s} | {a['control']} | {a['rule_users']} | {f(a['curve'])} | {f(a['instant'])} | "
                          f"{'yes' if a['passes'] else 'no'} | {'OK' if a['integrity']['ok'] else 'FAIL'} |")
            else:
                md.append(f"| {c['what']} | {s} | {a['control']} | 0 | - | - | - | {'OK' if a['integrity']['ok'] else 'FAIL'} |")
    g = lambda r: f"{r['value']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] ({r['relative_pct']:+.3f}%)"  # noqa: E731
    md += ["", "Secondary metric, report only (not part of the rule): AUC, arm minus control, paired user bootstrap.", "",
           "| model | set | head | users | overall AUC control -> arm | overall change | within-user AUC control -> arm | within-user change |",
           "|---|---|---|---|---|---|---|---|"]
    for chain, c in out["chains"].items():
        for s, a in c["sets"].items():
            au = a.get("auc_secondary") or {}
            for h, r in au.get("heads", {}).items():
                md.append(f"| {c['what']} | {s} | {h} | {au['users_with_dumps']} (within-user {r['within_user_users']}) | "
                          f"{r['control_overall_exact']:.4f} -> {r['arm_overall_exact']:.4f} | {g(r['overall_change'])} | "
                          f"{r['control_within_user']:.4f} -> {r['arm_within_user']:.4f} | {g(r['within_user_change'])} |")
    md += ["", "| training folder | reviews | reviews/s | minutes | peak allocated / reserved MiB | update offset end | resumed |",
           "|---|---|---|---|---|---|---|"]
    for chain, c in out["chains"].items():
        for t, r in c["training"].items():
            md.append(f"| {t} | {r['review_presentations'] or 0:,} | {r['reviews_per_second'] or 0:,.0f} | "
                      f"{(r['training_seconds'] or 0) / 60:.1f} | {r['peak_allocated_mib'] or 0:,.0f} / {r['peak_reserved_mib'] or 0:,.0f} | "
                      f"{r['update_offset_end']} | {'yes' if r['resumed'] else 'no'} |")
    (HERE / "sm183-result.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    for line in md[2:2 + 1 + len(out["chains"]) + 1]:
        if line:
            print(line)
    return 2 if pend else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report")
    r.add_argument("--no-auc", action="store_true", help="skip the AUC (dumps) for a quick look")
    global ARGS
    ARGS = p.parse_args()
    return report(ARGS)


if __name__ == "__main__":
    sys.exit(main())
