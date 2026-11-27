#!/usr/bin/env python
"""Session inputs confirmation (SM181): read-out (design.md "Rules"). CPU only, numpy; a few seconds, < 1 GB.

  python sm181_report.py report   -> sm181-result.json, sm181-result.md; exit 0 decided, 2 pending

Statistic (as SM179): % = 100 x (sum over users of LogLoss x reviews of the arm / the same sum of the control - 1),
per head (curve = "ahead", instant), REF labels (w8+s8hx8) on show-time inputs, paired user bootstrap, 2,000 draws,
numpy default_rng(0), 95% CI, on each axis's rule users (a finite row for both models, equal benchmark sizes).

Axis 1 (decisive users, six-layer Tiny two-head (SM130)): SM179's session arm vs SM171's control on the 765 users
with id >= 5000 of the 1,500 pool (users-decisive.json). Axis 2 (Small two-head (SM118)): sm181-small-ses vs SM149's
control on the 732 set.
Confirmed (per axis): instant change <= -0.05% with the CI upper bound below 0, and curve point estimate <= +0.05%.
Verdict: CONFIRMED (both axes), PARTIAL (one axis; the main session decides), NOT CONFIRMED (neither).
Pending (exit 2): axis 1 fewer than 743 rule users (97% of 765), axis 2 fewer than 720, a non-finite row, or an
integrity check fails (axis 1: every benchmark size equal to SM164's Tiny REF rows of the same user; axis 2: the
control's rows equal SM149's rows of that model, max |LogLoss difference| <= 1e-6).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
W = Path("C:/Users/jonat/workspace/wt")
ROWS_DECISIVE = HERE / "gpu" / "sm181-rows-decisive.jsonl"
ROWS_732 = HERE / "gpu" / "sm181-rows-732.jsonl"
ROWS_TINY732 = HERE / "gpu" / "sm181-rows-tiny732.jsonl"
SM164_ROWS = W / "sm164/artifacts/small-model/sm164/gpu/sm164-rows-new.jsonl"
SM149_ROWS = W / "sm149/artifacts/small-model/sm149/gpu/sm149-rows-732.jsonl"
SM179_ROWS = W / "sm179/artifacts/small-model/sm179/gpu/sm179-rows-732.jsonl"
RUN = W / "sm179-run/artifacts/small-model"
LABEL = "w8+s8hx8@e2s-query"
HEADS = {"curve": "ahead", "instant": "instant"}
GAIN = -0.05
GUARD = 0.05
DRAWS = 2000
AXES = {
    "decisive": {"what": "six-layer Tiny two-head (SM130), decisive users (ids >= 5000 of the 1,500 pool)",
                 "rows": ROWS_DECISIVE, "ctl": "sm171-tiny-ctl", "arm": "sm179-tiny-ses", "min_users": 743,
                 "list": HERE / "users-decisive.json", "dump": "decisive"},
    "small": {"what": "Small two-head (SM118), 732 set", "rows": ROWS_732, "ctl": "sm149-sm118-ctl",
              "arm": "sm181-small-ses", "min_users": 720, "list": None, "dump": "small732"},
}
# report only (no verdict): SM179's two Tiny models rescored on the 732 set with the per-review dump, for the AUC
REPORT_ONLY = {
    "tiny732": {"what": "six-layer Tiny two-head (SM130), 732 set (SM179's models rescored with the dump; report only)",
                "rows": ROWS_TINY732, "ctl": "sm171-tiny-ctl", "arm": "sm179-tiny-ses", "min_users": 720, "list": None,
                "dump": "tiny732"},
}
TRAINING = {"sm149-sm118-ctl": W / "sm149-run/artifacts/small-model/sm149-sm118-ctl/expertium/E89-F65e-F77a/result.json",
            "sm181-small-ses": RUN / "sm181-small-ses/expertium/E89-F65e-F77a/result.json"}


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


def draws_for(users):
    return np.random.default_rng(0).integers(0, len(users), size=(DRAWS, len(users)))


def compare(arm, ctl, users, idx=None):
    idx = draws_for(users) if idx is None else idx
    res = {h: change(sums(arm, users, h), sums(ctl, users, h), idx) for h in HEADS}
    res["confirmed"] = bool(res["instant"]["value"] <= GAIN and res["instant"]["ci"][1] < 0
                            and res["curve"]["value"] <= GUARD)
    bc = {}
    for h in HEADS:
        x, c = buckets(arm, users, h), buckets(ctl, users, h)
        bc[h] = {str(e): {"reviews": c[e][0], "change_pct": (x[e][1] / c[e][1] - 1) * 100 if c[e][1] else None}
                 for e in sorted(c) if e in x}
    res["buckets_all_predicted"] = bc
    res["control_logloss"] = {h: float(sums(ctl, users, h).sum() / sum(ctl[u]["benchmark"][HEADS[h]]["size"] for u in users))
                              for h in HEADS}
    return res


DUMPS = W / "sm181/work/sm181-dump"
BINS = 16384          # logit bins of the bootstrap's pooled AUC (the point estimate is also computed exactly)
LOGIT_RANGE = 12.0
AUC_CHUNK = 250
ARGS = None


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
    """pos, neg: (draws, BINS). Ties inside a bin count one half."""
    below = np.cumsum(neg, 1) - neg
    return (pos * (below + 0.5 * neg)).sum(1) / (pos.sum(1) * neg.sum(1))


def auc_metrics(name, spec, users, idx):
    """Secondary metric (report only, never part of the rule): overall (pooled) AUC and within-user AUC (mean of the
    per-user AUCs over users with both outcomes) per head, arm minus control, paired user bootstrap (the LogLoss draws)."""
    folder = DUMPS / spec["dump"]
    tags = {"control": spec["ctl"], "arm": spec["arm"]}
    keep, data = [], {}
    for i, u in enumerate(users):
        got = {k: load_dump(folder, u, t) for k, t in tags.items()}
        if any(v is None for v in got.values()):
            continue
        if not np.array_equal(got["control"][0], got["arm"][0]):
            raise SystemExit(f"{name}: user {u}: the two dumps have different labels")
        keep.append(i)
        data[u] = got
    if not keep:
        return {"users_with_dumps": 0}
    out = {"users_with_dumps": len(keep), "users_rule": len(users), "bins": BINS, "heads": {}}
    # remap the LogLoss draws to the users that have dumps (all of them when every dump exists)
    full = len(keep) == len(users)
    draws = idx if full else np.random.default_rng(0).integers(0, len(keep), size=(DRAWS, len(keep)))
    us = [users[i] for i in keep]
    for head in HEADS:
        res = {}
        per_user, H = {}, {}
        for k in tags:
            ys = [data[u][k][0] for u in us]
            ps = [data[u][k][1][head] for u in us]
            per_user[k] = np.array([auc_exact(y, p) for y, p in zip(ys, ps)])
            hs = [hist(y, p) for y, p in zip(ys, ps)]
            H[k] = (np.stack([h[0] for h in hs]), np.stack([h[1] for h in hs]))
            res[f"{k}_overall_exact"] = auc_exact(np.concatenate(ys), np.concatenate(ps))
            res[f"{k}_overall_binned"] = float(auc_from_hist(H[k][0].sum(0, keepdims=True), H[k][1].sum(0, keepdims=True))[0])
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
            wa, wc = per_user["arm"][d], per_user["control"][d]
            boot_w.append(np.nanmean(wa, 1) - np.nanmean(wc, 1))
        boot_o, boot_w = np.concatenate(boot_o), np.concatenate(boot_w)
        diff_o = res["arm_overall_exact"] - res["control_overall_exact"]
        diff_w = res["arm_within_user"] - res["control_within_user"]
        res["overall_change"] = {"value": float(diff_o), "ci": [float(np.percentile(boot_o, 2.5)), float(np.percentile(boot_o, 97.5))],
                                 "relative_pct": float(diff_o / res["control_overall_exact"] * 100)}
        res["within_user_change"] = {"value": float(diff_w), "ci": [float(np.percentile(boot_w, 2.5)), float(np.percentile(boot_w, 97.5))],
                                     "relative_pct": float(diff_w / res["control_within_user"] * 100)}
        out["heads"][head] = res
    return out


def training(tag):
    p = TRAINING.get(tag)
    if not p or not p.exists():
        return None
    r = json.loads(p.read_text())
    return {k: r.get(k) for k in ("review_presentations", "training_seconds", "reviews_per_second",
                                  "peak_allocated_mib", "peak_reserved_mib")}


def axis(name, spec):
    rows, bad = read_rows(spec["rows"])
    lc, la = f"{spec['ctl']}/{LABEL}", f"{spec['arm']}/{LABEL}"
    ctl, arm = rows.get(lc, {}), rows.get(la, {})
    want = set(json.loads(spec["list"].read_text())["users"]) if spec["list"] else None
    common = set(ctl) & set(arm)
    if want is not None:
        common &= want
    nonfinite = sorted((bad.get(lc, set()) | bad.get(la, set())) | {u for u in common if not (finite(ctl[u]) and finite(arm[u]))})
    users = sorted(u for u in common if u not in nonfinite and all(
        arm[u]["benchmark"][h]["size"] == ctl[u]["benchmark"][h]["size"] for h in HEADS.values()))
    if name == "decisive":
        ref, _ = read_rows(SM164_ROWS)
        ref = ref.get(f"sm130-th-dec2ep32k-t91s2/{LABEL}", {})
        both = [u for u in ctl if u in ref]
        bad_size = [u for u in both if any(ctl[u]["benchmark"][h]["size"] != ref[u]["benchmark"][h]["size"]
                                           for h in HEADS.values())]
        integrity = {"what": "benchmark sizes equal SM164's Tiny REF rows", "common_users": len(both),
                     "size_mismatch_users": bad_size, "ok": len(both) >= 600 and not bad_size}
    else:   # small: the control's rows equal SM149's; tiny732: both models' rows equal SM179's
        src, labs = (SM149_ROWS, (lc,)) if name == "small" else (SM179_ROWS, (lc, la))
        ref, _ = read_rows(src)
        mine = {(lab, u): r for lab, rr in ((lc, ctl), (la, arm)) if lab in labs for u, r in rr.items()}
        both = [k for k in mine if k[1] in ref.get(k[0], {})]
        diff = max((abs(mine[k]["benchmark"][h]["LogLoss"] - ref[k[0]][k[1]]["benchmark"][h]["LogLoss"])
                    for k in both for h in HEADS.values()), default=None)
        integrity = {"what": f"rows of {', '.join(labs)} equal {src.name}", "common_rows": len(both),
                     "max_abs_logloss_diff": diff, "ok": diff is not None and diff <= 1e-6 and len(both) >= 720 * len(labs)}
    out = {"what": spec["what"], "control": spec["ctl"], "arm": spec["arm"], "rule_users": len(users),
           "users_per_label": {"control": len(ctl), "arm": len(arm)}, "nonfinite_users": nonfinite,
           "integrity": integrity, "min_users": spec["min_users"]}
    out["pending"] = len(users) < spec["min_users"] or bool(nonfinite) or not integrity["ok"]
    if users:
        idx = draws_for(users)
        out.update(compare(arm, ctl, users, idx))
        if not getattr(ARGS, "no_auc", False):
            out["auc_secondary"] = auc_metrics(name, spec, users, idx)
    out["training"] = {t: training(t) for t in (spec["ctl"], spec["arm"]) if training(t)}
    return out, (ctl, arm, users)


def pooled_tiny(decisive):
    """Report only: six-layer Tiny two-head (SM130) on the 732 set (SM179's rows) and the decisive users together."""
    rows, _ = read_rows(SM179_ROWS)
    c732, a732 = rows.get(f"sm171-tiny-ctl/{LABEL}", {}), rows.get(f"sm179-tiny-ses/{LABEL}", {})
    ctl, arm, users = decisive
    cc = {**{u: c732[u] for u in c732 if u in a732 and finite(c732[u]) and finite(a732[u])}, **{u: ctl[u] for u in users}}
    aa = {**{u: a732[u] for u in c732 if u in a732 and finite(c732[u]) and finite(a732[u])}, **{u: arm[u] for u in users}}
    us = sorted(cc)
    if not us:
        return None
    r = compare(aa, cc, us)
    r.pop("buckets_all_predicted")
    return dict(r, users=len(us))


def report(args):
    out = {"label": LABEL, "axes": {}}
    data = {}
    for name, spec in AXES.items():
        out["axes"][name], data[name] = axis(name, spec)
    pend = [n for n, a in out["axes"].items() if a["pending"]]
    conf = [n for n, a in out["axes"].items() if a.get("confirmed")]
    verdict = "CONFIRMED" if len(conf) == 2 else "PARTIAL" if conf else "NOT CONFIRMED"
    out["confirmed_axes"] = conf
    out["status"] = "pending" if pend else "decided"
    out["verdict"] = verdict if not pend else f"pending ({verdict} on the users so far; pending axes: {', '.join(pend)})"
    if data["decisive"][2]:
        out["report_only_tiny_732_plus_decisive"] = pooled_tiny(data["decisive"])
    out["report_only"] = {}
    for name, spec in REPORT_ONLY.items():
        out["report_only"][name], _ = axis(name, spec)
    (HERE / "sm181-result.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    f = lambda r: f"{r['value']:+.3f}% [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]"  # noqa: E731
    md = ["# Session inputs confirmation (SM181): result tables", "",
          f"**Verdict: {out['verdict']}** ({out['status']}; label REF {LABEL}).", "",
          "| axis | rule users | curve vs control | instant vs control | confirmed | integrity |", "|---|---|---|---|---|---|"]
    for name, a in list(out["axes"].items()) + list(out["report_only"].items()):
        if "curve" in a:
            md.append(f"| {a['what']} | {a['rule_users']} | {f(a['curve'])} | {f(a['instant'])} | "
                      f"{'yes' if a['confirmed'] else 'no'} | {'OK' if a['integrity']['ok'] else 'FAIL'} |")
        else:
            md.append(f"| {a['what']} | 0 | - | - | - | {'OK' if a['integrity']['ok'] else 'FAIL'} |")
    p = out.get("report_only_tiny_732_plus_decisive")
    if p:
        md += ["", f"Report only, six-layer Tiny two-head (SM130) on the 732 set and the decisive users together "
                   f"({p['users']} users): curve {f(p['curve'])}, instant {f(p['instant'])}."]
    g = lambda r: f"{r['value']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] ({r['relative_pct']:+.3f}%)"  # noqa: E731
    md += ["", "Secondary metric, report only (not part of the rule): AUC, arm minus control, paired user bootstrap.", "",
           "| axis | head | users | overall AUC control -> arm | overall change | within-user AUC control -> arm | within-user change |",
           "|---|---|---|---|---|---|---|"]
    for name, a in list(out["axes"].items()) + list(out["report_only"].items()):
        au = a.get("auc_secondary") or {}
        for h, r in au.get("heads", {}).items():
            md.append(f"| {name} | {h} | {au['users_with_dumps']} (within-user {r['within_user_users']}) | "
                      f"{r['control_overall_exact']:.4f} -> {r['arm_overall_exact']:.4f} | {g(r['overall_change'])} | "
                      f"{r['control_within_user']:.4f} -> {r['arm_within_user']:.4f} | {g(r['within_user_change'])} |")
    md += ["", "| model | reviews/s | minutes | peak allocated / reserved MiB |", "|---|---|---|---|"]
    for t, r in out["axes"]["small"]["training"].items():
        md.append(f"| {t} | {r['reviews_per_second'] or 0:,.0f} | {(r['training_seconds'] or 0) / 60:.1f} | "
                  f"{r['peak_allocated_mib'] or 0:,.0f} / {r['peak_reserved_mib'] or 0:,.0f} |")
    (HERE / "sm181-result.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(md[2])
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
