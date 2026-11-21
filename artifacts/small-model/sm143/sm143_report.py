"""Per-user adaptation (SM143): the pre-registered read-out (design.md "Metrics" and "Go rule").

  report  [--small ROWS] [--tiny ROWS] [--small-npz DIR] [--big-dump DIR --big-label L] [--out-json F --out-md F]
  selftest

Per user the adaptation rows (scripts/sm143_adapt.py) hold, on the evaluation reviews (the 5 FSRS-7 test folds of the
published protocol that are also benchmark reviews), n, the LogLoss sum, AUC and ECE of every predictor: none (the
unadapted model), a / b / c / d (adapted) and fsrs7 (FSRS-7 fitted per user on the reviews before each fold). The big
instant-only model (SM99), unadapted, comes from the stock scorer's --dump files, looked up on the same reviews.

Comparisons X vs R, paired over users (the same reviews), 2,000 user-bootstrap resamples, seed 0, 95% percentile CI:
  LogLoss   review-weighted, % change: (sum LL_X - sum LL_R) / sum LL_R * 100
  AUC       review-weighted mean of the per-user AUC difference (users with both classes)
  ECE       review-weighted mean of the per-user ECE difference (10 equal-width bins)
Go rule (per candidate variant b, c, d of the primary model, vs none): LogLoss CI upper bound < 0 and the point
estimate <= -0.05%. Variant a is the recalibration control (it cannot change AUC). The pilot is a screen: a variant
that passes goes to the full 722-user run, which decides with the same rule.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

B, SEED = 2000, 0
MIN_EFFECT = 0.05
FIT_BINS = [(0, 1000), (1000, 5000), (5000, 20000), (20000, 10 ** 9)]
NAMES = {"none": "unadapted", "a": "(a) Platt control", "b": "(b) user vector", "c": "(c) instant head",
         "d": "(d) full fine-tune", "fsrs7": "FSRS-7 fitted per user", "big": "big instant-only model (SM99), unadapted"}
MODEL_NAMES = {"small": "the longer Small Instant-Only model (SM113)", "tiny": "the six-layer Tiny Instant-Only model (SM130)"}


def load_rows(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if "error" in r or "metrics" not in r:
            continue
        out[r["user_id"]] = r   # the last complete row of a user wins
    return out


def per_user(rows, names, users=None):
    """{name: (n [U], ll [U], auc [U] (nan if undefined), ece [U])} over `users` (rows order)."""
    users = users if users is not None else sorted(rows)
    out = {}
    for name in names:
        m = [rows[u]["metrics"][name] for u in users]
        out[name] = (np.array([x["n"] for x in m], dtype=np.float64), np.array([x["ll_sum"] for x in m]),
                     np.array([np.nan if x["auc"] is None else x["auc"] for x in m]), np.array([x["ece"] for x in m]))
    return users, out


def compare(stat_x, stat_r, rng_seed=SEED):
    """Point and 95% CI of LogLoss %, AUC difference and ECE difference, X vs R, user bootstrap."""
    n, llx, aucx, ecex = stat_x
    _, llr, aucr, ecer = stat_r
    U = len(n)
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, U, size=(B, U))
    w = np.zeros((B, U))
    np.add.at(w, (np.repeat(np.arange(B), U), idx.reshape(-1)), 1.0)
    def ll(wt):
        return (wt @ llx - wt @ llr) / (wt @ llr) * 100
    da = aucx - aucr
    ok = ~np.isnan(da)
    def auc_d(wt):
        wn = wt[..., ok] * n[ok]
        return (wn @ da[ok]) / wn.sum(-1)
    de = ecex - ecer
    def ece_d(wt):
        wn = wt * n
        return (wn @ de) / wn.sum(-1)
    one = np.ones(U)
    res = {}
    for key, fn in (("logloss_pct", ll), ("auc_diff", auc_d), ("ece_diff", ece_d)):
        point = float(fn(one))
        boot = fn(w)
        res[key] = {"point": point, "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5))}
    res["users"], res["reviews"] = int(U), int(n.sum())
    res["auc_users"] = int(ok.sum())
    return res


def absolute(stat):
    n, ll, a, e = stat
    ok = ~np.isnan(a)
    return {"logloss": float(ll.sum() / n.sum()), "logloss_user_mean": float(np.mean(ll / n)),
            "auc": float((a[ok] * n[ok]).sum() / n[ok].sum()), "auc_user_mean": float(np.nanmean(a)),
            "ece": float((e * n).sum() / n.sum())}


def go(c):
    lo = c["logloss_pct"]
    return bool(lo["hi"] < 0 and lo["point"] <= -MIN_EFFECT)


def big_stats(rows, users, npz_dir, dump_dir, big_label):
    """Big model stats on each user's evaluation reviews (users with every review present); returns (users, stat)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.sm143_adapt import summary as summary_arrays
    kept, n, ll, au, ec = [], [], [], [], []
    for u in users:
        f = Path(npz_dir) / f"{u}.npz"
        g = Path(dump_dir) / f"{u}_{big_label.replace('/', '_')}.npz"
        if not f.exists() or not g.exists():
            continue
        a, d = np.load(f), np.load(g)
        pos = np.searchsorted(d["review_th"], a["review_th"])
        pos = np.minimum(pos, len(d["review_th"]) - 1)
        if not np.array_equal(d["review_th"][pos], a["review_th"].astype(d["review_th"].dtype)):
            continue
        if not np.array_equal(np.clip(d["y"][pos], 0, 1).astype(np.int64), a["y"].astype(np.int64)):
            raise SystemExit(f"user {u}: big dump y differs")
        s = summary_arrays(a["y"], d["p_instant"][pos])
        kept.append(u)
        n.append(s["n"]), ll.append(s["ll_sum"]), au.append(np.nan if s["auc"] is None else s["auc"]), ec.append(s["ece"])
    return kept, (np.array(n, dtype=np.float64), np.array(ll), np.array(au), np.array(ec))


def by_fit_bins(rows, users, x, r):
    """X vs R by bin of the fold's fit rows (per user and fold), user bootstrap within the bin."""
    out = []
    for lo, hi in FIT_BINS:
        per = {}
        for u in users:
            for f in rows[u]["folds"]:
                k = str(f["fold"])
                if lo <= f["n_fit"] < hi and k in rows[u]["fold_metrics"]:
                    fm = rows[u]["fold_metrics"][k]
                    acc = per.setdefault(u, [0.0, 0.0, 0.0, 0.0, 0.0])
                    acc[0] += fm[x]["n"]
                    acc[1] += fm[x]["ll_sum"]
                    acc[2] += fm[r]["ll_sum"]
                    if fm[x]["auc"] is not None and fm[r]["auc"] is not None:
                        acc[3] += fm[x]["n"] * (fm[x]["auc"] - fm[r]["auc"])
                        acc[4] += fm[x]["n"]
        if len(per) < 5:
            out.append({"bin": [lo, hi], "users": len(per), "note": "fewer than 5 users"})
            continue
        a = np.array(list(per.values()))
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, len(a), size=(B, len(a)))
        s = a[idx].sum(1)
        ll = (s[:, 1] - s[:, 2]) / s[:, 2] * 100
        au = s[:, 3] / np.maximum(s[:, 4], 1e-9)
        t = a.sum(0)
        out.append({"bin": [lo, hi], "users": len(per), "reviews": int(t[0]),
                    "logloss_pct": {"point": float((t[1] - t[2]) / t[2] * 100), "lo": float(np.percentile(ll, 2.5)),
                                    "hi": float(np.percentile(ll, 97.5))},
                    "auc_diff": {"point": float(t[3] / max(t[4], 1e-9)), "lo": float(np.percentile(au, 2.5)),
                                 "hi": float(np.percentile(au, 97.5))}})
    return out


def cost(rows, users, variants):
    out = {}
    for v in variants:
        secs = [sum(f["variants"][v]["seconds"] for f in rows[u]["folds"] if v in f.get("variants", {})) for u in users
                if v in rows[u]["variants"]]
        steps = [f["variants"][v]["step"] for u in users for f in rows[u]["folds"] if v in f.get("variants", {})]
        if secs:
            out[v] = {"users": len(secs), "seconds_per_user_mean": float(np.mean(secs)),
                      "seconds_per_user_p90": float(np.percentile(secs, 90)),
                      "share_folds_step0": float(np.mean(np.array(steps) == 0)) if steps else None}
    fw = [rows[u]["forward_seconds"] for u in users]
    out["forward_seconds_per_user_mean"] = float(np.mean(fw))
    return out


def model_block(rows, variants, npz_dir=None, dump_dir=None, big_label=None):
    users_all = sorted(rows)
    block = {"users": len(users_all), "variants": variants, "absolute": {}, "vs_none": {}, "vs_fsrs7": {}, "vs_big": {},
             "bins": {}, "go": {}}
    names = ["none", "fsrs7"] + [v for v in variants if v != "d"]
    users, st = per_user(rows, names, users_all)
    for k in names:
        block["absolute"][k] = absolute(st[k])
    for v in [v for v in variants if v != "d"]:
        block["vs_none"][v] = compare(st[v], st["none"])
        block["bins"][v] = by_fit_bins(rows, users, v, "none")
    for k in [k for k in names if k != "fsrs7"]:
        block["vs_fsrs7"][k] = compare(st[k], st["fsrs7"])
    if "d" in variants:
        du = [u for u in users_all if "d" in rows[u]["variants"]]
        if du:
            _, sd = per_user(rows, ["none", "fsrs7", "d", *[v for v in variants if v != "d"]], du)
            block["d_subset"] = {"users": len(du), "absolute": {k: absolute(s) for k, s in sd.items()},
                                 "vs_none": {k: compare(sd[k], sd["none"]) for k in sd if k not in ("none", "fsrs7")},
                                 "vs_fsrs7": {k: compare(sd[k], sd["fsrs7"]) for k in sd if k != "fsrs7"},
                                 "bins": by_fit_bins(rows, du, "d", "none")}
            block["vs_none"]["d"] = block["d_subset"]["vs_none"]["d"]
    for v in ("b", "c", "d"):
        if v in block["vs_none"]:
            block["go"][v] = go(block["vs_none"][v])
    if npz_dir and dump_dir and big_label:
        bu, bst = big_stats(rows, users_all, npz_dir, dump_dir, big_label)
        if bu:
            _, sb = per_user(rows, names, bu)
            block["vs_big"] = {"users": len(bu), "big_absolute": absolute(bst),
                               **{k: compare(sb[k], bst) for k in names if k != "fsrs7"}}
            if "d" in variants:
                du = [u for u in bu if "d" in rows[u]["variants"]]
                if du:
                    keep = np.isin(np.array(bu), du)
                    _, sdd = per_user(rows, ["d"], du)
                    block["vs_big"]["d"] = compare(sdd["d"], tuple(a[keep] for a in bst))
    block["cost"] = cost(rows, users_all, variants)
    return block


def fmt(c, key, pct=True):
    x = c[key]
    if pct:
        return f"{x['point']:+.2f}% [{x['lo']:+.2f}, {x['hi']:+.2f}]"
    return f"{x['point']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}]"


def render(res):
    out = ["# Per-user adaptation (SM143): result", "",
           "Generated by `artifacts/small-model/sm143/sm143_report.py report` from the pre-registered design "
           "(`design.md`). Evaluation reviews: per user, the 5 FSRS-7 test folds of the published protocol that are also "
           "benchmark reviews; each fold's adaptation saw only the user's reviews before that fold. Differences are paired "
           "over users with a 2,000-resample user bootstrap (seed 0, 95% CI). LogLoss is review-weighted; AUC and ECE are "
           "review-weighted means of per-user values (AUC is within-user discrimination).", ""]
    for mk, block in res["models"].items():
        name = MODEL_NAMES[mk]
        out += [f"## {name[0].upper() + name[1:]}", "", f"{block['users']} users.", "",
                "| Predictor | LogLoss | AUC | ECE |", "|---|---|---|---|"]
        for k, a in block["absolute"].items():
            out.append(f"| {NAMES[k]} | {a['logloss']:.5f} | {a['auc']:.4f} | {a['ece']:.4f} |")
        out += ["", "Adapted vs unadapted (same model):", "", "| Variant | LogLoss % | AUC difference | ECE difference | Go |",
                "|---|---|---|---|---|"]
        for v, c in block["vs_none"].items():
            g = block["go"].get(v)
            out.append(f"| {NAMES[v]} ({c['users']} users) | {fmt(c, 'logloss_pct')} | {fmt(c, 'auc_diff', False)} | "
                       f"{fmt(c, 'ece_diff', False)} | {'control' if v == 'a' else ('go' if g else 'no go')} |")
        out += ["", "Vs FSRS-7 fitted per user on the same early reviews (negative LogLoss % = better than FSRS-7):", "",
                "| Predictor | LogLoss % | AUC difference |", "|---|---|---|"]
        for k, c in block["vs_fsrs7"].items():
            out.append(f"| {NAMES[k]} | {fmt(c, 'logloss_pct')} | {fmt(c, 'auc_diff', False)} |")
        if block.get("d_subset"):
            ds = block["d_subset"]
            out += ["", f"Full fine-tune subset ({ds['users']} users), vs FSRS-7:", "", "| Predictor | LogLoss % | AUC difference |",
                    "|---|---|---|"]
            for k, c in ds["vs_fsrs7"].items():
                out.append(f"| {NAMES[k]} | {fmt(c, 'logloss_pct')} | {fmt(c, 'auc_diff', False)} |")
        if block.get("vs_big"):
            vb = block["vs_big"]
            out += ["", f"Vs the {NAMES['big']} on the same reviews ({vb['users']} users; negative = better than Big):", "",
                    "| Predictor | LogLoss % | AUC difference |", "|---|---|---|"]
            for k, c in vb.items():
                if isinstance(c, dict) and "logloss_pct" in c:
                    out.append(f"| {NAMES[k]} | {fmt(c, 'logloss_pct')} | {fmt(c, 'auc_diff', False)} |")
        out += ["", "By the fold's amount of earlier history (labelled reviews before the fold), vs unadapted:", "",
                "| Variant | Fit reviews | Users | LogLoss % | AUC difference |", "|---|---|---|---|---|"]
        bins = dict(block["bins"])
        if block.get("d_subset"):
            bins["d"] = block["d_subset"]["bins"]
        for v, rows_ in bins.items():
            for b in rows_:
                lab = f"{b['bin'][0]:,}-{b['bin'][1]:,}" if b["bin"][1] < 10 ** 9 else f">= {b['bin'][0]:,}"
                if "logloss_pct" in b:
                    out.append(f"| {NAMES[v]} | {lab} | {b['users']} | {fmt(b, 'logloss_pct')} | {fmt(b, 'auc_diff', False)} |")
                else:
                    out.append(f"| {NAMES[v]} | {lab} | {b['users']} | - | - |")
        co = block["cost"]
        out += ["", "Cost (this run's device; seconds of adaptation per user, all 5 folds):", ""]
        for v in block["variants"]:
            if v in co:
                out.append(f"- {NAMES[v]}: {co[v]['seconds_per_user_mean']:.2f} s mean, {co[v]['seconds_per_user_p90']:.2f} s "
                           f"90th percentile ({co[v]['users']} users); the validation slice kept the base weights in "
                           f"{co[v]['share_folds_step0'] * 100:.0f}% of the folds.")
        out.append(f"- full-history forward: {co['forward_seconds_per_user_mean']:.2f} s per user.")
        out.append("")
    out += ["## Verdict", "", res["verdict"], ""]
    return "\n".join(out)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        return selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("report",))
    ap.add_argument("--small")
    ap.add_argument("--tiny")
    ap.add_argument("--small-npz")
    ap.add_argument("--big-dump")
    ap.add_argument("--big-label", default="sm99-big-inst-e2/fp32@e2s-query")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    a = ap.parse_args()
    res = {"models": {}}
    if a.small:
        rows = load_rows(a.small)
        variants = sorted({v for r in rows.values() for v in r["variants"]})
        res["models"]["small"] = model_block(rows, variants, a.small_npz, a.big_dump, a.big_label)
    if a.tiny:
        rows = load_rows(a.tiny)
        variants = sorted({v for r in rows.values() for v in r["variants"]})
        res["models"]["tiny"] = model_block(rows, variants)
    s = res["models"].get("small", {})
    gos = [v for v, g in s.get("go", {}).items() if g]
    res["status"] = "go" if gos else "no go"
    res["go_variants"] = gos
    if gos:
        res["verdict"] = (f"Go on the pilot for {', '.join(NAMES[v] for v in gos)} of {MODEL_NAMES['small']}: the LogLoss "
                          "improvement's CI is below 0 and the point estimate is at least 0.05%. Next: the full 722-user "
                          "run with the same rule.")
    else:
        res["verdict"] = (f"No go on the pilot for {MODEL_NAMES['small']}: no candidate variant improved LogLoss with a CI "
                          "below 0 and at least 0.05%.")
    Path(a.out_json).write_text(json.dumps(res, indent=1), encoding="utf-8")
    Path(a.out_md).write_text(render(res), encoding="utf-8")
    print(f"[sm143-report] {res['status']}: {res['verdict'][:300]}")


def selftest():
    rng = np.random.default_rng(1)
    U = 40
    n = rng.integers(100, 1000, U).astype(float)
    llr = n * 0.3
    stat_r = (n, llr, np.full(U, 0.7), np.full(U, 0.02))
    stat_x = (n, llr * 0.99, np.full(U, 0.71), np.full(U, 0.01))
    c = compare(stat_x, stat_r)
    assert abs(c["logloss_pct"]["point"] + 1.0) < 1e-9 and c["logloss_pct"]["hi"] < 0, c
    assert abs(c["auc_diff"]["point"] - 0.01) < 1e-9 and go(c)
    c2 = compare(stat_r, stat_r)
    assert c2["logloss_pct"]["point"] == 0 and not go(c2)
    print("sm143 report selftest OK")


if __name__ == "__main__":
    main()
