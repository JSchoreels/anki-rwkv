"""SM71 report: LogLoss of the big model and the small model(s) with and without the deleted-card signals.

Series (repeatable): --series MODEL/COND=SOURCE::LABEL
  MODEL  big | small | small8k (big = the reference for the gap)
  COND   keep (today's inputs) | flags | neighbour | reseed (keep with --seed-offset)
  SOURCE a .jsonl of scripts/small_model_heldout.py rows (LABEL = the row label; --split-existing rows carry
         benchmark_existing / benchmark_deleted) or a --dump directory (LABEL = file suffix, e.g. fork_w8+s8;
         subsets computed from the per-review predictions and scripts/sm71_leak.deleted_review_ths).
Views: "all" = every benchmark test review (view 1 when COND != keep); "existing" = reviews of cards still in the
cards table (view 2 with COND = keep). Review-weighted means over the users common to every series; user bootstrap
(2,000 resamples, seed 0) on paired differences, 95% intervals.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

HEADS = {"ahead": "curve", "instant": "instant"}
SUBSETS = ("all", "existing", "deleted")


def logloss(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if len(y) else float("nan")


def load(source, label, users=None):
    """{user: {subset: {head: (LogLoss, size)}}}"""
    path = Path(source)
    out = {}
    if path.is_dir():
        from scripts import compress_common as cc
        from scripts import sm71_leak
        data = cc.data_path()
        for f in sorted(path.glob(f"*_{label}.npz")):
            user = int(f.name.split("_")[0])
            if users is not None and user not in users:
                continue
            d = np.load(f)
            gone = np.isin(d["review_th"], sm71_leak.deleted_review_ths(data, user))
            rec = {}
            for subset, m in (("all", np.ones_like(gone)), ("existing", ~gone), ("deleted", gone)):
                rec[subset] = {h: (logloss(d["y"][m], d[f"p_{h}"][m]), int(m.sum())) for h in HEADS}
            out[user] = rec
        return out
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("label") != label or "error" in r:
            continue
        rec = {}
        for subset, key in (("all", "benchmark"), ("existing", "benchmark_existing"), ("deleted", "benchmark_deleted")):
            b = r.get(key)
            if b is None:
                continue
            rec[subset] = {h: ((b[h]["LogLoss"], b[h]["size"]) if h in b else (float("nan"), 0)) for h in HEADS}
        out[r["user_id"]] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", action="append", required=True)
    ap.add_argument("--users-file", default="", help="restrict to these users (JSON with a 'users' list)")
    ap.add_argument("--title", default="SM71")
    ap.add_argument("--out", required=True, help="JSON output; a .md table is written next to it")
    args = ap.parse_args()
    wanted = set(json.loads(Path(args.users_file).read_text())["users"]) if args.users_file else None
    series = {}
    for item in args.series:
        name, _, rest = item.partition("=")
        source, _, label = rest.rpartition("::")
        series[name] = load(source, label, wanted)
    # users common to every series, and with the subset metrics wherever a series has them
    users = sorted(set.intersection(*(set(s) for s in series.values())))
    if wanted is not None:
        users = [u for u in users if u in wanted]
    n = len(users)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(2000, n))

    def arr(name, subset, head):
        s = series[name]
        if any(subset not in s[u] for u in users):
            return None
        v = np.array([s[u][subset][head][0] for u in users])
        w = np.array([s[u][subset][head][1] for u in users], dtype=float)
        v = np.where(w > 0, v, 0.0)
        return v, w

    def wmean(v, w, sel=None):
        if sel is None:
            return (v * w).sum() / w.sum()
        return (v[sel] * w[sel]).sum(1) / w[sel].sum(1)

    res = {"users": n, "series": list(series), "sizes": {}, "loss": {}, "change_vs_keep": {}, "gap_pct": {},
           "gap_change_pp": {}}
    for subset in SUBSETS:
        a = arr(next(iter(series)), subset, "ahead")
        if a is not None:
            res["sizes"][subset] = int(a[1].sum())
    for name in series:
        for subset in SUBSETS:
            for head, hn in HEADS.items():
                a = arr(name, subset, head)
                if a is not None:
                    res["loss"][f"{name}|{subset}|{hn}"] = round(float(wmean(*a)), 6)
    models = sorted({k.split("/")[0] for k in series})
    conds = sorted({k.split("/")[1] for k in series})

    def ci(x):
        return [round(float(np.percentile(x, 2.5)), 6), round(float(np.percentile(x, 97.5)), 6)]

    # each model's change vs its own keep series (paired users, same review weights per subset)
    for m in models:
        for c in conds:
            if c == "keep" or f"{m}/{c}" not in series or f"{m}/keep" not in series:
                continue
            for subset in SUBSETS:
                for head, hn in HEADS.items():
                    a, b = arr(f"{m}/{c}", subset, head), arr(f"{m}/keep", subset, head)
                    if a is None or b is None:
                        continue
                    w = b[1]
                    d = a[0] - b[0]
                    base = wmean(b[0], w)
                    point = wmean(d, w)
                    boot = (d[idx] * w[idx]).sum(1) / w[idx].sum(1)
                    res["change_vs_keep"][f"{m}/{c}|{subset}|{hn}"] = {
                        "delta": round(float(point), 6), "ci": ci(boot), "pct": round(float(100 * point / base), 3)}
    # gap of each small model vs big under each (cond, subset), and its change vs today (keep, all)
    for m in models:
        if m == "big":
            continue
        for c in conds:
            for subset in SUBSETS:
                for head, hn in HEADS.items():
                    s, b = (arr(f"{m}/{c}", subset, head) if f"{m}/{c}" in series else None,
                            arr(f"big/{c}", subset, head) if f"big/{c}" in series else None)
                    if s is None or b is None:
                        continue
                    w = b[1]
                    S, B = (s[0][idx] * w[idx]).sum(1) / w[idx].sum(1), (b[0][idx] * w[idx]).sum(1) / w[idx].sum(1)
                    gp = 100 * (wmean(s[0], w) - wmean(b[0], w)) / wmean(b[0], w)
                    boot_gap = 100 * (S - B) / B
                    key = f"{m}-big/{c}|{subset}|{hn}"
                    res["gap_pct"][key] = {"gap_pct": round(float(gp), 3), "ci": [round(x, 3) for x in ci(boot_gap)]}
                    s0, b0 = arr(f"{m}/keep", "all", head), arr("big/keep", "all", head)
                    if (c, subset) != ("keep", "all") and s0 is not None and b0 is not None:
                        w0 = b0[1]
                        S0 = (s0[0][idx] * w0[idx]).sum(1) / w0[idx].sum(1)
                        B0 = (b0[0][idx] * w0[idx]).sum(1) / w0[idx].sum(1)
                        g0 = 100 * (wmean(s0[0], w0) - wmean(b0[0], w0)) / wmean(b0[0], w0)
                        res["gap_change_pp"][key] = {"pp": round(float(gp - g0), 3),
                                                     "ci": [round(x, 3) for x in ci(boot_gap - 100 * (S0 - B0) / B0)]}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    lines = [f"# {args.title}: {n} users; benchmark test reviews all {res['sizes'].get('all', 0):,}, existing "
             f"{res['sizes'].get('existing', 0):,}, deleted {res['sizes'].get('deleted', 0):,} (review-weighted, user "
             f"bootstrap 2,000, seed 0, 95% CI)", "", "## LogLoss", "",
             "| series | all curve | all instant | existing curve | existing instant | deleted curve | deleted instant |",
             "|---|---|---|---|---|---|---|"]
    for name in series:
        cells = [str(res["loss"].get(f"{name}|{s}|{h}", "-")) for s in SUBSETS for h in HEADS.values()]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "## Change vs the same model with today's inputs (delta [95% CI], % of today)", "",
              "| series | subset | curve | instant |", "|---|---|---|---|"]
    for key in sorted({k.rsplit("|", 1)[0] for k in res["change_vs_keep"]}):
        cells = []
        for h in HEADS.values():
            v = res["change_vs_keep"].get(f"{key}|{h}")
            cells.append(f"{v['delta']:+.6f} [{v['ci'][0]:+.6f}, {v['ci'][1]:+.6f}] ({v['pct']:+.2f}%)" if v else "-")
        name, subset = key.split("|")
        lines.append(f"| {name} | {subset} | " + " | ".join(cells) + " |")
    lines += ["", "## Small vs big model (% vs big model [95% CI]; change vs today's all-review gap in points)", "",
              "| pair / inputs | subset | curve gap | instant gap | curve change | instant change |",
              "|---|---|---|---|---|---|"]
    for key in sorted({k.rsplit("|", 1)[0] for k in res["gap_pct"]}):
        g = [res["gap_pct"].get(f"{key}|{h}") for h in HEADS.values()]
        c = [res["gap_change_pp"].get(f"{key}|{h}") for h in HEADS.values()]
        gc = [f"{v['gap_pct']:+.2f}% [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]" if v else "-" for v in g]
        cc_ = [f"{v['pp']:+.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]" if v else "-" for v in c]
        name, subset = key.split("|")
        lines.append(f"| {name} | {subset} | " + " | ".join(gc + cc_) + " |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
