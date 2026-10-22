"""SM58 disagreement test, small model vs big model: UM+ and slope in the Search Stats Extended conventions.

Definitions are copied from anki-rwkv/artifacts/collection-eval/2026-10-14/graphs/make_graphs.py (see its README.md for the
add-on source references): 41 buckets round(20 * d) with d = p_small - p_big, UM+ = sqrt(sum_b n_b * err_b^2 / N) over all
buckets, slope = count-weighted regression of bucket error on bucket disagreement over buckets with >= 200 reviews.
Slot mapping: big model in the RWKV slot (seagreen), small model in the FSRS slot (steelblue).

Inputs
  benchmark users: per-review dumps from small_model_heldout.py --dump (review_th, bin, y, p_ahead, p_instant), label
    <tag>/w8+s8 (deployed w8 weights + int8 state, full-history protocol); eligibility needs the original revlog
    (elapsed_seconds >= 86400 at row review_th - 1; the first review of a card has elapsed_seconds -1).
  own collection: predictions parquets (review_th, rating, p_curve, p_instant), full history, float32 CPU. Aggregate
    output only: this script never writes a per-review or per-card row for the collection.
Samples: "all" = all benchmark test reviews; "cross_day" = the add-on's eligibility (not the first review, >= 1 day since the
previous review).
Paired user bootstrap (benchmark): users resampled with replacement, 2,000 resamples, seed 0; UM+ is recomputed from the pooled
bucket statistics of each resample.

  python sm58_umplus.py --out-dir DIR [--models sm36[,sm55]] [--skip-collection] [--skip-benchmark]
  (self-test / dry run: --selftest)
"""
import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SM = HERE.parent
MAIN = Path(r"C:\Users\jonat\workspace\anki-rwkv")
EVAL = MAIN / "artifacts" / "collection-eval" / "2026-10-14"
ORIG = MAIN / "data" / "anki-revlogs-10k-original-20261011"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BINS_PER_UNIT, SMALL, NB = 20, 200, 41
HEADS = {"curve": ("p_ahead", "p_curve", "Curve head"), "instant": ("p_instant", "p_instant", "Instant head")}
SAMPLES = {"all": "All benchmark test reviews", "cross_day": "Reviews >= 1 day after the previous one (add-on eligibility)"}
EPS = 1e-7
SM24_DUMP = Path("C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/sm24-dump")  # fork dumps, 464 users


# ---------------------------------------------------------------------------------------- UM+ (bucket statistics)
def bucket_stats(ps, pb, y):
    """(41, 4) array: count, sum d, sum (p_small - y), sum (p_big - y) per bucket (d = p_small - p_big)."""
    d = ps - pb
    idx = np.clip(np.floor(d * BINS_PER_UNIT + 0.5) + BINS_PER_UNIT, 0, NB - 1).astype(int)  # Math.round
    out = np.empty((NB, 4))
    for k, w in enumerate((None, d, ps - y, pb - y)):
        out[:, k] = np.bincount(idx, weights=w, minlength=NB)
    return out


def um_from_stats(stats):
    """Vectorised over leading dims: stats (..., 41, 4) -> dict of UM+ and slope for the small (a) and big (b) model."""
    n = stats[..., 0]
    N = n.sum(-1)
    safe = np.where(n > 0, n, 1.0)
    dis, ea, eb = (stats[..., k] / safe for k in (1, 2, 3))
    res = {"um_a": np.sqrt((n * ea ** 2).sum(-1) / N), "um_b": np.sqrt((n * eb ** 2).sum(-1) / N)}
    w = np.where(n >= SMALL, n, 0.0)
    ws = w.sum(-1)
    wsafe = np.where(ws > 0, ws, 1.0)
    mx = (w * dis).sum(-1) / wsafe
    var = (w * (dis - mx[..., None]) ** 2).sum(-1)
    vsafe = np.where(var > 1e-12, var, 1.0)
    for m, e in (("a", ea), ("b", eb)):
        mv = (w * e).sum(-1) / wsafe
        cov = (w * (dis - mx[..., None]) * (e - mv[..., None])).sum(-1)
        res[f"slope_{m}"] = np.where(var > 1e-12, cov / vsafe, np.nan)
    return res


def bins_table(stats):
    n = stats[:, 0]
    keep = n > 0
    return [{"index": int(i), "count": int(n[i]), "disagreement": round(float(stats[i, 1] / n[i]), 6),
             "error_small": round(float(stats[i, 2] / n[i]), 6), "error_big": round(float(stats[i, 3] / n[i]), 6)}
            for i in np.flatnonzero(keep)]


def loss_sums(ps, pb, y):
    f = lambda p: float(-(y * np.log(np.clip(p, EPS, 1)) + (1 - y) * np.log(np.clip(1 - p, EPS, 1))).sum())
    return np.array([len(y), f(ps), f(pb)])


# ------------------------------------------------------------------------------------------------- data loading
def load_benchmark(users, small_tag, small_dir, big_dirs):
    """Per user: for each head and sample, (bucket stats, loss sums). Returns (users used, {key: array}, skipped)."""
    used, skipped = [], {}
    per = {(h, s): ([], []) for h in HEADS for s in SAMPLES}
    for u in users:
        sf = Path(small_dir) / f"{u}_{small_tag}_w8+s8.npz"
        bf = next((Path(d) / f"{u}_fork_w8+s8.npz" for d in big_dirs if (Path(d) / f"{u}_fork_w8+s8.npz").exists()), None)
        if not sf.exists() or bf is None:
            skipped[u] = "no small dump" if not sf.exists() else "no big dump"
            continue
        s, b = np.load(sf), np.load(bf)
        if not (np.array_equal(s["review_th"], b["review_th"]) and np.array_equal(s["y"], b["y"])):
            skipped[u] = "review_th / y mismatch between the two dumps"
            continue
        ths = s["review_th"]
        el = pd.read_parquet(ORIG / "revlogs" / f"user_id={u}", columns=["elapsed_seconds"])["elapsed_seconds"].to_numpy()
        elig = el[ths - 1] >= 86400
        y = s["y"].astype(float)
        used.append(u)
        for h, (col, _, _) in HEADS.items():
            ps, pb = s[col].astype(float), b[col].astype(float)
            for key, mask in (("all", slice(None)), ("cross_day", elig)):
                per[(h, key)][0].append(bucket_stats(ps[mask], pb[mask], y[mask]))
                per[(h, key)][1].append(loss_sums(ps[mask], pb[mask], y[mask]))
    return used, {k: (np.array(v[0]), np.array(v[1])) for k, v in per.items()}, skipped


def load_collection(small_path):
    import lmdb
    import torch
    env = lmdb.open(str(EVAL / "collection-label-db"), readonly=True, lock=False)
    with env.begin() as txn:
        ths = torch.load(io.BytesIO(txn.get("1_review_ths".encode())), weights_only=True).tolist()
    env.close()
    revlogs = pd.read_parquet(EVAL / "collection-dataset" / "revlogs" / "user_id=1")
    revlogs["review_th"] = np.arange(1, len(revlogs) + 1)
    big = pd.read_parquet(EVAL / "preds-model-B-full-s0.parquet").rename(columns={"p_curve": "b_curve", "p_instant": "b_instant"})
    small = pd.read_parquet(small_path).rename(columns={"p_curve": "a_curve", "p_instant": "a_instant", "rating": "rating_a"})
    data = (pd.DataFrame({"review_th": ths}).merge(big, on="review_th").merge(small, on="review_th")
            .merge(revlogs[["review_th", "elapsed_seconds", "rating"]].rename(columns={"rating": "rating_revlog"}), on="review_th"))
    assert len(data) == len(ths), "not all test reviews have both predictions"
    assert (data.rating == data.rating_a).all() and (data.rating + 1 == data.rating_revlog).all(), "review_th mismatch"
    data["y"] = (data.rating > 0).astype(float)
    out = {}
    for h in HEADS:
        for key, frame in (("all", data), ("cross_day", data[data.elapsed_seconds >= 86400])):
            ps, pb, y = (frame[f"a_{h}"].to_numpy(float), frame[f"b_{h}"].to_numpy(float), frame.y.to_numpy())
            out[(h, key)] = (bucket_stats(ps, pb, y), loss_sums(ps, pb, y))
    return out


# --------------------------------------------------------------------------------------------------- summaries
def summarise(stats, loss):
    um = um_from_stats(stats)
    ll_a, ll_b = loss[1] / loss[0], loss[2] / loss[0]
    return {"reviews": int(loss[0]), "small": {"UM+": float(um["um_a"]), "slope": float(um["slope_a"]), "LogLoss": float(ll_a)},
            "big": {"UM+": float(um["um_b"]), "slope": float(um["slope_b"]), "LogLoss": float(ll_b)},
            "um_plus_diff_small_minus_big": float(um["um_a"] - um["um_b"]),
            "logloss_delta_pct_small_vs_big": float(100 * (ll_a / ll_b - 1)), "um_plus_bins": bins_table(stats)}


def bootstrap(per_user_stats, per_user_loss, resamples=2000, seed=0):
    rng = np.random.default_rng(seed)
    U = len(per_user_stats)
    flat = per_user_stats.reshape(U, -1)
    lflat = per_user_loss
    diffs, slopes, lls = [], [], []
    for start in range(0, resamples, 250):
        m = min(250, resamples - start)
        W = np.stack([np.bincount(rng.integers(0, U, U), minlength=U) for _ in range(m)]).astype(float)
        st = (W @ flat).reshape(m, NB, 4)
        ls = W @ lflat
        um = um_from_stats(st)
        diffs.append(um["um_a"] - um["um_b"])
        slopes.append(um["slope_a"])
        lls.append(100 * ((ls[:, 1] / ls[:, 0]) / (ls[:, 2] / ls[:, 0]) - 1))
    q = lambda a: [float(np.nanquantile(np.concatenate(a), 0.025)), float(np.nanquantile(np.concatenate(a), 0.975))]
    return {"resamples": resamples, "seed": seed, "users": U, "um_plus_diff_ci95": q(diffs), "slope_small_ci95": q(slopes),
            "logloss_delta_pct_ci95": q(lls)}


# ---------------------------------------------------------------------------------------------------- drawing
sys.path.insert(0, str(HERE))


def drawing():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter
    return plt, Line2D, FixedLocator, FuncFormatter


def signed_percent(v, digits=0):
    v = 0 if abs(v) < 1e-12 else v
    return f"{'+' if v > 0 else ''}{v * 100:.{digits}f}%"


def rounded_tenth_domain(values, fallback):
    if len(values) == 0:
        return fallback
    lower = max(-1, np.floor((min(values) + 1e-10) / 0.1) * 0.1)
    upper = min(1, np.ceil((max(values) - 1e-10) / 0.1) * 0.1)
    if abs(upper - lower) < 1e-10:
        lower, upper = max(-1, lower - 0.1), min(1, upper + 0.1)
    return round(lower, 1), round(upper, 1)


def ticks_by_tenth(domain):
    return [round(k / 10, 1) for k in range(int(np.ceil(domain[0] * 10 - 1e-10)), int(np.floor(domain[1] * 10 + 1e-10)) + 1)]


def draw_um_plus(ax, stats, names, width_in):
    """renderUmPlusGraph as in make_graphs.py draw_um_plus (small model = FSRS slot a, big model = RWKV slot b)."""
    plt, Line2D, FixedLocator, FuncFormatter = drawing()
    n = stats[:, 0]
    keep = n >= SMALL
    idx = np.flatnonzero(keep)
    dis, ea, eb, cnt = stats[idx, 1] / n[idx], stats[idx, 2] / n[idx], stats[idx, 3] / n[idx], n[idx]
    um = um_from_stats(stats)
    N = n.sum()
    xdom = rounded_tenth_domain(dis.tolist(), (-1, 1))
    ylow, yhigh = rounded_tenth_domain([0, *ea, *eb], (-0.1, 0.1))
    ydom = (-0.1 if ylow == 0 else ylow, 0.1 if yhigh == 0 else yhigh)
    volume = ax.twinx()
    half = 0.5 / BINS_PER_UNIT
    for i, c in zip(idx, cnt):
        centre = (i - BINS_PER_UNIT) / BINS_PER_UNIT
        lo, hi = max(xdom[0], centre - half), min(xdom[1], centre + half)
        volume.bar(lo, c, width=hi - lo, align="edge", color="black", alpha=0.12, linewidth=0)
    volume.set_ylim(0, max(cnt.max(), 1))
    volume.set_ylabel("Paired reviews")
    volume.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"))
    volume.locator_params(axis="y", nbins=4)
    ax.set_zorder(volume.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_xlim(xdom)
    ax.set_ylim(ydom)
    ax.xaxis.set_major_locator(FixedLocator(ticks_by_tenth(xdom)))
    ax.yaxis.set_major_locator(FixedLocator(ticks_by_tenth(ydom)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: signed_percent(v)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: signed_percent(v)))
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.axhline(0, color="black", linewidth=1.5)
    if xdom[0] <= 0 <= xdom[1]:
        ax.axvline(0, color="black", alpha=0.35, linewidth=1)
    scale = width_in * 72 / 600
    handles = []
    for m, e, colour in (("a", ea, "steelblue"), ("b", eb, "seagreen")):
        ax.plot(dis, e, color=colour, linewidth=1.5)
        radius = np.maximum(2.5, np.sqrt(cnt / N * 1024 / np.pi))
        ax.scatter(dis, e, s=(2 * radius * scale) ** 2, color=colour, alpha=0.45, linewidths=0)
        slope = float(um[f"slope_{m}"])
        text = "n/a" if np.isnan(slope) else f"{'+' if slope >= 5e-4 else ''}{0 if abs(slope) < 5e-4 else slope:.3f}"
        handles.append(Line2D([], [], color=colour, linewidth=2,
                              label=f"{names[m]}, UM+={float(um[f'um_{m}']):.4f}, slope={text}"))
    ax.legend(handles=handles, loc="upper left", fontsize=8, frameon=False)
    ax.set_xlabel("Difference in predicted p(recall): small minus big")
    ax.set_ylabel("B-W Metric")
    for a in (ax, volume):
        for spine in a.spines.values():
            spine.set_color("#999999")
        a.tick_params(colors="#666666", labelsize=8)
        a.yaxis.label.set_color("#444444")
        a.xaxis.label.set_color("#444444")


def figure(title, panels, names, path):
    plt, *_ = drawing()
    width_in, height_in = 6.0, 6.0 * 325 / 600
    fig = plt.figure(figsize=(2 * width_in + 3.2, height_in + 1.45), dpi=110)
    fig.suptitle(title, fontsize=11, y=0.985)
    fw, fh = fig.get_size_inches()
    for k, (subtitle, stats) in enumerate(panels):
        left = 0.8 + k * (width_in + 1.6)
        ax = fig.add_axes([left / fw, 0.6 / fh, width_in / fw, height_in / fh])
        draw_um_plus(ax, stats, names, width_in)
        ax.set_title(subtitle, fontsize=9, color="#333333")
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------ main
def fmt_slope(s):
    return "n/a" if s is None or (isinstance(s, float) and np.isnan(s)) else f"{s:+.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(SM / "sm58"))
    ap.add_argument("--models", default="sm36")
    ap.add_argument("--users-file", default=str(SM / "sm58" / "users.json"))
    ap.add_argument("--big-dumps", default=f"{SM24_DUMP},{SM / 'sm58' / 'dump'}")
    ap.add_argument("--small-dump-dir", default=str(SM / "sm58" / "dump"))
    ap.add_argument("--collection-cache", default=str(HERE / "collection-cache"))
    ap.add_argument("--skip-collection", action="store_true")
    ap.add_argument("--skip-benchmark", action="store_true")
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--tag-map", default="", help="selftest: model=dumptag,... (use another dump label as the small model)")
    ap.add_argument("--collection-map", default="", help="selftest: model=path.parquet,...")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    models = [m for m in args.models.split(",") if m]
    tagmap = dict(kv.split("=") for kv in args.tag_map.split(",") if kv)
    cmap = dict(kv.split("=", 1) for kv in args.collection_map.split(",") if kv)
    NAMES = {"sm36": "Muon best small model (SM36)", "sm55": "8k small model (SM55)"}
    summary = {"definition": {"buckets": "round(20*d), d = p_small - p_big, 41 buckets", "UM+": "sqrt(sum n_b err_b^2 / N)",
                              "slope": "count-weighted regression of bucket error on disagreement, buckets >= 200 reviews",
                              "note": "slope_small = slope_big + 1 exactly in this construction: +0.5 for the small model (-0.5 for the big one) is a tie",
                              "logloss": f"pooled review-weighted, p clipped to [{EPS}, 1-{EPS}]"},
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "sets": {}}
    figures = []
    sets = {}
    if not args.skip_benchmark:
        users = json.loads(Path(args.users_file).read_text())["users"]
        sets["benchmark"] = {"label": "732-user pool (benchmark users, w8+s8 deployed variants)", "models": {}, "pool_users": len(users)}
        for m in models:
            used, per, skipped = load_benchmark(users, tagmap.get(m, m), args.small_dump_dir, args.big_dumps.split(","))
            entry = {"users_used": len(used), "users_skipped": {str(k): v for k, v in skipped.items()}, "heads": {}}
            for h in HEADS:
                entry["heads"][h] = {}
                for key in SAMPLES:
                    st, ls = per[(h, key)]
                    tot = summarise(st.sum(0), ls.sum(0))
                    tot["bootstrap"] = bootstrap(st, ls, args.resamples, 0)
                    entry["heads"][h][key] = tot
            sets["benchmark"]["models"][m] = entry
    if not args.skip_collection:
        sets["collection"] = {"label": "user's own collection (aggregate only; full history, float32 CPU)", "models": {}}
        for m in models:
            path = Path(cmap.get(m, Path(args.collection_cache) / f"preds-{m}-full-s0.parquet"))
            if not path.exists():
                sets["collection"]["models"][m] = {"error": f"missing {path.name}"}
                continue
            per = load_collection(path)
            sets["collection"]["models"][m] = {"heads": {h: {k: summarise(*per[(h, k)]) for k in SAMPLES} for h in HEADS}}
    summary["sets"] = sets
    # graphs: rebuild each panel from the bins table (stats array) to keep one code path
    for sname, sset in sets.items():
        for m, entry in sset["models"].items():
            if "heads" not in entry:
                continue
            names = {"a": f"{NAMES.get(m, m)}", "b": "Big model (fork RWKV)"}
            for h, (_, _, title) in HEADS.items():
                panels = []
                for key, label in SAMPLES.items():
                    r = entry["heads"][h][key]
                    st = np.zeros((NB, 4))
                    for b in r["um_plus_bins"]:
                        n = b["count"]
                        st[b["index"]] = [n, b["disagreement"] * n, b["error_small"] * n, b["error_big"] * n]
                    panels.append((f"{label}: {r['reviews']:,} reviews", st))
                path = out / f"um-plus-{sname}-{m}-{h}.png"
                figure(f"Small vs big UM+, {title.lower()} ({sset['label']})", panels, names, path)
                figures.append((sname, m, h, path))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    write_md(out / "summary.md", summary, NAMES)
    write_html(out / "graphs.html", summary, figures, NAMES)
    print(f"wrote {out}/summary.json, summary.md, graphs.html and {len(figures)} PNGs")


def write_md(path, summary, names):
    lines = ["# SM58 disagreement test: small vs big model (UM+ and slope)", "",
             "UM+ lower is better; slope closer to 0 is better. In this construction the two slopes always differ by exactly 1 "
             "(slope small = slope big + 1), so a tie is +0.5 for the small model and -0.5 for the big one (the sign of the half depends on which model is x's plus side; here x = p_small - p_big). x = p_small - p_big.", ""]
    for sname, sset in summary["sets"].items():
        lines += [f"## {sset['label']}", ""]
        for m, entry in sset["models"].items():
            if "heads" not in entry:
                lines += [f"- {names.get(m, m)} ({m}): {entry.get('error')}", ""]
                continue
            extra = f" ({entry['users_used']} users used; skipped: {sorted(entry['users_skipped']) or 'none'})" if sname == "benchmark" else ""
            lines += [f"### {names.get(m, m)} ({m}) vs big model{extra}", "",
                      "| Head | Sample | Reviews | UM+ small | slope small | UM+ big | slope big | UM+ diff (small - big) [95% CI] | "
                      "LogLoss small | LogLoss big | LogLoss delta [95% CI] |", "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|"]
            for h in HEADS:
                for key in SAMPLES:
                    r = entry["heads"][h][key]
                    bt = r.get("bootstrap")
                    ci = lambda v: f" [{v[0]:+.4f}, {v[1]:+.4f}]" if bt else ""
                    cip = f" [{bt['logloss_delta_pct_ci95'][0]:+.2f}%, {bt['logloss_delta_pct_ci95'][1]:+.2f}%]" if bt else ""
                    lines.append(f"| {h} | {key} | {r['reviews']:,} | {r['small']['UM+']:.4f} | {fmt_slope(r['small']['slope'])} | "
                                 f"{r['big']['UM+']:.4f} | {fmt_slope(r['big']['slope'])} | {r['um_plus_diff_small_minus_big']:+.4f}"
                                 f"{ci(bt['um_plus_diff_ci95']) if bt else ''} | {r['small']['LogLoss']:.4f} | "
                                 f"{r['big']['LogLoss']:.4f} | {r['logloss_delta_pct_small_vs_big']:+.2f}%{cip} |")
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_html(path, summary, figures, names):
    md = (path.parent / "summary.md").read_text(encoding="utf-8")
    import html as H
    figs = "".join(f'<figure><img alt="{p.stem}" src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}">'
                   f'<figcaption><b>{p.name}</b>: {H.escape(names.get(m, m))} vs the big model, {h} head, {s}. Reviews are grouped '
                   f'by the disagreement x = p<sub>small</sub> - p<sub>big</sub> (5-point buckets); the y axis is each model\'s mean '
                   f'prediction minus the observed recall rate. UM+ and slope as in Search Stats Extended; buckets under 200 '
                   f'reviews hidden.</figcaption></figure>' for s, m, h, p in figures)
    path.write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SM58 small vs big UM+</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;margin:0 auto;max-width:1400px;padding:16px;color:#1f2328;background:#fff}}
img{{max-width:100%;height:auto;border:1px solid #d0d7de}}figure{{margin:24px 0}}figcaption{{color:#57606a;font-size:13px;max-width:1100px}}
pre{{overflow-x:auto;font-size:12px;background:#f6f8fa;padding:8px}}</style></head><body>
<h1>Disagreement test: small model vs big model (SM58)</h1>
<p>Generated {summary['utc']}. Tables (markdown):</p><pre>{H.escape(md)}</pre>{figs}</body></html>
""", encoding="utf-8")


if __name__ == "__main__":
    main()
