"""Today's models on donor data (SM178): read-out. CPU, a few seconds, well under 1 GB.

  python sm178_report.py report   -> sm178-result.json, sm178-result.md (tables for RESULT.md); exit 2 if coverage < 95%

Rows: donors = gpu/sm178-rows.jsonl (REF w8+s8hx8, 218 donor users); 732 set = SM148's REF rows
(wt/sm164/artifacts/small-model/sm148/gpu/sm148-rows-732.jsonl), the 732-pool users with a row for all six models.
Statistic: benchmark LogLoss, review-weighted (ratio of sums over the benchmark test reviews), user bootstrap 2,000 draws,
seed 0, 95% CI. Size comparisons are paired (same users): Small / Tiny vs the matching Big model at REF, in % of Big's
LogLoss. Instant-Only models: instant head only (their curve head is not trained). Aggregates only; no donor is named.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROWS = HERE / "gpu" / "sm178-rows.jsonl"
ROWS732 = Path("C:/Users/jonat/workspace/wt/sm164/artifacts/small-model/sm148/gpu/sm148-rows-732.jsonl")
SOURCES = Path("C:/Users/jonat/workspace/anki-rwkv/data/donations-20261126-timefix/sources.parquet")
REF = "w8+s8hx8@e2s-query"
MODELS = [  # tag, plain name, heads, matching Big tag
    ("sm91s2-big-showtime", "Big two-head (SM91s2)", ("ahead", "instant"), None),
    ("sm99-big-inst-e2", "Big Instant-Only (SM99)", ("instant",), None),
    ("sm118-e26-dec2ep32k-t91s2", "Small two-head (SM118)", ("ahead", "instant"), "sm91s2-big-showtime"),
    ("sm113-inst-e35st-st-dec2ep32k-t99", "Small Instant-Only (SM113)", ("instant",), "sm99-big-inst-e2"),
    ("sm130-th-dec2ep32k-t91s2", "six-layer Tiny two-head (SM130)", ("ahead", "instant"), "sm91s2-big-showtime"),
    ("sm130-io-dec2ep32k-t99", "six-layer Tiny Instant-Only (SM130)", ("instant",), "sm99-big-inst-e2"),
]
HEAD = {"ahead": "curve", "instant": "instant"}
SRC_ORDER = ("campaign", "LMSherlock", "Expertium")


def load(path, users=None):
    out = {}
    for raw in open(path, "rb"):
        try:
            r = json.loads(raw.decode("utf-8", errors="replace"))
        except ValueError:
            continue
        if "benchmark" not in r or not r["label"].endswith(REF) or (users is not None and r["user_id"] not in users):
            continue
        b = r["benchmark"]
        if not all(np.isfinite(b[h]["LogLoss"]) for h in HEAD):
            continue
        out.setdefault(r["label"].split("/")[0], {})[r["user_id"]] = r
    return out


def arr(rows, users, head):
    ll = np.array([rows[u]["benchmark"][head]["LogLoss"] for u in users])
    w = np.array([rows[u]["benchmark"]["size"] for u in users], float)
    return ll, w


def level(rows, users, head):
    if not users:
        return None
    ll, w = arr(rows, users, head)
    idx = np.random.default_rng(0).integers(0, len(users), size=(2000, len(users)))
    boot = (ll[idx] * w[idx]).sum(1) / w[idx].sum(1)
    return {"users": len(users), "reviews": int(w.sum()), "logloss": round(float((ll * w).sum() / w.sum()), 6),
            "ci": [round(float(np.percentile(boot, q)), 6) for q in (2.5, 97.5)]}


def paired(new, big, users, head):
    a, w = arr(new, users, head)
    c, _ = arr(big, users, head)
    idx = np.random.default_rng(0).integers(0, len(users), size=(2000, len(users)))
    boot = ((a - c)[idx] * w[idx]).sum(1) / (c[idx] * w[idx]).sum(1) * 100
    return {"users": len(users), "pct": round(float(((a - c) * w).sum() / (c * w).sum() * 100), 3),
            "ci": [round(float(np.percentile(boot, q)), 3) for q in (2.5, 97.5)]}


def profile(rows, users):
    """Benchmark-set shape (from the Big two-head rows): reviews per user, deleted-card share, prior-answer shares."""
    if not users:
        return {"users": 0, "benchmark_reviews": 0, "median_per_user": 0, "users_1M_plus_reviews": 0,
                "deleted_card_share_pct": 0, "prior_16_plus_pct": 0, "prior_32_plus_pct": 0}
    size = np.array([rows[u]["benchmark"]["size"] for u in users], float)
    dele = np.array([rows[u].get("benchmark_deleted", {}).get("size", 0) for u in users], float)
    n = np.zeros(10)
    for u in users:
        n += np.array(rows[u]["prior_buckets"]["benchmark"]["ahead"]["n"], float)
    edges = rows[users[0]]["prior_buckets"]["edges"]
    tot = n.sum()
    share = lambda lo: float(n[[i for i, e in enumerate(edges) if e >= lo]].sum() / tot * 100)  # noqa: E731
    return {"users": len(users), "benchmark_reviews": int(size.sum()), "median_per_user": int(np.median(size)),
            "mean_per_user": int(size.mean()), "users_1M_plus_reviews": int(sum(rows[u]["reviews"] >= 1_000_000 for u in users)),
            "deleted_card_share_pct": round(float(dele.sum() / size.sum() * 100), 2),
            "prior_16_plus_pct": round(share(16), 2), "prior_32_plus_pct": round(share(32), 2)}


def fmt(x):
    if x is None:
        return "-"
    return f"{x['logloss']:.4f} [{x['ci'][0]:.4f}, {x['ci'][1]:.4f}]"


def fpc(x):
    return f"{x['pct']:+.2f}% [{x['ci'][0]:+.2f}, {x['ci'][1]:+.2f}]"


def report():
    import pandas as pd
    want = set(json.loads((HERE / "users-all.json").read_text())["users"])
    src = dict(pd.read_parquet(SOURCES, columns=["user_id", "source"]).itertuples(index=False, name=None))
    don = load(ROWS, want)
    sys.path.insert(0, "C:/Users/jonat/workspace/wt/sm164")
    from scripts.eval_pools import pool_users
    s732 = load(ROWS732, set(pool_users(732)))
    cover = {t: len(don.get(t, {})) for t, *_ in MODELS}
    common = sorted(set.intersection(*(set(don.get(t, {})) for t, *_ in MODELS)))
    c732 = sorted(set.intersection(*(set(s732[t]) for t, *_ in MODELS)))
    res = {"coverage": cover, "donor_users_all_models": len(common), "set732_users_all_models": len(c732),
           "missing_donor_users": len(want) - len(common), "models": {}, "by_source": {}, "sizes": {}, "profile": {}}
    for tag, name, heads, big in MODELS:
        m = res["models"][tag] = {"name": name}
        for h in heads:
            d, s = level(don[tag], common, h), level(s732[tag], c732, h)
            m[HEAD[h]] = {"donors": d, "set732": s, "donors_vs_732_pct": round((d["logloss"] / s["logloss"] - 1) * 100, 2)}
            for so in SRC_ORDER:
                us = [u for u in common if src.get(u) == so]
                res["by_source"].setdefault(so, {}).setdefault(tag, {})[HEAD[h]] = level(don[tag], us, h)
            if big:
                res["sizes"].setdefault(tag, {})[HEAD[h]] = {"donors": paired(don[tag], don[big], common, h),
                                                            "set732": paired(s732[tag], s732[big], c732, h)}
    res["profile"]["set732"] = profile(s732["sm91s2-big-showtime"], c732)
    res["profile"]["donors"] = profile(don["sm91s2-big-showtime"], common)
    for so in SRC_ORDER:
        res["profile"][so] = profile(don["sm91s2-big-showtime"], [u for u in common if src.get(u) == so])
    (HERE / "sm178-result.json").write_text(json.dumps(res, indent=1) + "\n")

    L = ["# Today's models on donor data (SM178): tables", "",
         f"Donors: {len(common)} of 218 users with a finite REF row for all six models (coverage per model {cover}); "
         f"732 set: {len(c732)} users with SM148 REF rows for all six. Benchmark LogLoss, review-weighted, user bootstrap "
         "2,000 draws, 95% CI.", "",
         "## Per model and head", "",
         "| model | head | donors | 732 set | donors vs 732 |", "|---|---|---|---|---|"]
    for tag, name, heads, _ in MODELS:
        for h in heads:
            x = res["models"][tag][HEAD[h]]
            L.append(f"| {name} | {HEAD[h]} | {fmt(x['donors'])} | {fmt(x['set732'])} | {x['donors_vs_732_pct']:+.2f}% |")
    L += ["", "## By source (donors)", "", "| model | head | " + " | ".join(
        f"{so} ({res['profile'][so]['users']} users)" for so in SRC_ORDER) + " |", "|---|---|---|---|---|"]
    for tag, name, heads, _ in MODELS:
        for h in heads:
            L.append(f"| {name} | {HEAD[h]} | " + " | ".join(fmt(res["by_source"][so][tag][HEAD[h]]) for so in SRC_ORDER) + " |")
    L += ["", "## Sizes: Small / Tiny vs the matching Big model (paired, % of Big's LogLoss)", "",
          "| model | head | donors | 732 set |", "|---|---|---|---|"]
    for tag, name, heads, big in MODELS:
        if big:
            for h in heads:
                x = res["sizes"][tag][HEAD[h]]
                L.append(f"| {name} | {HEAD[h]} | {fpc(x['donors'])} | {fpc(x['set732'])} |")
    L += ["", "## Benchmark-set shape (Big two-head rows)", "",
          "| set | users | benchmark reviews | median per user | users 1M+ reviews | deleted-card share | 16+ prior answers | 32+ prior answers |",
          "|---|---|---|---|---|---|---|---|"]
    for k in ("set732", "donors") + SRC_ORDER:
        p = res["profile"][k]
        L.append(f"| {k} | {p['users']} | {p['benchmark_reviews']:,} | {p['median_per_user']:,} | {p['users_1M_plus_reviews']} | "
                 f"{p['deleted_card_share_pct']:.2f}% | {p['prior_16_plus_pct']:.2f}% | {p['prior_32_plus_pct']:.2f}% |")
    (HERE / "sm178-result.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0 if len(common) >= 0.95 * len(want) else 2


if __name__ == "__main__":
    if sys.argv[1:] == ["report"]:
        sys.exit(report())
    raise SystemExit(__doc__)
