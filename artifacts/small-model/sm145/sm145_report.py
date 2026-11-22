#!/usr/bin/env python
"""Int3 young tier on the early switch (SM145): GPU smoke check, the pre-registered rules, the go decision, the extra
cost vs AN16 and the "worth its bytes" recommendation (design.md "Rules", fixed 11-22 before any scoring). CPU only.

  python sm145_report.py smoke --gpu gpu/smoke-gpu-rows.jsonl --cpu cpu/smoke-cpu-rows.jsonl   exit 0 = smoke OK
  python sm145_report.py report      gpu/sm145-rows-732.jsonl + cpu/sm145-longcard-rows.jsonl + sm145-dr.json
                                     -> sm145-result.{json,md}; exit 0 = decided, 2 = pending (coverage), 3 = bucket mismatch
  python sm145_report.py selftest    synthetic rows
  python sm145_report.py check       pins (showtime_report.py sha256, users files, bytes)

Statistic: % = 100 x (sum of the label's LogLoss / sum of the reference's LogLoss - 1), review-weighted, paired user
bootstrap 2,000 draws, seed 0, 95% CI. Dataset-wide (R2) on the benchmark LogLoss of the 722 rule users of the 732 set
(showtime_report.compare, as every 732 readout); buckets (R3, R4) on the "all" scope of --prior-buckets (every review
the scorer predicts), by the answers of the card before the review (= the updates of the state the review reads).
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
G, C = HERE / "gpu", HERE / "cpu"
W = Path("C:/Users/jonat/workspace/wt")
REPORT = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/small-model/showtime/showtime_report.py")
REPORT_SHA = "3cdfed03afd40734d4b1931c244e8c1e73453d281a40c3843e80cd310d70dd9d"
U732 = W / "sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json"
ULONG = HERE / "users-longcard.json"
ROWS732, LONGROWS = G / "sm145-rows-732.jsonl", C / "sm145-longcard-rows.jsonl"
DR = HERE / "sm145-dr.json"
SM144_GPU, SM144_LONG = HERE.parent / "sm144" / "gpu" / "sm144-rows-732.jsonl", HERE.parent / "sm144" / "cpu" / "sm144-longcard-rows.jsonl"
E2S = "@e2s-query"
LABELS = {"REF": "w8+s8hx8", "AN16": "w8+t4.16.8c4.16.8nhx8", "Y2N": "w8+t3.3.4.16.8c3.3.4.16.8nhx8",
          "Y4C": "w8+t3.5.4.16.8c4.16.8nhx8", "Y4N": "w8+t3.5.4.16.8c3.5.4.16.8nhx8",
          "U8": "w8+t8.5.8.128.8c8.5.8.128.8nhx8"}
# in ascending state bytes for every size (sm145-bytes.json): the go rule takes the smallest passing label
SCHEMES = ["Y4N", "Y4C", "Y2N", "AN16"]
INT3 = ["Y4N", "Y4C", "Y2N"]
YOUNG = {"Y2N": 2, "Y4C": 4, "Y4N": 4}       # R4: reviews whose card had 1-M earlier answers (the int3 tier)
CEIL_ALL, CEIL_YOUNG = 0.05, 0.10
LONG_POINT, LONG_UPPER = 0.5, 1.0            # R3, the user's 11-22 long-card tolerance (point and CI upper bound)
WORTH = {"bytes_min_pct": 2.0, "extra_point_max": 0.02, "extra_upper_max": 0.03, "card_share_factor": 2.0,
         "card_share_floor": 1e-4, "card_limit": "0.10"}
BUCKETS = ((1, 3), (3, 5), (5, 9), (9, 16), (16, 32), (32, 128), (128, None))   # report only; edges of --prior-buckets
MIN_732, MIN_LONG = 720, 48
TOL_STRICT, TOL_DISCRETE, TOL_DISCRETE_MEAN = 1e-5, 3e-4, 1e-4
TOL_EPS = 1e-9   # the rows store LogLoss to 6 decimals (SM141 smoke fix)
KEY = {"curve": "ahead", "instant": "instant"}
TWO, IO = ("curve", "instant"), ("instant",)
MODELS = {   # key: (tag, plain name, heads, size)
    "sm91s2": ("sm91s2-big-showtime", "the show-time big model (SM91s2)", TWO, "Big"),
    "sm99": ("sm99-big-inst-e2", "the show-time instant-only big model (SM99)", IO, "Big"),
    "sm118": ("sm118-e26-dec2ep32k-t91s2", "the show-time two-head cool-down for Small (SM118)", TWO, "Small"),
    "sm113": ("sm113-inst-e35st-st-dec2ep32k-t99", "the longer Small Instant-Only model (SM113)", IO, "Small"),
    "sm130th": ("sm130-th-dec2ep32k-t91s2", "the six-layer Tiny two-head model (SM130)", TWO, "Tiny"),
    "sm130io": ("sm130-io-dec2ep32k-t99", "the six-layer Tiny Instant-Only model (SM130)", IO, "Tiny"),
}
BYTES = HERE / "sm145-bytes.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def showtime():
    if sha(REPORT) != REPORT_SHA:
        raise SystemExit("[sm145] showtime_report.py changed: refusing")
    spec = importlib.util.spec_from_file_location("showtime_report", REPORT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_rows(path):
    """{label: {user: row}} of benchmark rows, {label: set(users)} of non-finite rows, torn lines skipped."""
    rows, nonfinite, torn = {}, {}, 0
    if Path(path).exists():
        for raw in open(path, "rb"):
            try:
                r = json.loads(raw.decode("utf-8", errors="replace"))
            except ValueError:
                torn += 1
                continue
            if "error" in r:
                continue
            if "benchmark" in r:
                rows.setdefault(r["label"], {})[r["user_id"]] = r
            elif "nonfinite" in r:
                nonfinite.setdefault(r["label"], set()).add(r["user_id"])
    return rows, nonfinite, torn


def as_loss(rows):
    out = {}
    for u, r in rows.items():
        b = r["benchmark"]
        out[u] = {h: ((b[h]["LogLoss"], b[h]["size"]) if h in b else (float("nan"), 0)) for h in ("ahead", "instant")}
    return out


def bucket_sums(row, head, lo, hi=None):
    """(n, ll) of the all-scope reviews of `row` whose card had lo <= prior answers (< hi); bucket edges must align."""
    pb = row["prior_buckets"]
    edges = pb["edges"]
    h = pb["all"].get(head)
    if h is None:
        return 0, 0.0
    if lo not in edges or (hi is not None and hi not in edges):
        raise ValueError(f"bucket {lo}-{hi} does not align with edges {edges}")
    idx = [i for i, e in enumerate(edges) if e >= lo and (hi is None or (edges[i + 1] if i + 1 < len(edges) else math.inf) <= hi)]
    return sum(h["n"][i] for i in idx), sum(h["ll"][i] for i in idx)


class BucketMismatch(Exception):
    pass


def bucket_cmp(new, ref, head, lo, hi=None, draws=2000):
    """Paired user bootstrap of the % cost of `new` vs `ref` ({user: row}) on one bucket; counts must match."""
    users, a, b = [], [], []
    for u in sorted(set(new) & set(ref)):
        n1, l1 = bucket_sums(new[u], head, lo, hi)
        n0, l0 = bucket_sums(ref[u], head, lo, hi)
        if n1 != n0:
            raise BucketMismatch(f"user {u} head {head} bucket {lo}-{hi}: {n1} vs {n0} reviews")
        if n0:
            users.append(u); a.append(l1); b.append(l0)
    if not users:
        return None
    a, b = np.array(a), np.array(b)
    idx = np.random.default_rng(0).integers(0, len(users), size=(draws, len(users)))
    boot = (a[idx].sum(1) / b[idx].sum(1) - 1) * 100
    reviews = sum(bucket_sums(ref[u], head, lo, hi)[0] for u in users)
    return {"value": float((a.sum() / b.sum() - 1) * 100), "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "users": len(users), "reviews": int(reviews), "finite": bool(np.isfinite(a).all())}


def _bench_cmp(new, ref, h, users, draws=2000):
    a = np.array([new[u]["benchmark"][h]["LogLoss"] * new[u]["benchmark"][h]["size"] for u in users])
    b = np.array([ref[u]["benchmark"][h]["LogLoss"] * ref[u]["benchmark"][h]["size"] for u in users])
    idx = np.random.default_rng(0).integers(0, len(users), size=(draws, len(users)))
    boot = (a[idx].sum(1) / b[idx].sum(1) - 1) * 100
    return {"value": float((a.sum() / b.sum() - 1) * 100), "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "users": len(users)}


def dataset_cmp(new, ref, heads, common, st):
    if not common:
        return {}
    if st is not None:
        c = st.compare(as_loss(new), as_loss(ref), as_loss(ref), common)   # keyed by "curve" / "instant" (as in SM144), not by the row key
        return {h: {"value": c[h]["pct_of_ref"], "ci": c[h]["pct_of_ref_ci"], "users": c["users"]} for h in heads}
    return {h: _bench_cmp(new, ref, KEY[h], common) for h in heads}   # selftest: the same ratio-of-sums bootstrap


def fmt(x):
    return "n/a" if x is None else f"{x['value']:+.3f}% [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]"


def smoke(gpu, cpu):
    """GPU vs CPU on the smoke users: REF and U8 strict (1e-5), U8 == REF on the GPU within 2e-6 per field; the tiered
    labels by the discrete-codec rule (per user <= 3e-4, review-weighted mean signed difference <= 1e-4, per head)."""
    g, gnf, _ = read_rows(gpu)
    c, cnf, _ = read_rows(cpu)
    users = json.loads((HERE / "users-smoke.json").read_text())["users"]
    ok, lines, rep = True, [], {}
    for key, (tag, name, heads, size) in MODELS.items():
        for lab in ["REF", "U8", *SCHEMES]:
            L = f"{tag}/{LABELS[lab]}{E2S}"
            if gnf.get(L) or cnf.get(L):
                ok = False; lines.append(f"{L}: non-finite rows GPU {sorted(gnf.get(L, ()))} CPU {sorted(cnf.get(L, ()))}")
                continue
            if set(g.get(L, {})) != set(users) or set(c.get(L, {})) != set(users):
                ok = False; lines.append(f"{L}: rows missing (GPU {sorted(g.get(L, {}))}, CPU {sorted(c.get(L, {}))})")
                continue
            for h in sorted({KEY[x] for x in heads}):
                d = [g[L][u]["benchmark"][h]["LogLoss"] - c[L][u]["benchmark"][h]["LogLoss"] for u in users]
                w = [c[L][u]["benchmark"][h]["size"] for u in users]
                mean = sum(x * y for x, y in zip(d, w)) / sum(w)
                worst = max(abs(x) for x in d)
                if lab in ("REF", "U8"):
                    good = worst <= TOL_STRICT + TOL_EPS
                else:
                    good = worst <= TOL_DISCRETE + TOL_EPS and abs(mean) <= TOL_DISCRETE_MEAN + TOL_EPS
                young = [abs(bucket_sums(g[L][u], h, 1, 5)[1] / max(bucket_sums(g[L][u], h, 1, 5)[0], 1)
                             - bucket_sums(c[L][u], h, 1, 5)[1] / max(bucket_sums(c[L][u], h, 1, 5)[0], 1)) for u in users]
                long = [abs(bucket_sums(g[L][u], h, 128)[1] / max(bucket_sums(g[L][u], h, 128)[0], 1)
                            - bucket_sums(c[L][u], h, 128)[1] / max(bucket_sums(c[L][u], h, 128)[0], 1)) for u in users]
                rep[f"{L} {h}"] = {"max_abs": worst, "mean_signed": mean, "young_1_4_mean_abs_max": max(young),
                                   "long_128p_mean_abs_max": max(long), "pass": good}
                ok &= good
            if lab == "U8":   # the tier kernel with every tier int8 == the stock kernel, on the GPU too (2e-6 per field)
                R = f"{tag}/{LABELS['REF']}{E2S}"

                def close(a, b):
                    if isinstance(a, dict) and isinstance(b, dict):
                        return set(a) == set(b) and all(close(a[k], b[k]) for k in a)
                    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                        return abs(a - b) <= 2e-6
                    return a == b
                exact = all(g[L][u]["benchmark"] == g[R][u]["benchmark"] for u in users if u in g.get(R, {}))
                same = all(close(g[L][u]["benchmark"], g[R][u]["benchmark"]) for u in users if u in g.get(R, {}))
                rep[f"{L} == REF on the GPU"] = same
                rep[f"{L} == REF on the GPU, every digit (report only)"] = exact
                ok &= same
    (G / "smoke-check.json").write_text(json.dumps({"ok": ok, "problems": lines, "checks": rep}, indent=1) + "\n",
                                        encoding="utf-8")
    worst_t = max((v["max_abs"] for k, v in rep.items() if isinstance(v, dict) and "/w8+t" in k and "t8.5" not in k), default=0)
    worst_s = max((v["max_abs"] for k, v in rep.items() if isinstance(v, dict) and ("s8hx8" in k or "t8.5" in k)), default=0)
    print(f"[sm145] smoke {'OK' if ok else 'FAILED'}: {len(rep)} checks; int8 labels max |GPU - CPU| {worst_s:.2e} "
          f"(tol {TOL_STRICT:g}); tiered labels max {worst_t:.2e} (tol {TOL_DISCRETE:g}, mean {TOL_DISCRETE_MEAN:g})"
          + (f"; {lines[:3]}" if lines else ""))
    return 0 if ok else 1


def card_share(dr, key, lab, h):
    """Share of cards whose worst review has |dR| > 0.10 (all cards of the long-card users), or None."""
    try:
        return dr["models"][key]["labels"][lab]["heads"][h]["cards"]["all"][f"share_gt_{WORTH['card_limit']}"]
    except (KeyError, TypeError):
        return None


def decide(rows732, nf732, longrows, longnf, users732, longusers, st=None, bytes_pct=None, dr=None):
    """The rules per model and label, the go per size, the extra cost vs AN16 and the worth-its-bytes recommendation."""
    res = {"ceilings_pct": {"R2_dataset_upper": CEIL_ALL, "R3_long_point": LONG_POINT, "R3_long_upper": LONG_UPPER,
                            "R4_young_upper": CEIL_YOUNG}, "worth_criterion": WORTH,
           "min_users": {"732": MIN_732, "long": MIN_LONG}, "models": {}, "go": {}, "pending": []}
    for key, (tag, name, heads, size) in MODELS.items():
        m = {"name": name, "size": size, "heads": list(heads), "schemes": {}}
        lab = lambda s: f"{tag}/{LABELS[s]}{E2S}"   # noqa: E731
        ref = {u: r for u, r in rows732.get(lab("REF"), {}).items() if u in users732}
        lref = {u: r for u, r in longrows.get(lab("REF"), {}).items() if u in longusers}
        an = {u: r for u, r in rows732.get(lab("AN16"), {}).items() if u in users732}
        lan = {u: r for u, r in longrows.get(lab("AN16"), {}).items() if u in longusers}
        for s in SCHEMES:
            new = {u: r for u, r in rows732.get(lab(s), {}).items() if u in users732}
            lnew = {u: r for u, r in longrows.get(lab(s), {}).items() if u in longusers}
            x = {}
            nonfin = sorted(nf732.get(lab(s), set()) | longnf.get(lab(s), set()))
            common = sorted(set(new) & set(ref))
            lcommon = sorted(set(lnew) & set(lref))
            x["users_732"], x["users_long"] = len(common), len(lcommon)
            if len(common) < MIN_732:
                res["pending"].append(f"{tag} {s}: {len(common)} of {MIN_732} users on the 732 set")
            if len(lcommon) < MIN_LONG:
                res["pending"].append(f"{tag} {s}: {len(lcommon)} of {MIN_LONG} long-card users")
            fin = not nonfin and all(math.isfinite(new[u]["benchmark"][KEY[h]]["LogLoss"]) for u in common for h in heads)
            x["R1_finite"], x["nonfinite_users"] = fin, nonfin
            r2 = dataset_cmp(new, ref, heads, common, st)
            x["R2_dataset"] = r2
            x["R2"] = bool(r2) and all(r2[h]["ci"][1] <= CEIL_ALL for h in heads)
            r3 = {h: bucket_cmp(lnew, lref, KEY[h], 128) for h in heads} if lcommon else {}
            x["R3_long_128p"] = r3
            x["R3"] = bool(r3) and all(r3[h] is not None and r3[h]["finite"] and r3[h]["value"] <= LONG_POINT
                                       and r3[h]["ci"][1] <= LONG_UPPER for h in heads)
            if s in YOUNG:
                r4 = {h: bucket_cmp(new, ref, KEY[h], 1, YOUNG[s] + 1) for h in heads} if common else {}
                x["R4_young"] = r4
                x["R4"] = bool(r4) and all(r4[h] is not None and r4[h]["ci"][1] <= CEIL_YOUNG for h in heads)
            else:
                x["R4"] = True
            x["pass"] = x["R1_finite"] and x["R2"] and x["R3"] and x["R4"]
            # report only: the extra cost vs AN16 (paired), every bucket, the all scope total
            if s in INT3:
                ca = sorted(set(new) & set(an))
                la = sorted(set(lnew) & set(lan))
                x["vs_AN16"] = {"dataset": dataset_cmp(new, an, heads, ca, st),
                                "young": {h: bucket_cmp(new, an, KEY[h], 1, YOUNG[s] + 1) for h in heads} if ca else {},
                                "long_128p": {h: bucket_cmp(lnew, lan, KEY[h], 128) for h in heads} if la else {}}
            x["buckets_gpu"] = {f"{lo}-{hi - 1 if hi else ''}": {h: bucket_cmp(new, ref, KEY[h], lo, hi) for h in heads}
                                for lo, hi in BUCKETS} if common else {}
            x["buckets_long"] = {f"{lo}-{hi - 1 if hi else ''}": {h: bucket_cmp(lnew, lref, KEY[h], lo, hi) for h in heads}
                                 for lo, hi in BUCKETS} if lcommon else {}
            x["all_scope_gpu"] = {h: bucket_cmp(new, ref, KEY[h], 0) for h in heads} if common else {}
            m["schemes"][s] = x
        res["models"][key] = m
    res["status"] = "pending" if res["pending"] else "decided"
    sizes = {}
    for key, m in res["models"].items():
        sizes.setdefault(m["size"], []).append((key, m))
    for size, ms in sizes.items():
        passing = [s for s in SCHEMES if all(m["schemes"][s]["pass"] for _, m in ms)]
        rank = bytes_pct[size] if bytes_pct else {s: i for i, s in enumerate(SCHEMES)}
        best = min(passing, key=lambda s: rank[s]) if passing else "REF"
        worth = {}
        for s in INT3:   # the worth-its-bytes criterion (design.md), per int3 label of the size
            saving = (bytes_pct[size]["AN16"] - bytes_pct[size][s]) if bytes_pct else None
            parts = {"bytes_saving_vs_AN16_pct": saving,
                     "bytes": saving is not None and saving >= WORTH["bytes_min_pct"], "extra": True, "cards": True}
            for key, m in ms:
                va = m["schemes"][s].get("vs_AN16", {}).get("dataset", {})
                for h in m["heads"]:
                    v = va.get(h)
                    if v is None or v["value"] > WORTH["extra_point_max"] or v["ci"][1] > WORTH["extra_upper_max"]:
                        parts["extra"] = False
                    cs, c0 = card_share(dr, key, s, h), card_share(dr, key, "AN16", h)
                    if cs is None or c0 is None:
                        parts["cards"] = None if parts["cards"] is not False else False
                    elif cs > max(WORTH["card_share_factor"] * c0, WORTH["card_share_floor"]):
                        parts["cards"] = False
            parts["worth"] = (s in passing and parts["bytes"] and parts["extra"] and parts["cards"] is True)
            worth[s] = parts
        allowed = [s for s in passing if s == "AN16" or worth[s]["worth"]]
        rec = min(allowed, key=lambda s: rank[s]) if allowed else "REF"
        res["go"][size] = {"passing": passing, "pick": best,
                           "bytes_vs_REF_pct": (bytes_pct[size][best] if bytes_pct and best != "REF" else 0.0),
                           "worth": worth, "recommended": rec,
                           "recommended_bytes_vs_REF_pct": (bytes_pct[size][rec] if bytes_pct and rec != "REF" else 0.0)}
    return res


def bytes_rank():
    d = json.loads(BYTES.read_text())["scheme_bytes_732"]
    return {size: {s: d[s][size]["vs_REF_pct"] for s in SCHEMES} for size in ("Big", "Small", "Tiny")}


def integrity(r732, rl):
    """Report only: REF and AN16 rows vs the earlier int8 switch (SM144)'s rows, GPU (732 set) and CPU (long cards)."""
    out = {}
    g44, _, _ = read_rows(SM144_GPU)
    l44, _, _ = read_rows(SM144_LONG)
    for key, (tag, *_rest) in MODELS.items():
        for s in ("REF", "AN16"):
            L = f"{tag}/{LABELS[s]}{E2S}"
            for name, old, new in (("gpu_732", g44, r732), ("cpu_long", l44, rl)):
                common = set(old.get(L, {})) & set(new.get(L, {}))
                out[f"{key} {s} {name}"] = {"users": len(common), "max_abs_benchmark_logloss": max(
                    (abs(old[L][u]["benchmark"][h]["LogLoss"] - new[L][u]["benchmark"][h]["LogLoss"])
                     for u in common for h in old[L][u]["benchmark"]
                     if isinstance(old[L][u]["benchmark"][h], dict) and isinstance(new[L][u]["benchmark"].get(h), dict)),
                    default=None)}
        L = f"{tag}/{LABELS['REF']}{E2S}"
        diffs = []
        for u in set(rl.get(L, {})) & set(r732.get(L, {})):
            for h in ("ahead", "instant"):
                n1, l1 = bucket_sums(r732[L][u], h, 128)
                n0, l0 = bucket_sums(rl[L][u], h, 128)
                if n0 and n1 == n0:
                    diffs.append(abs(l1 / n1 - l0 / n0))
        out[f"{key} REF gpu_vs_cpu_long_128p_mean_logloss_max_abs"] = max(diffs) if diffs else None
    return out


def report():
    st = showtime()
    users732, _ = st.pools()
    longusers = set(json.loads(ULONG.read_text())["users"])
    r732, nf732, t1 = read_rows(ROWS732)
    rl, nfl, t2 = read_rows(LONGROWS)
    dr = json.loads(DR.read_text()) if DR.exists() else None
    try:
        res = decide(r732, nf732, rl, nfl, set(users732), longusers, st, bytes_rank(), dr)
    except BucketMismatch as e:
        print(f"[sm145] STOP bucket-count mismatch: {e}")
        return 3
    res["torn_lines"] = {"732": t1, "long": t2}
    res["dr_file"] = str(DR.name) if dr else None
    if dr is None or not dr.get("complete"):
        res["pending"].append("per-review |dR| read-out (sm145-dr.json) missing or incomplete")
        res["status"] = "pending"
    res["report_only"] = {"integrity_vs_sm144": integrity(r732, rl)}
    (HERE / "sm145-result.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    write_md(res, dr)
    if res["status"] == "pending":
        print(f"[sm145] report pending: {res['pending'][:6]}")
        return 2
    print("[sm145] report decided: " + "; ".join(
        f"{s} -> {g['pick']} ({g['bytes_vs_REF_pct']:+.1f}% bytes), recommended {g['recommended']} "
        f"({g['recommended_bytes_vs_REF_pct']:+.1f}%)" for s, g in res["go"].items()))
    return 0


def _p(x, k, scale=1.0, f="{:.4f}"):
    return "-" if not x or k not in x else f.format(x[k] * scale)


def write_md(res, dr):
    L = ["# Int3 young tier on the early switch (SM145): result", "",
         f"Status: **{res['status']}**. Rules (design.md, pre-registered 11-22): R1 every user finite; R2 dataset-wide cost vs "
         f"int8 cards and notes (w8+s8hx8), benchmark LogLoss, 95% CI upper bound <= +{CEIL_ALL}% per trained head; R3 "
         f"reviews of cards with 128+ earlier answers on the long-card users (CPU, every label scored directly): point "
         f"estimate <= +{LONG_POINT}% and CI upper bound <= +{LONG_UPPER}% per head (the user's 11-22 tolerance); R4 reviews "
         f"of young cards (1-M earlier answers, the int3 tier) CI upper bound <= +{CEIL_YOUNG}%. Go: per size the label "
         f"with the smallest state bytes that passes for both models; recommended: the same among AN16 and the int3 "
         f"labels that are worth their bytes (design.md).", "",
         "## Go per size", "",
         "| size | passing labels | go pick | bytes vs int8 | recommended | bytes vs int8 |", "|---|---|---|---|---|---|"]
    for size, g in res["go"].items():
        L.append(f"| {size} | {', '.join(g['passing']) or 'none'} | {g['pick']} | {g['bytes_vs_REF_pct']:+.1f}% | "
                 f"{g['recommended']} | {g['recommended_bytes_vs_REF_pct']:+.1f}% |")
    L += ["", "## Worth its bytes (int3 labels; design.md criterion)", "",
          "| size | label | passes R1-R4 | bytes saved vs AN16 (% of int8) | extra cost vs AN16 within limits | per-card share > 0.10 within limits | worth |",
          "|---|---|---|---|---|---|---|"]
    for size, g in res["go"].items():
        for s, w in g["worth"].items():
            sv = w["bytes_saving_vs_AN16_pct"]
            L.append(f"| {size} | {s} | {s in g['passing']} | {'-' if sv is None else f'{sv:.2f}%'} | {w['extra']} | {w['cards']} | {w['worth']} |")
    L += ["", "## Per model and label (% vs int8 cards and notes)", "",
          "| model | label | head | R2 dataset-wide | R3 long cards 128+ (CPU) | R4 young (1-M) | R1 | pass |",
          "|---|---|---|---|---|---|---|---|"]
    for key, m in res["models"].items():
        for s, x in m["schemes"].items():
            for h in m["heads"]:
                r4 = fmt(x.get("R4_young", {}).get(h)) if "R4_young" in x else "-"
                L.append(f"| {m['name']} | {s} | {h} | {fmt(x['R2_dataset'].get(h))} | {fmt(x['R3_long_128p'].get(h))} "
                         f"| {r4} | {x['R1_finite']} | {x['pass']} |")
    L += ["", "## Extra cost of each int3 label vs AN16 (paired, report only; % of AN16's LogLoss)", "",
          "| model | label | head | dataset-wide (GPU) | young 1-M (GPU) | long cards 128+ (CPU) |", "|---|---|---|---|---|---|"]
    for key, m in res["models"].items():
        for s, x in m["schemes"].items():
            if "vs_AN16" not in x:
                continue
            for h in m["heads"]:
                v = x["vs_AN16"]
                L.append(f"| {m['name']} | {s} | {h} | {fmt(v['dataset'].get(h))} | {fmt(v['young'].get(h))} | {fmt(v['long_128p'].get(h))} |")
    for title, key in (("732 set (GPU, all users)", "buckets_gpu"), ("long-card users (CPU)", "buckets_long")):
        L += ["", f"## Cost per answer-count bucket, {title} (% vs int8; card answers before the review)", "",
              "| model | label | head | " + " | ".join(f"{lo}-{hi - 1 if hi else ''}" for lo, hi in BUCKETS) + " |",
              "|---|---|---|" + "---|" * len(BUCKETS)]
        for k, m in res["models"].items():
            for s, x in m["schemes"].items():
                for h in m["heads"]:
                    L.append(f"| {m['name']} | {s} | {h} | " + " | ".join(
                        fmt((x.get(key, {}).get(f"{lo}-{hi - 1 if hi else ''}") or {}).get(h)) for lo, hi in BUCKETS) + " |")
    if dr:
        L += ["", "## Per-review and per-card |dR| vs int8 (long-card users, CPU, every predicted review; report only)", "",
              "|dR| = |predicted R of the label - predicted R of int8 cards and notes| on the same review. Reviews: pooled "
              "over the users. Cards: each card's worst review. Shares in %.", "",
              "| model | head | label | reviews | P50 | P90 | P99 | P99.9 | max | > 0.05 | > 0.10 | cards | card P99 | card max | cards > 0.05 | cards > 0.10 | users with a review > 0.10 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for key, m in dr["models"].items():
            for h in m["heads"]:
                for lab, lb in m["labels"].items():
                    hd = lb["heads"].get(h, {})
                    rv, cd = hd.get("reviews", {}).get("all"), hd.get("cards", {}).get("all")
                    L.append(f"| {m['name']} | {h} | {lab} | {_p(rv, 'n', f='{:.0f}')} | {_p(rv, 'P50')} | {_p(rv, 'P90')} | "
                             f"{_p(rv, 'P99')} | {_p(rv, 'P99.9')} | {_p(rv, 'max')} | {_p(rv, 'share_gt_0.05', 100, '{:.3f}%')} | "
                             f"{_p(rv, 'share_gt_0.10', 100, '{:.4f}%')} | {_p(cd, 'n', f='{:.0f}')} | {_p(cd, 'P99')} | {_p(cd, 'max')} | "
                             f"{_p(cd, 'share_gt_0.05', 100, '{:.3f}%')} | {_p(cd, 'share_gt_0.10', 100, '{:.4f}%')} | "
                             f"{hd.get('users_with_a_review_gt_0.10', '-')} of {hd.get('users', '-')} |")
        names = dr["review_buckets"]
        L += ["", "### |dR| by the card's answers before the review (P99 / max / share > 0.05)", "",
              "| model | head | label | " + " | ".join(names) + " |", "|---|---|---|" + "---|" * len(names)]
        for key, m in dr["models"].items():
            for h in m["heads"]:
                for lab, lb in m["labels"].items():
                    rvs = lb["heads"].get(h, {}).get("reviews", {})
                    L.append(f"| {m['name']} | {h} | {lab} | " + " | ".join(
                        (f"{rvs[nm]['P99']:.4f} / {rvs[nm]['max']:.3f} / {100 * rvs[nm]['share_gt_0.05']:.3f}%" if rvs.get(nm, {}).get("n") else "-")
                        for nm in names) + " |")
        names = dr["card_buckets"]
        L += ["", "### Worst review per card, by the card's answers in the whole history (P99 / max / share of cards > 0.10)", "",
              "| model | head | label | " + " | ".join(names) + " |", "|---|---|---|" + "---|" * len(names)]
        for key, m in dr["models"].items():
            for h in m["heads"]:
                for lab, lb in m["labels"].items():
                    cds = lb["heads"].get(h, {}).get("cards", {})
                    L.append(f"| {m['name']} | {h} | {lab} | " + " | ".join(
                        (f"{cds[nm]['P99']:.4f} / {cds[nm]['max']:.3f} / {100 * cds[nm]['share_gt_0.10']:.4f}%" if cds.get(nm, {}).get("n") else "-")
                        for nm in names) + " |")
    (HERE / "sm145-result.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def selftest():
    edges = [0, 1, 3, 5, 9, 16, 32, 128, 300, 1000]
    users = list(range(1, 761))
    longusers = set(users[:50])

    def row(u, lab, k_all=1.0, k_young=1.0, k_long=1.0):
        base = {h: 0.3 + 0.01 * ((u * 7 + i) % 5) for i, h in enumerate(("ahead", "instant"))}
        n = [0, 40, 30, 30, 20, 10, 8, 3 if u in longusers else 0, 1 if u in longusers else 0, 0]
        pb = {"edges": edges, "all": {}}
        for h in ("ahead", "instant"):
            ll = [ni * base[h] * (k_young if 1 <= edges[i] < 5 else k_long if edges[i] >= 128 else k_all) for i, ni in enumerate(n)]
            pb["all"][h] = {"n": n, "ll": ll}
        return {"user_id": u, "label": lab, "benchmark": {h: {"LogLoss": base[h] * k_all, "size": 100} for h in base},
                "prior_buckets": pb}

    def build(mod):
        rows, longrows = {}, {}
        for key, (tag, *_r) in MODELS.items():
            for s in ["REF", *SCHEMES]:
                L = f"{tag}/{LABELS[s]}{E2S}"
                kw = mod.get((key, s), {})
                rows[L] = {u: row(u, L, **kw) for u in users}
                longrows[L] = {u: row(u, L, **kw) for u in longusers}
        return rows, longrows

    rk = {size: {"AN16": -40.0, "Y2N": -42.5, "Y4C": -43.2, "Y4N": -45.0} for size in ("Big", "Small", "Tiny")}

    def fake_dr(over=None):
        d = {"models": {}}
        for key, (tag, name, heads, size) in MODELS.items():
            d["models"][key] = {"labels": {lab: {"heads": {h: {"cards": {"all": {"share_gt_0.10": (over or {}).get((key, lab), 0.0)}}}
                                                         for h in heads}} for lab in ("AN16", *INT3)}}
        return d

    def run(mod, nf=None, dr=None, rank=rk):
        rows, longrows = build(mod)
        return decide(rows, nf or {}, longrows, {}, set(users[:722]), longusers, None, rank, dr if dr is not None else fake_dr())

    ok = True
    r = run({})
    ok &= r["status"] == "decided" and all(g["pick"] == "Y4N" and g["recommended"] == "Y4N" for g in r["go"].values())
    ok &= abs(r["models"]["sm118"]["schemes"]["Y4N"]["R2_dataset"]["curve"]["value"]) < 1e-12
    r = run({(k, s): {"k_all": 1.001} for k in MODELS for s in ("Y4N", "Y4C")})   # +0.1% dataset-wide: R2 fails, Y2N picked
    ok &= all(g["pick"] == "Y2N" for g in r["go"].values())
    r = run({("sm99", "Y4N"): {"k_long": 1.01}})                 # long cards +1% point on Big Instant-Only: R3 fails
    ok &= (r["go"]["Big"]["pick"] == "Y4C" and r["models"]["sm99"]["schemes"]["Y4N"]["R3"] is False
           and r["go"]["Small"]["pick"] == "Y4N")
    r = run({(k, "Y4N"): {"k_long": 1.004} for k in MODELS})     # +0.4% on long cards: inside the 11-22 tolerance
    ok &= all(g["pick"] == "Y4N" for g in r["go"].values())
    r = run({("sm130io", "Y4N"): {"k_young": 1.0015}, ("sm130io", "Y4C"): {"k_young": 1.0015}})   # young +0.15%: R4 fails
    ok &= r["go"]["Tiny"]["pick"] == "Y2N" and r["models"]["sm130io"]["schemes"]["Y4N"]["R4"] is False
    nf = {f"{MODELS['sm130io'][0]}/{LABELS['Y4N']}{E2S}": {5}}
    r = run({}, nf)                                              # a non-finite user fails Y4N for Tiny
    ok &= "Y4N" not in r["go"]["Tiny"]["passing"] and "Y4N" in r["go"]["Big"]["passing"]
    # worth: an extra cost of +0.03% vs AN16 (passes R2) makes Y4N not worth it; Y4C recommended
    r = run({(k, "Y4N"): {"k_all": 1.0003} for k in MODELS})
    ok &= all(g["pick"] == "Y4N" and g["recommended"] == "Y4C" and not g["worth"]["Y4N"]["extra"] for g in r["go"].values())
    # worth: the per-card share above 0.10 at 3x AN16's makes Y4C not worth it on Small
    r = run({}, dr=fake_dr({("sm113", "AN16"): 0.001, ("sm113", "Y4N"): 0.003, ("sm113", "Y4C"): 0.003}))
    ok &= r["go"]["Small"]["recommended"] == "Y2N" and r["go"]["Big"]["recommended"] == "Y4N"
    # worth: bytes saving below 2% (Y2N at -41.5 vs AN16 -40) is not worth it -> AN16 when the others fail
    rk2 = {size: {"AN16": -40.0, "Y2N": -41.5, "Y4C": -43.2, "Y4N": -45.0} for size in ("Big", "Small", "Tiny")}
    r = run({(k, s): {"k_all": 1.001} for k in MODELS for s in ("Y4N", "Y4C")}, rank=rk2)
    ok &= all(g["pick"] == "Y2N" and g["recommended"] == "AN16" for g in r["go"].values())
    # no dR file: cards None, not worth (recommended AN16)
    rows, longrows = build({})
    r = decide(rows, {}, longrows, {}, set(users[:722]), longusers, None, rk, None)
    ok &= all(g["recommended"] == "AN16" and g["worth"]["Y4N"]["cards"] is None for g in r["go"].values())
    # pending on missing users
    rows, longrows = build({})
    for u in range(1, 4):
        rows[f"{MODELS['sm91s2'][0]}/{LABELS['Y2N']}{E2S}"].pop(u)
    r = decide(rows, {}, longrows, {}, set(users[:722]), longusers, None, rk, fake_dr())
    ok &= r["status"] == "pending"
    # bucket mismatch raises
    rows, longrows = build({})
    bad = rows[f"{MODELS['sm91s2'][0]}/{LABELS['Y4C']}{E2S}"][1]["prior_buckets"]["all"]["ahead"]
    bad["n"] = [x + 1 for x in bad["n"]]
    try:
        decide(rows, {}, longrows, {}, set(users[:722]), longusers, None, rk, fake_dr())
        ok = False
    except BucketMismatch:
        pass
    print(f"[sm145] selftest {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


def check():
    ok = sha(REPORT) == REPORT_SHA and U732.exists() and ULONG.exists() and BYTES.exists()
    if ok:
        r = bytes_rank()
        ok = all(sorted(SCHEMES, key=lambda s: r[size][s]) == SCHEMES for size in r)   # the byte order of SCHEMES holds
    print(f"[sm145] pins {'OK' if ok else 'CHANGED'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke"); s.add_argument("--gpu", required=True); s.add_argument("--cpu", required=True)
    sub.add_parser("report"); sub.add_parser("selftest"); sub.add_parser("check")
    a = ap.parse_args()
    if a.cmd == "smoke":
        return smoke(a.gpu, a.cpu)
    return {"report": report, "selftest": selftest, "check": check}[a.cmd]()


if __name__ == "__main__":
    raise SystemExit(main())
