#!/usr/bin/env python
"""Earlier int8 switch (SM144): GPU smoke check, the pre-registered rules and the go decision
(design.md "Rules", fixed 11-21 before any GPU ticket). CPU only.

  python sm144_report.py smoke --gpu gpu/smoke-gpu-rows.jsonl --cpu cpu/smoke-cpu-rows.jsonl   exit 0 = smoke OK
  python sm144_report.py report      gpu/sm144-rows-732.jsonl + cpu/sm144-longcard-rows.jsonl -> sm144-result.{json,md}
                                     exit 0 = decided, 2 = pending (coverage), 3 = bucket-count mismatch
  python sm144_report.py selftest    synthetic rows
  python sm144_report.py check       pins (showtime_report.py sha256, users files)

Statistic: % = 100 x (sum of the scheme's LogLoss / sum of REF's LogLoss - 1), review-weighted, paired user bootstrap
2,000 draws, seed 0, 95% CI. Dataset-wide (R2) on the benchmark LogLoss of the 722 rule users of the 732 set
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
ROWS732, LONGROWS = G / "sm144-rows-732.jsonl", C / "sm144-longcard-rows.jsonl"
SM139_ROWS = HERE.parent / "sm139" / "gpu" / "sm139-rows-732.jsonl"
E2S = "@e2s-query"
LABELS = {"REF": "w8+s8hx8", "A": "w8+t4.128.8c8nhx8", "A32": "w8+t4.32.8c8nhx8", "A16": "w8+t4.16.8c8nhx8",
          "AN32": "w8+t4.32.8c4.32.8nhx8", "AN16": "w8+t4.16.8c4.16.8nhx8", "U8": "w8+t8.5.8.128.8c8.5.8.128.8nhx8"}
# in ascending state bytes (notes int4 below the switch is smaller than notes int8; a later switch is smaller): the
# go rule takes the first that passes when no bytes ranking is given (selftest); the report uses sm144-bytes.json
SCHEMES = ["AN32", "AN16", "A", "A32", "A16"]
YOUNG = {}                                   # no young tier in this experiment (no R4)
LONG_FROM = {s: s for s in SCHEMES}          # R3: every label is scored directly on the long-card users (no dominance)
CEIL_ALL, CEIL_LONG, CEIL_YOUNG = 0.05, 0.15, 0.10
BUCKETS = ((1, 3), (3, 5), (5, 9), (9, 16), (16, 32), (32, 128), (128, None))   # report only; edges of --prior-buckets
MIN_732, MIN_LONG = 720, 48
TOL_STRICT, TOL_DISCRETE, TOL_DISCRETE_MEAN = 1e-5, 3e-4, 1e-4
TOL_EPS = 1e-9   # 11-21 17:47 smoke: the rows store LogLoss to 6 decimals, so one rounding unit read back as 1.000000000001e-05 > 1e-5
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
BYTES = HERE / "sm144-bytes.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def showtime():
    if sha(REPORT) != REPORT_SHA:
        raise SystemExit("[sm144] showtime_report.py changed: refusing")
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
    idx = [i for i, e in enumerate(edges) if e >= lo and (hi is None or (edges[i + 1] if i + 1 < len(edges) else math.inf) <= hi)]
    if lo not in edges or (hi is not None and hi not in edges):
        raise ValueError(f"bucket {lo}-{hi} does not align with edges {edges}")
    return sum(h["n"][i] for i in idx), sum(h["ll"][i] for i in idx)


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


class BucketMismatch(Exception):
    pass


def fmt(x):
    return "n/a" if x is None else f"{x['value']:+.3f}% [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]"


def smoke(gpu, cpu):
    """GPU vs CPU on the smoke users: REF and U8 strict (1e-5), U8 == REF on the GPU exactly; the tiered labels by the
    discrete-codec rule (per user <= 3e-4, review-weighted mean signed difference <= 1e-4, per head)."""
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
                young = [abs(bucket_sums(g[L][u], h, 1, 9)[1] / max(bucket_sums(g[L][u], h, 1, 9)[0], 1)
                             - bucket_sums(c[L][u], h, 1, 9)[1] / max(bucket_sums(c[L][u], h, 1, 9)[0], 1)) for u in users]
                long = [abs(bucket_sums(g[L][u], h, 128)[1] / max(bucket_sums(g[L][u], h, 128)[0], 1)
                            - bucket_sums(c[L][u], h, 128)[1] / max(bucket_sums(c[L][u], h, 128)[0], 1)) for u in users]
                rep[f"{L} {h}"] = {"max_abs": worst, "mean_signed": mean, "young_1_8_mean_abs_max": max(young),
                                   "long_128p_mean_abs_max": max(long), "pass": good}
                ok &= good
            if lab == "U8":   # the tier kernel with every tier int8 == the stock kernel, on the GPU too
                R = f"{tag}/{LABELS['REF']}{E2S}"
                # 11-21 17:47 smoke: LogLoss, RMSE and size were equal to every stored digit, but the Big AUC of one user
                # differed in its 6th decimal (GPU run-to-run order). The kernel itself is bitwise-checked by gpu-selftest;
                # here every field must agree within 2e-6 (AUC 1e-6 apart is reported, not failed).
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
    print(f"[sm144] smoke {'OK' if ok else 'FAILED'}: {len(rep)} checks; int8 labels max |GPU - CPU| {worst_s:.2e} "
          f"(tol {TOL_STRICT:g}); tiered labels max {worst_t:.2e} (tol {TOL_DISCRETE:g}, mean {TOL_DISCRETE_MEAN:g})"
          + (f"; {lines[:3]}" if lines else ""))
    return 0 if ok else 1


def decide(rows732, nf732, longrows, longnf, users732, longusers, st=None, bytes_rank=None):
    """The rules per model and scheme, then the go per size. Returns the result dict (status decided / pending)."""
    res = {"ceilings_pct": {"R2_dataset": CEIL_ALL, "R3_long": CEIL_LONG, "R4_young": CEIL_YOUNG},
           "min_users": {"732": MIN_732, "long": MIN_LONG}, "models": {}, "go": {}, "pending": []}
    for key, (tag, name, heads, size) in MODELS.items():
        m = {"name": name, "size": size, "heads": list(heads), "schemes": {}}
        lab = lambda s: f"{tag}/{LABELS[s]}{E2S}"   # noqa: E731
        ref = {u: r for u, r in rows732.get(lab("REF"), {}).items() if u in users732}
        lref = {u: r for u, r in longrows.get(lab("REF"), {}).items() if u in longusers}
        for s in SCHEMES:
            new = {u: r for u, r in rows732.get(lab(s), {}).items() if u in users732}
            ls = LONG_FROM[s]
            lnew = {u: r for u, r in longrows.get(lab(ls), {}).items() if u in longusers}
            x = {"long_label": ls}
            nonfin = sorted(nf732.get(lab(s), set()) | longnf.get(lab(ls), set()))
            common = sorted(set(new) & set(ref))
            lcommon = sorted(set(lnew) & set(lref))
            x["users_732"], x["users_long"] = len(common), len(lcommon)
            if len(common) < MIN_732:
                res["pending"].append(f"{tag} {s}: {len(common)} of {MIN_732} users on the 732 set")
            if len(lcommon) < MIN_LONG:
                res["pending"].append(f"{tag} {ls}: {len(lcommon)} of {MIN_LONG} long-card users")
            fin = not nonfin and all(math.isfinite(new[u]["benchmark"][KEY[h]]["LogLoss"]) for u in common for h in heads)
            x["R1_finite"], x["nonfinite_users"] = fin, nonfin
            r2 = {}
            if common and st is not None:
                cmpx = st.compare(as_loss(new), as_loss(ref), as_loss(ref), common)
                r2 = {h: {"value": cmpx[h]["pct_of_ref"], "ci": cmpx[h]["pct_of_ref_ci"], "users": cmpx["users"]} for h in heads}
            elif common:   # selftest: the same ratio-of-sums bootstrap on the benchmark sums
                r2 = {h: _bench_cmp(new, ref, KEY[h], common) for h in heads}
            x["R2_dataset"] = r2
            x["R2"] = bool(r2) and all(r2[h]["ci"][1] <= CEIL_ALL for h in heads)
            r3 = {h: bucket_cmp(lnew, lref, KEY[h], 128) for h in heads} if lcommon else {}
            x["R3_long_128p"] = r3
            x["R3"] = bool(r3) and all(r3[h] is not None and r3[h]["ci"][1] <= CEIL_LONG and r3[h]["finite"] for h in heads)
            if s in YOUNG:
                r4 = {h: bucket_cmp(new, ref, KEY[h], 1, YOUNG[s] + 1) for h in heads}
                x["R4_young"] = r4
                x["R4"] = all(r4[h] is not None and r4[h]["ci"][1] <= CEIL_YOUNG for h in heads)
            else:
                x["R4"] = True
            # report only: every bucket on the GPU rows, the all scope total
            x["buckets_gpu"] = {f"{lo}-{hi - 1 if hi else ''}": {h: bucket_cmp(new, ref, KEY[h], lo, hi) for h in heads}
                                for lo, hi in BUCKETS} if common else {}
            x["buckets_long"] = {f"{lo}-{hi - 1 if hi else ''}": {h: bucket_cmp(lnew, lref, KEY[h], lo, hi) for h in heads}
                                 for lo, hi in BUCKETS} if lcommon else {}
            x["all_scope_gpu"] = {h: bucket_cmp(new, ref, KEY[h], 0) for h in heads} if common else {}
            x["pass"] = x["R1_finite"] and x["R2"] and x["R3"] and x["R4"]
            m["schemes"][s] = x
        res["models"][key] = m
    res["status"] = "pending" if res["pending"] else "decided"
    sizes = {}
    for key, m in res["models"].items():
        sizes.setdefault(m["size"], []).append(m)
    for size, ms in sizes.items():
        passing = [s for s in SCHEMES if all(m["schemes"][s]["pass"] for m in ms)]
        rank = bytes_rank[size] if bytes_rank else {s: i for i, s in enumerate(SCHEMES)}
        best = min(passing, key=lambda s: rank[s]) if passing else "REF"
        res["go"][size] = {"passing": passing, "pick": best,
                           "bytes_vs_REF_pct": (bytes_rank[size][best] if bytes_rank and best != "REF" else 0.0)}
    return res


def _bench_cmp(new, ref, h, users, draws=2000):
    a = np.array([new[u]["benchmark"][h]["LogLoss"] * new[u]["benchmark"][h]["size"] for u in users])
    b = np.array([ref[u]["benchmark"][h]["LogLoss"] * ref[u]["benchmark"][h]["size"] for u in users])
    idx = np.random.default_rng(0).integers(0, len(users), size=(draws, len(users)))
    boot = (a[idx].sum(1) / b[idx].sum(1) - 1) * 100
    return {"value": float((a.sum() / b.sum() - 1) * 100), "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "users": len(users)}


def bytes_rank():
    d = json.loads(BYTES.read_text())["scheme_bytes_732"]
    return {size: {s: d[s][size]["vs_REF_pct"] for s in SCHEMES} for size in ("Big", "Small", "Tiny")}


def report():
    st = showtime()
    users732, _ = st.pools()
    longusers = set(json.loads(ULONG.read_text())["users"])
    r732, nf732, t1 = read_rows(ROWS732)
    rl, nfl, t2 = read_rows(LONGROWS)
    try:
        res = decide(r732, nf732, rl, nfl, set(users732), longusers, st, bytes_rank())
    except BucketMismatch as e:
        print(f"[sm144] STOP bucket-count mismatch: {e}")
        return 3
    res["torn_lines"] = {"732": t1, "long": t2}
    # report only: REF vs SM139's GPU rows (integrity), GPU vs CPU on the long-card users (REF, 128+ bucket)
    old, _, _ = read_rows(SM139_ROWS)
    integ = {}
    for key, (tag, *_rest) in MODELS.items():
        L = f"{tag}/{LABELS['REF']}{E2S}"
        common = set(old.get(L, {})) & set(r732.get(L, {}))
        integ[key] = {"users": len(common), "max_abs_benchmark_logloss": max(
            (abs(old[L][u]["benchmark"][h]["LogLoss"] - r732[L][u]["benchmark"][h]["LogLoss"])
             for u in common for h in old[L][u]["benchmark"]
             if isinstance(old[L][u]["benchmark"][h], dict) and isinstance(r732[L][u]["benchmark"].get(h), dict)),
            default=None)}   # 11-22 00:49: skip the scalar fields (size, *_coverage) of the benchmark dict
        lc = set(rl.get(L, {})) & set(r732.get(L, {}))
        diffs = []
        for u in lc:
            for h in ("ahead", "instant"):
                n1, l1 = bucket_sums(r732[L][u], h, 128)
                n0, l0 = bucket_sums(rl[L][u], h, 128)
                if n0 and n1 == n0:
                    diffs.append(abs(l1 / n1 - l0 / n0))
        integ[key]["gpu_vs_cpu_long_128p_mean_logloss_max_abs"] = max(diffs) if diffs else None
    res["report_only"] = {"ref_vs_sm139_gpu_rows": integ}
    (HERE / "sm144-result.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    write_md(res)
    if res["status"] == "pending":
        print(f"[sm144] report pending: {res['pending'][:6]}")
        return 2
    print("[sm144] report decided: " + "; ".join(f"{s} -> {g['pick']} ({g['bytes_vs_REF_pct']:+.1f}% bytes)" for s, g in res["go"].items()))
    return 0


def write_md(res):
    L = ["# Earlier int8 switch (SM144): result", "",
         f"Status: **{res['status']}**. Rules (design.md): R1 every user finite; R2 dataset-wide cost vs int8 cards and notes "
         f"(w8+s8hx8), benchmark LogLoss, 95% CI upper bound <= +{CEIL_ALL}% per trained head; R3 reviews of cards with 128+ "
         f"earlier answers on the long-card users (every label scored directly, CPU) <= +{CEIL_LONG}%. Go: per size the "
         f"smallest scheme (by bytes) that passes for both models.", "",
         "## Go per size", "", "| size | passing schemes | pick | state bytes vs int8 (732 set) |", "|---|---|---|---|"]
    for size, g in res["go"].items():
        L.append(f"| {size} | {', '.join(g['passing']) or 'none'} | {g['pick']} | {g['bytes_vs_REF_pct']:+.1f}% |")
    L += ["", "## Per model and scheme (% vs int8 cards and notes)", "",
          "| model | scheme | head | R2 dataset-wide | R3 long cards 128+ (CPU, label) | R4 young | R1 | pass |", "|---|---|---|---|---|---|---|---|"]
    for key, m in res["models"].items():
        for s, x in m["schemes"].items():
            for h in m["heads"]:
                r4 = fmt(x.get("R4_young", {}).get(h)) if "R4_young" in x else "-"
                L.append(f"| {m['name']} | {s} | {h} | {fmt(x['R2_dataset'].get(h))} | {fmt(x['R3_long_128p'].get(h))} ({x['long_label']}) "
                         f"| {r4} | {x['R1_finite']} | {x['pass']} |")
    for title, key in (("732 set (GPU, all users)", "buckets_gpu"), ("long-card users (CPU)", "buckets_long")):
        L += ["", f"## Cost per answer-count bucket, {title} (% vs int8; card answers before the review)", "",
              "| model | scheme | head | " + " | ".join(f"{lo}-{hi - 1 if hi else ''}" for lo, hi in BUCKETS) + " |",
              "|---|---|---|" + "---|" * len(BUCKETS)]
        for k, m in res["models"].items():
            for s, x in m["schemes"].items():
                for h in m["heads"]:
                    L.append(f"| {m['name']} | {s} | {h} | " + " | ".join(
                        fmt((x.get(key, {}).get(f"{lo}-{hi - 1 if hi else ''}") or {}).get(h)) for lo, hi in BUCKETS) + " |")
    (HERE / "sm144-result.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def selftest():
    rng = np.random.default_rng(1)
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

    def run(mod, nf=None):
        rows, longrows = build(mod)
        return decide(rows, nf or {}, longrows, {}, set(users[:722]), longusers)

    ok = True
    r = run({})
    ok &= r["status"] == "decided" and all(g["pick"] == "AN32" for g in r["go"].values())   # all pass: first in rank order
    ok &= abs(r["models"]["sm118"]["schemes"]["AN16"]["R2_dataset"]["curve"]["value"]) < 1e-12
    r = run({(k, s): {"k_all": 1.001} for k in MODELS for s in ("AN32", "AN16")})   # +0.1% on both AN labels: they fail R2, A picked
    ok &= all(g["pick"] == "A" for g in r["go"].values())
    r = run({("sm99", "AN32"): {"k_long": 1.01}, ("sm99", "A"): {"k_long": 1.01}})   # long cards +1% on Big Instant-Only
    ok &= (r["go"]["Big"]["pick"] == "AN16" and r["models"]["sm99"]["schemes"]["AN32"]["R3"] is False
           and r["models"]["sm99"]["schemes"]["A"]["R3"] is False and r["go"]["Small"]["pick"] == "AN32")   # no dominance: own rows
    nf = {f"{MODELS['sm130io'][0]}/{LABELS['AN16']}{E2S}": {5}}
    r = run({}, nf)                                              # a non-finite user fails AN16 for Tiny
    ok &= "AN16" not in r["go"]["Tiny"]["passing"] and "AN16" in r["go"]["Big"]["passing"]
    # pending on missing users
    rows, longrows = build({})
    for u in range(1, 4):
        rows[f"{MODELS['sm91s2'][0]}/{LABELS['A16']}{E2S}"].pop(u)
    r = decide(rows, {}, longrows, {}, set(users[:722]), longusers)
    ok &= r["status"] == "pending"
    # bucket mismatch raises
    rows, longrows = build({})
    bad = rows[f"{MODELS['sm91s2'][0]}/{LABELS['A']}{E2S}"][1]["prior_buckets"]["all"]["ahead"]
    bad["n"] = [x + 1 for x in bad["n"]]
    try:
        decide(rows, {}, longrows, {}, set(users[:722]), longusers)
        ok = False
    except BucketMismatch:
        pass
    # bytes ranking: with every scheme passing, the smallest by bytes wins
    rk = {size: {"A": -27, "A32": -26, "A16": -25, "AN32": -43, "AN16": -41} for size in ("Big", "Small", "Tiny")}
    rows, longrows = build({})
    r = decide(rows, {}, longrows, {}, set(users[:722]), longusers, None, rk)
    ok &= all(g["pick"] == "AN32" for g in r["go"].values())
    print(f"[sm144] selftest {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


def check():
    ok = sha(REPORT) == REPORT_SHA and U732.exists() and ULONG.exists() and BYTES.exists()
    print(f"[sm144] pins {'OK' if ok else 'CHANGED'}")
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
