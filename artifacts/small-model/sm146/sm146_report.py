#!/usr/bin/env python
"""Aggressive state tiers for Tiny (SM146): GPU smoke check, the pre-registered rules and the go decision for the two
six-layer Tiny models (design.md "Rules", fixed 11-22 before any scoring of the new labels). CPU only.

  python sm146_report.py smoke --gpu gpu/smoke-gpu-rows.jsonl --cpu cpu/smoke-cpu-rows.jsonl   exit 0 = smoke OK
  python sm146_report.py report      gpu/sm146-rows-732.jsonl + gpu/small-ref-rows.jsonl + cpu/sm146-longcard-rows.jsonl
                                     + sm146-dr.json -> sm146-result.{json,md}; exit 0 = decided, 2 = pending, 3 = bucket mismatch
  python sm146_report.py selftest    synthetic rows
  python sm146_report.py check       pins (showtime_report.py, the Small int8 rows, users files, bytes order)

Statistic: % = 100 x (sum of the label's LogLoss / sum of the reference's LogLoss - 1), review-weighted, paired user
bootstrap 2,000 draws, seed 0, 95% CI. Dataset-wide (R2, and the cost vs Tiny int8) on the benchmark LogLoss of the 722 rule
users of the 732 set (showtime_report.compare, as every 732 readout); buckets (R3 and the read-outs) on the "all" scope of
--prior-buckets (every review the scorer predicts), by the answers of the card before the review.
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
ROWS732, LONGROWS = G / "sm146-rows-732.jsonl", C / "sm146-longcard-rows.jsonl"
SMALLREF = G / "small-ref-rows.jsonl"      # Small int8 (w8+s8hx8) GPU rows of the two Small models, taken from SM145 (identical to SM144's)
SMALLREF_SHA = "6b285f92987b1bb0760209768a900df9925ea9a5e8dfd6ddde0ce7dfdfee3594"
DR = HERE / "sm146-dr.json"
BYTES = HERE / "sm146-bytes.json"
OLD = {"SM145": (W / "sm145/artifacts/small-model/sm145/gpu/sm145-rows-732.jsonl", W / "sm145/artifacts/small-model/sm145/cpu/sm145-longcard-rows.jsonl"),
       "SM144": (W / "sm144/artifacts/small-model/sm144/gpu/sm144-rows-732.jsonl", W / "sm144/artifacts/small-model/sm144/cpu/sm144-longcard-rows.jsonl")}
E2S = "@e2s-query"
LABELS = {"REF": "w8+s8hx8", "AN16": "w8+t4.16.8c4.16.8nhx8", "I3a": "w8+t3.5.4.16.8c3.5.4.16.8nhx8",
          "I3b": "w8+t3.9.4.16.8c3.9.4.16.8nhx8", "I2a": "w8+t2.5.4.16.8c2.5.4.16.8nhx8",
          "I2b": "w8+t2.5.3.16.8c2.5.3.16.8nhx8", "U8": "w8+t8.5.8.128.8c8.5.8.128.8nhx8",
          "P4": "w8+t4.5.4.16.8c4.5.4.16.8nhx8"}
# in ascending state bytes (sm146-bytes.json): the go rule takes the smallest passing label
SCHEMES = ["I2b", "I2a", "I3b", "I3a", "AN16"]
CEIL_SMALL = 1.00                            # R2: Tiny label vs Small int8, paired by user, 95% CI upper bound per head (user decision 11-22)
LONG_POINT, LONG_UPPER = 0.5, 1.0            # R3, the user's 11-22 long-card tolerance (point and CI upper bound), vs Tiny int8
CARD_GUARD = 0.01                            # G: share of long-card-user cards whose worst review has |dR| > 0.10
CARD_LIMIT = "0.10"
BUCKETS = ((1, 3), (3, 5), (5, 9), (9, 16), (16, 32), (32, 128), (128, None))   # report only; edges of --prior-buckets
MIN_732, MIN_LONG = 720, 48
TOL_STRICT, TOL_DISCRETE, TOL_DISCRETE_MEAN = 1e-5, 3e-4, 1e-4
TOL_EPS = 1e-9   # the rows store LogLoss to 6 decimals
KEY = {"curve": "ahead", "instant": "instant"}
TWO, IO = ("curve", "instant"), ("instant",)
MODELS = {   # key: (tag, plain name, heads, size, Small tag it is compared with, Small plain name)
    "sm130th": ("sm130-th-dec2ep32k-t91s2", "the six-layer Tiny two-head model (SM130)", TWO, "Tiny",
                "sm118-e26-dec2ep32k-t91s2", "the show-time two-head cool-down for Small (SM118)"),
    "sm130io": ("sm130-io-dec2ep32k-t99", "the six-layer Tiny Instant-Only model (SM130)", IO, "Tiny",
                "sm113-inst-e35st-st-dec2ep32k-t99", "the longer Small Instant-Only model (SM113)"),
}


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
        c = st.compare(as_loss(new), as_loss(ref), as_loss(ref), common)
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
    for key, (tag, name, heads, size, *_s) in MODELS.items():
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
                rep[f"{L} {h}"] = {"max_abs": worst, "mean_signed": mean, "pass": good}
                ok &= good
            if lab == "U8":   # the tier kernel with every tier int8 == the stock kernel, on the GPU too (2e-6 per field)
                R = f"{tag}/{LABELS['REF']}{E2S}"

                def close(a, b):
                    if isinstance(a, dict) and isinstance(b, dict):
                        return set(a) == set(b) and all(close(a[k], b[k]) for k in a)
                    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                        return abs(a - b) <= 2e-6
                    return a == b
                same = all(close(g[L][u]["benchmark"], g[R][u]["benchmark"]) for u in users if u in g.get(R, {}))
                rep[f"{L} == REF on the GPU"] = same
                ok &= same
    (G / "smoke-check.json").write_text(json.dumps({"ok": ok, "problems": lines, "checks": rep}, indent=1) + "\n",
                                        encoding="utf-8")
    worst_t = max((v["max_abs"] for k, v in rep.items() if isinstance(v, dict) and "/w8+t" in k and "t8.5" not in k), default=0)
    worst_s = max((v["max_abs"] for k, v in rep.items() if isinstance(v, dict) and ("s8hx8" in k or "t8.5" in k)), default=0)
    print(f"[sm146] smoke {'OK' if ok else 'FAILED'}: {len(rep)} checks; int8 labels max |GPU - CPU| {worst_s:.2e} "
          f"(tol {TOL_STRICT:g}); tiered labels max {worst_t:.2e} (tol {TOL_DISCRETE:g}, mean {TOL_DISCRETE_MEAN:g})"
          + (f"; {lines[:3]}" if lines else ""))
    return 0 if ok else 1


def card_share(dr, key, lab, h):
    """Share of cards whose worst review has |dR| > 0.10 (all cards of the long-card users), or None."""
    try:
        return dr["models"][key]["labels"][lab]["heads"][h]["cards"]["all"][f"share_gt_{CARD_LIMIT}"]
    except (KeyError, TypeError):
        return None


def decide(rows732, nf732, longrows, longnf, small, users732, longusers, st=None, bytes_pct=None, dr=None):
    """The rules per model and label and the go pick (Tiny only)."""
    res = {"ceilings_pct": {"R2_vs_small_upper": CEIL_SMALL, "R3_long_point": LONG_POINT, "R3_long_upper": LONG_UPPER,
                            "G_card_share_gt_0.10_max": CARD_GUARD},
           "min_users": {"732": MIN_732, "long": MIN_LONG}, "models": {}, "go": {}, "pending": []}
    for key, (tag, name, heads, size, stag, sname) in MODELS.items():
        m = {"name": name, "size": size, "heads": list(heads), "small": sname, "schemes": {}}
        lab = lambda s: f"{tag}/{LABELS[s]}{E2S}"   # noqa: E731
        ref = {u: r for u, r in rows732.get(lab("REF"), {}).items() if u in users732}
        sref = {u: r for u, r in small.get(f"{stag}/{LABELS['REF']}{E2S}", {}).items() if u in users732}
        lref = {u: r for u, r in longrows.get(lab("REF"), {}).items() if u in longusers}
        an = {u: r for u, r in rows732.get(lab("AN16"), {}).items() if u in users732}
        lan = {u: r for u, r in longrows.get(lab("AN16"), {}).items() if u in longusers}
        c0 = sorted(set(ref) & set(sref))
        m["REF_vs_small"] = dataset_cmp(ref, sref, heads, c0, st)
        if len(sref) < MIN_732 or len(ref) < MIN_732:
            res["pending"].append(f"{tag} REF: {len(ref)} Tiny / {len(sref)} Small int8 users of {MIN_732}")
        for s in SCHEMES:
            new = {u: r for u, r in rows732.get(lab(s), {}).items() if u in users732}
            lnew = {u: r for u, r in longrows.get(lab(s), {}).items() if u in longusers}
            x = {}
            nonfin = sorted(nf732.get(lab(s), set()) | longnf.get(lab(s), set()))
            common = sorted(set(new) & set(ref) & set(sref))
            lcommon = sorted(set(lnew) & set(lref))
            x["users_732"], x["users_long"] = len(common), len(lcommon)
            if len(common) < MIN_732:
                res["pending"].append(f"{tag} {s}: {len(common)} of {MIN_732} users on the 732 set")
            if len(lcommon) < MIN_LONG:
                res["pending"].append(f"{tag} {s}: {len(lcommon)} of {MIN_LONG} long-card users")
            fin = not nonfin and all(math.isfinite(new[u]["benchmark"][KEY[h]]["LogLoss"]) for u in common for h in heads)
            x["R1_finite"], x["nonfinite_users"] = fin, nonfin
            r2 = dataset_cmp(new, sref, heads, common, st)            # Tiny label vs the matching Small model at int8
            x["R2_vs_small"] = r2
            x["R2"] = bool(r2) and all(r2[h]["ci"][1] <= CEIL_SMALL for h in heads)
            r3 = {h: bucket_cmp(lnew, lref, KEY[h], 128) for h in heads} if lcommon else {}
            x["R3_long_128p"] = r3
            x["R3"] = bool(r3) and all(r3[h] is not None and r3[h]["finite"] and r3[h]["value"] <= LONG_POINT
                                       and r3[h]["ci"][1] <= LONG_UPPER for h in heads)
            shares = {h: card_share(dr, key, s, h) for h in heads}
            x["G_card_share_gt_0.10"] = shares
            x["G"] = all(v is not None and v <= CARD_GUARD for v in shares.values())
            x["pass"] = x["R1_finite"] and x["R2"] and x["R3"] and x["G"]
            # report only: the cost vs Tiny int8 (paired), dataset-wide and per bucket; vs AN16
            x["cost_vs_tiny_ref"] = dataset_cmp(new, ref, heads, sorted(set(new) & set(ref)), st)
            if s != "AN16":
                ca = sorted(set(new) & set(an))
                la = sorted(set(lnew) & set(lan))
                x["vs_AN16"] = {"dataset": dataset_cmp(new, an, heads, ca, st),
                                "long_128p": {h: bucket_cmp(lnew, lan, KEY[h], 128) for h in heads} if la else {}}
            x["buckets_gpu"] = {f"{lo}-{hi - 1 if hi else ''}": {h: bucket_cmp(new, ref, KEY[h], lo, hi) for h in heads}
                                for lo, hi in BUCKETS} if common else {}
            x["buckets_long"] = {f"{lo}-{hi - 1 if hi else ''}": {h: bucket_cmp(lnew, lref, KEY[h], lo, hi) for h in heads}
                                 for lo, hi in BUCKETS} if lcommon else {}
            x["all_scope_gpu"] = {h: bucket_cmp(new, ref, KEY[h], 0) for h in heads} if common else {}
            m["schemes"][s] = x
        res["models"][key] = m
    res["status"] = "pending" if res["pending"] else "decided"
    ms = list(res["models"].items())
    passing = [s for s in SCHEMES if all(m["schemes"][s]["pass"] for _, m in ms)]
    rank = bytes_pct if bytes_pct else {s: i for i, s in enumerate(SCHEMES)}
    best = min(passing, key=lambda s: rank[s]) if passing else "REF"
    res["go"] = {"passing": passing, "pick": best, "bytes_vs_REF_pct": (bytes_pct[best] if bytes_pct and best != "REF" else 0.0),
                 "failing": {s: [k for k, m in ms if not m["schemes"][s]["pass"]] for s in SCHEMES if s not in passing}}
    return res


def bytes_rank():
    d = json.loads(BYTES.read_text())["labels"]
    return {s: d[s]["vs_REF_pct"] for s in SCHEMES}


def integrity(r732, rl, small):
    """Report only: Tiny REF and AN16 rows vs SM145's and SM144's (GPU 732 set, CPU long cards), Small int8 rows vs SM145's,
    GPU vs CPU on the long-card users (REF, 128+ bucket)."""
    out = {}
    for name, (gp, lp) in OLD.items():
        og, _, _ = read_rows(gp)
        ol, _, _ = read_rows(lp)
        for key, (tag, *_rest) in MODELS.items():
            for s in ("REF", "AN16"):
                L = f"{tag}/{LABELS[s]}{E2S}"
                for where, old, new in (("gpu_732", og, r732), ("cpu_long", ol, rl)):
                    common = set(old.get(L, {})) & set(new.get(L, {}))
                    out[f"{key} {s} vs {name} {where}"] = {"users": len(common), "max_abs_benchmark_logloss": max(
                        (abs(old[L][u]["benchmark"][h]["LogLoss"] - new[L][u]["benchmark"][h]["LogLoss"])
                         for u in common for h in old[L][u]["benchmark"]
                         if isinstance(old[L][u]["benchmark"][h], dict) and isinstance(new[L][u]["benchmark"].get(h), dict)),
                        default=None)}
        if name == "SM145":   # the Small int8 rows used as the R2 reference vs SM145's own file
            for key, (tag, nm, heads, size, stag, sname) in MODELS.items():
                L = f"{stag}/{LABELS['REF']}{E2S}"
                common = set(og.get(L, {})) & set(small.get(L, {}))
                out[f"Small int8 rows of {sname} vs {name} gpu_732"] = {"users": len(common), "max_abs_benchmark_logloss": max(
                    (abs(og[L][u]["benchmark"][h]["LogLoss"] - small[L][u]["benchmark"][h]["LogLoss"])
                     for u in common for h in og[L][u]["benchmark"] if isinstance(og[L][u]["benchmark"][h], dict)), default=None)}
    for key, (tag, *_rest) in MODELS.items():
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
    if sha(SMALLREF) != SMALLREF_SHA:
        raise SystemExit("[sm146] gpu/small-ref-rows.jsonl changed: refusing")
    users732, _ = st.pools()
    longusers = set(json.loads(ULONG.read_text())["users"])
    r732, nf732, t1 = read_rows(ROWS732)
    rl, nfl, t2 = read_rows(LONGROWS)
    small, _, _ = read_rows(SMALLREF)
    dr = json.loads(DR.read_text()) if DR.exists() else None
    try:
        res = decide(r732, nf732, rl, nfl, small, set(users732), longusers, st, bytes_rank(), dr)
    except BucketMismatch as e:
        print(f"[sm146] STOP bucket-count mismatch: {e}")
        return 3
    res["torn_lines"] = {"732": t1, "long": t2}
    res["dr_file"] = str(DR.name) if dr else None
    if dr is None or not dr.get("complete"):
        res["pending"].append("per-review |dR| read-out (sm146-dr.json) missing or incomplete")
        res["status"] = "pending"
    res["report_only"] = {"integrity": integrity(r732, rl, small)}
    (HERE / "sm146-result.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    write_md(res, dr)
    if res["status"] == "pending":
        print(f"[sm146] report pending: {res['pending'][:6]}")
        return 2
    g = res["go"]
    print(f"[sm146] report decided: Tiny -> {g['pick']} ({g['bytes_vs_REF_pct']:+.2f}% bytes); passing {g['passing']}; failing {g['failing']}")
    return 0


def _p(x, k, scale=1.0, f="{:.4f}"):
    return "-" if not x or k not in x else f.format(x[k] * scale)


def write_md(res, dr):
    L = ["# Aggressive state tiers for Tiny (SM146): result", "",
         f"Status: **{res['status']}**. Rules (design.md, pre-registered 11-22): R1 every user finite; R2 dataset-wide cost of the Tiny label vs "
         f"the matching Small model at int8 (w8+s8hx8), benchmark LogLoss paired by user on the same users, 95% CI upper bound <= +{CEIL_SMALL:.2f}% per head; "
         f"R3 reviews of cards with 128+ earlier answers on the long-card users (CPU, every label scored directly) vs Tiny int8: point "
         f"estimate <= +{LONG_POINT}% and CI upper bound <= +{LONG_UPPER}% per head; G the share of long-card-user cards whose worst review has |dR| > 0.10 "
         f"is at most {100 * CARD_GUARD:.0f}% per head. Go: the label with the smallest state bytes that passes for both models.", "",
         "## Go", "", "| passing labels | go pick | bytes vs int8 | failing labels (models) |", "|---|---|---|---|"]
    g = res["go"]
    L.append(f"| {', '.join(g['passing']) or 'none'} | {g['pick']} | {g['bytes_vs_REF_pct']:+.2f}% | "
             f"{'; '.join(f'{s} ({chr(44).join(v)})' for s, v in g['failing'].items()) or 'none'} |")
    L += ["", "## Tiny int8 vs Small int8 (the starting point of R2; SM130's own result was +0.467% / +0.524% / +0.564%)", "",
          "| Tiny model | compared with | head | Tiny int8 vs Small int8 |", "|---|---|---|---|"]
    for key, m in res["models"].items():
        for h in m["heads"]:
            L.append(f"| {m['name']} | {m['small']} | {h} | {fmt(m['REF_vs_small'].get(h))} |")
    L += ["", "## Per model and label", "",
          "| model | label | head | R2: vs Small int8 (rule) | cost vs Tiny int8 (GPU, no verdict) | R3: long cards 128+ vs Tiny int8 (CPU) | G: cards with a review > 0.10 | R1 | pass |",
          "|---|---|---|---|---|---|---|---|---|"]
    for key, m in res["models"].items():
        for s, x in m["schemes"].items():
            for h in m["heads"]:
                sh = x["G_card_share_gt_0.10"].get(h)
                L.append(f"| {m['name']} | {s} | {h} | {fmt(x['R2_vs_small'].get(h))} | {fmt(x['cost_vs_tiny_ref'].get(h))} | "
                         f"{fmt(x['R3_long_128p'].get(h))} | {'-' if sh is None else f'{100 * sh:.4f}%'} | {x['R1_finite']} | {x['pass']} |")
    L += ["", "## Extra cost of each label vs AN16 (paired, report only; % of AN16's LogLoss)", "",
          "| model | label | head | dataset-wide (GPU) | long cards 128+ (CPU) |", "|---|---|---|---|---|"]
    for key, m in res["models"].items():
        for s, x in m["schemes"].items():
            if "vs_AN16" in x:
                for h in m["heads"]:
                    v = x["vs_AN16"]
                    L.append(f"| {m['name']} | {s} | {h} | {fmt(v['dataset'].get(h))} | {fmt(v['long_128p'].get(h))} |")
    for title, key in (("732 set (GPU, all users)", "buckets_gpu"), ("long-card users (CPU)", "buckets_long")):
        L += ["", f"## Cost per answer-count bucket vs Tiny int8, {title} (% ; card answers before the review)", "",
              "| model | label | head | " + " | ".join(f"{lo}-{hi - 1 if hi else ''}" for lo, hi in BUCKETS) + " |",
              "|---|---|---|" + "---|" * len(BUCKETS)]
        for k, m in res["models"].items():
            for s, x in m["schemes"].items():
                for h in m["heads"]:
                    L.append(f"| {m['name']} | {s} | {h} | " + " | ".join(
                        fmt((x.get(key, {}).get(f"{lo}-{hi - 1 if hi else ''}") or {}).get(h)) for lo, hi in BUCKETS) + " |")
    if dr:
        L += ["", "## Per-review and per-card |dR| vs Tiny int8 (long-card users, CPU, every predicted review; report only)", "",
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
    (HERE / "sm146-result.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def selftest():
    edges = [0, 1, 3, 5, 9, 16, 32, 128, 300, 1000]
    users = list(range(1, 761))
    longusers = set(users[:50])

    def row(u, lab, k_all=1.0, k_long=1.0):
        base = {h: 0.3 + 0.01 * ((u * 7 + i) % 5) for i, h in enumerate(("ahead", "instant"))}
        n = [0, 40, 30, 30, 20, 10, 8, 3 if u in longusers else 0, 1 if u in longusers else 0, 0]
        pb = {"edges": edges, "all": {}}
        for h in ("ahead", "instant"):
            ll = [ni * base[h] * (k_long if edges[i] >= 128 else k_all) for i, ni in enumerate(n)]
            pb["all"][h] = {"n": n, "ll": ll}
        return {"user_id": u, "label": lab, "benchmark": {h: {"LogLoss": base[h] * k_all, "size": 100} for h in base},
                "prior_buckets": pb}

    def build(mod, small_k=1.0):
        rows, longrows, small = {}, {}, {}
        for key, (tag, _n, _h, _s, stag, _sn) in MODELS.items():
            for s in ["REF", *SCHEMES]:
                L = f"{tag}/{LABELS[s]}{E2S}"
                kw = mod.get((key, s), {})
                rows[L] = {u: row(u, L, **kw) for u in users}
                longrows[L] = {u: row(u, L, **kw) for u in longusers}
            LS = f"{stag}/{LABELS['REF']}{E2S}"
            small[LS] = {u: row(u, LS, k_all=small_k) for u in users}
        return rows, longrows, small

    rk = {"AN16": -35.3, "I3a": -39.8, "I3b": -42.7, "I2a": -44.3, "I2b": -48.6}

    def fake_dr(over=None):
        d = {"models": {}}
        for key, (tag, name, heads, *_r) in MODELS.items():
            d["models"][key] = {"labels": {lab: {"heads": {h: {"cards": {"all": {"share_gt_0.10": (over or {}).get((key, lab), 0.0)}}}
                                                         for h in heads}} for lab in SCHEMES}}
        return d

    def run(mod, nf=None, dr=None, small_k=1.0):
        rows, longrows, small = build(mod, small_k)
        return decide(rows, nf or {}, longrows, {}, small, set(users[:722]), longusers, None, rk, dr if dr is not None else fake_dr())

    ok = True
    r = run({})
    ok &= r["status"] == "decided" and r["go"]["pick"] == "I2b" and r["go"]["passing"] == SCHEMES
    ok &= abs(r["models"]["sm130th"]["schemes"]["I2b"]["R2_vs_small"]["curve"]["value"]) < 1e-12
    r = run({(k, "I2b"): {"k_all": 1.012} for k in MODELS})            # +1.2% vs Small int8: R2 fails, I2a picked
    ok &= r["go"]["pick"] == "I2a" and r["models"]["sm130io"]["schemes"]["I2b"]["R2"] is False
    r = run({}, small_k=0.995)                                           # Small better by 0.5%: Tiny starts at +0.5%
    ok &= abs(r["models"]["sm130th"]["REF_vs_small"]["curve"]["value"] - 0.5025) < 0.01 and r["go"]["pick"] == "I2b"
    r = run({(k, "I2b"): {"k_all": 1.006} for k in MODELS}, small_k=0.995)   # +1.1% in all vs Small: fails
    ok &= r["go"]["pick"] == "I2a"
    r = run({("sm130io", "I2b"): {"k_long": 1.006}, ("sm130io", "I2a"): {"k_long": 1.006}})   # long cards +0.6% point: R3 fails
    ok &= r["go"]["pick"] == "I3b" and r["models"]["sm130io"]["schemes"]["I2a"]["R3"] is False
    r = run({}, dr=fake_dr({("sm130th", "I2b"): 0.02, ("sm130io", "I2a"): 0.0101, ("sm130th", "I3b"): 0.0099}))   # card guard 1%
    ok &= r["go"]["pick"] == "I3b" and r["models"]["sm130th"]["schemes"]["I2b"]["G"] is False
    nf = {f"{MODELS['sm130io'][0]}/{LABELS['I2b']}{E2S}": {5}}
    r = run({}, nf)                                                      # a non-finite user fails I2b
    ok &= "I2b" not in r["go"]["passing"] and r["go"]["pick"] == "I2a"
    rows, longrows, small = build({})                                    # no dR file: G fails everywhere -> REF
    r = decide(rows, {}, longrows, {}, small, set(users[:722]), longusers, None, rk, None)
    ok &= r["go"]["pick"] == "REF"
    rows, longrows, small = build({})                                    # pending on missing users
    for u in range(1, 4):
        rows[f"{MODELS['sm130th'][0]}/{LABELS['I2a']}{E2S}"].pop(u)
    r = decide(rows, {}, longrows, {}, small, set(users[:722]), longusers, None, rk, fake_dr())
    ok &= r["status"] == "pending"
    rows, longrows, small = build({})                                    # missing Small int8 rows: pending
    small.pop(f"{MODELS['sm130io'][4]}/{LABELS['REF']}{E2S}")
    r = decide(rows, {}, longrows, {}, small, set(users[:722]), longusers, None, rk, fake_dr())
    ok &= r["status"] == "pending"
    rows, longrows, small = build({})                                    # bucket mismatch raises
    bad = rows[f"{MODELS['sm130th'][0]}/{LABELS['I3b']}{E2S}"][1]["prior_buckets"]["all"]["ahead"]
    bad["n"] = [x + 1 for x in bad["n"]]
    try:
        decide(rows, {}, longrows, {}, small, set(users[:722]), longusers, None, rk, fake_dr())
        ok = False
    except BucketMismatch:
        pass
    print(f"[sm146] selftest {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


def check():
    ok = sha(REPORT) == REPORT_SHA and U732.exists() and ULONG.exists() and BYTES.exists() and sha(SMALLREF) == SMALLREF_SHA
    if ok:
        r = bytes_rank()
        ok = sorted(SCHEMES, key=lambda s: r[s]) == SCHEMES   # the byte order of SCHEMES holds
    print(f"[sm146] pins {'OK' if ok else 'CHANGED'}")
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
