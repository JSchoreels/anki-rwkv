#!/usr/bin/env python
"""Rank-1 int8 states (SM150): GPU smoke check, the pre-registered rules and the go per size
(design.md "Rules", fixed 11-23 before any scoring of the new labels on the 732 set or the long-card users). CPU only.
Rows: the new labels R1 and R12 from this run; REF, Q4 and Q2 are SM148's rows (gpu / cpu rows files of ../sm148).

  python sm150_report.py smoke --gpu gpu/smoke-gpu-rows.jsonl --cpu cpu/smoke-cpu-rows.jsonl   exit 0 = smoke OK
  python sm150_report.py report      gpu/sm150-rows-732.jsonl + cpu/sm150-longcard-rows.jsonl + sm150-dr.json
                                     -> sm150-result.{json,md}; exit 0 = decided, 2 = pending (coverage), 3 = bucket mismatch
  python sm150_report.py selftest    synthetic rows
  python sm150_report.py check       pins (showtime_report.py sha256, users files, bytes order)

Statistic: % = 100 x (sum of the label's LogLoss / sum of the reference's LogLoss - 1), review-weighted, paired user
bootstrap 2,000 draws, seed 0, 95% CI. Dataset-wide (R2, R2T) on the benchmark LogLoss of the 722 rule users of the 732
set (showtime_report.compare, as every 732 read-out); buckets (R3) on the "all" scope of --prior-buckets (every review the
scorer predicts), by the answers of the card before the review. The helpers below are copied from the aggressive state
tiers for Tiny (SM146)'s sm146_report.py (the version with the curve / instant keying fix of SM145).
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
ROWS732, LONGROWS = G / "sm150-rows-732.jsonl", C / "sm150-longcard-rows.jsonl"
OLD732, OLDLONG = HERE.parent / "sm148" / "gpu" / "sm148-rows-732.jsonl", HERE.parent / "sm148" / "cpu" / "sm148-longcard-rows.jsonl"
DR = HERE / "sm150-dr.json"
BYTES = HERE / "sm150-bytes.json"
E2S = "@e2s-query"
LABELS = {"REF": "w8+s8hx8", "Q4": "w8+lr4q8x8", "Q2": "w8+lr2q8x8", "R12": "w8+lr1.16.2q8x8", "R1": "w8+lr1q8x8",
          "F1": "w8+lr1fx8"}
NEWL = ("R12", "R1")                          # scored in this run (SM148's rows hold REF, Q4, Q2)
SCHEMES = ["R1", "R12", "Q2", "Q4"]           # ascending state bytes for every size (sm150-bytes.json): go = smallest passing
CEIL_BIG = 0.15                               # Big: vs its own int8, 95% CI upper bound per head (user 11-23)
CEIL_VS_BIG = {"Small": 0.5, "Tiny": 1.0}     # Small / Tiny: TOTAL cost vs the matching Big model at int8, CI upper per head
LONG_POINT, LONG_UPPER = 0.5, 1.0             # long-card (128+) warning levels, vs the model's own int8 (not a rule)
CARD_GUARD = 0.01                             # per-card warning level: share of cards whose worst review has |dR| > 0.10
CARD_LIMIT = "0.10"
BUCKETS = ((1, 3), (3, 5), (5, 9), (9, 16), (16, 32), (32, 128), (128, None))   # report only; edges of --prior-buckets
MIN_732, MIN_LONG = 715, 48
TOL_STRICT, TOL_DISCRETE, TOL_DISCRETE_MEAN = 1e-5, 3e-4, 1e-4
TOL_EPS = 1e-9   # the rows store LogLoss to 6 decimals
KEY = {"curve": "ahead", "instant": "instant"}
TWO, IO = ("curve", "instant"), ("instant",)
MODELS = {   # key: (tag, plain name, heads, size, key of the matching Big model (Small / Tiny) or None)
    "sm91s2": ("sm91s2-big-showtime", "Big two-head (SM91s2)", TWO, "Big", None),
    "sm99": ("sm99-big-inst-e2", "Big Instant-Only (SM99)", IO, "Big", None),
    "sm118": ("sm118-e26-dec2ep32k-t91s2", "Small two-head (SM118)", TWO, "Small", "sm91s2"),
    "sm113": ("sm113-inst-e35st-st-dec2ep32k-t99", "Small Instant-Only (SM113)", IO, "Small", "sm99"),
    "sm130th": ("sm130-th-dec2ep32k-t91s2", "six-layer Tiny two-head (SM130)", TWO, "Tiny", "sm91s2"),
    "sm130io": ("sm130-io-dec2ep32k-t99", "six-layer Tiny Instant-Only (SM130)", IO, "Tiny", "sm99"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def showtime():
    if sha(REPORT) != REPORT_SHA:
        raise SystemExit("[sm150] showtime_report.py changed: refusing")
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
    """GPU vs CPU on the smoke users: REF strict (1e-5 per user and head); the low-rank labels (R1, R12, F1)
    by the discrete-codec rule (per user <= 3e-4, review-weighted mean signed difference <= 1e-4, per head)."""
    g, gnf, _ = read_rows(gpu)
    c, cnf, _ = read_rows(cpu)
    users = json.loads((HERE / "users-smoke.json").read_text())["users"]
    ok, lines, rep = True, [], {}
    for key, (tag, name, heads, size, _s) in MODELS.items():
        for lab in ["REF", "F1", *NEWL]:
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
                if lab == "REF":
                    good = worst <= TOL_STRICT + TOL_EPS
                else:
                    good = worst <= TOL_DISCRETE + TOL_EPS and abs(mean) <= TOL_DISCRETE_MEAN + TOL_EPS
                rep[f"{L} {h}"] = {"max_abs": worst, "mean_signed": mean, "pass": good}
                ok &= good
    (G / "smoke-check.json").write_text(json.dumps({"ok": ok, "problems": lines, "checks": rep}, indent=1) + "\n",
                                        encoding="utf-8")
    worst_i = max((v["max_abs"] for k, v in rep.items() if "s8hx8" in k), default=0)
    worst_l = max((v["max_abs"] for k, v in rep.items() if "s8hx8" not in k), default=0)
    print(f"[sm150] smoke {'OK' if ok else 'FAILED'}: {len(rep)} checks; REF max |GPU - CPU| "
          f"{worst_i:.2e} (tol {TOL_STRICT:g}); low-rank labels max {worst_l:.2e} (tol {TOL_DISCRETE:g}, mean "
          f"{TOL_DISCRETE_MEAN:g})" + (f"; {lines[:3]}" if lines else ""))
    return 0 if ok else 1


def card_share(dr, key, lab, h):
    """Share of cards whose worst review has |dR| > 0.10 (all cards of the long-card users), or None (Big: not run)."""
    try:
        return dr["models"][key]["labels"][lab]["heads"][h]["cards"]["all"][f"share_gt_{CARD_LIMIT}"]
    except (KeyError, TypeError):
        return None


def sub(rows, label, users):
    return {u: r for u, r in rows.get(label, {}).items() if u in users}


def decide(rows732, nf732, longrows, longnf, users732, longusers, st=None, bytes_pct=None, dr=None):
    """Rules per model and label and the go per size (design.md "Rules").
    Rule C (cost, the go rule): Big vs its own int8, CI upper <= 0.15% per head; Small / Tiny vs the matching Big model at
    int8 (two-head vs two-head, Instant-Only vs Instant-Only, the model's own heads), CI upper <= 0.5% / 1.0% per head.
    Finite rows and the coverage (>= 715 users) are conditions. Long-card (128+) and per-card |dR| are warnings only."""
    res = {"ceilings_pct": {"Big_vs_own_int8_upper": CEIL_BIG, "Small_vs_Big_int8_upper": CEIL_VS_BIG["Small"],
                            "Tiny_vs_Big_int8_upper": CEIL_VS_BIG["Tiny"], "warn_long_point": LONG_POINT,
                            "warn_long_upper": LONG_UPPER, "warn_card_share_gt_0.10_max": CARD_GUARD},
           "min_users": {"732": MIN_732, "long_warning": MIN_LONG}, "models": {}, "go": {}, "pending": [], "warnings": []}
    for key, (tag, name, heads, size, bkey) in MODELS.items():
        m = {"name": name, "size": size, "heads": list(heads), "schemes": {}}
        lab = lambda s, t=tag: f"{t}/{LABELS[s]}{E2S}"   # noqa: E731
        ref = sub(rows732, lab("REF"), users732)
        lref = sub(longrows, lab("REF"), longusers)
        bref = sub(rows732, lab("REF", MODELS[bkey][0]), users732) if bkey else None
        if len(ref) < MIN_732:
            res["pending"].append(f"{tag} REF: {len(ref)} of {MIN_732} users on the 732 set")
        if bkey:
            m["big_match"] = MODELS[bkey][1]
            m["REF_vs_big"] = dataset_cmp(ref, bref, heads, sorted(set(ref) & set(bref)), st)
        for s in SCHEMES:
            new = sub(rows732, lab(s), users732)
            lnew = sub(longrows, lab(s), longusers)
            x = {}
            nonfin = sorted(nf732.get(lab(s), set()) | longnf.get(lab(s), set()))
            common = sorted(set(new) & set(ref))
            lcommon = sorted(set(lnew) & set(lref))
            x["users_732"], x["users_long"] = len(common), len(lcommon)
            if len(common) < MIN_732:
                res["pending"].append(f"{tag} {s}: {len(common)} of {MIN_732} users on the 732 set")
            fin = not nonfin and all(math.isfinite(new[u]["benchmark"][KEY[h]]["LogLoss"]) for u in common for h in heads) \
                and all(math.isfinite(lnew[u]["benchmark"][KEY[h]]["LogLoss"]) for u in lcommon for h in heads)
            x["R1_finite"], x["nonfinite_users"] = fin, nonfin
            own = dataset_cmp(new, ref, heads, common, st)
            x["vs_own_int8"] = own
            if bkey:
                cb = sorted(set(new) & set(bref))
                if len(cb) < MIN_732:
                    res["pending"].append(f"{tag} {s}: {len(cb)} of {MIN_732} users with the Big int8 rows")
                vb = dataset_cmp(new, bref, heads, cb, st)
                x["vs_big_int8"] = vb
                x["C"] = bool(vb) and all(vb[h]["ci"][1] <= CEIL_VS_BIG[size] for h in heads)
            else:
                x["C"] = bool(own) and all(own[h]["ci"][1] <= CEIL_BIG for h in heads)
            x["pass"] = bool(x["R1_finite"] and x["C"] and len(common) >= MIN_732)
            # warnings (not rules): cards with 128+ answers, per-card |dR|
            r3 = {h: bucket_cmp(lnew, lref, KEY[h], 128) for h in heads} if lcommon else {}
            x["long_128p"] = r3
            x["long_not_run"] = size == "Big"
            x["warn_long"] = False if size == "Big" else (not r3) or not all(
                r3[h] is not None and r3[h]["value"] <= LONG_POINT and r3[h]["ci"][1] <= LONG_UPPER for h in heads)
            shares = {h: card_share(dr, key, s, h) for h in heads}
            x["card_share_gt_0.10"] = shares
            x["warn_card"] = size != "Big" and not all(v is not None and v <= CARD_GUARD for v in shares.values())
            if size != "Big" and len(lcommon) < MIN_LONG:
                res["warnings"].append(f"{tag} {s}: {len(lcommon)} of {MIN_LONG} long-card users")
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
        rank = bytes_pct[size] if bytes_pct else {s: i for i, s in enumerate(SCHEMES)}
        passing = [s for s in SCHEMES if all(m["schemes"][s]["pass"] for _, m in ms)]
        best = min(passing, key=lambda s: rank[s]) if passing else "REF"
        res["go"][size] = {"rule": "vs own int8, upper <= 0.15%" if size == "Big" else f"vs Big at int8, upper <= {CEIL_VS_BIG[size]}%",
                           "passing": passing, "pick": best,
                           "bytes_vs_REF_pct": (bytes_pct[size][best] if bytes_pct and best != "REF" else 0.0),
                           "failing": {s: [k for k, m in ms if not m["schemes"][s]["pass"]] for s in SCHEMES if s not in passing},
                           "warnings": {s: [k for k, m in ms if m["schemes"][s]["warn_long"] or m["schemes"][s]["warn_card"]]
                                        for s in SCHEMES}}
    return res


def bytes_rank():
    d = json.loads(BYTES.read_text())["sizes"]
    return {size: {s: d[size][s]["vs_int8_pct"] for s in [*SCHEMES, "REF"]} for size in ("Big", "Small", "Tiny")}


def speed(r732, rl):
    """Report only: review evaluations per second (model time of the scorer) per model and label, GPU and CPU rows."""
    out = {}
    for where, rows in (("gpu_732", r732), ("cpu_long", rl)):
        for key, (tag, *_r) in MODELS.items():
            for s in ["REF", *SCHEMES]:
                rr = list(rows.get(f"{tag}/{LABELS[s]}{E2S}", {}).values())
                sec = sum(r.get("model_seconds", 0) for r in rr)
                rev = sum(r.get("reviews", 0) for r in rr)
                out[f"{where} {key} {s}"] = {"review_evals_per_s": round(rev / sec) if sec else None,
                                             "model_seconds": round(sec, 1), "users": len(rr)}
            ref = out[f"{where} {key} REF"]["review_evals_per_s"]
            for s in SCHEMES:
                v = out[f"{where} {key} {s}"]["review_evals_per_s"]
                out[f"{where} {key} {s}"]["vs_REF"] = round(v / ref, 3) if v and ref else None
    return out


def merged(old, new):
    """({label: {user: row}}, {label: nonfinite users}, torn) of SM148's rows (REF, Q4, Q2) plus this run's rows."""
    ro, no, to = read_rows(old)
    rn, nn, tn = read_rows(new)
    rows = {**ro, **rn}
    nf = {k: set(no.get(k, ())) | set(nn.get(k, ())) for k in set(no) | set(nn)}
    return rows, nf, to + tn


def report():
    st = showtime()
    users732, _ = st.pools()
    longusers = set(json.loads(ULONG.read_text())["users"])
    r732, nf732, t1 = merged(OLD732, ROWS732)
    rl, nfl, t2 = merged(OLDLONG, LONGROWS)
    dr = json.loads(DR.read_text()) if DR.exists() else None
    try:
        res = decide(r732, nf732, rl, nfl, set(users732), longusers, st, bytes_rank(), dr)
    except BucketMismatch as e:
        print(f"[sm150] STOP bucket-count mismatch: {e}")
        return 3
    res["torn_lines"] = {"732": t1, "long": t2}
    res["dr_file"] = str(DR.name) if dr else None
    if dr is None or not dr.get("complete"):
        res["warnings"].append("per-review |dR| read-out (sm150-dr.json) missing or incomplete (Small and Tiny only)")
    res["report_only"] = {"speed": speed(r732, rl), "bytes": json.loads(BYTES.read_text())["sizes"],
                          "ref_rows_source": "SM148 (gpu/sm148-rows-732.jsonl, cpu/sm148-longcard-rows.jsonl); REF rescored on the 3 smoke users only"}
    (HERE / "sm150-result.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    write_md(res, dr)
    if res["status"] == "pending":
        print(f"[sm150] report pending: {res['pending'][:6]}")
        return 2
    print("[sm150] report decided: " + "; ".join(
        f"{s} -> {g['pick']} ({g['bytes_vs_REF_pct']:+.1f}% bytes)" for s, g in res["go"].items()))
    return 0


def write_md(res, dr):
    L = ["# Rank-1 int8 states (SM150): result tables", "",
         f"Status: **{res['status']}**. Rules (design.md, pre-registered 11-23): the cost is the 95% CI upper bound per head of the "
         f"LogLoss change on the 722 rule users of the 732 set. Big: vs its own int8 (w8+s8hx8) <= +{CEIL_BIG}%. Small / Tiny: TOTAL "
         f"cost vs the matching Big model at int8 (two-head vs Big two-head (SM91s2), Instant-Only vs Big Instant-Only (SM99), the "
         f"model's own heads) <= +{CEIL_VS_BIG['Small']}% (Small) / +{CEIL_VS_BIG['Tiny']}% (Tiny); every user finite; at least {MIN_732} "
         f"users. Go per size: the passing label with the smallest state bytes for both models among rank 1 (R1), rank 1 then 2 "
         f"from 16 answers (R12), rank 2 (Q2), rank 4 (Q4) (int8 factors). Long-card (128+ answers) and per-card |dR| are warnings "
         f"only (long-card runs: Small and Tiny only).", "", "## Go per size", "",
         "| size | rule | passing | go pick | bytes vs int8 | labels with a warning |", "|---|---|---|---|---|---|"]
    for size, g in res["go"].items():
        L.append(f"| {size} | {g['rule']} | {', '.join(g['passing']) or 'none'} | {g['pick']} | {g['bytes_vs_REF_pct']:+.1f}% | "
                 f"{'; '.join(f'{s}: ' + ', '.join(v) for s, v in g['warnings'].items() if v) or 'none'} |")
    L += ["", "## Cost per model, label and head", "",
          "| model | label | head | vs own int8 | vs Big int8 | long-card 128+ (warning) | cards > 0.10 (warning) | pass |",
          "|---|---|---|---|---|---|---|---|"]
    for key, m in res["models"].items():
        for s, x in m["schemes"].items():
            for h in m["heads"]:
                g = x["card_share_gt_0.10"].get(h)
                L.append(f"| {m['name']} | {s} | {h} | {fmt(x['vs_own_int8'].get(h))} | "
                         f"{fmt(x.get('vs_big_int8', {}).get(h)) if 'vs_big_int8' in x else '-'} | "
                         f"{'not run' if x['long_not_run'] else fmt(x['long_128p'].get(h))} | "
                         f"{'-' if g is None else f'{100 * g:.4f}%'} | {'yes' if x['pass'] else 'no'} |")
    L += ["", "## By answer count before the review (GPU, 732 set, % vs own int8)", "",
          "| model | label | head | " + " | ".join(f"{lo}-{hi - 1 if hi else ''}" for lo, hi in BUCKETS) + " |",
          "|---|---|---|" + "---|" * len(BUCKETS)]
    for key, m in res["models"].items():
        for s, x in m["schemes"].items():
            for h in m["heads"]:
                cells = []
                for lo, hi in BUCKETS:
                    v = x["buckets_gpu"].get(f"{lo}-{hi - 1 if hi else ''}", {}).get(h)
                    cells.append("n/a" if v is None else f"{v['value']:+.3f}%")
                L.append(f"| {m['name']} | {s} | {h} | " + " | ".join(cells) + " |")
    if dr:
        L += ["", "## Per-review and per-card |dR| vs int8 (Small and Tiny long-card users, CPU)", "",
              "| model | label | head | reviews P50 / P99 / max | reviews > 0.10 | cards > 0.05 | cards > 0.10 |", "|---|---|---|---|---|---|---|"]
        for key, m in dr["models"].items():
            for s, lb in m["labels"].items():
                for h, hd in lb["heads"].items():
                    if "reviews" not in hd:
                        continue
                    a, c = hd["reviews"]["all"], hd["cards"]["all"]
                    L.append(f"| {res['models'][key]['name']} | {s} | {h} | {a['P50']:.4f} / {a['P99']:.4f} / {a['max']:.4f} | "
                             f"{a['count_gt_0.10']} ({100 * a['share_gt_0.10']:.4f}%) | {100 * c['share_gt_0.05']:.4f}% | "
                             f"{100 * c['share_gt_0.10']:.4f}% |")
    sp = res["report_only"]["speed"]
    L += ["", "## Speed (report only): review evaluations per second, model time", "",
          "| model | where | REF | " + " | ".join(SCHEMES) + " |", "|---|---|---|" + "---|" * len(SCHEMES)]
    for key, m in res["models"].items():
        for where in ("gpu_732", "cpu_long"):
            cells = [str(sp[f"{where} {key} REF"]["review_evals_per_s"])]
            for s in SCHEMES:
                v = sp[f"{where} {key} {s}"]
                cells.append(f"{v['review_evals_per_s']} ({v['vs_REF']}x)" if v["vs_REF"] else str(v["review_evals_per_s"]))
            L.append(f"| {m['name']} | {where} | " + " | ".join(cells) + " |")
    (HERE / "sm150-result.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def selftest():
    edges = [0, 1, 3, 5, 9, 16, 32, 128, 300, 1000]
    users = list(range(1, 761))
    longusers = set(users[:50])

    def row(u, lab, k_all=1.0, k_long=1.0, base_k=1.0):
        base = {h: (0.3 + 0.01 * ((u * 7 + i) % 5)) * base_k for i, h in enumerate(("ahead", "instant"))}
        n = [0, 40, 30, 30, 20, 10, 8, 3 if u in longusers else 0, 1 if u in longusers else 0, 0]
        pb = {"edges": edges, "all": {}}
        for h in ("ahead", "instant"):
            ll = [ni * base[h] * (k_long if edges[i] >= 128 else k_all) for i, ni in enumerate(n)]
            pb["all"][h] = {"n": n, "ll": ll}
        return {"user_id": u, "label": lab, "benchmark": {h: {"LogLoss": base[h] * k_all, "size": 100} for h in base},
                "prior_buckets": pb, "model_seconds": 1.0, "reviews": 100}

    def build(mod):   # mod: {(key, label): kwargs}; small_k / tiny_k: Small / Tiny losses are this factor of Big's
        rows, longrows = {}, {}
        for key, (tag, name, heads, size, bkey) in MODELS.items():
            for s in ["REF", *SCHEMES]:
                L = f"{tag}/{LABELS[s]}{E2S}"
                kw = dict(mod.get((key, s), {}))
                if size != "Big":
                    kw.setdefault("base_k", mod.get("tiny_k", 1.0) if size == "Tiny" else mod.get("small_k", 1.0))
                rows[L] = {u: row(u, L, **kw) for u in users}
                if size != "Big":
                    longrows[L] = {u: row(u, L, **kw) for u in longusers}
        return rows, longrows

    rk = {size: {"R1": -88.0, "R12": -87.0, "Q2": -83.0, "Q4": -74.0, "REF": 0.0} for size in ("Big", "Small", "Tiny")}

    def fake_dr(over=None):
        d = {"models": {}}
        for key, (tag, name, heads, size, _b) in MODELS.items():
            if size != "Big":
                d["models"][key] = {"labels": {lab: {"heads": {h: {"cards": {"all": {"share_gt_0.10": (over or {}).get((key, lab), 0.0)}}}
                                                             for h in heads}} for lab in SCHEMES}}
        return d

    def run(mod, nf=None, dr=None, nusers=722):
        rows, longrows = build(mod)
        return decide(rows, nf or {}, longrows, {}, set(users[:nusers]), longusers, None, rk, dr if dr is not None else fake_dr())

    ok = True
    r = run({})
    ok &= r["status"] == "decided" and all(r["go"][s]["pick"] == "R1" for s in ("Big", "Small", "Tiny"))
    r = run({("sm99", "R1"): {"k_all": 1.002}})                              # Big Instant-Only +0.2% > 0.15%: Big -> R12
    ok &= r["go"]["Big"]["pick"] == "R12" and r["go"]["Small"]["pick"] == "R1"
    r = run({("sm99", "R1"): {"k_all": 1.0012}})                             # +0.12% < 0.15%: passes
    ok &= r["go"]["Big"]["pick"] == "R1"
    r = run({"small_k": 1.002, ("sm118", "R1"): {"k_all": 1.004}})            # Small int8 +0.2% vs Big, R1 +0.4% more: +0.6% > 0.5
    ok &= r["go"]["Small"]["pick"] == "R12" and r["models"]["sm118"]["schemes"]["R1"]["C"] is False
    ok &= abs(r["models"]["sm118"]["REF_vs_big"]["curve"]["value"] - 0.2) < 0.01
    r = run({"tiny_k": 1.0075, ("sm130io", "R1"): {"k_all": 1.004}})          # Tiny +0.75% then +0.4%: 1.15% fails 1.0%
    ok &= r["go"]["Tiny"]["pick"] == "R12" and r["models"]["sm130io"]["schemes"]["R1"]["C"] is False
    r = run({}, dr=fake_dr({("sm118", "R1"): 0.02}))                          # per-card share is a warning, not a rule
    ok &= r["go"]["Small"]["pick"] == "R1" and r["go"]["Small"]["warnings"]["R1"] == ["sm118"]
    r = run({("sm113", "R1"): {"k_long": 1.02}})                              # long-card +2%: warning only
    ok &= r["go"]["Small"]["pick"] == "R1" and r["go"]["Small"]["warnings"]["R1"] == ["sm113"]
    nf = {f"{MODELS['sm113'][0]}/{LABELS['R1']}{E2S}": {5}}
    r = run({}, nf)                                                          # a non-finite user fails R1 (Small)
    ok &= r["go"]["Small"]["pick"] == "R12"
    r = run({}, nusers=700)                                                  # fewer than 715 users: pending
    ok &= r["status"] == "pending"
    rows, longrows = build({})                                               # bucket mismatch raises
    bad = rows[f"{MODELS['sm130th'][0]}/{LABELS['R1']}{E2S}"][1]["prior_buckets"]["all"]["ahead"]
    bad["n"] = [x + 1 for x in bad["n"]]
    try:
        decide(rows, {}, longrows, {}, set(users[:722]), longusers, None, rk, fake_dr())
        ok = False
    except BucketMismatch:
        pass
    print(f"[sm150] selftest {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


def check():
    ok = sha(REPORT) == REPORT_SHA and U732.exists() and ULONG.exists() and BYTES.exists() and OLD732.exists() and OLDLONG.exists()
    if ok:
        r = bytes_rank()
        ok = all(sorted(SCHEMES, key=lambda s: r[size][s]) == SCHEMES for size in r)
    print(f"[sm150] pins {'OK' if ok else 'CHANGED'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("smoke"); s.add_argument("--gpu", required=True); s.add_argument("--cpu", required=True)
    sp.add_parser("report"); sp.add_parser("selftest"); sp.add_parser("check")
    a = ap.parse_args()
    if a.cmd == "smoke":
        return smoke(a.gpu, a.cpu)
    return {"report": report, "selftest": selftest, "check": check}[a.cmd]()


if __name__ == "__main__":
    raise SystemExit(main())
