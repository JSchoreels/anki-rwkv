#!/usr/bin/env python
"""Low-rank int8 card and note states, post-training (SM148): GPU smoke check, the pre-registered rules and the go per size
(design.md "Rules", fixed 11-23 before any scoring of the 732 set or the long-card users). CPU only.

  python sm148_report.py smoke --gpu gpu/smoke-gpu-rows.jsonl --cpu cpu/smoke-cpu-rows.jsonl   exit 0 = smoke OK
  python sm148_report.py report      gpu/sm148-rows-732.jsonl + cpu/sm148-longcard-rows.jsonl + sm148-dr.json
                                     -> sm148-result.{json,md}; exit 0 = decided, 2 = pending (coverage), 3 = bucket mismatch
  python sm148_report.py selftest    synthetic rows
  python sm148_report.py check       pins (showtime_report.py sha256, users files, bytes order)

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
ROWS732, LONGROWS = G / "sm148-rows-732.jsonl", C / "sm148-longcard-rows.jsonl"
DR = HERE / "sm148-dr.json"
BYTES = HERE / "sm148-bytes.json"
OLD = {"SM145": (W / "sm145/artifacts/small-model/sm145/gpu/sm145-rows-732.jsonl",
                 W / "sm145/artifacts/small-model/sm145/cpu/sm145-longcard-rows.jsonl")}
E2S = "@e2s-query"
LABELS = {"REF": "w8+s8hx8", "Q4": "w8+lr4q8x8", "T24": "w8+lr2.16.4q8x8", "Q2": "w8+lr2q8x8", "U8": "w8+lrRq8x8",
          "F4": "w8+lr4fx8"}
SCHEMES = ["Q2", "T24", "Q4"]                 # ascending state bytes for every size (sm148-bytes.json): go = smallest passing
CEIL_ALL = 0.05                               # R2: vs the model's own int8 (REF), 95% CI upper bound per head
CEIL_SMALL = 1.00                             # R2T (Tiny): vs the matching Small model at int8, upper bound (user 11-22)
LONG_POINT, LONG_UPPER = 0.5, 1.0             # R3, the user's 11-22 long-card tolerance (point and CI upper bound)
CARD_GUARD = 0.01                             # G: share of long-card-user cards whose worst review has |dR| > 0.10
CARD_LIMIT = "0.10"
BUCKETS = ((1, 3), (3, 5), (5, 9), (9, 16), (16, 32), (32, 128), (128, None))   # report only; edges of --prior-buckets
MIN_732, MIN_LONG = 720, 48
TOL_STRICT, TOL_DISCRETE, TOL_DISCRETE_MEAN = 1e-5, 3e-4, 1e-4
TOL_EPS = 1e-9   # the rows store LogLoss to 6 decimals
KEY = {"curve": "ahead", "instant": "instant"}
TWO, IO = ("curve", "instant"), ("instant",)
MODELS = {   # key: (tag, plain name, heads, size, matching Small key for the Tiny rule or None)
    "sm91s2": ("sm91s2-big-showtime", "Big two-head (SM91s2)", TWO, "Big", None),
    "sm99": ("sm99-big-inst-e2", "Big Instant-Only (SM99)", IO, "Big", None),
    "sm118": ("sm118-e26-dec2ep32k-t91s2", "Small two-head (SM118)", TWO, "Small", None),
    "sm113": ("sm113-inst-e35st-st-dec2ep32k-t99", "Small Instant-Only (SM113)", IO, "Small", None),
    "sm130th": ("sm130-th-dec2ep32k-t91s2", "six-layer Tiny two-head (SM130)", TWO, "Tiny", "sm118"),
    "sm130io": ("sm130-io-dec2ep32k-t99", "six-layer Tiny Instant-Only (SM130)", IO, "Tiny", "sm113"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def showtime():
    if sha(REPORT) != REPORT_SHA:
        raise SystemExit("[sm148] showtime_report.py changed: refusing")
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
    """GPU vs CPU on the smoke users: REF and U8 strict (1e-5 per user and head); the low-rank labels (Q4, T24, Q2, F4)
    by the discrete-codec rule (per user <= 3e-4, review-weighted mean signed difference <= 1e-4, per head)."""
    g, gnf, _ = read_rows(gpu)
    c, cnf, _ = read_rows(cpu)
    users = json.loads((HERE / "users-smoke.json").read_text())["users"]
    ok, lines, rep = True, [], {}
    for key, (tag, name, heads, size, _s) in MODELS.items():
        for lab in ["REF", "U8", "F4", *SCHEMES]:
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
    (G / "smoke-check.json").write_text(json.dumps({"ok": ok, "problems": lines, "checks": rep}, indent=1) + "\n",
                                        encoding="utf-8")
    worst_i = max((v["max_abs"] for k, v in rep.items() if "s8hx8" in k or "lrR" in k), default=0)
    worst_l = max((v["max_abs"] for k, v in rep.items() if "s8hx8" not in k and "lrR" not in k), default=0)
    print(f"[sm148] smoke {'OK' if ok else 'FAILED'}: {len(rep)} checks; int8 labels (REF, U8) max |GPU - CPU| "
          f"{worst_i:.2e} (tol {TOL_STRICT:g}); low-rank labels max {worst_l:.2e} (tol {TOL_DISCRETE:g}, mean "
          f"{TOL_DISCRETE_MEAN:g})" + (f"; {lines[:3]}" if lines else ""))
    return 0 if ok else 1


def card_share(dr, key, lab, h):
    """Share of cards whose worst review has |dR| > 0.10 (all cards of the long-card users), or None."""
    try:
        return dr["models"][key]["labels"][lab]["heads"][h]["cards"]["all"][f"share_gt_{CARD_LIMIT}"]
    except (KeyError, TypeError):
        return None


def sub(rows, label, users):
    return {u: r for u, r in rows.get(label, {}).items() if u in users}


def decide(rows732, nf732, longrows, longnf, users732, longusers, st=None, bytes_pct=None, dr=None):
    """The rules per model and label and the go per size (Tiny: under the user's Tiny rule, the strict go reported too)."""
    res = {"ceilings_pct": {"R2_dataset_upper": CEIL_ALL, "R2T_tiny_vs_small_upper": CEIL_SMALL, "R3_long_point": LONG_POINT,
                            "R3_long_upper": LONG_UPPER, "G_card_share_gt_0.10_max": CARD_GUARD},
           "min_users": {"732": MIN_732, "long": MIN_LONG}, "models": {}, "go": {}, "pending": []}
    for key, (tag, name, heads, size, skey) in MODELS.items():
        m = {"name": name, "size": size, "heads": list(heads), "schemes": {}}
        lab = lambda s, t=tag: f"{t}/{LABELS[s]}{E2S}"   # noqa: E731
        ref = sub(rows732, lab("REF"), users732)
        lref = sub(longrows, lab("REF"), longusers)
        sref = sub(rows732, lab("REF", MODELS[skey][0]), users732) if skey else None
        if len(ref) < MIN_732:
            res["pending"].append(f"{tag} REF: {len(ref)} of {MIN_732} users on the 732 set")
        if skey:
            m["small"] = MODELS[skey][1]
            m["REF_vs_small"] = dataset_cmp(ref, sref, heads, sorted(set(ref) & set(sref)), st)
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
            if len(lcommon) < MIN_LONG:
                res["pending"].append(f"{tag} {s}: {len(lcommon)} of {MIN_LONG} long-card users")
            fin = not nonfin and all(math.isfinite(new[u]["benchmark"][KEY[h]]["LogLoss"]) for u in common for h in heads) \
                and all(math.isfinite(lnew[u]["benchmark"][KEY[h]]["LogLoss"]) for u in lcommon for h in heads)
            x["R1_finite"], x["nonfinite_users"] = fin, nonfin
            r2 = dataset_cmp(new, ref, heads, common, st)
            x["R2_dataset"] = r2
            x["R2"] = bool(r2) and all(r2[h]["ci"][1] <= CEIL_ALL for h in heads)
            if skey:
                cs = sorted(set(new) & set(sref))
                if len(cs) < MIN_732:
                    res["pending"].append(f"{tag} {s}: {len(cs)} of {MIN_732} users with the Small int8 rows")
                r2t = dataset_cmp(new, sref, heads, cs, st)
                x["R2T_vs_small"] = r2t
                x["R2T"] = bool(r2t) and all(r2t[h]["ci"][1] <= CEIL_SMALL for h in heads)
            r3 = {h: bucket_cmp(lnew, lref, KEY[h], 128) for h in heads} if lcommon else {}
            x["R3_long_128p"] = r3
            x["R3"] = bool(r3) and all(r3[h] is not None and r3[h]["finite"] and r3[h]["value"] <= LONG_POINT
                                       and r3[h]["ci"][1] <= LONG_UPPER for h in heads)
            shares = {h: card_share(dr, key, s, h) for h in heads}
            x["G_card_share_gt_0.10"] = shares
            x["G"] = all(v is not None and v <= CARD_GUARD for v in shares.values())
            x["pass_strict"] = x["R1_finite"] and x["R2"] and x["R3"] and x["G"]
            x["pass"] = (x["R1_finite"] and x["R2T"] and x["R3"] and x["G"]) if skey else x["pass_strict"]
            # report only: per bucket, the all scope, the long-card users' dataset-wide cost
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
        g = {}
        for mode in ("pass", "pass_strict"):
            passing = [s for s in SCHEMES if all(m["schemes"][s][mode] for _, m in ms)]
            best = min(passing, key=lambda s: rank[s]) if passing else "REF"
            g[mode] = {"passing": passing, "pick": best,
                       "bytes_vs_REF_pct": (bytes_pct[size][best] if bytes_pct and best != "REF" else 0.0)}
        res["go"][size] = {"rule": "Tiny rule (R2T)" if size == "Tiny" else "R2", **g["pass"],
                           "strict": g["pass_strict"],
                           "failing": {s: [k for k, m in ms if not m["schemes"][s]["pass"]] for s in SCHEMES
                                       if s not in g["pass"]["passing"]}}
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


def integrity(r732, rl):
    """Report only: REF rows vs the int3 young tier on the early switch (SM145)'s rows (GPU 732 set, CPU long cards);
    GPU vs CPU on the long-card users (REF, 128+ bucket)."""
    out = {}
    for name, (gp, lp) in OLD.items():
        og, _, _ = read_rows(gp)
        ol, _, _ = read_rows(lp)
        for key, (tag, *_rest) in MODELS.items():
            L = f"{tag}/{LABELS['REF']}{E2S}"
            for where, old, new in (("gpu_732", og, r732), ("cpu_long", ol, rl)):
                common = set(old.get(L, {})) & set(new.get(L, {}))
                out[f"{key} REF vs {name} {where}"] = {"users": len(common), "max_abs_benchmark_logloss": max(
                    (abs(old[L][u]["benchmark"][h]["LogLoss"] - new[L][u]["benchmark"][h]["LogLoss"])
                     for u in common for h in old[L][u]["benchmark"]
                     if isinstance(old[L][u]["benchmark"][h], dict) and isinstance(new[L][u]["benchmark"].get(h), dict)),
                    default=None)}
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
    users732, _ = st.pools()
    longusers = set(json.loads(ULONG.read_text())["users"])
    r732, nf732, t1 = read_rows(ROWS732)
    rl, nfl, t2 = read_rows(LONGROWS)
    dr = json.loads(DR.read_text()) if DR.exists() else None
    try:
        res = decide(r732, nf732, rl, nfl, set(users732), longusers, st, bytes_rank(), dr)
    except BucketMismatch as e:
        print(f"[sm148] STOP bucket-count mismatch: {e}")
        return 3
    res["torn_lines"] = {"732": t1, "long": t2}
    res["dr_file"] = str(DR.name) if dr else None
    if dr is None or not dr.get("complete"):
        res["pending"].append("per-review |dR| read-out (sm148-dr.json) missing or incomplete")
        res["status"] = "pending"
    res["report_only"] = {"integrity_vs_sm145": integrity(r732, rl), "speed": speed(r732, rl),
                          "bytes": json.loads(BYTES.read_text())["sizes"]}
    (HERE / "sm148-result.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    write_md(res, dr)
    if res["status"] == "pending":
        print(f"[sm148] report pending: {res['pending'][:6]}")
        return 2
    print("[sm148] report decided: " + "; ".join(
        f"{s} -> {g['pick']} ({g['bytes_vs_REF_pct']:+.1f}% bytes; strict {g['strict']['pick']})" for s, g in res["go"].items()))
    return 0


def write_md(res, dr):
    L = ["# Low-rank int8 card and note states, post-training (SM148): result tables", "",
         f"Status: **{res['status']}**. Rules (design.md, pre-registered 11-23): R1 every user finite; R2 dataset-wide cost vs "
         f"int8 cards and notes (w8+s8hx8), benchmark LogLoss on the 722 rule users of the 732 set, 95% CI upper bound <= "
         f"+{CEIL_ALL}% per head; R2T (six-layer Tiny models) the label vs the matching Small model at int8, upper bound <= "
         f"+{CEIL_SMALL:.2f}%; R3 reviews of cards with 128+ earlier answers on the 50 long-card users (CPU): point <= "
         f"+{LONG_POINT}% and upper bound <= +{LONG_UPPER}% per head; G at most {CARD_GUARD:.0%} of the long-card users' cards with "
         f"a review |dR| > 0.10. Go per size: the label with the smallest state bytes passing for both models (Tiny under "
         f"R2T; the strict go under R2 is listed too).", "",
         "## Go per size", "",
         "| size | rule | passing | go pick | bytes vs int8 | strict (R2) passing | strict pick |", "|---|---|---|---|---|---|---|"]
    for size, g in res["go"].items():
        L.append(f"| {size} | {g['rule']} | {', '.join(g['passing']) or 'none'} | {g['pick']} | {g['bytes_vs_REF_pct']:+.1f}% | "
                 f"{', '.join(g['strict']['passing']) or 'none'} | {g['strict']['pick']} |")
    L += ["", "## Rules per model, label and head", "",
          "| model | label | head | R2 vs int8 | R2T vs Small int8 | R3 128+ (long-card users) | G cards > 0.10 | pass |",
          "|---|---|---|---|---|---|---|---|"]
    for key, m in res["models"].items():
        for s, x in m["schemes"].items():
            for h in m["heads"]:
                g = x["G_card_share_gt_0.10"].get(h)
                L.append(f"| {m['name']} | {s} | {h} | {fmt(x['R2_dataset'].get(h))} | "
                         f"{fmt(x.get('R2T_vs_small', {}).get(h)) if 'R2T_vs_small' in x else '-'} | "
                         f"{fmt(x['R3_long_128p'].get(h))} | {'-' if g is None else f'{100 * g:.4f}%'} | "
                         f"{'yes' if x['pass'] else 'no'}{'' if x['pass'] == x['pass_strict'] else ' (strict: ' + ('yes' if x['pass_strict'] else 'no') + ')'} |")
    L += ["", "## By answer count before the review (GPU, 732 set, % vs int8, read heads)", "",
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
        L += ["", "## Per-review and per-card |dR| vs int8 (long-card users, CPU)", "",
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
          "| model | where | REF | Q4 | T24 | Q2 |", "|---|---|---|---|---|---|"]
    for key, m in res["models"].items():
        for where in ("gpu_732", "cpu_long"):
            cells = [str(sp[f"{where} {key} REF"]["review_evals_per_s"])]
            for s in SCHEMES[::-1]:
                v = sp[f"{where} {key} {s}"]
                cells.append(f"{v['review_evals_per_s']} ({v['vs_REF']}x)" if v["vs_REF"] else str(v["review_evals_per_s"]))
            L.append(f"| {m['name']} | {where} | " + " | ".join(cells) + " |")
    (HERE / "sm148-result.md").write_text("\n".join(L) + "\n", encoding="utf-8")


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
                "prior_buckets": pb, "model_seconds": 1.0, "reviews": 100}

    def build(mod):
        rows, longrows = {}, {}
        for key, (tag, *_r) in MODELS.items():
            for s in ["REF", *SCHEMES]:
                L = f"{tag}/{LABELS[s]}{E2S}"
                kw = mod.get((key, s), {})
                if s == "REF" and key in ("sm118", "sm113") and "small_k" in mod:
                    kw = {"k_all": mod["small_k"]}
                rows[L] = {u: row(u, L, **kw) for u in users}
                longrows[L] = {u: row(u, L, **kw) for u in longusers}
        return rows, longrows

    rk = {size: {"Q2": -80.0, "T24": -78.0, "Q4": -70.0, "REF": 0.0} for size in ("Big", "Small", "Tiny")}

    def fake_dr(over=None):
        d = {"models": {}}
        for key, (tag, name, heads, *_r) in MODELS.items():
            d["models"][key] = {"labels": {lab: {"heads": {h: {"cards": {"all": {"share_gt_0.10": (over or {}).get((key, lab), 0.0)}}}
                                                         for h in heads}} for lab in SCHEMES}}
        return d

    def run(mod, nf=None, dr=None):
        rows, longrows = build(mod)
        return decide(rows, nf or {}, longrows, {}, set(users[:722]), longusers, None, rk, dr if dr is not None else fake_dr())

    ok = True
    r = run({})
    ok &= r["status"] == "decided" and all(r["go"][s]["pick"] == "Q2" for s in ("Big", "Small", "Tiny"))
    r = run({("sm99", "Q2"): {"k_all": 1.0006}})                              # +0.06% on one Big model: Big -> T24
    ok &= r["go"]["Big"]["pick"] == "T24" and r["go"]["Small"]["pick"] == "Q2"
    r = run({("sm118", "Q2"): {"k_long": 1.006}, ("sm118", "T24"): {"k_long": 1.006}})   # long cards +0.6% point: R3
    ok &= r["go"]["Small"]["pick"] == "Q4" and r["models"]["sm118"]["schemes"]["T24"]["R3"] is False
    r = run({}, dr=fake_dr({("sm91s2", "Q2"): 0.0101, ("sm91s2", "T24"): 0.0099}))       # card guard 1%
    ok &= r["go"]["Big"]["pick"] == "T24" and r["models"]["sm91s2"]["schemes"]["Q2"]["G"] is False
    r = run({("sm130th", "Q2"): {"k_all": 1.003}, ("sm130io", "Q2"): {"k_all": 1.003}})  # Tiny +0.3%: strict fails, R2T passes
    ok &= r["go"]["Tiny"]["pick"] == "Q2" and r["go"]["Tiny"]["strict"]["pick"] == "T24"
    r = run({"small_k": 0.992})                                                # Small int8 0.8% better: Tiny starts at +0.8%
    ok &= r["go"]["Tiny"]["pick"] == "Q2"
    ok &= abs(r["models"]["sm130th"]["REF_vs_small"]["curve"]["value"] - 0.806) < 0.01
    r = run({"small_k": 0.992, ("sm130io", "Q2"): {"k_all": 1.003}})           # +1.1% vs Small: R2T fails for Q2
    ok &= r["go"]["Tiny"]["pick"] == "T24" and r["models"]["sm130io"]["schemes"]["Q2"]["R2T"] is False
    nf = {f"{MODELS['sm113'][0]}/{LABELS['Q2']}{E2S}": {5}}
    r = run({}, nf)                                                            # a non-finite user fails Q2 (Small)
    ok &= r["go"]["Small"]["pick"] == "T24"
    rows, longrows = build({})                                                 # no dR file: G fails -> REF everywhere
    r = decide(rows, {}, longrows, {}, set(users[:722]), longusers, None, rk, None)
    ok &= all(r["go"][s]["pick"] == "REF" for s in ("Big", "Small", "Tiny"))
    rows, longrows = build({})                                                 # pending on missing users
    for u in range(1, 4):
        rows[f"{MODELS['sm91s2'][0]}/{LABELS['T24']}{E2S}"].pop(u)
    r = decide(rows, {}, longrows, {}, set(users[:722]), longusers, None, rk, fake_dr())
    ok &= r["status"] == "pending"
    rows, longrows = build({})                                                 # bucket mismatch raises
    bad = rows[f"{MODELS['sm130th'][0]}/{LABELS['Q4']}{E2S}"][1]["prior_buckets"]["all"]["ahead"]
    bad["n"] = [x + 1 for x in bad["n"]]
    try:
        decide(rows, {}, longrows, {}, set(users[:722]), longusers, None, rk, fake_dr())
        ok = False
    except BucketMismatch:
        pass
    print(f"[sm148] selftest {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


def check():
    ok = sha(REPORT) == REPORT_SHA and U732.exists() and ULONG.exists() and BYTES.exists()
    if ok:
        r = bytes_rank()
        ok = all(sorted(SCHEMES, key=lambda s: r[size][s]) == SCHEMES for size in r)
    print(f"[sm148] pins {'OK' if ok else 'CHANGED'}")
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
