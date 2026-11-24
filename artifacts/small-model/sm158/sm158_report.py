#!/usr/bin/env python
"""Output-aware T24 on the 732 set (SM158): GPU smoke check, the ETA rule, the pre-registered rules and the go
(design.md "Rules", fixed 11-24 before any 732-set row of a weighted label was scored). CPU only.
Rows: T24RK and Q2RK from this run (gpu/sm158-rows-732.jsonl); REF, Q2 and T24 are the low-rank int8 states (SM148)'s 732
rows (../sm148/gpu/sm148-rows-732.jsonl). M is output-aware low-rank truncation (SM157)'s (../sm157/sm157-stats.json).

  python sm158_report.py smoke --gpu gpu/smoke-gpu-rows.jsonl --cpu cpu/smoke-cpu-rows.jsonl   exit 0 = smoke OK
  python sm158_report.py eta   --gpu gpu/smoke-gpu-rows.jsonl -> gpu/eta.json, prints "LABELS <names>" and the minutes
  python sm158_report.py report      -> sm158-result.{json,md}; exit 0 = decided, 2 = pending, 3 = invalid rows
  python sm158_report.py selftest    synthetic rows
  python sm158_report.py check       pins (showtime_report.py, SM148 rows, SM157 statistics)

Statistic: % = 100 x (sum of the label's LogLoss x size / the same for the reference - 1), review-weighted, paired user
bootstrap 2,000 draws, seed 0, 95% CI, on the benchmark LogLoss of the 722 rule users of the 732 set
(showtime_report.compare, as every 732 read-out). Helpers adapted from SM152's sm152_report.py.
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
REPORT = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/small-model/showtime/showtime_report.py")
REPORT_SHA = "3cdfed03afd40734d4b1931c244e8c1e73453d281a40c3843e80cd310d70dd9d"
OLD732 = HERE.parent / "sm148" / "gpu" / "sm148-rows-732.jsonl"
OLDSMOKE_GPU = HERE.parent / "sm148" / "gpu" / "smoke-gpu-rows.jsonl"
ROWS732 = G / "sm158-rows-732.jsonl"
STATS = HERE.parent / "sm157" / "sm157-stats.json"
STATS_SHA = "d0394c73e44d442a276e6e86031cde939abb942319f3068598ac2d8cae7c54fe"
BENCH = HERE.parent / "sm157" / "check-cpu-sm157.json"
BYTES = HERE.parent / "sm148" / "sm148-bytes.json"
ETA = G / "eta.json"
E2S = "@e2s-query"
LABELS = {"REF": "w8+s8hx8", "Q2": "w8+lr2q8x8", "T24": "w8+lr2.16.4q8x8",
          "T24RK": "w8+lr2.16.4q8x8wrk", "Q2RK": "w8+lr2q8x8wrk"}
NEWL = ("T24RK", "Q2RK")                      # scored in this run
PLAIN = {"T24RK": "T24", "Q2RK": "Q2"}        # the plain label of the same rank tiers (SM148's rows)
CEIL_BIG = 0.15                               # Big: vs its own int8, 95% CI upper bound per head (user 11-23)
CEIL_VS_BIG = {"Small": 0.5, "Tiny": 1.0}     # Small / Tiny: TOTAL cost vs the matching Big model at int8, CI upper per head
MIN_GAIN = 0.05                               # the go: D <= -0.05% with CI upper < 0 on a head of each Big model
MIN_732 = 715
ETA_LIMIT_H, WALL_OVER_MODEL, RETRY_H = 8.0, 1.3, 0.25
BUCKETS = ((0, 1), (1, 3), (3, 5), (5, 9), (9, 16), (16, 32), (32, 128), (128, None))   # report only
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
SIZE_MODELS = {s: [k for k, v in MODELS.items() if v[3] == s] for s in ("Tiny", "Small", "Big")}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def showtime():
    if sha(REPORT) != REPORT_SHA:
        raise SystemExit("[sm158] showtime_report.py changed: refusing")
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


def merged(old, new):
    ro, no, to = read_rows(old)
    rn, nn, tn = read_rows(new)
    rows = {**ro, **rn}
    nf = {k: set(no.get(k, ())) | set(nn.get(k, ())) for k in set(no) | set(nn)}
    return rows, nf, to + tn


def lab(s, tag):
    return f"{tag}/{LABELS[s]}{E2S}"


def as_loss(rows):
    out = {}
    for u, r in rows.items():
        b = r["benchmark"]
        out[u] = {h: ((b[h]["LogLoss"], b[h]["size"]) if h in b else (float("nan"), 0)) for h in ("ahead", "instant")}
    return out


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


def bucket_sums(row, head, lo, hi=None):
    pb = row.get("prior_buckets") or {}
    edges = pb.get("edges", [])
    h = (pb.get("all") or {}).get(head)
    if h is None or lo not in edges:
        return 0, 0.0
    idx = [i for i, e in enumerate(edges) if e >= lo and (hi is None or (edges[i + 1] if i + 1 < len(edges) else math.inf) <= hi)]
    return sum(h["n"][i] for i in idx), sum(h["ll"][i] for i in idx)


def bucket_cmp(new, ref, head, lo, hi=None, draws=2000):
    """Paired user bootstrap of the % of `new` vs `ref` on one answer-count bucket (all scope); report only."""
    users, a, b, mism = [], [], [], 0
    for u in sorted(set(new) & set(ref)):
        n1, l1 = bucket_sums(new[u], head, lo, hi)
        n0, l0 = bucket_sums(ref[u], head, lo, hi)
        if n1 != n0:
            mism += 1
            continue
        if n0:
            users.append(u); a.append(l1); b.append(l0)
    if not users:
        return None
    a, b = np.array(a), np.array(b)
    idx = np.random.default_rng(0).integers(0, len(users), size=(draws, len(users)))
    boot = (a[idx].sum(1) / b[idx].sum(1) - 1) * 100
    return {"value": float((a.sum() / b.sum() - 1) * 100), "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "users": len(users), "count_mismatch_users": mism}


def fmt(x):
    return "n/a" if not x else f"{x['value']:+.3f}% [{x['ci'][0]:+.3f}%, {x['ci'][1]:+.3f}%]"


def sub(rows, label, users):
    return {u: r for u, r in rows.get(label, {}).items() if u in users}


def stats_sha_problems(rows):
    bad = []
    for s in NEWL:
        for key, (tag, *_r) in MODELS.items():
            for u, r in rows.get(lab(s, tag), {}).items():
                d = r.get("state_lowrank") or {}
                if d.get("weight") != "rk" or d.get("weight_stats_sha256") != STATS_SHA:
                    bad.append([tag, s, u, d.get("weight"), d.get("weight_stats_sha256")])
    return bad


def decide(rows, nf, users732, scored=NEWL, st=None):
    """Budgets per model and label, weighted vs plain, and the go (design.md "Rules")."""
    res = {"rules": {"Big_vs_own_int8_upper_pct": CEIL_BIG, "Small_vs_Big_int8_upper_pct": CEIL_VS_BIG["Small"],
                     "Tiny_vs_Big_int8_upper_pct": CEIL_VS_BIG["Tiny"], "min_gain_pct": MIN_GAIN, "min_users": MIN_732},
           "scored": list(scored), "models": {}, "pending": [], "go": {}}
    for key, (tag, name, heads, size, bkey) in MODELS.items():
        m = {"name": name, "size": size, "heads": list(heads), "labels": {}}
        ref = sub(rows, lab("REF", tag), users732)
        bref = sub(rows, lab("REF", MODELS[bkey][0]), users732) if bkey else None
        if len(ref) < MIN_732:
            res["pending"].append(f"{tag} REF: {len(ref)} of {MIN_732} users")
        if bkey:
            m["big_match"] = MODELS[bkey][1]
            m["REF_vs_big"] = dataset_cmp(ref, bref, heads, sorted(set(ref) & set(bref)), st)
        for s in ["T24", "Q2", *scored]:
            new = sub(rows, lab(s, tag), users732)
            common = sorted(set(new) & set(ref))
            nonfin = sorted(nf.get(lab(s, tag), set()) & set(users732))
            x = {"users": len(common), "nonfinite_users": nonfin}
            if len(common) < MIN_732:
                res["pending"].append(f"{tag} {s}: {len(common)} of {MIN_732} users")
            x["finite"] = not nonfin and all(math.isfinite(new[u]["benchmark"][KEY[h]]["LogLoss"]) for u in common for h in heads)
            x["covered"] = len(common) >= MIN_732
            x["vs_own_int8"] = own = dataset_cmp(new, ref, heads, common, st)
            if bkey:
                cb = sorted(set(new) & set(bref))
                x["vs_big_int8"] = vb = dataset_cmp(new, bref, heads, cb, st)
                x["budget_ok"] = bool(vb) and all(vb[h]["ci"][1] <= CEIL_VS_BIG[size] for h in heads)
            else:
                x["budget_ok"] = bool(own) and all(own[h]["ci"][1] <= CEIL_BIG for h in heads)
            x["budget_pass"] = bool(x["finite"] and x["covered"] and x["budget_ok"])
            if s in NEWL:
                plain = sub(rows, lab(PLAIN[s], tag), users732)
                cp = sorted(set(new) & set(plain))
                x["vs_plain"] = d = dataset_cmp(new, plain, heads, cp, st)
                x["gain_heads"] = [h for h in heads if d and d[h]["ci"][1] < 0 and d[h]["value"] <= -MIN_GAIN]
                x["worse_heads"] = [h for h in heads if d and d[h]["ci"][0] > 0]
                x["buckets_vs_plain"] = {f"{lo}-{hi - 1 if hi else ''}": {h: bucket_cmp(new, plain, KEY[h], lo, hi) for h in heads}
                                         for lo, hi in BUCKETS} if cp else {}
                if s == "Q2RK":
                    for other in ("T24", "T24RK"):
                        o = sub(rows, lab(other, tag), users732)
                        x[f"vs_{other}"] = dataset_cmp(new, o, heads, sorted(set(new) & set(o)), st) if o else {}
            m["labels"][s] = x
        res["models"][key] = m
    res["status"] = "pending" if res["pending"] else "decided"
    M = res["models"]
    t = {k: M[k]["labels"]["T24RK"] for k in MODELS}
    a = {k: bool(t[k]["gain_heads"]) for k in SIZE_MODELS["Big"]}
    b = {k: t[k]["worse_heads"] for k in MODELS if t[k]["worse_heads"]}
    c = {k: bool(t[k]["finite"] and t[k]["covered"]) for k in MODELS}
    go = all(a.values()) and not b and all(c.values())
    res["go"] = {"go": go, "a_gain_each_big": a, "b_significantly_worse": b, "c_finite_and_covered": c, "per_size": {}}
    for size, keys in SIZE_MODELS.items():
        rk_ok = all(M[k]["labels"]["T24RK"]["budget_pass"] for k in keys)
        plain_ok = all(M[k]["labels"]["T24"]["budget_pass"] for k in keys)
        res["go"]["per_size"][size] = {"T24RK_meets_budgets": rk_ok, "T24_meets_budgets": plain_ok,
                                       "format": "T24RK" if (go and rk_ok) else "T24"}
    if "Q2RK" in scored:
        res["q2rk_info"] = {size: {"Q2RK_meets_budgets": all(M[k]["labels"]["Q2RK"]["budget_pass"] for k in keys),
                                   "Q2_meets_budgets": all(M[k]["labels"]["Q2"]["budget_pass"] for k in keys)}
                            for size, keys in SIZE_MODELS.items()}
    return res


def speed(rows, scored):
    """Report only: review evaluations per second (model time) per model and label on the 732 set."""
    out = {}
    for key, (tag, *_r) in MODELS.items():
        for s in ["REF", "T24", "Q2", *scored]:
            rr = list(rows.get(lab(s, tag), {}).values())
            sec = sum(r.get("model_seconds", 0) for r in rr)
            rev = sum(r.get("reviews", 0) for r in rr)
            out[f"{key} {s}"] = {"review_evals_per_s": round(rev / sec) if sec else None, "model_seconds": round(sec, 1),
                                 "users": len(rr)}
        for s in scored:
            v, p = out[f"{key} {s}"]["review_evals_per_s"], out[f"{key} {PLAIN[s]}"]["review_evals_per_s"]
            out[f"{key} {s}"]["vs_plain"] = round(v / p, 3) if v and p else None
    return out


def write_md(res):
    L = ["# Output-aware T24 on the 732 set (SM158): result tables", "",
         f"Status: **{res['status']}**. Generated by `sm158_report.py report`. Labels scored here: {', '.join(res['scored'])}; "
         "REF, Q2 and T24 rows are the low-rank int8 states (SM148)'s. 95% paired user bootstrap on the 722 rule users of the "
         "732 set.", "",
         f"Rules (design.md): budgets per head on the CI upper bound, Big vs its own int8 <= +{CEIL_BIG}%, Small <= "
         f"+{CEIL_VS_BIG['Small']}% and Tiny <= +{CEIL_VS_BIG['Tiny']}% total vs the matching Big model at int8; every user "
         f"finite; at least {MIN_732} users. Go for T24RK over T24: on each Big model a head with D = T24RK vs T24 at most "
         f"-{MIN_GAIN}% and CI upper below 0; no head of any model with the CI lower bound of D above 0; T24RK finite and "
         "covered on every model.", ""]
    g = res["go"]
    L += [f"**Go: {'YES' if g['go'] else 'NO'}.** (a) gain on each Big model: "
          + ", ".join(f"{res['models'][k]['name']} {'yes' if v else 'no'}" for k, v in g["a_gain_each_big"].items())
          + "; (b) significantly worse heads: "
          + ("; ".join(f"{res['models'][k]['name']} {', '.join(v)}" for k, v in g["b_significantly_worse"].items()) or "none")
          + "; (c) finite and covered: " + ("all" if all(g["c_finite_and_covered"].values()) else
                                             ", ".join(res['models'][k]['name'] for k, v in g["c_finite_and_covered"].items() if not v) + " not")
          + ".", "", "| size | T24 meets budgets | T24RK meets budgets | format |", "|---|---|---|---|"]
    for size, v in g["per_size"].items():
        L.append(f"| {size} | {'yes' if v['T24_meets_budgets'] else 'no'} | {'yes' if v['T24RK_meets_budgets'] else 'no'} | {v['format']} |")
    L += ["", "## Per model, label and head", "",
          "| model | label | head | vs own int8 | vs Big int8 | vs plain (D) | budget | users |", "|---|---|---|---|---|---|---|---|"]
    for key, m in res["models"].items():
        for s, x in m["labels"].items():
            for h in m["heads"]:
                L.append(f"| {m['name']} | {s} | {h} | {fmt(x['vs_own_int8'].get(h))} | "
                         f"{fmt(x['vs_big_int8'].get(h)) if 'vs_big_int8' in x else '-'} | "
                         f"{fmt(x['vs_plain'].get(h)) if x.get('vs_plain') else '-'} | {'pass' if x['budget_pass'] else 'fail'} | {x['users']} |")
    if "Q2RK" in res["scored"]:
        L += ["", "## Q2RK (information): against T24 and T24RK", "", "| model | head | Q2RK vs T24 | Q2RK vs T24RK |", "|---|---|---|---|"]
        for key, m in res["models"].items():
            x = m["labels"]["Q2RK"]
            for h in m["heads"]:
                L.append(f"| {m['name']} | {h} | {fmt(x['vs_T24'].get(h))} | {fmt(x['vs_T24RK'].get(h))} |")
    L += ["", "## T24RK vs T24 by answers of the card before the review (all scope, report only)", "",
          "| model | head | " + " | ".join(f"{lo}-{hi - 1 if hi else ''}" for lo, hi in BUCKETS) + " |",
          "|---|---|" + "---|" * len(BUCKETS)]
    for key, m in res["models"].items():
        bk = m["labels"]["T24RK"].get("buckets_vs_plain") or {}
        for h in m["heads"]:
            cells = []
            for lo, hi in BUCKETS:
                v = bk.get(f"{lo}-{hi - 1 if hi else ''}", {}).get(h)
                cells.append("n/a" if v is None else f"{v['value']:+.3f}%")
            L.append(f"| {m['name']} | {h} | " + " | ".join(cells) + " |")
    ro = res.get("report_only", {})
    sp = ro.get("speed", {})
    if sp:
        L += ["", "## Speed (report only): GPU review evaluations per second, model time, 732 set", "",
              "| model | REF | T24 | Q2 | " + " | ".join(res["scored"]) + " |", "|---|---|---|---|" + "---|" * len(res["scored"])]
        for key, m in res["models"].items():
            cells = [str(sp[f"{key} {s}"]["review_evals_per_s"]) for s in ("REF", "T24", "Q2")]
            cells += [f"{sp[f'{key} {s}']['review_evals_per_s']} ({sp[f'{key} {s}']['vs_plain']}x plain)" for s in res["scored"]]
            L.append(f"| {m['name']} | " + " | ".join(cells) + " |")
    if ro.get("codec_bench_cpu"):
        L += ["", "Codec cost per write (CPU microbenchmark of SM157, one thread, rank 2, int8 factors, 4 heads): "
              + "; ".join(f"{k.split('_')[0]} plain {v['plain']:.1f} us, weighted {v['weighted']:.1f} us ({v['extra_pct']:+.0f}%)"
                          for k, v in ro["codec_bench_cpu"].items()) + "."]
    if ro.get("eta"):
        e = ro["eta"]
        L += ["", f"ETA rule: estimated {e['eta_hours']:.2f} GPU hours for both labels (limit {ETA_LIMIT_H} h): scored {', '.join(e['labels'])}."]
    (HERE / "sm158-result.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def scored_labels():
    if ETA.exists():
        return tuple(json.loads(ETA.read_text())["labels"])
    return NEWL


def report():
    st = showtime()
    users732, _ = st.pools()
    users732 = set(users732)
    rows, nf, torn = merged(OLD732, ROWS732)
    scored = scored_labels()
    bad = stats_sha_problems(rows)
    res = decide(rows, nf, users732, scored, st)
    res["torn_lines"] = torn
    res["stats_sha_problems"] = bad[:20]
    res["report_only"] = {"speed": speed(rows, scored),
                          "codec_bench_cpu": json.loads(BENCH.read_text())["info"]["bench"] if BENCH.exists() else None,
                          "eta": json.loads(ETA.read_text()) if ETA.exists() else None,
                          "bytes": {s: {k: json.loads(BYTES.read_text())["sizes"][s][k]["vs_int8_pct"] for k in ("T24", "Q2")}
                                    for s in ("Big", "Small", "Tiny")}}
    if bad:
        res["status"] = "invalid"
    (HERE / "sm158-result.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    write_md(res)
    if bad:
        print(f"[sm158] STOP: {len(bad)} weighted rows without the SM157 statistics sha256: {bad[:3]}")
        return 3
    if res["status"] == "pending":
        print(f"[sm158] report pending: {res['pending'][:6]}")
        return 2
    print(f"[sm158] report decided: go {'YES' if res['go']['go'] else 'NO'}; "
          + "; ".join(f"{s} -> {v['format']}" for s, v in res["go"]["per_size"].items()))
    return 0


def smoke(gpu, cpu):
    """GPU vs CPU on the smoke users: REF strict (1e-5 per user and head); T24, Q2RK, T24RK by the discrete-codec rule
    (per user <= 3e-4, review-weighted mean signed difference <= 1e-4, per head); REF and T24 on the GPU equal SM148's GPU
    smoke rows within 1e-5 per user and head (same counts)."""
    g, gnf, _ = read_rows(gpu)
    c, cnf, _ = read_rows(cpu)
    o, _onf, _ = read_rows(OLDSMOKE_GPU)
    users = json.loads((HERE / "users-smoke.json").read_text())["users"]
    names = ["REF", "T24", "Q2RK", "T24RK"]
    ok, lines, rep = True, [], {}
    for key, (tag, name, heads, size, _b) in MODELS.items():
        for s in names:
            L = lab(s, tag)
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
                good = worst <= TOL_STRICT + TOL_EPS if s == "REF" else (worst <= TOL_DISCRETE + TOL_EPS and abs(mean) <= TOL_DISCRETE_MEAN + TOL_EPS)
                rep[f"{L} {h}"] = {"max_abs": worst, "mean_signed": mean, "pass": good}
                ok &= good
                if s in ("REF", "T24"):
                    if set(o.get(L, {})) != set(users):
                        ok = False; lines.append(f"{L}: SM148 GPU smoke rows missing")
                        continue
                    d2 = max(abs(g[L][u]["benchmark"][h]["LogLoss"] - o[L][u]["benchmark"][h]["LogLoss"]) for u in users)
                    n2 = all(g[L][u]["benchmark"][h]["size"] == o[L][u]["benchmark"][h]["size"] for u in users)
                    good2 = n2 and d2 <= TOL_STRICT + TOL_EPS
                    rep[f"{L} {h} vs SM148 GPU"] = {"max_abs": d2, "same_counts": n2, "pass": good2}
                    ok &= good2
    bad = stats_sha_problems(g)
    if bad:
        ok = False; lines.append(f"weighted GPU rows without the SM157 statistics sha256: {bad[:3]}")
    G.mkdir(exist_ok=True)
    (G / "smoke-check.json").write_text(json.dumps({"ok": ok, "problems": lines, "checks": rep}, indent=1) + "\n", encoding="utf-8")
    wi = max((v["max_abs"] for k, v in rep.items() if "s8hx8" in k and "SM148" not in k), default=0)
    wl = max((v["max_abs"] for k, v in rep.items() if "s8hx8" not in k and "SM148" not in k), default=0)
    wo = max((v["max_abs"] for k, v in rep.items() if "SM148" in k), default=0)
    print(f"[sm158] smoke {'OK' if ok else 'FAILED'}: {len(rep)} checks; REF max |GPU - CPU| {wi:.2e} (tol {TOL_STRICT:g}); "
          f"low-rank labels max {wl:.2e} (tol {TOL_DISCRETE:g}, mean {TOL_DISCRETE_MEAN:g}); REF and T24 vs SM148's GPU rows "
          f"max {wo:.2e} (tol {TOL_STRICT:g})" + (f"; {lines[:3]}" if lines else ""))
    return 0 if ok else 1


def eta_doc(smoke_rows, old_smoke, old732):
    """The ETA rule (design.md "ETA rule"): slowdown per model and weighted label vs SM148's plain label on the same smoke
    users (floored at 1) x SM148's 732 model seconds of that plain label, x 1.3 wall, + 0.25 h; > 8 h -> T24RK only."""
    def spd(rows, L):
        rr = list(rows.get(L, {}).values())
        sec, rev = sum(r.get("model_seconds", 0) for r in rr), sum(r.get("reviews", 0) for r in rr)
        return (rev / sec if sec else None), sec, len(rr)
    per, size_s = {}, {s: {l: 0.0 for l in NEWL} for s in SIZE_MODELS}
    for key, (tag, name, heads, size, _b) in MODELS.items():
        for s in NEWL:
            v_new, _, n_new = spd(smoke_rows, lab(s, tag))
            v_old, _, _ = spd(old_smoke, lab(PLAIN[s], tag))
            _, sec732, n732 = spd(old732, lab(PLAIN[s], tag))
            if not v_new or not v_old or not sec732:
                raise SystemExit(f"[sm158] eta: missing speed for {tag} {s} (smoke {v_new}, SM148 smoke {v_old}, SM148 732 {sec732})")
            slow = max(1.0, v_old / v_new)
            est = sec732 * slow * WALL_OVER_MODEL
            per[f"{key} {s}"] = {"smoke_review_evals_per_s": round(v_new), "sm148_smoke_plain_per_s": round(v_old),
                                 "slowdown": round(slow, 3), "sm148_732_model_seconds": round(sec732, 1),
                                 "est_wall_minutes": round(est / 60, 1), "smoke_users": n_new}
            size_s[size][s] += est
    total_h = sum(v for d in size_s.values() for v in d.values()) / 3600 + RETRY_H
    labels = list(NEWL) if total_h <= ETA_LIMIT_H else ["T24RK"]
    minutes = {s: max(20, math.ceil(sum(size_s[s][l] for l in labels) / 60 * 1.15)) for s in SIZE_MODELS}
    t24_h = sum(size_s[s]["T24RK"] for s in SIZE_MODELS) / 3600 + RETRY_H
    return {"eta_hours": round(total_h, 2), "eta_hours_T24RK_only": round(t24_h, 2), "limit_hours": ETA_LIMIT_H,
            "labels": labels, "dropped": [l for l in NEWL if l not in labels], "minutes_per_size": minutes,
            "rule": "design.md ETA rule: slowdown vs SM148's plain label on the smoke users (floor 1) x SM148's 732 model "
                    "seconds x 1.3 + 0.25 h; > 8 h -> T24RK only", "per_model": per}


def eta(gpu):
    g, _, _ = read_rows(gpu)
    o, _, _ = read_rows(OLDSMOKE_GPU)
    r, _, _ = read_rows(OLD732)
    doc = eta_doc(g, o, r)
    G.mkdir(exist_ok=True)
    ETA.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(f"LABELS {' '.join(doc['labels'])}")
    print("MINUTES " + " ".join(f"{s}={m}" for s, m in doc["minutes_per_size"].items()))
    print(f"[sm158] ETA {doc['eta_hours']:.2f} GPU hours for {', '.join(NEWL)} (limit {ETA_LIMIT_H} h; T24RK only "
          f"{doc['eta_hours_T24RK_only']:.2f} h): scoring {', '.join(doc['labels'])}")
    return 0


def selftest():
    users = list(range(1, 761))
    rule = set(users[:722])

    def row(u, L, k=1.0, base_k=1.0, ms=1.0):
        base = {h: (0.3 + 0.01 * ((u * 7 + i) % 5)) * base_k for i, h in enumerate(("ahead", "instant"))}
        d = {"weight": "rk", "weight_stats_sha256": STATS_SHA} if L.split("/")[1].startswith(("w8+lr2.16.4q8x8wrk", "w8+lr2q8x8wrk")) else {}
        return {"user_id": u, "label": L, "benchmark": {h: {"LogLoss": base[h] * k, "size": 100} for h in base},
                "model_seconds": ms, "reviews": 100, "state_lowrank": d}

    def build(mod, scored=NEWL):
        rows = {}
        for key, (tag, name, heads, size, bkey) in MODELS.items():
            for s in ["REF", "T24", "Q2", *scored]:
                L = lab(s, tag)
                kw = dict(mod.get((key, s), {}))
                if size != "Big":
                    kw.setdefault("base_k", mod.get("tiny_k" if size == "Tiny" else "small_k", 1.0))
                rows[L] = {u: row(u, L, **kw) for u in users}
        return rows

    def run(mod, nf=None, scored=NEWL, n=722):
        return decide(build(mod, scored), nf or {}, set(users[:n]), scored, None)

    ok = True
    g = {("sm91s2", "T24RK"): {"k": 0.999}, ("sm99", "T24RK"): {"k": 0.999}}   # -0.1% on every head of both Big models
    r = run(g)
    ok &= r["status"] == "decided" and r["go"]["go"] and all(v["format"] == "T24RK" for v in r["go"]["per_size"].values())
    r = run({("sm91s2", "T24RK"): {"k": 0.999}})                                # no gain on Big Instant-Only: no go
    ok &= not r["go"]["go"] and r["go"]["a_gain_each_big"] == {"sm91s2": True, "sm99": False}
    ok &= all(v["format"] == "T24" for v in r["go"]["per_size"].values())
    r = run({("sm91s2", "T24RK"): {"k": 0.9996}, ("sm99", "T24RK"): {"k": 0.999}})   # -0.04% < 0.05%: no go
    ok &= not r["go"]["go"]
    r = run({**g, ("sm118", "T24RK"): {"k": 1.001}})                            # Small significantly worse: no go
    ok &= not r["go"]["go"] and "sm118" in r["go"]["b_significantly_worse"]
    r = run({**g, "small_k": 1.003, ("sm118", "T24RK"): {"k": 0.9985}})         # Small T24RK +0.15% vs Big, budget 0.5: passes
    ok &= r["go"]["go"] and r["go"]["per_size"]["Small"]["format"] == "T24RK"
    ok &= abs(r["models"]["sm118"]["REF_vs_big"]["curve"]["value"] - 0.3) < 0.01
    r = run({**g, "small_k": 1.006})                                            # Small at +0.6% vs Big: T24 and T24RK fail, T24 stays
    ok &= r["go"]["go"] and r["go"]["per_size"]["Small"]["format"] == "T24" and not r["go"]["per_size"]["Small"]["T24RK_meets_budgets"]
    ok &= r["go"]["per_size"]["Big"]["format"] == "T24RK"
    r = run({**g, ("sm99", "T24"): {"k": 1.002}, ("sm99", "T24RK"): {"k": 1.0012}})   # Big: plain +0.2% fails, rk +0.12% passes
    ok &= r["go"]["go"] and not r["go"]["per_size"]["Big"]["T24_meets_budgets"] and r["go"]["per_size"]["Big"]["T24RK_meets_budgets"]
    nf = {lab("T24RK", MODELS["sm130io"][0]): {5}}
    r = run(g, nf)                                                              # a non-finite T24RK user: no go (c)
    ok &= not r["go"]["go"] and r["go"]["c_finite_and_covered"]["sm130io"] is False
    r = run(g, n=700)                                                           # fewer than 715 users: pending
    ok &= r["status"] == "pending"
    r = run(g, scored=("T24RK",))                                               # Q2RK dropped by the ETA rule: still decided
    ok &= r["status"] == "decided" and r["go"]["go"] and "q2rk_info" not in r
    bad = build(g); bad[lab("T24RK", MODELS["sm99"][0])][3]["state_lowrank"]["weight_stats_sha256"] = "x"
    ok &= len(stats_sha_problems(bad)) == 1 and not stats_sha_problems(build(g))
    # ETA rule: twice as slow as SM148's plain labels
    sm = build({}); old = build({})
    for L in sm:
        for u in sm[L]:
            sm[L][u]["model_seconds"] = 2.0
    e = eta_doc(sm, old, old)
    ok &= e["labels"] == list(NEWL) and all(v["slowdown"] == 2.0 for v in e["per_model"].values())
    for L in old:
        for u in old[L]:
            old[L][u]["model_seconds"] = 30.0                                     # 732 seconds large: over 8 h -> T24RK only
    e = eta_doc(sm, build({}), old)
    ok &= e["labels"] == ["T24RK"] and e["dropped"] == ["Q2RK"]
    print(f"[sm158] selftest {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


def check():
    ok = sha(REPORT) == REPORT_SHA and OLD732.exists() and OLDSMOKE_GPU.exists() and BENCH.exists() and BYTES.exists() \
        and STATS.exists() and sha(STATS) == STATS_SHA
    print(f"[sm158] pins {'OK' if ok else 'CHANGED'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("smoke"); s.add_argument("--gpu", required=True); s.add_argument("--cpu", required=True)
    e = sp.add_parser("eta"); e.add_argument("--gpu", required=True)
    sp.add_parser("report"); sp.add_parser("selftest"); sp.add_parser("check")
    a = ap.parse_args()
    if a.cmd == "smoke":
        return smoke(a.gpu, a.cpu)
    if a.cmd == "eta":
        return eta(a.gpu)
    return {"report": report, "selftest": selftest, "check": check}[a.cmd]()


if __name__ == "__main__":
    raise SystemExit(main())
