#!/usr/bin/env python
"""Long-card state drift on real users (SM136): pre-registered rule (fixed 2026-11-20 before any GPU ticket; CPU, seconds).

Question: on real users, how much LogLoss do int4 card rows (the fork default w8+s4c8n and the compact w8+s4c8nhx8) lose
against int8 card rows (the published w8+s8) on reviews of cards with many prior answers? Decides whether the fork should
store int8 card rows (all sizes or Big only) instead of int4 for the 11-28 release candidate.

Models (one per size and head chain, the family candidates; no training; show-time inputs, leak-free, w8 weights):
  Big    the show-time big model (SM91s2)                        sm91s2-big-showtime                 curve, instant
         the show-time instant-only big model (SM99)             sm99-big-inst-e2                    instant
  Small  the show-time two-head cool-down for Small (SM118)      sm118-e26-dec2ep32k-t91s2           curve, instant
         the longer Small Instant-Only model (SM113)             sm113-inst-e35st-st-dec2ep32k-t99   instant
  Tiny   the show-time two-head cool-down for Tiny (SM124)       sm124-c1-e29st-dec2ep32k-t91s2      curve, instant
         the longer cool-down for Tiny (SM115)                   sm115-c1-e38st-st-dec2ep32k-t99     instant
Formats: w8+s8 (int8 everywhere, the published format, the reference), w8+s4c8n (fork default), w8+s4c8nhx8 (compact),
w8+s8hx8 (int8 card rows with the compact blocks, report only), w8 (float32 state, report only).

Readout: the scorer's --prior-buckets 0,32,128,300,1000 rows (scripts/small_model_heldout.py, SM136 flag): per user,
head and bucket of the card's prior answers, the count and LogLoss sum of every review the scorer predicts ("all" scope;
the benchmark set stops at 127 prior answers, so it cannot see long cards). Buckets 0-31, 32-127, 128-299, 300-999,
1000+, pooled 128+ and 300+, and "all".
Statistic: per head and bucket, % = 100 x (sum LogLoss of the format / sum LogLoss of w8+s8 - 1) over the bucket's
reviews (review-weighted), paired user bootstrap over the users with reviews in the bucket (2,000 draws, seed 0), 95% CI.
RULE (per size and per int4 format; the release decision follows the compact format):
  rule buckets = the buckets with >= 128 prior answers (128-299, 300-999, 1000+, 300+, 128+) that have >= 20 users of the
                 main pool (pool-sm136.json fixes this before scoring: 128-299 and 128+ with 66 users; 300-999, 300+ (6
                 users) and 1000+ (1 user) are report only)
  acceptable(model, format) = no main-pool user has a non-finite row for the format (an overflowing card: the fork errors
                 or panics) AND in every rule bucket every head's 95% CI upper bound <= +0.25%
                 (two-head: curve and instant; Instant-Only: instant)
  int4 card rows acceptable for a size = both models acceptable; otherwise recommend int8 card rows for that size
Users: the main pool (users-sm136.json, 66 users); per model and format the users with finite rows for the format and for
w8+s8; at least MIN_USERS = 64 of 66 must have a row (finite or non-finite) for every format, otherwise `decide` exits 2
(pending). The bucket counts must be identical across formats (they depend only on the inputs), otherwise exit 3.
Reported, no verdict: every format vs w8+s8 and vs float32 (w8), per bucket and head; the "all" and benchmark totals of
the scored pool; non-finite users per format; the extension pool (users-sm136-ext.json, report only) and the combined
main + extension 300+ bucket; w8+s8 rows vs the published GPU rows (max |difference| of the benchmark LogLoss); the bytes
(sm136-bytes.json); the whole-pool context of the compact states for Tiny (SM129) and for Small and Big (SM131).

  python sm136_rule.py selftest   -> the rule's code on synthetic rows (CPU, seconds)
  python sm136_rule.py decide     -> sm136-decision.{json,md}; exit 2 while rows are missing
"""
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
G = HERE / "gpu"
SMG = HERE.parent
W = Path("C:/Users/jonat/workspace/wt")
ROWS, EXT_ROWS = G / "sm136-rows.jsonl", G / "sm136-ext-rows.jsonl"
POOL, USERS, EXT_USERS = HERE / "pool-sm136.json", HERE / "users-sm136.json", HERE / "users-sm136-ext.json"
E2S = "@e2s-query"
REF, DEFAULT, COMPACT, INT8C, F32 = "w8+s8", "w8+s4c8n", "w8+s4c8nhx8", "w8+s8hx8", "w8"
FORMATS = (REF, DEFAULT, COMPACT, INT8C, F32)
INT4 = (COMPACT, DEFAULT)
FORMAT_NAMES = {REF: "int8 everywhere (published w8+s8)", DEFAULT: "fork default (w8+s4c8n)", COMPACT: "compact (w8+s4c8nhx8)",
                INT8C: "int8 cards, compact blocks (w8+s8hx8)", F32: "float32 state (w8)"}
MODELS = {
    "sm91s2": {"size": "Big", "tag": "sm91s2-big-showtime", "heads": ("curve", "instant"),
               "name": "the show-time big model (SM91s2)",
               "pub": W / "sm91s2/artifacts/small-model/sm91s2/gpu/sm91s2-inputs.jsonl"},
    "sm99": {"size": "Big", "tag": "sm99-big-inst-e2", "heads": ("instant",),
             "name": "the show-time instant-only big model (SM99)", "pub": W / "sm99/artifacts/small-model/sm99/gpu/sm99-rows.jsonl"},
    "sm118": {"size": "Small", "tag": "sm118-e26-dec2ep32k-t91s2", "heads": ("curve", "instant"),
              "name": "the show-time two-head cool-down for Small (SM118)", "pub": SMG / "sm118/gpu/sm118-rows.jsonl"},
    "sm113": {"size": "Small", "tag": "sm113-inst-e35st-st-dec2ep32k-t99", "heads": ("instant",),
              "name": "the longer Small Instant-Only model (SM113)",
              "pub": W / "sm113/artifacts/small-model/sm113/gpu/sm113-rows.jsonl"},
    "sm124": {"size": "Tiny", "tag": "sm124-c1-e29st-dec2ep32k-t91s2", "heads": ("curve", "instant"),
              "name": "the show-time two-head cool-down for Tiny (SM124)", "pub": SMG / "sm124/gpu/sm124-rows.jsonl"},
    "sm115": {"size": "Tiny", "tag": "sm115-c1-e38st-st-dec2ep32k-t99", "heads": ("instant",),
              "name": "the longer cool-down for Tiny (SM115)", "pub": SMG / "sm115/gpu/sm115-rows.jsonl"},
}
SIZES = {"Big": ("sm91s2", "sm99"), "Small": ("sm118", "sm113"), "Tiny": ("sm124", "sm115")}
BYTES_ARCH = {"Big": "Big (fork)", "Small": "Small (w64)", "Tiny": "Tiny (w64-c1)"}
HKEY = {"curve": "ahead", "instant": "instant"}
BUCKETS = ["0-31", "32-127", "128-299", "300-999", "1000+"]
EDGES = [0, 32, 128, 300, 1000]
GROUPS = {**{b: [i] for i, b in enumerate(BUCKETS)}, "128+": [2, 3, 4], "300+": [3, 4], "all": [0, 1, 2, 3, 4]}
RULE_CANDIDATES = ("128-299", "300-999", "1000+", "300+", "128+")
CEILING = 0.25
MIN_BUCKET_USERS = 20
MIN_USERS = 64


def label(model, fmt):
    return f"{MODELS[model]['tag']}/{fmt}{E2S}"


def load(path, counts):
    """{(user, label): row} of the parseable rows ('nonfinite' rows -> {'nonfinite': True}); error rows skipped; the
    number of unparseable lines goes to counts[path]."""
    out, bad = {}, 0
    path = Path(path)
    if path.exists():
        for raw in open(path, "rb"):
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                key = (r["user_id"], r["label"])
            except (ValueError, KeyError, TypeError):
                bad += 1
                continue
            if "error" in r:
                continue
            if "nonfinite" in r and "benchmark" not in r:
                out[key] = {"nonfinite": True}
            elif "benchmark" in r:
                out[key] = r
    counts[str(path)] = bad
    return out


def sums(row, head, scope="all"):
    """(n[5], ll[5]) of a row's prior buckets, or None (no buckets / non-finite numbers)."""
    pb = row.get("prior_buckets", {}).get(scope, {}).get(HKEY[head])
    if pb is None or pb.get("n") is None:
        return None
    n, ll = np.asarray(pb["n"], dtype=np.int64), np.asarray(pb["ll"], dtype=np.float64)
    if len(n) != len(BUCKETS) or not np.all(np.isfinite(ll)):
        return None
    return n, ll


def finite(row, heads):
    return row is not None and "nonfinite" not in row and all(sums(row, h) is not None for h in heads)


def compare(new, ref, users, head, group, scope="all"):
    """100 x (sum new / sum ref - 1) over the group's reviews of `users` with reviews there; paired user bootstrap."""
    idx = GROUPS[group]
    rows = []
    for u in users:
        sa, sb = sums(new[u], head, scope), sums(ref[u], head, scope)
        if not np.array_equal(sa[0], sb[0]):
            raise ValueError(f"bucket counts differ for user {u} ({head}): {sa[0].tolist()} vs {sb[0].tolist()}")
        n = int(sa[0][idx].sum())
        if n:
            rows.append((n, float(sa[1][idx].sum()), float(sb[1][idx].sum())))
    if not rows:
        return {"users": 0, "reviews": 0}
    n, a, b = (np.array(x) for x in zip(*rows))
    k = len(n)
    boot = np.random.default_rng(0).integers(0, k, size=(2000, k))
    pct_boot = (a[boot].sum(1) / b[boot].sum(1) - 1) * 100
    return {"users": k, "reviews": int(n.sum()), "new": round(float(a.sum() / n.sum()), 6), "ref": round(float(b.sum() / n.sum()), 6),
            "delta": round(float((a.sum() - b.sum()) / n.sum()), 7), "pct": round(float((a.sum() / b.sum() - 1) * 100), 4),
            "ci": [round(float(np.percentile(pct_boot, q)), 4) for q in (2.5, 97.5)],
            "users_better": int((a < b).sum()), "users_worse": int((a > b).sum())}


def table(rows, model, fmt, ref_fmt, users, scope="all"):
    m = MODELS[model]
    new, ref = (rows.get(label(model, f)) for f in (fmt, ref_fmt))
    ok = [u for u in users if finite(new.get(u), m["heads"]) and finite(ref.get(u), m["heads"])]
    return {h: {g: compare(new, ref, ok, h, g, scope) for g in GROUPS} for h in m["heads"]}, ok


def by_label(raw):
    out = {}
    for (u, lab), r in raw.items():
        out.setdefault(lab, {})[u] = r
    return out


def bucket_users(rows, model, users):
    """users with reviews per group (from the w8+s8 rows; the counts are format independent)."""
    ref = rows.get(label(model, REF), {})
    out = {}
    for g, idx in GROUPS.items():
        out[g] = sum(1 for u in users if finite(ref.get(u), ("instant",)) and sums(ref[u], "instant")[0][idx].sum() > 0)
    return out


def vs_published(rows, model, users):
    """max |w8+s8 benchmark LogLoss - published GPU row| per head over the users both have (integrity, report only)."""
    pub = {}
    path = MODELS[model]["pub"]
    if not Path(path).exists():
        return None
    want = label(model, REF)
    for line in open(path, encoding="utf-8", errors="replace"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("label") == want and "benchmark" in r and r["user_id"] in users:
            pub[r["user_id"]] = r["benchmark"]
    ours = rows.get(want, {})
    common = [u for u in users if u in pub and finite(ours.get(u), MODELS[model]["heads"])]
    return {"users": len(common), **{h: max((abs(ours[u]["benchmark"][HKEY[h]]["LogLoss"] - pub[u][HKEY[h]]["LogLoss"])
                                              for u in common), default=None) for h in MODELS[model]["heads"]}}


def decide(rows_path=ROWS, ext_path=EXT_ROWS, out_dir=HERE, users=None, ext_users=None, write=True):
    users = users if users is not None else json.loads(USERS.read_text())["users"]
    ext_users = ext_users if ext_users is not None else (json.loads(EXT_USERS.read_text())["users"] if EXT_USERS.exists() else [])
    counts = {}
    rows, ext = by_label(load(rows_path, counts)), by_label(load(ext_path, counts))
    res = {"rule": (f"int4 card rows acceptable for a size iff, for both models, no main-pool user is non-finite for the format "
                    f"and in every rule bucket (>= 128 prior answers, >= {MIN_BUCKET_USERS} users) every head's 95% CI "
                    f"upper bound of 100 x (format / w8+s8 - 1) <= +{CEILING:.2f}%; otherwise recommend int8 card rows"),
           "pool_users": len(users), "extension_users": len(ext_users), "unparseable_lines": counts, "models": {}, "sizes": {}}
    pending = []
    for model, m in MODELS.items():
        heads = m["heads"]
        out = {"name": m["name"], "heads": list(heads), "formats": {}}
        cover = {}
        for f in FORMATS:
            got = rows.get(label(model, f), {})
            cover[f] = {"rows": sum(1 for u in users if u in got),
                        "nonfinite_users": sorted(u for u in users if u in got and not finite(got[u], heads)),
                        "missing_users": sorted(u for u in users if u not in got)}
            if cover[f]["rows"] < MIN_USERS:
                pending.append(f"{model} {f}: {cover[f]['rows']} of {len(users)} users")
        out["coverage"] = cover
        if any(p.startswith(model + " ") for p in pending):
            res["models"][model] = out
            continue
        if cover[REF]["nonfinite_users"]:
            out["note"] = "w8+s8 non-finite for some users: excluded from every comparison"
        bu = bucket_users(rows, model, users)
        out["bucket_users"] = bu
        out["rule_buckets"] = [g for g in RULE_CANDIDATES if bu[g] >= MIN_BUCKET_USERS]
        for f in FORMATS[1:]:
            t, ok = table(rows, model, f, REF, users)
            entry = {"users": len(ok), "nonfinite_users": cover[f]["nonfinite_users"], "vs_w8s8": t,
                     "benchmark_vs_w8s8": table(rows, model, f, REF, users, scope="benchmark")[0]}
            if f != F32:
                entry["vs_float32"] = table(rows, model, f, F32, users)[0]
            if f in INT4:
                worst = {h: max(t[h][g]["ci"][1] for g in out["rule_buckets"]) for h in heads} if out["rule_buckets"] else {}
                entry["worst_rule_ci_upper"] = worst
                entry["acceptable"] = (not cover[f]["nonfinite_users"]) and bool(out["rule_buckets"]) and \
                    all(v <= CEILING for v in worst.values())
            if ext_users:
                te, oke = table(ext, model, f, REF, ext_users)
                entry["extension_vs_w8s8"] = {"users": len(oke), "nonfinite_users": sorted(
                    u for u in ext_users if u in ext.get(label(model, f), {}) and not finite(ext[label(model, f)][u], heads)), **te}
                comb = {**{("m", u): rows[label(model, f)][u] for u in ok}, **{("e", u): ext[label(model, f)][u] for u in oke}}
                combr = {**{("m", u): rows[label(model, REF)][u] for u in ok}, **{("e", u): ext[label(model, REF)][u] for u in oke}}
                entry["main_plus_extension_300plus"] = {h: compare(comb, combr, list(comb), h, "300+") for h in heads}
            out["formats"][f] = entry
        out["w8s8_vs_published_max_abs"] = vs_published(rows, model, users)
        res["models"][model] = out
    if pending:
        print("[sm136-rule] decide pending: " + "; ".join(pending))
        return 2, res
    for size, ms in SIZES.items():
        s = {}
        for f in INT4:
            s[f] = {mm: res["models"][mm]["formats"][f]["acceptable"] for mm in ms}
        comp_ok = all(s[COMPACT].values())
        s["verdict"] = ("int4 card rows acceptable (keep the compact format)" if comp_ok else
                        "recommend int8 card rows (" + ", ".join(MODELS[mm]["name"] for mm, v in s[COMPACT].items() if not v) +
                        " not acceptable with compact int4 cards)")
        s["fork_default_verdict"] = "acceptable" if all(s[DEFAULT].values()) else \
            "not acceptable (" + ", ".join(MODELS[mm]["name"] for mm, v in s[DEFAULT].items() if not v) + ")"
        res["sizes"][size] = s
    res["verdict"] = {size: res["sizes"][size]["verdict"] for size in SIZES}
    bpath = HERE / "sm136-bytes.json"
    if bpath.exists():
        b = json.loads(bpath.read_text())["arms"]
        res["bytes"] = {size: {f: {"card": b[a][f]["card"]["bytes"], "note": b[a][f]["note"]["bytes"]}
                               for f in (REF, DEFAULT, COMPACT, INT8C)} for size, a in BYTES_ARCH.items()}
    res["context_722_users"] = context()
    if write:
        (out_dir / "sm136-decision.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
        (out_dir / "sm136-decision.md").write_text(markdown(res), encoding="utf-8")
    print("[sm136-rule] verdict " + "; ".join(f"{k}: {v}" for k, v in res["verdict"].items()))
    return 0, res


def context():
    """Whole-pool numbers of earlier runs (722 users, benchmark LogLoss vs the published w8+s8), report only."""
    out = {}
    try:
        d = json.loads((SMG / "sm131/sm131-decision.json").read_text())
        for mm in ("sm91s2", "sm99", "sm118", "sm113"):
            x = d["models"][mm]
            out[mm] = {"source": "compact state for Small and Big (SM131)",
                       "compact_vs_w8s8": {h: [v["value"], v["ci"]] for h, v in x["compact_vs_published"].items()},
                       "fork_default_vs_w8s8": {h: [v["value"], v["ci"]] for h, v in x["fork_default_vs_published"].items()}}
    except (OSError, KeyError, ValueError):
        pass
    try:
        d = json.loads((SMG / "sm129/sm129-decision.json").read_text())
        for kind, mm in (("two-head", "sm124"), ("instant-only", "sm115")):
            arms = d["models"][kind]["arms"]
            out[mm] = {"source": "compact state for Tiny (SM129)",
                       "compact_vs_w8s8": {h: [v["value"], v["ci"]] for h, v in arms["compact"]["cost_vs_tiny_published"].items()}}
            if "fork_default" in arms and "cost_vs_tiny_published" in arms["fork_default"]:
                out[mm]["fork_default_vs_w8s8"] = {h: [v["value"], v["ci"]] for h, v in arms["fork_default"]["cost_vs_tiny_published"].items()}
    except (OSError, KeyError, ValueError):
        pass
    return out


def fmt_pct(x):
    if not x or not x.get("users"):
        return "n/a"
    s = lambda v: f"{v:+.3f}".replace("-", "&minus;")
    return f"{s(x['pct'])}% [{s(x['ci'][0])}, {s(x['ci'][1])}]"


def markdown(res):
    L = ["# Long-card state drift on real users (SM136): decision", "", res["rule"], ""]
    L.append("## Verdict per size (release decision on the compact format; the fork default is reported too)")
    L.append("")
    for size, s in res["sizes"].items():
        L.append(f"- **{size}: {s['verdict']}**; fork default: {s['fork_default_verdict']}")
    L.append("")
    if "bytes" in res:
        L += ["## Bytes per card / note", "", "| size | " + " | ".join(FORMAT_NAMES[f] for f in (REF, DEFAULT, COMPACT, INT8C)) + " |",
              "|---|" + "---:|" * 4]
        for size, b in res["bytes"].items():
            L.append(f"| {size} | " + " | ".join(f"{b[f]['card']:,} / {b[f]['note']:,}" for f in (REF, DEFAULT, COMPACT, INT8C)) + " |")
        L.append("")
    for model, m in res["models"].items():
        L += [f"## {m['name']}", ""]
        if "bucket_users" not in m:
            L += ["(pending)", ""]
            continue
        L.append("Users per bucket (main pool): " + ", ".join(f"{g} {n}" for g, n in m["bucket_users"].items()) +
                 f". Rule buckets: {', '.join(m['rule_buckets']) or 'none'}.")
        L.append("")
        for f, e in m["formats"].items():
            nf = e["nonfinite_users"]
            acc = f"; acceptable: {'yes' if e['acceptable'] else 'no'}" if "acceptable" in e else ""
            L.append(f"### {FORMAT_NAMES[f]} vs w8+s8 ({e['users']} users; non-finite users: {len(nf)}{acc})")
            L.append("")
            L.append("| head | " + " | ".join(GROUPS) + " |")
            L.append("|---|" + "---|" * len(GROUPS))
            for h, t in e["vs_w8s8"].items():
                L.append(f"| {h} | " + " | ".join(fmt_pct(t[g]) for g in GROUPS) + " |")
            for h, t in e.get("vs_float32", {}).items():
                L.append(f"| {h}, vs float32 | " + " | ".join(fmt_pct(t[g]) for g in GROUPS) + " |")
            for h, t in e["benchmark_vs_w8s8"].items():
                L.append(f"| {h}, benchmark reviews | " + " | ".join(fmt_pct(t[g]) for g in GROUPS) + " |")
            if "extension_vs_w8s8" in e:
                x = e["extension_vs_w8s8"]
                for h in m["heads"]:
                    L.append(f"| {h}, extension ({x['users']} users, report only) | " + " | ".join(fmt_pct(x[h][g]) for g in GROUPS) + " |")
                L.append("")
                L.append("Main + extension, 300+ bucket (report only): " + "; ".join(
                    f"{h} {fmt_pct(v)} ({v.get('users', 0)} users, {v.get('reviews', 0):,} reviews)"
                    for h, v in e["main_plus_extension_300plus"].items()))
            L.append("")
        L.append(f"w8+s8 rows vs the published GPU rows, max |benchmark LogLoss difference|: {m.get('w8s8_vs_published_max_abs')}")
        L.append("")
    L.append(f"Unparseable lines: {res['unparseable_lines']}")
    if res.get("context_722_users"):
        L += ["", "## Context: the whole 722-user pool (benchmark reviews, earlier runs, % vs w8+s8 [95% CI])", ""]
        for mm, c in res["context_722_users"].items():
            L.append(f"- {MODELS[mm]['name']} ({c['source']}): compact {c.get('compact_vs_w8s8')}, fork default {c.get('fork_default_vs_w8s8')}")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- selftest

def synth(path, users, scale=None, nonfinite=(), drop=(), torn=False, seed=1, mismatch=None, ext=False):
    """Synthetic rows: every model x format; bucket counts shaped like the pool (6 users with 300-999, 1 with 1000+);
    scale = {(model, fmt, group): factor} applied to the LogLoss sums of those buckets."""
    rng = np.random.default_rng(seed)
    base = {u: rng.uniform(0.2, 0.45) for u in users}
    shape = {}
    for i, u in enumerate(users):
        n = [int(rng.integers(2000, 60000)), int(rng.integers(300, 20000)), int(rng.integers(20, 3000)),
             int(rng.integers(20, 1500)) if (i < 6 or ext) else 0, int(rng.integers(50, 3000)) if i == 0 else 0]
        shape[u] = n
    noise = {(u, model, k): rng.normal(0, 0.002) for u in users for model in MODELS for k in range(5)}
    lines = []
    for model, m in MODELS.items():
        for f in FORMATS:
            for u in users:
                if (model, f, u) in drop:
                    continue
                if (model, f, u) in nonfinite:
                    lines.append(json.dumps({"user_id": u, "label": label(model, f), "nonfinite": "Non-finite evaluation loss"}))
                    continue
                n = list(shape[u])
                if mismatch == (model, f, u):
                    n[0] += 1
                buckets = {}
                for head in ("ahead", "instant"):
                    ll = []
                    for k in range(5):
                        v = n[k] * (base[u] + noise[(u, model, k)] + (0.02 if head == "ahead" else 0))
                        for g, idx in GROUPS.items():
                            if scale and k in idx and scale.get((model, f, g)) and g != "all":
                                v *= scale[(model, f, g)]
                        ll.append(v)
                    buckets[head] = {"n": n, "ll": ll, "min_prior": 1, "max_prior": 1500}
                bench = {h: {"LogLoss": round(sum(buckets[h]["ll"][:2]) / sum(buckets[h]["n"][:2]), 6), "size": sum(n[:2])}
                         for h in ("ahead", "instant")}
                lines.append(json.dumps({"user_id": u, "label": label(model, f), "benchmark": {"size": sum(n[:2]), **bench},
                                         "prior_buckets": {"edges": EDGES, "all": buckets,
                                                           "benchmark": {h: {"n": n[:2] + [0, 0, 0], "ll": buckets[h]["ll"][:2] + [0, 0, 0]}
                                                                         for h in buckets}}}))
    text = "\n".join(lines) + "\n"
    if torn:
        text += '{"user_id": 1, "label": "x/w8+s8@e2s-q\n\x00\x00\x00\n'
    Path(path).write_text(text, encoding="utf-8")


def selftest():
    users = list(range(100, 166))
    ext_users = list(range(9000, 9010))
    checks = {}
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        r, e = t / "rows.jsonl", t / "ext.jsonl"
        synth(e, ext_users, ext=True, seed=2)
        # 1. identical formats -> 0 cost, every size acceptable
        synth(r, users, scale=None, torn=True)
        code, res = decide(r, e, t, users, ext_users, write=False)
        z = res["models"]["sm91s2"]["formats"][COMPACT]["vs_w8s8"]["curve"]["128+"]
        checks["identical: acceptable everywhere"] = code == 0 and all("acceptable" in v for v in res["verdict"].values())
        checks["identical: zero cost"] = z["pct"] == 0 and z["ci"] == [0, 0]
        checks["rule buckets = 128-299, 128+"] = res["models"]["sm91s2"]["rule_buckets"] == ["128-299", "128+"]
        checks["bucket users 66/66/66/6/1"] = [res["models"]["sm118"]["bucket_users"][g] for g in BUCKETS] == [66, 66, 66, 6, 1]
        checks["torn and NUL lines counted"] = res["unparseable_lines"][str(r)] == 2
        checks["extension reported"] = res["models"]["sm124"]["formats"][COMPACT]["extension_vs_w8s8"]["users"] == 10 and \
            res["models"]["sm124"]["formats"][COMPACT]["main_plus_extension_300plus"]["instant"]["users"] == 16
        # 2. hand statistic: two users
        a = {1: {"prior_buckets": {"all": {"instant": {"n": [10, 0, 5, 0, 0], "ll": [3.0, 0, 2.0, 0, 0]}}}},
             2: {"prior_buckets": {"all": {"instant": {"n": [10, 0, 5, 0, 0], "ll": [3.0, 0, 3.0, 0, 0]}}}}}
        b = {1: {"prior_buckets": {"all": {"instant": {"n": [10, 0, 5, 0, 0], "ll": [3.0, 0, 1.0, 0, 0]}}}},
             2: {"prior_buckets": {"all": {"instant": {"n": [10, 0, 5, 0, 0], "ll": [3.0, 0, 3.0, 0, 0]}}}}}
        x = compare(a, b, [1, 2], "instant", "128-299")
        checks["hand statistic 25%"] = x["pct"] == 25.0 and x["users"] == 2 and x["reviews"] == 10 and x["users_worse"] == 1
        # 3. the instant-only big model's compact instant x 1.004 on 128-299 -> Big not acceptable (compact), others fine
        synth(r, users, scale={("sm99", COMPACT, "128-299"): 1.004})
        code, res = decide(r, e, t, users, ext_users, write=False)
        checks["Big compact +0.4% on 128-299 -> int8 for Big"] = code == 0 and res["verdict"]["Big"].startswith("recommend int8") \
            and "acceptable" in res["verdict"]["Small"] and "acceptable" in res["verdict"]["Tiny"] \
            and res["sizes"]["Big"][DEFAULT]["sm99"] is True
        # 4. a report-only bucket (300-999, 6 users) x 1.01 decides nothing
        synth(r, users, scale={("sm118", COMPACT, "300-999"): 1.01})
        code, res = decide(r, e, t, users, ext_users, write=False)
        rep = res["models"]["sm118"]["formats"][COMPACT]["vs_w8s8"]["curve"]["300-999"]
        checks["report-only bucket does not decide"] = code == 0 and "acceptable" in res["verdict"]["Small"] and rep["pct"] > 0.9
        # 5. a non-finite fork-default user on the show-time big model -> fork default not acceptable for Big, compact fine
        synth(r, users, nonfinite={("sm91s2", DEFAULT, 100)})
        code, res = decide(r, e, t, users, ext_users, write=False)
        checks["non-finite default user"] = code == 0 and res["sizes"]["Big"][DEFAULT]["sm91s2"] is False and \
            "acceptable" in res["verdict"]["Big"] and res["models"]["sm91s2"]["formats"][DEFAULT]["nonfinite_users"] == [100] \
            and res["models"]["sm91s2"]["formats"][DEFAULT]["users"] == 65
        # 6. Tiny curve +0.3% on 128+ for the two-head model -> Tiny int8
        synth(r, users, scale={("sm124", COMPACT, "128+"): 1.003})
        code, res = decide(r, e, t, users, ext_users, write=False)
        checks["Tiny curve +0.3% -> int8 for Tiny"] = res["verdict"]["Tiny"].startswith("recommend int8") and \
            "acceptable" in res["verdict"]["Big"]
        # 7. 3 users missing for one label -> pending (exit 2); 2 missing -> decided on 64
        synth(r, users, drop={("sm113", INT8C, u) for u in users[:3]})
        code, _ = decide(r, e, t, users, ext_users, write=False)
        checks["3 missing -> pending"] = code == 2
        synth(r, users, drop={("sm113", INT8C, u) for u in users[:2]})
        code, res = decide(r, e, t, users, ext_users, write=False)
        checks["2 missing -> decided"] = code == 0 and res["models"]["sm113"]["formats"][INT8C]["users"] == 64
        # 8. bucket counts that differ between formats -> error
        synth(r, users, mismatch=("sm115", COMPACT, 101))
        try:
            decide(r, e, t, users, ext_users, write=False)
            checks["count mismatch raises"] = False
        except ValueError:
            checks["count mismatch raises"] = True
        # 9. markdown renders
        synth(r, users)
        code, res = decide(r, e, t, users, ext_users, write=True)
        md = (t / "sm136-decision.md").read_text(encoding="utf-8")
        checks["markdown"] = "the show-time big model (SM91s2)" in md and "| head |" in md
    ok = all(checks.values())
    for k, v in checks.items():
        print(f"{'PASS' if v else 'FAIL'} {k}")
    print(f"[sm136-rule] selftest {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)})")
    return 0 if ok else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "selftest":
        raise SystemExit(selftest())
    if cmd == "decide":
        try:
            code, _ = decide()
        except ValueError as error:
            print(f"[sm136-rule] decide failed: {error}")
            raise SystemExit(3)
        raise SystemExit(code)
    print(__doc__)
    raise SystemExit(1)
