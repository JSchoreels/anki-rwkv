#!/usr/bin/env python
"""Release-candidate re-check of the final state formats (SM164): user lists, GPU smoke check, the pre-registered rules and
the verdict per size (design.md "Rules", committed before any row of this run existed). CPU only, under 1 GB of RAM.

Final formats (user 11-24): Big and Small on F6 (`w8+lr2.16.4q6x8`, T24 with int6 factors), the six-layer Tiny on T24
(`w8+lr2.16.4q8x8`); reference REF int8 (`w8+s8hx8`). Rows, the first source wins per (label, user):
  1 SM148 `../sm148/gpu/sm148-rows-732.jsonl`  REF, T24 on the 732 set (723 fit users, Big 718)
  2 SM152 `../sm152/gpu/sm152-rows-732.jsonl`  F6 on the 732 set
  3 SM160 `wt/sm160/.../gpu/sm160-rows-wfit.jsonl` (sha256 pinned)  REF, T24 on 100 users with id >= 5000 of the 1,500 pool
  4 SM164 `gpu/sm164-rows-new.jsonl`  this run: every missing (model, label, user) of the ids >= 5000 part

  python sm164_report.py users [--write]   the scoring lists (rule in design.md); without --write: check they are unchanged
  python sm164_report.py smoke             gpu/smoke-gpu-rows.jsonl vs SM152's GPU smoke rows and SM160's rows; exit 0 = OK
  python sm164_report.py report            -> sm164-result.{json,md}; exit 0 decided, 2 pending (coverage), 3 invalid
  python sm164_report.py selftest          synthetic rows

Statistic (as SM148 / SM152): % = 100 x (sum of the label's LogLoss / sum of the reference's LogLoss - 1), review-weighted
over the benchmark reviews, paired user bootstrap 2,000 draws, seed 0, 95% CI (showtime_report.compare, sha256 pinned).
"""
import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SM = HERE.parent
ROOT = SM.parents[1]
G = HERE / "gpu"
W = Path("C:/Users/jonat/workspace/wt")
REPORT = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/small-model/showtime/showtime_report.py")
REPORT_SHA = "3cdfed03afd40734d4b1931c244e8c1e73453d281a40c3843e80cd310d70dd9d"
SRC148 = SM / "sm148" / "gpu" / "sm148-rows-732.jsonl"
SRC152 = SM / "sm152" / "gpu" / "sm152-rows-732.jsonl"
SRC160 = W / "sm160" / "artifacts" / "small-model" / "sm160" / "gpu" / "sm160-rows-wfit.jsonl"
SRC160_SHA = "c03f1ac776dd4b07187474d0e1c731f4c67567419234e80de4ca11ddf37931e5"   # SM160 GPU DONE 11-24 15:26 (complete)
U160 = W / "sm160" / "artifacts" / "small-model" / "sm160" / "users-wfit.json"
NEWROWS = G / "sm164-rows-new.jsonl"
SMOKEROWS = G / "smoke-gpu-rows.jsonl"
SMOKE152 = SM / "sm152" / "gpu" / "smoke-gpu-rows.jsonl"
POOL1500 = SM / "eval-users-1500.json"
U732FIT = SM / "sm72s" / "eval-users-732-fit.json"
E2S = "@e2s-query"
LABELS = {"REF": "w8+s8hx8", "T24": "w8+lr2.16.4q8x8", "F6": "w8+lr2.16.4q6x8"}
FINAL = {"Big": "F6", "Small": "F6", "Tiny": "T24"}      # the user's 11-24 choice per size
CEIL_BIG = 0.15                                          # Big: vs its own int8, CI upper per head
CEIL_VS_BIG = {"Small": 0.5, "Tiny": 1.0}                # Small / Tiny: total vs the matching Big model at int8, CI upper
MAX_REVIEWS = 650000          # the 1,500 pool scoring rule of the instant model check (SM93): larger users never fit
MAX_REVIEWS_BIG = 450701      # the largest user the Big models fit on the GPU (SM148 / SM152 rows of the 732 set)
MIN_732 = 715                 # as SM152
MIN_NEW_SHARE = 0.97          # ids >= 5000 part: at least 97% of the scoring list per (model, label)
TOL = 1e-5
TOL_EPS = 1e-9
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
LISTS = {"new-all": "users-new-all.json", "new-rest": "users-new-rest.json", "big-all": "users-big-all.json",
         "big-rest": "users-big-rest.json"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lab(tag, s):
    return f"{tag}/{LABELS[s]}{E2S}"


def showtime():
    if sha(REPORT) != REPORT_SHA:
        raise SystemExit("[sm164] showtime_report.py changed: refusing")
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
                rows.setdefault(r["label"], {}).setdefault(r["user_id"], r)
            elif "nonfinite" in r:
                nonfinite.setdefault(r["label"], set()).add(r["user_id"])
    return rows, nonfinite, torn


def merged(paths):
    """First source wins per (label, user); returns rows, non-finite sets, torn count, {label: {user: source index}}."""
    rows, nf, torn, src = {}, {}, 0, {}
    for i, p in enumerate(paths):
        r, n, t = read_rows(p)
        torn += t
        for L, d in r.items():
            for u, row in d.items():
                if u not in rows.setdefault(L, {}):
                    rows[L][u] = row
                    src.setdefault(L, {})[u] = i
        for L, s in n.items():
            nf.setdefault(L, set()).update(s)
    return rows, nf, torn, src


# ---------------------------------------------------------------- user lists

def counts():
    tiers = json.loads((ROOT / "artifacts" / "accuracy" / "tiers.json").read_text())
    return json.loads(Path(tiers["split"]).read_text())["review_counts"]


def build_lists():
    c = counts()
    pool = json.loads(POOL1500.read_text())["users"]
    new = sorted(u for u in pool if u >= 5000)
    done160 = set(json.loads(U160.read_text())["users"])
    na = [u for u in new if c[str(u)] <= MAX_REVIEWS]
    nb = [u for u in na if c[str(u)] <= MAX_REVIEWS_BIG]
    lists = {"new-all": na, "new-rest": [u for u in na if u not in done160],
             "big-all": nb, "big-rest": [u for u in nb if u not in done160]}
    rules = {"new-all": f"the 1,500 pool's users with id >= 5000 and at most {MAX_REVIEWS:,} reviews (SM93's rule); "
                        "Small and Tiny, labels without earlier rows (F6)",
             "new-rest": "new-all minus the 100 users that SM160 scored at REF and T24 (users-wfit.json): Small and "
                         "Tiny labels T24 and REF",
             "big-all": f"new-all with at most {MAX_REVIEWS_BIG:,} reviews (the largest user the Big models fit on the "
                        "GPU in SM148 / SM152): Big F6",
             "big-rest": "big-all minus SM160's 100 users: Big REF"}
    out = {}
    for k, us in lists.items():
        out[k] = {"note": f"Release-candidate re-check of the final state formats (SM164): scoring list {k}. Public "
                          "benchmark user ids only.", "rule": rules[k], "users_count": len(us),
                  "reviews_total": sum(c[str(u)] for u in us), "users": us}
    smoke_users = [2388, 4676, 1866] + sorted(done160, key=lambda u: (c[str(u)], u))[:2]
    out["smoke"] = {"note": "SM164 GPU smoke users: SM152's 3 smoke users (2388, 4676, 1866; rows vs SM152's GPU smoke rows) "
                            "and the 2 smallest of SM160's 100 users with id >= 5000 (rows vs SM160's REF and T24 rows). "
                            "Public benchmark user ids only.", "users": smoke_users}
    return out


def users_cmd(write):
    lists = build_lists()
    names = dict(LISTS, smoke="users-smoke.json")
    bad = []
    for k, doc in lists.items():
        p = HERE / names[k]
        if write:
            p.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        elif not p.exists() or json.loads(p.read_text())["users"] != doc["users"]:
            bad.append(k)
        print(f"[sm164] {names[k]}: {len(doc['users'])} users, {doc.get('reviews_total', 0):,} reviews")
    if bad:
        print(f"[sm164] users lists differ from their rule: {bad}")
        return 1
    print("[sm164] users OK")
    return 0


# ---------------------------------------------------------------- smoke

def smoke():
    g, gnf, _ = read_rows(SMOKEROWS)
    o, _, _ = read_rows(SMOKE152)
    w, _, _ = read_rows(SRC160)
    us = json.loads((HERE / "users-smoke.json").read_text())["users"]
    old, new = us[:3], us[3:]
    ok, lines, rep = True, [], {}
    for key, (tag, name, heads, size, _b) in MODELS.items():
        for s in LABELS:
            L = lab(tag, s)
            if gnf.get(L):
                ok = False; lines.append(f"{L}: non-finite rows {sorted(gnf[L])}"); continue
            if set(g.get(L, {})) != set(us):
                ok = False; lines.append(f"{L}: rows missing (have {sorted(g.get(L, {}))})"); continue
            checks = [("SM152", o, old)] + ([("SM160", w, new)] if s in ("REF", "T24") else [])
            for src, d, uu in checks:
                if not set(uu) <= set(d.get(L, {})):
                    ok = False; lines.append(f"{L}: {src} rows missing for {uu}"); continue
                for h in sorted({KEY[x] for x in heads}):
                    diffs = [abs(g[L][u]["benchmark"][h]["LogLoss"] - d[L][u]["benchmark"][h]["LogLoss"]) for u in uu]
                    sizes = all(g[L][u]["benchmark"][h]["size"] == d[L][u]["benchmark"][h]["size"] for u in uu)
                    fin = all(math.isfinite(g[L][u]["benchmark"][h]["LogLoss"]) for u in uu)
                    good = max(diffs) <= TOL + TOL_EPS and sizes and fin
                    rep[f"{L} {h} vs {src}"] = {"max_abs": max(diffs), "sizes_equal": sizes, "pass": good}
                    ok &= good
                    if not good:
                        lines.append(f"{L} {h} vs {src}: max |d| {max(diffs):.2e}, sizes equal {sizes}, finite {fin}")
    worst = max((v["max_abs"] for v in rep.values()), default=float("nan"))
    (G / "smoke-check.json").write_text(json.dumps({"ok": ok, "cells": rep, "problems": lines}, indent=1) + "\n",
                                        encoding="utf-8")
    msg = (f"[sm164] smoke OK: {len(rep)} (model, label, head, source) cells within {TOL:g} (worst {worst:.2e})" if ok else
           f"[sm164] smoke FAIL: " + "; ".join(lines[:8]))
    print(msg)
    return 0 if ok else 1


# ---------------------------------------------------------------- rules

def as_loss(rows):
    return {u: {h: ((r["benchmark"][h]["LogLoss"], r["benchmark"][h]["size"]) if h in r["benchmark"] else (float("nan"), 0))
                for h in ("ahead", "instant")} for u, r in rows.items()}


def _bench_cmp(new, ref, h, users, draws=2000):
    a = np.array([new[u]["benchmark"][h]["LogLoss"] * new[u]["benchmark"][h]["size"] for u in users])
    b = np.array([ref[u]["benchmark"][h]["LogLoss"] * ref[u]["benchmark"][h]["size"] for u in users])
    idx = np.random.default_rng(0).integers(0, len(users), size=(draws, len(users)))
    boot = (a[idx].sum(1) / b[idx].sum(1) - 1) * 100
    return {"value": float((a.sum() / b.sum() - 1) * 100),
            "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], "users": len(users)}


def cmp(new, ref, heads, users, st):
    users = sorted(u for u in users if u in new and u in ref)
    if not users:
        return {}
    if st is not None:
        c = st.compare(as_loss(new), as_loss(ref), as_loss(ref), users)
        return {h: {"value": c[h]["pct_of_ref"], "ci": c[h]["pct_of_ref_ci"], "users": c["users"]} for h in heads}
    return {h: _bench_cmp(new, ref, KEY[h], users) for h in heads}   # selftest: the same ratio-of-sums bootstrap


def fmt(x):
    return "n/a" if not x else f"{x['value']:+.3f}% [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]"


def decide(rows, nf, u732, unew, lists, st=None):
    """Per model and head (design.md "Rules").
    732 rule (the user's budget, all fork-unseen users): Big final label vs its own int8; Small / Tiny final label vs the
    matching Big model at int8 (TOTAL); CI upper bound <= ceiling.
    1,500 rule (decisive): Big final label vs its own int8 on the 1,500 set, CI upper <= 0.15%. Small / Tiny: the users
    with id >= 5000 are training users of the Big models, so a direct Small / Tiny vs Big comparison there measures Big's
    in-sample advantage, not the format. The total is decomposed: upper(own int8 vs Big int8, 732 set) + upper(final label
    vs own int8, 1,500 set) <= ceiling (sum of the two upper bounds: conservative).
    Cell PASS if both rules pass; FAIL if the 732 rule fails or a row is non-finite, or (Big) the 1,500 rule fails; HOLD if
    a Small / Tiny cell passes the 732 rule and fails only the 1,500 rule (the main session and the user decide).
    Size GO if every cell of both models passes, NO if one fails, HOLD otherwise. The direct Small / Tiny vs Big
    comparison on the 1,500 set and on the ids >= 5000 alone is reported, never a verdict."""
    u1500 = set(u732) | set(unew)
    res = {"ceilings_pct": {"Big_vs_own_int8_upper": CEIL_BIG, "Small_vs_Big_int8_upper": CEIL_VS_BIG["Small"],
                            "Tiny_vs_Big_int8_upper": CEIL_VS_BIG["Tiny"]},
           "final_labels": {k: f"{v} ({LABELS[v]})" for k, v in FINAL.items()}, "users": {"732": len(u732)},
           "models": {}, "sizes": {}, "pending": [], "warnings": []}
    sub = lambda L, us: {u: r for u, r in rows.get(L, {}).items() if u in us}   # noqa: E731
    for key, (tag, name, heads, size, bkey) in MODELS.items():
        s = FINAL[size]
        L, Lref = lab(tag, s), lab(tag, "REF")
        Lbig = Lref if bkey is None else lab(MODELS[bkey][0], "REF")
        ceil = CEIL_BIG if bkey is None else CEIL_VS_BIG[size]
        m = {"name": name, "size": size, "label": f"{s} ({LABELS[s]})", "heads": list(heads), "ceiling_upper_pct": ceil,
             "baseline": "own int8" if bkey is None else f"{MODELS[bkey][1]} at int8"}
        lk = "big-all" if size == "Big" else "new-all"
        need = {L: lk, Lref: lk, Lbig: "big-all"}
        for LL, k in need.items():
            have732 = len(set(rows.get(LL, {})) & set(u732))
            havenew = len(set(rows.get(LL, {})) & set(lists[k]))
            want = math.ceil(MIN_NEW_SHARE * len(lists[k]))
            m.setdefault("coverage", {})[LL] = {"732": have732, "new": havenew, "new_list": len(lists[k]), "new_min": want}
            if have732 < MIN_732:
                res["pending"].append(f"{LL}: {have732} of {MIN_732} users on the 732 set")
            if havenew < want:
                res["pending"].append(f"{LL}: {havenew} of {want} users with id >= 5000 ({k})")
        new, own, big = sub(L, u1500), sub(Lref, u1500), sub(Lbig, u1500)
        nonfin = sorted(set().union(*[nf.get(x, set()) for x in need]) & u1500)
        finite = not nonfin and all(math.isfinite(r["benchmark"][KEY[h]]["LogLoss"]) for d in (new, own, big)
                                    for r in d.values() for h in heads)
        m["nonfinite_users"], m["finite"] = nonfin, finite
        m["total_732"] = cmp(new, big, heads, u732, st)          # the 732 rule
        m["own_int8_1500"] = cmp(new, own, heads, u1500, st)     # Big: the 1,500 rule; Small / Tiny: its format part
        m["own_int8_732"] = cmp(new, own, heads, u732, st)       # report only
        m["own_int8_new_only"] = cmp(new, own, heads, unew, st)  # report only
        if bkey:
            m["int8_vs_big_732"] = cmp(own, big, heads, u732, st)          # the gap part of the 1,500 rule
            m["direct_total_1500"] = cmp(new, big, heads, u1500, st)       # report only (Big in-sample on ids >= 5000)
            m["direct_total_new_only"] = cmp(new, big, heads, unew, st)    # report only
            m["int8_vs_big_new_only"] = cmp(own, big, heads, unew, st)     # report only: the in-sample effect itself
        cells = {}
        for h in heads:
            a = m["total_732"].get(h)
            ok7 = bool(a) and a["ci"][1] <= ceil + TOL_EPS
            f = m["own_int8_1500"].get(h)
            if bkey is None:
                up15 = f["ci"][1] if f else None
            else:
                g = m["int8_vs_big_732"].get(h)
                up15 = (g["ci"][1] + f["ci"][1]) if (f and g) else None
            ok15 = up15 is not None and up15 <= ceil + TOL_EPS
            if not finite or not ok7:
                v = "FAIL"
            elif ok15:
                v = "PASS"
            else:
                v = "FAIL" if bkey is None else "HOLD"
            cells[h] = {"pass_732": ok7, "pass_1500": ok15, "verdict": v, "upper_732": a["ci"][1] if a else None,
                        "upper_1500": up15, "margin_1500": (ceil - up15) if up15 is not None else None}
        m["cells"] = cells
        res["models"][key] = m
    for size in ("Big", "Small", "Tiny"):
        vs = [c["verdict"] for m in res["models"].values() if m["size"] == size for c in m["cells"].values()]
        res["sizes"][size] = {"label": f"{FINAL[size]} ({LABELS[FINAL[size]]})",
                              "verdict": "NO" if "FAIL" in vs else ("HOLD" if "HOLD" in vs else "GO")}
    res["users"]["1500_final_labels_min"] = min(len(set(rows.get(lab(MODELS[k][0], FINAL[MODELS[k][3]]), {})) & u1500)
                                                for k in MODELS)
    res["status"] = "pending" if res["pending"] else "decided"
    return res


def fup(x):
    return "n/a" if x is None else f"{x:+.3f}%"


def write_md(res):
    md = ["# Release-candidate re-check of the final state formats (SM164): result", "",
          f"Status: **{res['status']}**. Rows: SM148 (REF, T24) and SM152 (F6) on the 732 set, SM160 (REF, T24) on 100 users "
          "with id >= 5000, this run for the rest of the ids >= 5000 part of the 1,500 pool. Statistic: % change of the "
          "benchmark LogLoss vs the baseline, review-weighted, paired user bootstrap 2,000 draws, 95% CI.", "",
          "## Verdict per size", "", "| size | final label | verdict |", "|---|---|---|"]
    md += [f"| {s} | `{v['label']}` | **{v['verdict']}** |" for s, v in res["sizes"].items()]
    md += ["", "## Per model and head (budget = CI upper bound)", "",
           "732 rule: the final label vs the baseline on the 722 rule users. 1,500 rule: Big vs its own int8 on the 1,500 "
           "set; Small / Tiny: upper(own int8 vs Big int8, 732 set) + upper(final label vs own int8, 1,500 set).", "",
           "| model | head | baseline | ceiling | 732 rule | own int8 vs Big int8 (732) | vs own int8 (1,500) | 1,500 rule upper | "
           "verdict |", "|---|---|---|---|---|---|---|---|---|"]
    for m in res["models"].values():
        for h in m["heads"]:
            c = m["cells"][h]
            n = m["own_int8_1500"].get(h, {}).get("users", 0)
            gap = fmt(m["int8_vs_big_732"].get(h)) if "int8_vs_big_732" in m else "-"
            md.append(f"| {m['name']} | {h} | {m['baseline']} | +{m['ceiling_upper_pct']}% | {fmt(m['total_732'].get(h))} | "
                      f"{gap} | {fmt(m['own_int8_1500'].get(h))} (n={n}) | {fup(c['upper_1500'])} | {c['verdict']} |")
    md += ["", "## Report only", "", "| model | head | vs own int8, 732 | vs own int8, ids >= 5000 | direct vs Big int8, "
           "1,500 | direct vs Big int8, ids >= 5000 | own int8 vs Big int8, ids >= 5000 |", "|---|---|---|---|---|---|---|"]
    for m in res["models"].values():
        for h in m["heads"]:
            md.append(f"| {m['name']} | {h} | {fmt(m['own_int8_732'].get(h))} | {fmt(m['own_int8_new_only'].get(h))} | "
                      f"{fmt(m.get('direct_total_1500', {}).get(h))} | {fmt(m.get('direct_total_new_only', {}).get(h))} | "
                      f"{fmt(m.get('int8_vs_big_new_only', {}).get(h))} |")
    md += ["", "The users with id >= 5000 are training users of the Big models: a direct Small or Tiny vs Big comparison "
           "there mostly measures Big's in-sample advantage (last column), so it is reported, never a verdict.", ""]
    if res["pending"]:
        md += ["## Pending", ""] + [f"- {p}" for p in res["pending"]] + [""]
    (HERE / "sm164-result.md").write_text("\n".join(md), encoding="utf-8")


def load_lists():
    return {k: json.loads((HERE / v).read_text())["users"] for k, v in LISTS.items()}


def report():
    if not SRC160.exists() or sha(SRC160) != SRC160_SHA:
        print(f"[sm164] INVALID: SM160 rows {SRC160} missing or changed (sha256 pinned {SRC160_SHA[:12]})")
        return 3
    if not (G / "smoke-ok").exists():
        print("[sm164] INVALID: no gpu/smoke-ok")
        return 3
    st = showtime()
    u732, _ = st.pools()
    pool = json.loads(POOL1500.read_text())["users"]
    unew = {u for u in pool if u >= 5000}
    rows, nf, torn, src = merged([SRC148, SRC152, SRC160, NEWROWS])
    res = decide(rows, nf, set(u732), unew, load_lists(), st)
    res["torn_lines"] = torn
    names = ["SM148", "SM152", "SM160", "SM164"]
    res["row_sources"] = {}
    for key, (tag, *_r) in MODELS.items():
        for s in LABELS:
            L = lab(tag, s)
            if L in src:
                cnt = {}
                for i in src[L].values():
                    cnt[names[i]] = cnt.get(names[i], 0) + 1
                res["row_sources"][L] = cnt
    (HERE / "sm164-result.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    write_md(res)
    for s, v in res["sizes"].items():
        print(f"[sm164] {s} {v['label']}: {v['verdict']}")
    if res["status"] == "pending":
        print(f"[sm164] report pending: {res['pending'][:6]}")
        return 2
    return 0


def selftest():
    rng = np.random.default_rng(1)
    u732 = list(range(1, 730))
    unew = list(range(5001, 5770))
    lists = {"new-all": unew, "new-rest": unew, "big-all": unew[:-10], "big-rest": unew[:-10]}

    base = {u: 0.3 + 0.05 * rng.standard_normal() for u in u732 + unew}

    def mk():   # every label of every model equal to one per-user LogLoss (Big misses its 10 largest new users)
        rows = {}
        for key, (tag, _n, heads, size, bkey) in MODELS.items():
            for s in LABELS:
                rows[lab(tag, s)] = {u: {"benchmark": {h: {"LogLoss": base[u], "size": 1000} for h in ("ahead", "instant")}}
                                     for u in u732 + unew if not (size == "Big" and u in unew[-10:])}
        return rows
    # case 1: no cost anywhere -> GO everywhere
    r = mk()
    res = decide(r, {}, set(u732), set(unew), lists)
    assert all(v["verdict"] == "GO" for v in res["sizes"].values()), res["sizes"]
    assert res["status"] == "decided", res["pending"]
    # case 2: Small label +0.8% on the ids >= 5000 only -> Small HOLD (passes 732, fails 1500)
    r2 = {k: dict(v) for k, v in r.items()}
    for key in ("sm118", "sm113"):
        L = lab(MODELS[key][0], "F6")
        r2[L] = {u: ({"benchmark": {h: {"LogLoss": x["benchmark"][h]["LogLoss"] * (1.016 if u >= 5000 else 1.0),
                                        "size": 1000} for h in ("ahead", "instant")}}) for u, x in r2[L].items()}
    res = decide(r2, {}, set(u732), set(unew), lists)
    assert res["sizes"]["Small"]["verdict"] == "HOLD", res["sizes"]
    assert res["sizes"]["Big"]["verdict"] == "GO" and res["sizes"]["Tiny"]["verdict"] == "GO"
    # case 2b: Small / Tiny rows 3% worse than Big on the ids >= 5000 for every label (Big in-sample) -> still GO
    r2b = {k: dict(v) for k, v in r.items()}
    for key in ("sm118", "sm113", "sm130th", "sm130io"):
        for s in LABELS:
            L = lab(MODELS[key][0], s)
            r2b[L] = {u: ({"benchmark": {h: {"LogLoss": x["benchmark"][h]["LogLoss"] * (1.03 if u >= 5000 else 1.0),
                                             "size": 1000} for h in ("ahead", "instant")}}) for u, x in r2b[L].items()}
    res = decide(r2b, {}, set(u732), set(unew), lists)
    assert all(v["verdict"] == "GO" for v in res["sizes"].values()), res["sizes"]
    assert res["models"]["sm118"]["direct_total_1500"]["curve"]["ci"][1] > 0.5
    # case 3: Big label +0.4% on ids >= 5000 only -> Big NO (Big is judged on the 1,500 set too)
    r3 = {k: dict(v) for k, v in r.items()}
    for key in ("sm91s2",):
        L = lab(MODELS[key][0], "F6")
        r3[L] = {u: ({"benchmark": {h: {"LogLoss": x["benchmark"][h]["LogLoss"] * (1.008 if u >= 5000 else 1.0),
                                        "size": 1000} for h in ("ahead", "instant")}}) for u, x in r3[L].items()}
    res = decide(r3, {}, set(u732), set(unew), lists)
    assert res["sizes"]["Big"]["verdict"] == "NO", res["sizes"]
    # case 4: a non-finite row -> FAIL; coverage short -> pending
    res = decide(r, {lab(MODELS["sm130th"][0], "T24"): {5001}}, set(u732), set(unew), lists)
    assert res["sizes"]["Tiny"]["verdict"] == "NO", res["sizes"]
    r5 = {k: dict(v) for k, v in r.items()}
    L = lab(MODELS["sm118"][0], "F6")
    r5[L] = {u: x for u, x in r5[L].items() if u < 5700}
    res = decide(r5, {}, set(u732), set(unew), lists)
    assert res["status"] == "pending", res["pending"]
    print("[sm164] selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["users", "smoke", "report", "selftest"])
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    if a.cmd == "users":
        return users_cmd(a.write)
    return {"smoke": smoke, "report": report, "selftest": selftest}[a.cmd]()


if __name__ == "__main__":
    sys.exit(main())
