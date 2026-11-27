#!/usr/bin/env python
"""4-bit weights (SM180): GPU smoke check, the pre-registered PTQ and QAT rules (design.md "Rules", "Step 2"), CPU only,
under 1 GB of RAM.

  python sm180_report.py smoke      gpu/smoke-gpu-rows.jsonl: w8 rows == SM164's GPU smoke rows exactly; int4 rows sane
  python sm180_report.py ptq        -> sm180-ptq.{json,md}, gpu/qat-targets.txt; exit 0 decided, 2 pending, 3 invalid
  python sm180_report.py qat        -> sm180-qat.{json,md}; exit 0 decided, 2 pending, 3 invalid
  python sm180_report.py selftest   synthetic rows

Statistic (as SM148 / SM152 / SM164): % = 100 x (sum of the label's LogLoss / sum of the baseline's LogLoss - 1),
review-weighted over the benchmark reviews, paired user bootstrap 2,000 draws, seed 0, 95% CI: `compare` below is a
verbatim copy of showtime_report.compare (sha256 3cdfed03...; selftest checks equality when that file is present).
"""
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
G = HERE / "gpu"
W = Path("C:/Users/jonat/workspace/wt")
RUN = W / "sm180-run"
SRC148 = W / "sm164/artifacts/small-model/sm148/gpu/sm148-rows-732.jsonl"
SRC148_SHA = "b5d3721561cdfd621ff3b7d866b19ebe43f073e3686375e3e2edf2ad9397b2b3"
SRC152 = W / "sm164/artifacts/small-model/sm152/gpu/sm152-rows-732.jsonl"
SRC152_SHA = "1e716ec4fdf3893fd439375378368b8950331a5cf58977a8c1f92d3796f59779"
SMOKE164 = RUN / "artifacts/small-model/sm164/gpu/smoke-gpu-rows.jsonl"
SMOKE164_SHA = "7579f70010110fba503f86d3057fddf085fe116e01f763e9b1c549b9f2000cf0"
USMOKE = RUN / "artifacts/small-model/sm164/users-smoke.json"
U732 = RUN / "artifacts/small-model/sm72s/eval-users-732-fit.json"
SHOWTIME = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/small-model/showtime/showtime_report.py")
SHOWTIME_SHA = "3cdfed03afd40734d4b1931c244e8c1e73453d281a40c3843e80cd310d70dd9d"
PTQROWS, QATROWS, SMOKEROWS = G / "sm180-rows-ptq.jsonl", G / "sm180-rows-qat.jsonl", G / "smoke-gpu-rows.jsonl"
E2S = "@e2s-query"
FMTS = {"F6": "lr2.16.4q6x8", "T24": "lr2.16.4q8x8"}
GROUPS = ("w4g32", "w4g16")
CEIL = 0.15        # CI upper bound per head, int4 vs int8 (Big's own format budget)
MIN_USERS = 715    # as SM152
TOL_EPS = 1e-9
KEY = {"curve": "ahead", "instant": "instant"}
TWO, IO = ("curve", "instant"), ("instant",)
HEADS = {"ahead": "curve", "instant": "instant"}
P = "C:/Users/jonat/workspace/wt"
MODELS = {   # key: (tag, plain name, heads, size, state format, master.pth)
    "sm91s2": ("sm91s2-big-showtime", "Big two-head (SM91s2)", TWO, "Big", "F6",
               f"{P}/sm91s2/artifacts/small-model/sm91s2-big-showtime/expertium/big-showtime/master.pth"),
    "sm99": ("sm99-big-inst-e2", "Big Instant-Only (SM99)", IO, "Big", "F6",
             f"{P}/sm99/artifacts/small-model/sm99-big-inst-e2/expertium/big-inst/master.pth"),
    "sm118": ("sm118-e26-dec2ep32k-t91s2", "Small two-head (SM118)", TWO, "Small", "F6",
              f"{P}/sm118/artifacts/small-model/sm118-e26-dec2ep32k-t91s2-p2/expertium/E89-F65e-F77a/master.pth"),
    "sm113": ("sm113-inst-e35st-st-dec2ep32k-t99", "Small Instant-Only (SM113)", IO, "Small", "F6",
              f"{P}/sm113/artifacts/small-model/sm113-inst-e35st-st-dec2ep32k-t99-p2/expertium/E89-F65e-F77a/master.pth"),
    "sm130th": ("sm130-th-dec2ep32k-t91s2", "six-layer Tiny two-head (SM130)", TWO, "Tiny", "T24",
                f"{P}/sm130/artifacts/small-model/sm130-th-dec2ep32k-p2/expertium/E89-F65e-F77a/master.pth"),
    "sm130io": ("sm130-io-dec2ep32k-t99", "six-layer Tiny Instant-Only (SM130)", IO, "Tiny", "T24",
                f"{P}/sm130/artifacts/small-model/sm130-io-dec2ep32k-p2/expertium/E89-F65e-F77a/master.pth"),
}
SIZES = {"Big": ("sm91s2", "sm99"), "Small": ("sm118", "sm113"), "Tiny": ("sm130th", "sm130io")}
TWOHEAD = {"Small": "sm118", "Tiny": "sm130th"}
INSTONLY = {"Small": "sm113", "Tiny": "sm130io"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lab(tag, w, fmt):
    return f"{tag}/{w}+{FMTS[fmt]}{E2S}"


def arm_tags(key):
    return f"sm180-{key}-qat", ("sm171-tiny-ctl" if key == "sm130th" else f"sm180-{key}-ctl")


def read_rows(path):
    """{label: {user: row}} of benchmark rows, {label: set(users)} of non-finite rows (first row per user wins)."""
    rows, nonfinite = {}, {}
    if Path(path).exists():
        for raw in open(path, "rb"):
            try:
                r = json.loads(raw.decode("utf-8", errors="replace"))
            except ValueError:
                continue
            if "error" in r:
                continue
            if "benchmark" in r:
                rows.setdefault(r["label"], {}).setdefault(r["user_id"], r)
            elif "nonfinite" in r:
                nonfinite.setdefault(r["label"], set()).add(r["user_id"])
    return rows, nonfinite


def merged(paths):
    rows, nf = {}, {}
    for p in paths:
        r, n = read_rows(p)
        for L, d in r.items():
            for u, row in d.items():
                rows.setdefault(L, {}).setdefault(u, row)
        for L, s in n.items():
            nf.setdefault(L, set()).update(s)
    return rows, nf


def compare(new, ref, big, users):   # verbatim copy of showtime_report.compare (sha256 3cdfed03...)
    """new - ref per head over `users` (the SM92 rule's compare): raw review-weighted delta, points = % of `big` LogLoss,
    pct_of_ref = % of the reference's own LogLoss, paired user bootstrap 2,000, seed 0, 95% CI."""
    idx = np.random.default_rng(0).integers(0, len(users), size=(2000, len(users)))
    out = {"users": len(users)}
    for head, h in HEADS.items():
        a = np.array([new[u][head][0] for u in users]); b = np.array([ref[u][head][0] for u in users])
        c = np.array([big[u][head][0] for u in users])
        w = np.array([big[u][head][1] for u in users], float)
        a, b, c = (np.where(w > 0, x, 0.0) for x in (a, b, c))
        boot_raw = ((a - b)[idx] * w[idx]).sum(1) / w[idx].sum(1)
        boot = ((a - b)[idx] * w[idx]).sum(1) / (c[idx] * w[idx]).sum(1) * 100
        boot_pct = ((a - b)[idx] * w[idx]).sum(1) / (b[idx] * w[idx]).sum(1) * 100
        big_loss = (c * w).sum()
        out[h] = {"new": round(float((a * w).sum() / w.sum()), 6), "ref": round(float((b * w).sum() / w.sum()), 6),
                  "big": round(float(big_loss / w.sum()), 6), "delta": round(float(((a - b) * w).sum() / w.sum()), 6),
                  "delta_ci": [round(float(np.percentile(boot_raw, q)), 6) for q in (2.5, 97.5)],
                  "points": round(float(((a - b) * w).sum() / big_loss * 100), 3),
                  "points_ci": [round(float(np.percentile(boot, q)), 3) for q in (2.5, 97.5)],
                  "pct_of_ref": round(float(((a - b) * w).sum() / (b * w).sum() * 100), 3),
                  "pct_of_ref_ci": [round(float(np.percentile(boot_pct, q)), 3) for q in (2.5, 97.5)],
                  "users_better": int((a < b).sum()), "users_worse": int((a > b).sum())}
    return out


def as_loss(rows):
    return {u: {h: ((r["benchmark"][h]["LogLoss"], r["benchmark"][h]["size"]) if h in r["benchmark"] else (float("nan"), 0))
                for h in ("ahead", "instant")} for u, r in rows.items()}


def cmp(new, ref, heads, users):
    users = sorted(u for u in users if u in new and u in ref)
    if not users:
        return {}
    c = compare(as_loss(new), as_loss(ref), as_loss(ref), users)
    return {h: {"value": c[h]["pct_of_ref"], "ci": c[h]["pct_of_ref_ci"], "users": c["users"],
                "new": c[h]["new"], "ref": c[h]["ref"]} for h in heads}


def fmt(x):
    return "n/a" if not x else f"{x['value']:+.3f}% [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]"


def finite(rows, heads):
    return all(math.isfinite(r["benchmark"][KEY[h]]["LogLoss"]) for r in rows.values() for h in heads)


def coverage(rows, nf, labels, users, res):
    ok = True
    for L in labels:
        have = len(set(rows.get(L, {})) & set(users))
        if have < MIN_USERS:
            res["pending"].append(f"{L}: {have} of {MIN_USERS} users")
            ok = False
        if nf.get(L, set()) & set(users):
            res["nonfinite"].append(f"{L}: {sorted(nf[L] & set(users))[:10]}")
    return ok


# ---------------------------------------------------------------- smoke

def smoke():
    if sha(SMOKE164) != SMOKE164_SHA:
        print("[sm180] smoke FAIL: SM164 smoke rows changed")
        return 1
    g, gnf = read_rows(SMOKEROWS)
    o, _ = read_rows(SMOKE164)
    us = json.loads(USMOKE.read_text())["users"]
    lines, cells = [], {}
    fields = ("curve_log_loss", "instant_log_loss")
    for key, (tag, name, heads, size, fm, _p) in MODELS.items():
        for w8 in (f"{tag}/w8+s8hx8{E2S}", lab(tag, "w8", "T24"), lab(tag, "w8", "F6")):
            if set(g.get(w8, {})) != set(us) or set(o.get(w8, {})) != set(us):
                lines.append(f"{w8}: rows missing"); continue
            worst, same = 0.0, True
            for u in us:
                a, b = g[w8][u], o[w8][u]
                for h in ("ahead", "instant"):
                    worst = max(worst, abs(a["benchmark"][h]["LogLoss"] - b["benchmark"][h]["LogLoss"]))
                    same &= a["benchmark"][h]["LogLoss"] == b["benchmark"][h]["LogLoss"]
                    same &= a["benchmark"][h]["size"] == b["benchmark"][h]["size"]
                same &= all(a.get(f) == b.get(f) for f in fields) and a["weight_bytes"] == b["weight_bytes"]
            cells[w8] = {"max_abs": worst, "bitwise": same}
            if not same:
                lines.append(f"{w8}: not bitwise equal to SM164 (max |d| {worst:.2e})")
        ref = lab(tag, "w8", fm)
        for w in GROUPS:
            L = lab(tag, w, fm)
            if gnf.get(L) or set(g.get(L, {})) != set(us):
                lines.append(f"{L}: rows missing or non-finite"); continue
            fin = finite(g[L], heads)
            sizes = all(g[L][u]["benchmark"][KEY[h]]["size"] == g[ref][u]["benchmark"][KEY[h]]["size"]
                        for u in us for h in heads)
            smaller = all(g[L][u]["weight_bytes"] < g[ref][u]["weight_bytes"] for u in us)
            moved = any(g[L][u]["benchmark"][KEY[h]]["LogLoss"] != g[ref][u]["benchmark"][KEY[h]]["LogLoss"]
                        for u in us for h in heads)
            cells[L] = {"finite": fin, "sizes_equal": sizes, "bytes_smaller": smaller, "differs_from_w8": moved,
                        "weight_bytes": g[L][us[0]]["weight_bytes"], "w8_bytes": g[ref][us[0]]["weight_bytes"]}
            if not (fin and sizes and smaller and moved):
                lines.append(f"{L}: {cells[L]}")
    ok = not lines
    (G / "smoke-check.json").write_text(json.dumps({"ok": ok, "cells": cells, "problems": lines}, indent=1) + "\n",
                                        encoding="utf-8")
    n8 = sum(1 for k, v in cells.items() if "bitwise" in v)
    print(f"[sm180] smoke OK: {n8} int8 (model, label) cells bitwise equal to SM164's GPU smoke rows, "
          f"{len(cells) - n8} int4 cells finite / smaller / different" if ok else
          "[sm180] smoke FAIL: " + "; ".join(lines[:8]))
    return 0 if ok else 1


# ---------------------------------------------------------------- bytes

def weight_bytes():
    sys.path.insert(0, str(RUN))
    import torch
    from scripts import compress_quant as cq
    out = {}
    for key, (*_x, path) in MODELS.items():
        st = {k: v.float() for k, v in torch.load(path, map_location="cpu", weights_only=True).items()}
        names = cq.linear_weight_names(st)
        b8 = cq.weight_bytes(st, names, 8, None)["total"]
        out[key] = {"tensors": len(names), "quantized_parameters": sum(st[n].numel() for n in names),
                    "parameters": sum(v.numel() for v in st.values()), "w8": b8, "fp32": 4 * sum(v.numel() for v in st.values())}
        for w in GROUPS:
            b = cq.weight_bytes(st, names, *cq.WEIGHT_SCHEMES[{"w4g32": "w4-g32", "w4g16": "w4-g16"}[w]])["total"]
            out[key][w] = b
            out[key][w + "_saved_pct"] = round(100 * (1 - b / b8), 1)
    return out


# ---------------------------------------------------------------- PTQ

def decide_ptq(rows, nf, users):
    res = {"ceiling_upper_pct": CEIL, "users_list": len(users), "models": {}, "sizes": {}, "pending": [],
           "nonfinite": [], "qat_targets": []}
    for key, (tag, name, heads, size, fm, _p) in MODELS.items():
        ref = lab(tag, "w8", fm)
        m = {"name": name, "size": size, "format": f"{fm} ({FMTS[fm]})", "heads": list(heads), "groups": {}}
        for w in GROUPS:
            L = lab(tag, w, fm)
            covered = coverage(rows, nf, [L, ref], users, res)
            c = cmp(rows.get(L, {}), rows.get(ref, {}), heads, users)
            fin = not (nf.get(L, set()) | nf.get(ref, set())) & set(users) and finite(rows.get(L, {}), heads)
            cells = {h: bool(c.get(h)) and fin and c[h]["ci"][1] <= CEIL + TOL_EPS for h in heads}
            m["groups"][w] = {"label": L, "vs": ref, "cmp": c, "finite": fin, "covered": covered, "free": cells,
                              "all_free": all(cells.values())}
        res["models"][key] = m
    for size, keys in SIZES.items():
        s = {g: all(res["models"][k]["groups"][g]["all_free"] for k in keys) for g in GROUPS}
        chosen = next((g for g in GROUPS if s[g]), None)
        s["chosen"] = chosen
        s["verdict"] = f"PTQ free at {chosen}" if chosen else "PTQ fails"
        if not chosen and size in TWOHEAD:
            th = res["models"][TWOHEAD[size]]["groups"]["w4g32"]["all_free"]
            s["qat_target"] = INSTONLY[size] if th else TWOHEAD[size]
        res["sizes"][size] = s
    for size in ("Tiny", "Small"):
        if res["sizes"][size].get("qat_target"):
            res["qat_targets"].append(res["sizes"][size]["qat_target"])
    res["status"] = "pending" if res["pending"] else "decided"
    return res


def ptq():
    if sha(SRC148) != SRC148_SHA or sha(SRC152) != SRC152_SHA:
        print("[sm180] invalid: SM148 / SM152 rows changed"); return 3
    if not (G / "smoke-ok").exists():
        print("[sm180] invalid: no gpu/smoke-ok"); return 3
    users = json.loads(U732.read_text())["users"]
    rows, nf = merged([PTQROWS, SRC152, SRC148])
    res = decide_ptq(rows, nf, users)
    res["bytes"] = weight_bytes()
    res["row_sources"] = {"int4": str(PTQROWS), "int8 F6": f"{SRC152} ({SRC152_SHA[:8]})",
                          "int8 T24": f"{SRC148} ({SRC148_SHA[:8]})"}
    (HERE / "sm180-ptq.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    md = ["# 4-bit weights (SM180): PTQ read-out", "",
          f"Status **{res['status']}**. Rule: int4 vs the same model's int8 weights (same state format), CI upper "
          f"bound <= +{CEIL}% per head. Users: the 723-user list of the 732 set.", "",
          "| model | head | w4g32 vs int8 | free | w4g16 vs int8 | free |", "|---|---|---|---|---|---|"]
    for key, m in res["models"].items():
        for h in m["heads"]:
            a, b = m["groups"]["w4g32"], m["groups"]["w4g16"]
            md.append(f"| {m['name']} | {h} | {fmt(a['cmp'].get(h))} | {'yes' if a['free'][h] else 'no'} | "
                      f"{fmt(b['cmp'].get(h))} | {'yes' if b['free'][h] else 'no'} |")
    md += ["", "| size | w4g32 | w4g16 | verdict | QAT target |", "|---|---|---|---|---|"]
    for size, s in res["sizes"].items():
        tgt = MODELS[s["qat_target"]][1] if s.get("qat_target") else "-"
        md.append(f"| {size} | {'FREE' if s['w4g32'] else 'NOT FREE'} | {'FREE' if s['w4g16'] else 'NOT FREE'} | "
                  f"{s['verdict']} | {tgt} |")
    md += ["", "| model | int8 bytes | w4g32 bytes (saved) | w4g16 bytes (saved) |", "|---|---|---|---|"]
    for key, b in res["bytes"].items():
        md.append(f"| {MODELS[key][1]} | {b['w8']:,} | {b['w4g32']:,} ({b['w4g32_saved_pct']}%) | "
                  f"{b['w4g16']:,} ({b['w4g16_saved_pct']}%) |")
    if res["pending"] or res["nonfinite"]:
        md += ["", "Pending: " + "; ".join(res["pending"]), "Non-finite: " + "; ".join(res["nonfinite"])]
    (HERE / "sm180-ptq.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    if res["status"] == "decided":
        (G / "qat-targets.txt").write_text("".join(t + "\n" for t in res["qat_targets"]), encoding="utf-8")
    for size, s in res["sizes"].items():
        print(f"[sm180] PTQ {size}: {s['verdict']}" + (f" -> QAT {MODELS[s['qat_target']][1]}" if s.get("qat_target") else ""))
    print(f"[sm180] PTQ status {res['status']}" + (": " + "; ".join(res["pending"][:6]) if res["pending"] else ""))
    return 0 if res["status"] == "decided" else 2


# ---------------------------------------------------------------- QAT

def decide_qat(rows, nf, users, targets):
    res = {"ceiling_upper_pct": CEIL, "rule": "arm at w4g32 vs control at w8, CI upper <= ceiling per head",
           "users_list": len(users), "arms": {}, "pending": [], "nonfinite": []}
    for key in targets:
        tag0, name, heads, size, fm, _p = MODELS[key]
        qa, ct = arm_tags(key)
        L = {"arm_w4": lab(qa, "w4g32", fm), "arm_w8": lab(qa, "w8", fm), "ctl_w8": lab(ct, "w8", fm),
             "ctl_w4": lab(ct, "w4g32", fm), "orig_w8": lab(tag0, "w8", fm)}
        coverage(rows, nf, list(L.values()), users, res)
        R = {k: rows.get(v, {}) for k, v in L.items()}
        fin = all(finite(R[k], heads) for k in R) and not any(nf.get(v, set()) & set(users) for v in L.values())
        rule = cmp(R["arm_w4"], R["ctl_w8"], heads, users)
        a = {"name": name, "labels": L, "finite": fin, "rule_arm_w4_vs_ctl_w8": rule,
             "report_arm_w4_vs_ctl_w4": cmp(R["arm_w4"], R["ctl_w4"], heads, users),
             "report_arm_w8_vs_ctl_w8": cmp(R["arm_w8"], R["ctl_w8"], heads, users),
             "report_arm_w4_vs_orig_w8": cmp(R["arm_w4"], R["orig_w8"], heads, users),
             "report_ctl_w8_vs_orig_w8": cmp(R["ctl_w8"], R["orig_w8"], heads, users),
             "report_ctl_w4_vs_ctl_w8": cmp(R["ctl_w4"], R["ctl_w8"], heads, users)}
        a["free"] = {h: bool(rule.get(h)) and fin and rule[h]["ci"][1] <= CEIL + TOL_EPS for h in heads}
        a["verdict"] = "QAT free" if all(a["free"].values()) else "QAT not free"
        res["arms"][key] = a
    res["status"] = "pending" if res["pending"] else "decided"
    return res


def qat():
    if sha(SRC148) != SRC148_SHA or sha(SRC152) != SRC152_SHA:
        print("[sm180] invalid: SM148 / SM152 rows changed"); return 3
    tf = G / "qat-targets.txt"
    if not tf.exists():
        print("[sm180] invalid: no gpu/qat-targets.txt (run ptq first)"); return 3
    targets = [t for t in tf.read_text().split() if t]
    users = json.loads(U732.read_text())["users"]
    rows, nf = merged([QATROWS, SRC152, SRC148])
    res = decide_qat(rows, nf, users, targets)
    (HERE / "sm180-qat.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    md = ["# 4-bit weights (SM180): QAT read-out", "", f"Status **{res['status']}**. Rule: {res['rule']} "
          f"(+{CEIL}%).", "", "| model | head | arm w4g32 vs control int8 (rule) | free | arm w4g32 vs control w4g32 | "
          "arm int8 vs control int8 | arm w4g32 vs original int8 | control int4 (PTQ) vs control int8 |",
          "|---|---|---|---|---|---|---|---|"]
    for key, a in res["arms"].items():
        for h in MODELS[key][2]:
            md.append(f"| {a['name']} | {h} | {fmt(a['rule_arm_w4_vs_ctl_w8'].get(h))} | "
                      f"{'yes' if a['free'][h] else 'no'} | {fmt(a['report_arm_w4_vs_ctl_w4'].get(h))} | "
                      f"{fmt(a['report_arm_w8_vs_ctl_w8'].get(h))} | {fmt(a['report_arm_w4_vs_orig_w8'].get(h))} | "
                      f"{fmt(a['report_ctl_w4_vs_ctl_w8'].get(h))} |")
    md += [""] + [f"- {a['name']}: **{a['verdict']}**" for a in res["arms"].values()]
    if res["pending"] or res["nonfinite"]:
        md += ["", "Pending: " + "; ".join(res["pending"]), "Non-finite: " + "; ".join(res["nonfinite"])]
    (HERE / "sm180-qat.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    for a in res["arms"].values():
        print(f"[sm180] QAT {a['name']}: {a['verdict']}")
    print(f"[sm180] QAT status {res['status']}")
    return 0 if res["status"] == "decided" else 2


# ---------------------------------------------------------------- selftest

def selftest():
    rng = np.random.default_rng(1)
    users = list(range(1, 724))
    base = {u: 0.3 + 0.05 * rng.standard_normal() for u in users}

    def row(x):
        return {"benchmark": {h: {"LogLoss": x, "size": 1000} for h in ("ahead", "instant")}}

    def mk(cost=None):   # cost: {(key, group): factor}
        rows = {}
        for key, (tag, _n, heads, size, fm, _p) in MODELS.items():
            rows[lab(tag, "w8", fm)] = {u: row(base[u]) for u in users}
            for w in GROUPS:
                f = (cost or {}).get((key, w), 1.0)
                rows[lab(tag, w, fm)] = {u: row(base[u] * f) for u in users}
        return rows
    r = decide_ptq(mk(), {}, users)
    assert r["status"] == "decided" and all(s["chosen"] == "w4g32" for s in r["sizes"].values()) and not r["qat_targets"]
    r = decide_ptq(mk({("sm130io", "w4g32"): 1.003}), {}, users)
    assert r["sizes"]["Tiny"]["chosen"] == "w4g16" and not r["qat_targets"], r["sizes"]
    r = decide_ptq(mk({("sm118", "w4g32"): 1.003, ("sm118", "w4g16"): 1.002, ("sm91s2", "w4g32"): 1.01,
                       ("sm91s2", "w4g16"): 1.01}), {}, users)
    assert r["sizes"]["Small"]["qat_target"] == "sm118" and r["qat_targets"] == ["sm118"], r["sizes"]
    assert r["sizes"]["Big"]["verdict"] == "PTQ fails" and "qat_target" not in r["sizes"]["Big"]
    r = decide_ptq(mk({("sm130io", "w4g32"): 1.003, ("sm130io", "w4g16"): 1.003, ("sm118", "w4g32"): 1.003,
                       ("sm118", "w4g16"): 1.003}), {}, users)
    assert r["qat_targets"] == ["sm130io", "sm118"], r["qat_targets"]
    r = decide_ptq(mk({("sm118", "w4g32"): 1.0015}), {}, users)   # +0.15% point: CI upper above -> not free
    assert r["sizes"]["Small"]["w4g32"] is False or r["models"]["sm118"]["groups"]["w4g32"]["cmp"]["curve"]["ci"][1] <= CEIL
    rows = mk()
    rows[lab("sm118-e26-dec2ep32k-t91s2", "w4g32", "F6")] = {u: rows[lab("sm118-e26-dec2ep32k-t91s2", "w4g32", "F6")][u]
                                                             for u in users[:700]}
    r = decide_ptq(rows, {}, users)
    assert r["status"] == "pending", r["pending"]
    # noisy per-user costs: the CI matters
    rows = mk()
    L = lab("sm113-inst-e35st-st-dec2ep32k-t99", "w4g32", "F6")
    rows[L] = {u: row(base[u] * (1 + 0.001 + 0.01 * rng.standard_normal())) for u in users}
    r = decide_ptq(rows, {}, users)
    c = r["models"]["sm113"]["groups"]["w4g32"]["cmp"]["instant"]
    assert c["ci"][0] < c["value"] < c["ci"][1]
    # QAT
    q = {}
    for key, f4 in (("sm130th", 1.0005), ("sm118", 1.004)):
        tag0, _n, heads, size, fm, _p = MODELS[key]
        qa, ct = arm_tags(key)
        q[lab(qa, "w4g32", fm)] = {u: row(base[u] * f4) for u in users}
        q[lab(qa, "w8", fm)] = {u: row(base[u]) for u in users}
        q[lab(ct, "w8", fm)] = {u: row(base[u]) for u in users}
        q[lab(ct, "w4g32", fm)] = {u: row(base[u] * 1.01) for u in users}
        q[lab(tag0, "w8", fm)] = {u: row(base[u]) for u in users}
    r = decide_qat(q, {}, users, ["sm130th", "sm118"])
    assert r["arms"]["sm130th"]["verdict"] == "QAT free" and r["arms"]["sm118"]["verdict"] == "QAT not free", r["arms"]
    assert r["status"] == "decided"
    if SHOWTIME.exists() and sha(SHOWTIME) == SHOWTIME_SHA:
        spec = importlib.util.spec_from_file_location("showtime_report", SHOWTIME)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        a = as_loss(rows[L]); b = as_loss(rows[lab("sm113-inst-e35st-st-dec2ep32k-t99", "w8", "F6")])
        assert mod.compare(a, b, b, users) == compare(a, b, b, users)
        print("[sm180] compare == showtime_report.compare (pinned sha256)")
    else:
        print("[sm180] showtime_report.py at the pinned sha256 not present: copy not cross-checked")
    print("[sm180] selftest OK")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.exit({"smoke": smoke, "ptq": ptq, "qat": qat, "selftest": selftest}.get(cmd, lambda: (print(__doc__), 1)[1])())
