#!/usr/bin/env python
"""Energy-triggered rank growth (SM159): threshold selection, the pre-registered pointer rule (design.md "Rule") and
the read-out. CPU only, aggregates only.

  python sm159_report.py select     cpu/calib-train.json (training users)  -> sm159-taus.json (matched code per model)
  python sm159_report.py fallback   cpu/sm159-rows.jsonl (bytes ONLY)      -> sm159-taus.json "fallback" + prints the
                                    extra --only arguments (empty when every matched label is within 2% of T24)
  python sm159_report.py labels     prints TAG/LABEL lines of the probe grid (run-sm159.sh)
  python sm159_report.py report     rows + taus -> sm159-result.{json,md}; exit 0 decided, 2 rows missing, 3 proof failed
  python sm159_report.py selftest   synthetic rows

Cost of a label = 100 x (sum over users of its benchmark LogLoss x size / the same for REF - 1) per head, REF =
`w8+s8hx8`; paired user bootstrap 2,000 draws, seed 0, 95% CI (as the output-aware truncation pointer (SM157)).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sm159_bytes as sb  # noqa: E402

KEY = {"curve": "ahead", "instant": "instant"}
TWO, IO = ("curve", "instant"), ("instant",)
MODELS = {   # key: (tag, plain name, heads, size)
    "sm91s2": ("sm91s2-big-showtime", "Big two-head (SM91s2)", TWO, "big"),
    "sm99": ("sm99-big-inst-e2", "Big Instant-Only (SM99)", IO, "big"),
    "sm118": ("sm118-e26-dec2ep32k-t91s2", "Small two-head (SM118)", TWO, "small"),
    "sm113": ("sm113-inst-e35st-st-dec2ep32k-t99", "Small Instant-Only (SM113)", IO, "small"),
    "sm130th": ("sm130-th-dec2ep32k-t91s2", "six-layer Tiny two-head (SM130)", TWO, "tiny"),
    "sm130io": ("sm130-io-dec2ep32k-t99", "six-layer Tiny Instant-Only (SM130)", IO, "tiny"),
}
TWO_HEAD = ("sm91s2", "sm118", "sm130th")
PLAIN = {"REF": "w8+s8hx8", "Q2": "w8+lr2q8x8", "T24": "w8+lr2.16.4q8x8", "F6": "w8+lr2.16.4q6x8"}
TRAIN_GRID = (800, 1000, 1250, 1500, 1800, 2200)   # tau x 1e4, training-user calibration
FIXED = (500, 1000, 1500, 2000)                   # tau x 1e4, scored on the probe users for every model
MATCH_TOL, FLOOR, CUT, WORSE, MIN_INFO = 0.02, 0.01, 0.20, 0.01, 2
PROBE_USERS = [2388, 4676, 1866, 2883, 3882, 1215, 4407, 966]
SM157_ROWS = HERE.parent / "sm157" / "cpu" / "sm157-rows.jsonl"
IGNORE = {"model_seconds", "prepare_seconds", "provenance_commit", "provenance_dirty", "lowrank_ranks"}


def elabel(code, q="q8"):
    return f"w8+lr2{q}x8g4e{int(code)}"


def code_of(label):
    return int(label.rsplit("e", 1)[1]) if "g4e" in label else None


# ------------------------------------------------------------------------------------------------ bytes

def sum_readouts(ros):
    acc = {}
    for ro in ros:
        for name, st in ro.items():
            a = acc.get(name)
            if a is None:
                acc[name] = json.loads(json.dumps(st))
                continue
            for k in ("entities", "entities_n1", "entities_ge16", "sum_rank"):
                a[k] += st[k]
            for k in ("rank_hist", "rank_hist_ge16", "entity_max_hist", "entity_max_hist_ge16"):
                a[k] = [x + y for x, y in zip(a[k], st[k])]
    return acc


def bytes_of(size, label, ro):
    """{label, REF, T24 (at the label's factor bits: F6 for q6), Q2} totals and rho = label / T24 - 1."""
    fb = 6 if "q6x" in label else 8
    b = sb.label_bytes(size, ro, fb, energy="g4e" in label)
    b["rho"] = b["label"] / b["T24"] - 1
    b["vs_int8_pct"] = 100 * (b["label"] / b["REF"] - 1)
    b["card_mean"] = sb.stream_bytes(size, "card", ro["card"], fb, energy="g4e" in label) / ro["card"]["entities"]
    b["note_mean"] = sb.stream_bytes(size, "note", ro["note"], fb, energy="g4e" in label) / ro["note"]["entities"]
    return b


def match_code(points):
    """points {code: rho} (rho = bytes / T24 - 1, decreasing in tau) -> (code, clamped): log-linear interpolation of
    the code at rho = 0 between the first adjacent pair (from the smallest code) whose rho changes sign."""
    cs = sorted(points)
    for a, b in zip(cs, cs[1:]):
        ra, rb = points[a], points[b]
        if ra == 0:
            return a, False
        if (ra > 0) != (rb > 0) or rb == 0:
            t = ra / (ra - rb)
            return int(round(math.exp(math.log(a) + t * (math.log(b) - math.log(a))))), False
    return (cs[-1] if points[cs[-1]] > 0 else cs[0]), True


def select():
    doc = json.loads((HERE / "cpu" / "calib-train.json").read_text())
    out = {"note": "matched codes (tau x 1e4) from the training-user calibration (design.md 'Calibration')",
           "train_grid": list(TRAIN_GRID), "fixed": list(FIXED), "models": {}}
    for mk, (tag, name, _, size) in MODELS.items():
        labs = doc["models"][tag]
        pts, rows = {}, {}
        for c in TRAIN_GRID:
            b = bytes_of(size, elabel(c), labs[elabel(c)]["streams"])
            pts[c] = b["rho"]
            rows[str(c)] = {"rho_vs_T24": b["rho"], "vs_int8_pct": b["vs_int8_pct"]}
        code, clamped = match_code(pts)
        ro = labs[elabel(TRAIN_GRID[0])]["streams"]
        t24 = sb.label_bytes(size, ro, 8, energy=False)
        out["models"][mk] = {"tag": tag, "name": name, "grid": rows, "matched_code": code, "clamped": clamped,
                             "T24_vs_int8_pct": 100 * (t24["T24"] / t24["REF"] - 1),
                             "Q2_vs_T24_pct": 100 * (t24["Q2"] / t24["T24"] - 1),
                             "entities": {k: ro[k]["entities"] for k in ("card", "note")},
                             "reviews": doc["provenance"][tag]["reviews"]}
    (HERE / "sm159-taus.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for mk, m in out["models"].items():
        print(f"[sm159] {m['name']}: matched code {m['matched_code']}{' (clamped)' if m['clamped'] else ''}")
    return 0


def probe_labels(taus, mk, with_fallback=True):
    """{name: label} scored for model mk: the plain labels, the fixed thresholds, EM / EM6, and EMP / EMP6."""
    labs = dict(PLAIN)
    for c in FIXED:
        labs[f"E{c}"] = elabel(c)
    cm = taus["models"][mk]["matched_code"]
    labs["EM"], labs["EM6"] = elabel(cm), elabel(cm, "q6")
    fb = taus.get("fallback", {}).get(mk, {}).get("code") if with_fallback else None
    if fb is not None:
        labs["EMP"], labs["EMP6"] = elabel(fb), elabel(fb, "q6")
    return labs


def labels_cmd(group):
    taus = json.loads((HERE / "sm159-taus.json").read_text())
    for mk, (tag, _, _, _) in MODELS.items():
        if group != "all" and mk not in group.split(","):
            continue
        seen = set()
        for lab in probe_labels(taus, mk).values():
            if lab not in seen:
                seen.add(lab)
                print(f"{tag}/{lab}")
    return 0


def read_rows(path):
    rows = {}
    for raw in open(path, "rb"):
        try:
            r = json.loads(raw.decode("utf-8", errors="replace"))
        except ValueError:
            continue
        if "benchmark" in r or "nonfinite" in r:
            rows.setdefault((r["tag"], r["variant"].split("@")[0]), {})[r["user_id"]] = r
    return rows


def probe_bytes(rows, mk, label):
    tag, _, _, size = MODELS[mk]
    us = rows.get((tag, label), {})
    if sorted(us) != sorted(PROBE_USERS) or not all("lowrank_ranks" in r for r in us.values()):
        return None
    return bytes_of(size, label, sum_readouts([us[u]["lowrank_ranks"] for u in PROBE_USERS]))


def fallback():
    """Bytes only (never a LogLoss): a model whose EM probe bytes miss T24's by more than 2% gets a probe-matched
    code from the scored q8 energy labels (fixed + EM), interpolated as in select."""
    taus = json.loads((HERE / "sm159-taus.json").read_text())
    rows = read_rows(HERE / "cpu" / "sm159-rows.jsonl")
    fb, extra = {}, []
    for mk, (tag, name, _, _) in MODELS.items():
        labs = probe_labels(taus, mk, with_fallback=False)
        em = probe_bytes(rows, mk, labs["EM"])
        if em is None:
            raise SystemExit(f"SM159 fallback: {tag} EM rows missing")
        entry = {"EM_rho": em["rho"], "code": None}
        if abs(em["rho"]) > MATCH_TOL:
            pts = {}
            for n, lab in labs.items():
                if n.startswith("E") and "q6x" not in lab:
                    b = probe_bytes(rows, mk, lab)
                    if b is not None:
                        pts[code_of(lab)] = b["rho"]
            code, clamped = match_code(pts)
            entry.update({"probe_points": {str(k): v for k, v in sorted(pts.items())}, "code": code,
                          "clamped": clamped})
            if code not in pts:
                extra += [f"{tag}/{elabel(code)}"]
            extra += [f"{tag}/{elabel(code, 'q6')}"]
        fb[mk] = entry
    taus["fallback"] = fb
    (HERE / "sm159-taus.json").write_text(json.dumps(taus, indent=1), encoding="utf-8")
    for x in extra:
        print(x)
    return 0


# ------------------------------------------------------------------------------------------------ statistics

def finite(r):
    return "nonfinite" not in r and all(math.isfinite(r["benchmark"][h]["LogLoss"]) for h in ("ahead", "instant")
                                        if isinstance(r["benchmark"].get(h), dict))


def _sums(rows, h, users):
    return np.array([rows[u]["benchmark"][h]["LogLoss"] * rows[u]["benchmark"][h]["size"] for u in users])


def _idx(n, draws=2000):
    return np.random.default_rng(0).integers(0, n, size=(draws, n))


def cmp(new, ref, h, users):
    a, b = _sums(new, h, users), _sums(ref, h, users)
    idx = _idx(len(users))
    boot = (a[idx].sum(1) / b[idx].sum(1) - 1) * 100
    return {"value": float((a.sum() / b.sum() - 1) * 100),
            "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


def cmp_interp(xa, xb, lam, ref, h, users):
    """lam x cost(xa) + (1 - lam) x cost(xb) against ref, with the same bootstrap draws."""
    a, b, r = _sums(xa, h, users), _sums(xb, h, users), _sums(ref, h, users)
    idx = _idx(len(users))
    f = lambda x, y: (x / y - 1) * 100   # noqa: E731
    boot = lam * f(a[idx].sum(1), r[idx].sum(1)) + (1 - lam) * f(b[idx].sum(1), r[idx].sum(1))
    return {"value": float(lam * f(a.sum(), r.sum()) + (1 - lam) * f(b.sum(), r.sum())),
            "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], "lam": lam}


def proof(rows):
    """K2: REF, Q2 and T24 rows equal the output-aware truncation pointer (SM157)'s probe rows in every field except
    timing and provenance (identical JSON values); K3: T24 and Q2 read-outs equal the answer-count formula."""
    if not SM157_ROWS.exists():
        return {"ok": False, "error": f"{SM157_ROWS} missing"}
    old = read_rows(SM157_ROWS)
    checked, bad = 0, []
    for mk, (tag, _, _, _) in MODELS.items():
        for n in ("REF", "Q2", "T24"):
            lab = PLAIN[n]
            for u in PROBE_USERS:
                a, b = rows.get((tag, lab), {}).get(u), old.get((tag, lab), {}).get(u)
                if a is None or b is None:
                    bad.append(f"{tag}/{lab}/{u} missing")
                    continue
                keys = (set(a) | set(b)) - IGNORE
                diff = sorted(k for k in keys if a.get(k) != b.get(k))
                checked += 1
                if diff:
                    bad.append(f"{tag}/{lab}/{u}: {diff[:5]}")
        for n in ("Q2", "T24"):
            for u in PROBE_USERS:
                r = rows.get((tag, PLAIN[n]), {}).get(u)
                if r is None or "lowrank_ranks" not in r:
                    bad.append(f"{tag}/{PLAIN[n]}/{u}: no read-out")
                    continue
                for kind, st in r["lowrank_ranks"].items():
                    if st["sum_rank"] != sb.tier_sum_rank(st, n):
                        bad.append(f"{tag}/{PLAIN[n]}/{u} {kind}: read-out {st['sum_rank']} != formula")
    return {"ok": not bad, "rows_compared": checked, "problems": bad[:20]}


def rank_shares(ro):
    out = {}
    for kind in ("card", "note"):
        st = ro[kind]
        h, e = st["rank_hist"], st["entity_max_hist"]
        th, te = sum(h), sum(e)
        out[kind] = {"entities": st["entities"], "entities_ge16_share": st["entities_ge16"] / max(st["entities"], 1),
                     "head_rank_share": {str(r): h[r] / th for r in range(1, 5)},
                     "entity_max_rank_share": {str(r): e[r] / te for r in range(1, 5)},
                     "entity_max_rank_share_ge16": {str(r): st["entity_max_hist_ge16"][r]
                                                    / max(sum(st["entity_max_hist_ge16"]), 1) for r in range(1, 5)}}
    return out


def matched(mk, fmt, labs, nbytes):
    """('label', name) of the matched energy label within 2% of the reference format's bytes (smallest |rho|), else
    ('interp', (name_a, name_b, lam)) between the nearest bracketing scored energy labels, else ('none', None)."""
    q6 = fmt == "F6"
    cand = [n for n in (("EM6", "EMP6") if q6 else ("EM", "EMP")) if n in labs and nbytes.get(n)]
    ok = [n for n in cand if abs(nbytes[n]["rho"]) <= MATCH_TOL]
    if ok:
        return "label", min(ok, key=lambda n: abs(nbytes[n]["rho"]))
    pool = [n for n in labs if n.startswith("E") and (("q6x" in labs[n]) == q6) and nbytes.get(n)]
    above = [n for n in pool if nbytes[n]["rho"] > 0]
    below = [n for n in pool if nbytes[n]["rho"] <= 0]
    if not above or not below:
        return "none", None
    a = min(above, key=lambda n: nbytes[n]["rho"])
    b = max(below, key=lambda n: nbytes[n]["rho"])
    ba, bb, bt = nbytes[a]["label"], nbytes[b]["label"], nbytes[a]["T24"]
    return "interp", (a, b, (bt - bb) / (ba - bb))


def decide(rows, taus):
    cells, missing, nbytes, ranks, match = {}, [], {}, {}, {}
    for mk, (tag, name, heads, size) in MODELS.items():
        labs = probe_labels(taus, mk)
        got = {n: rows.get((tag, lab), {}) for n, lab in labs.items()}
        users = [u for u in PROBE_USERS if all(u in got[n] for n in labs)]
        if len(users) != len(PROBE_USERS) or not all(finite(got[n][u]) for n in labs for u in users):
            missing.append([mk, sorted(set(PROBE_USERS) - set(users))])
            continue
        nbytes[mk] = {n: probe_bytes(rows, mk, lab) for n, lab in labs.items() if n != "REF"}
        ranks[mk] = {n: rank_shares(sum_readouts([got[n][u]["lowrank_ranks"] for u in PROBE_USERS]))
                     for n in labs if n != "REF"}
        match[mk] = {fmt: matched(mk, fmt, labs, nbytes[mk]) for fmt in ("T24", "F6")}
        for h in heads:
            c = {n: cmp(got[n], got["REF"], KEY[h], users) for n in labs if n != "REF"}
            vs = {}
            for n in labs:
                if n.startswith("E"):
                    vs[n] = cmp(got[n], got["F6" if "q6x" in labs[n] else "T24"], KEY[h], users)
            for fmt, (kind, m) in match[mk].items():
                if kind == "label":
                    c[f"match_{fmt}"], vs[f"match_{fmt}"] = c[m], vs[m]
                elif kind == "interp":
                    a, b, lam = m
                    c[f"match_{fmt}"] = cmp_interp(got[a], got[b], lam, got["REF"], KEY[h], users)
                    vs[f"match_{fmt}"] = cmp_interp(got[a], got[b], lam, got[fmt], KEY[h], users)
            cells[f"{mk}|{h}"] = {"model": mk, "name": name, "head": h, "cost": c, "vs_ref_format": vs,
                                  "reviews": int(sum(got["REF"][u]["benchmark"][KEY[h]]["size"] for u in users))}
    rule = {}
    if not missing:
        for fmt in ("T24", "F6"):
            unmatched = [mk for mk in MODELS if match[mk][fmt][0] == "none"]
            primary, worse = {}, []
            for mk in TWO_HEAD:
                hs = {h: cells[f"{mk}|{h}"]["cost"][fmt]["value"] for h in TWO}
                h = max(hs, key=hs.get)
                key = f"{mk}|{h}"
                e = cells[key]["cost"].get(f"match_{fmt}")
                primary[key] = {"ref_cost": hs[h], "informative": hs[h] >= FLOOR,
                                "energy_cost": None if e is None else e["value"],
                                "reduction": None if (e is None or hs[h] < FLOOR) else 1 - e["value"] / hs[h]}
            for key, cl in cells.items():
                d = cl["vs_ref_format"].get(f"match_{fmt}")
                if d is not None and d["value"] > WORSE:
                    worse.append(key)
            info = [k for k, p in primary.items() if p["informative"]]
            inconclusive = bool(unmatched) or len(info) < MIN_INFO
            ok = (not inconclusive and all(primary[k]["reduction"] >= CUT for k in info) and not worse)
            rule[fmt] = {"unmatched_models": unmatched, "primary_cells": primary, "informative_cells": info,
                         "worse_cells": worse, "inconclusive": inconclusive, "pass": bool(ok)}
    verdict = ({"decided": False, "missing": missing} if missing else
               {"decided": True, "pass_T24": rule["T24"]["pass"], "pass_F6": rule["F6"]["pass"],
                "go": rule["T24"]["pass"] or rule["F6"]["pass"],
                "inconclusive": (not (rule["T24"]["pass"] or rule["F6"]["pass"]))
                and rule["T24"]["inconclusive"]})
    return {"cells": cells, "bytes": nbytes, "ranks": ranks,
            "match": {mk: {f: list(v) if v[0] != "interp" else ["interp", list(v[1])] for f, v in m.items()}
                      for mk, m in match.items()}, "rule": rule, "verdict": verdict}


def fewer_bytes_at_t24_cost(res, taus):
    """Descriptive (not in the rule): per model, the scored q8 energy label with the fewest bytes whose point cost is
    at most T24's on every head of the model."""
    out = {}
    for mk, (tag, name, heads, _) in MODELS.items():
        if mk not in res["bytes"]:
            continue
        labs = probe_labels(taus, mk)
        best = None
        for n, lab in labs.items():
            if not n.startswith("E") or "q6x" in lab:
                continue
            if all(res["cells"][f"{mk}|{h}"]["cost"][n]["value"] <= res["cells"][f"{mk}|{h}"]["cost"]["T24"]["value"]
                   for h in heads):
                b = res["bytes"][mk][n]
                if best is None or b["label"] < res["bytes"][mk][best]["label"]:
                    best = n
        out[mk] = None if best is None else {"name": best, "label": labs[best], "rho_vs_T24": res["bytes"][mk][best]["rho"]}
    return out


def cpu_minutes(rows):
    sec = sum(r.get("model_seconds", 0) + r.get("prepare_seconds", 0) for us in rows.values() for r in us.values())
    return round(sec / 60, 1)


def pct(x, d=3):
    return "n/a" if x is None else f"{x:+.{d}f}%"


def ci(c):
    return f"{pct(c['value'])} [{pct(c['ci'][0])}, {pct(c['ci'][1])}]"


def write_md(res, taus):
    L = ["# Energy-triggered rank growth (SM159): pointer result", "",
         "Generated by `sm159_report.py report` from `cpu/sm159-rows.jsonl` (8 probe users of the low-rank structure "
         "probe (SM147), CPU, show-time inputs, leak-free). Cost = % change of the review-weighted benchmark LogLoss "
         "against int8 rows (REF, `w8+s8hx8`), paired user bootstrap 95% CI. Bytes: all answered cards and notes of "
         "the 8 users, end-of-history ranks, the fork's compact layout.", ""]
    v = res["verdict"]
    L += [f"Proof (REF, Q2 and T24 equal SM157's rows; read-out equals the answer-count formula): "
          f"{'OK' if res['proof']['ok'] else 'FAILED ' + json.dumps(res['proof']['problems'][:5])} "
          f"({res['proof'].get('rows_compared', 0)} rows).", ""]
    if not v.get("decided"):
        L += ["**Not decided: rows missing** " + json.dumps(v.get("missing")), ""]
    else:
        L += [f"**Verdict: {'GO' if v['go'] else ('INCONCLUSIVE' if v['inconclusive'] else 'NO GO')}** "
              f"(int8 factors vs T24: {'pass' if v['pass_T24'] else 'no'}; int6 factors vs F6: "
              f"{'pass' if v['pass_F6'] else 'no'}).", ""]
    L += ["## Matched comparison per model and head", "",
          "| Model | Head | T24 | matched energy (int8) | vs T24 (paired) | F6 | matched energy (int6) | vs F6 (paired) |",
          "|---|---|---|---|---|---|---|---|"]
    for key, c in res["cells"].items():
        g = lambda n: ci(c["cost"][n]) if n in c["cost"] else "n/a"   # noqa: E731
        p = lambda n: ci(c["vs_ref_format"][n]) if n in c["vs_ref_format"] else "n/a"   # noqa: E731
        L.append(f"| {c['name']} | {c['head']} | {g('T24')} | {g('match_T24')} | {p('match_T24')} | {g('F6')} | "
                 f"{g('match_F6')} | {p('match_F6')} |")
    L += ["", "## Bytes vs int8 rows and vs T24 (8 probe users)", "",
          "| Model | Label | Variant | vs int8 | vs T24 (or F6 for int6) | card mean (B) | note mean (B) |",
          "|---|---|---|---|---|---|---|"]
    for mk, bb in res["bytes"].items():
        labs = probe_labels(taus, mk)
        for n, b in bb.items():
            L.append(f"| {MODELS[mk][1]} | {n} | `{labs[n]}` | {pct(b['vs_int8_pct'], 2)} | {pct(100 * b['rho'], 2)} | "
                     f"{b['card_mean']:,.0f} | {b['note_mean']:,.0f} |")
    L += ["", "## Cost vs int8 rows of every label", "", "| Model | Head | " + " | ".join(
        ["Q2", "T24", "F6"] + [f"E{c}" for c in FIXED] + ["EM", "EM6", "EMP", "EMP6"]) + " |",
          "|---|---|" + "---|" * (7 + len(FIXED))]
    for key, c in res["cells"].items():
        L.append(f"| {c['name']} | {c['head']} | " + " | ".join(
            pct(c["cost"][n]["value"]) if n in c["cost"] else "-" for n in
            ["Q2", "T24", "F6"] + [f"E{x}" for x in FIXED] + ["EM", "EM6", "EMP", "EMP6"]) + " |")
    L += ["", "## Rank distribution (share of answered entities by their largest head rank, card / note)", "",
          "| Model | Label | card r2 / r3 / r4 | note r2 / r3 / r4 | card heads r2 / r3 / r4 | note heads r2 / r3 / r4 |",
          "|---|---|---|---|---|---|"]
    for mk, rr in res["ranks"].items():
        for n, s in rr.items():
            f = lambda d: " / ".join(f"{100 * d[str(r)]:.1f}%" for r in (2, 3, 4))   # noqa: E731
            L.append(f"| {MODELS[mk][1]} | {n} | {f(s['card']['entity_max_rank_share'])} | "
                     f"{f(s['note']['entity_max_rank_share'])} | {f(s['card']['head_rank_share'])} | "
                     f"{f(s['note']['head_rank_share'])} |")
    if res.get("rule"):
        L += ["", "## Rule", ""]
        for fmt, r in res["rule"].items():
            L.append(f"- {fmt}: informative cells {r['informative_cells']}, primary " + json.dumps(
                {k: (None if p["reduction"] is None else round(p["reduction"], 3)) for k, p in r["primary_cells"].items()})
                + f", worse cells {r['worse_cells'] or '-'}, unmatched {r['unmatched_models'] or '-'}, "
                f"pass {r['pass']}")
    L += ["", f"Probe scoring CPU time (model + preparation): {res.get('cpu_minutes')} min."]
    (HERE / "sm159-result.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def report():
    path = HERE / "cpu" / "sm159-rows.jsonl"
    if not path.exists():
        print("rows missing")
        return 2
    taus = json.loads((HERE / "sm159-taus.json").read_text())
    rows = read_rows(path)
    res = decide(rows, taus)
    res["proof"] = proof(rows)
    if res["verdict"].get("decided"):
        res["fewer_bytes_at_t24_cost"] = fewer_bytes_at_t24_cost(res, taus)
    if not res["proof"]["ok"]:
        res["verdict"] = {"decided": False, "proof_failed": True, **res["verdict"]}
    res["cpu_minutes"] = cpu_minutes(rows)
    res["taus"] = taus
    (HERE / "sm159-result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    write_md(res, taus)
    print(json.dumps(res["verdict"]))
    if not res["proof"]["ok"]:
        return 3
    return 0 if res["verdict"].get("decided") else 2


# ------------------------------------------------------------------------------------------------ selftest

def selftest():
    global SM157_ROWS
    assert match_code({1000: 0.07, 2000: -0.02}) == (int(round(math.exp(math.log(1000) + 0.07 / 0.09 * math.log(2)))),
                                                      False)
    assert match_code({1000: 0.1, 2000: 0.05}) == (2000, True) and match_code({1000: -0.1, 2000: -0.2}) == (1000, True)
    taus = {"models": {mk: {"matched_code": 1600} for mk in MODELS}}

    def ro(size, extra):
        H, K, D, card, note = sb.ARCH[size]
        out = {}
        for kind, (L, _) in (("card", card), ("note", note)):
            E, n1, e16 = 100, 10, 8
            base = L * H * (n1 + 2 * (E - n1 - e16) + 4 * e16)
            out[kind] = {"entities": E, "entities_n1": n1, "entities_ge16": e16, "H": H, "K": K,
                         "layers": list(range(L)), "sum_rank": int(base * extra), "rank_hist": [0, 1, 1, 1, 1, 0, 0, 0, 0],
                         "rank_hist_ge16": [0] * 9, "entity_max_hist": [0, 1, 1, 1, 1, 0, 0, 0, 0],
                         "entity_max_hist_ge16": [0, 0, 0, 1, 1, 0, 0, 0, 0]}
        return out

    def build(mult, extra):
        rows = {}
        for mk, (tag, _, _, size) in MODELS.items():
            for n, lab in probe_labels(taus, mk).items():
                us = {}
                for u in PROBE_USERS:
                    base = {h: 0.3 + 0.01 * ((u * 7 + i) % 5) for i, h in enumerate(("ahead", "instant"))}
                    k = mult.get(n, 1.0)
                    f = (lambda h, k=k, mk=mk: k(mk, h)) if callable(k) else (lambda h, k=k: k)
                    us[u] = {"user_id": u, "tag": tag, "variant": lab + "@e2s-query",
                             "benchmark": {h: {"LogLoss": base[h] * f(h), "size": 1000 + u % 7} for h in base}}
                    if n != "REF":
                        us[u]["lowrank_ranks"] = ro(size, extra.get(n, 1.0))
                rows[(tag, lab)] = us
        return rows
    extra = {"E500": 1.2, "E1000": 1.08, "E1500": 1.0, "E2000": 0.95, "EM": 1.0, "EM6": 1.0, "Q2": 0.9}
    # EM matched (same sum_rank as T24, the 2-bit cap field adds a few bytes: within 2%), 50% cheaper than T24
    r = decide(build({"T24": 1.0004, "F6": 1.0005, "Q2": 1.001, "EM": 1.0002, "EM6": 1.00025, "E500": 1.0001,
                      "E1000": 1.0002, "E1500": 1.0002, "E2000": 1.0003}, extra), taus)
    assert r["verdict"]["decided"] and r["verdict"]["pass_T24"] and r["verdict"]["pass_F6"], r["rule"]
    assert r["match"]["sm91s2"]["T24"] == ["label", "EM"]
    red = r["rule"]["T24"]["primary_cells"]["sm91s2|curve"]["reduction"]
    assert abs(red - 0.5) < 1e-3, red
    # only 10% cheaper: no go
    r = decide(build({"T24": 1.0004, "F6": 1.0005, "EM": 1.00036, "EM6": 1.00045}, extra), taus)
    assert r["verdict"]["decided"] and not r["verdict"]["go"]
    # one Instant-Only cell worse by more than +0.01%: no go
    r = decide(build({"T24": 1.0004, "F6": 1.0005, "EM": lambda mk, h: 1.0003 if mk != "sm99" else 1.0006,
                      "EM6": 1.0002}, extra), taus)
    assert not r["verdict"]["pass_T24"] and r["rule"]["T24"]["worse_cells"] == ["sm99|instant"], r["rule"]["T24"]
    # T24 cost below the floor everywhere: inconclusive
    r = decide(build({"T24": 1.00005, "F6": 1.00005}, extra), taus)
    assert r["verdict"]["inconclusive"] and not r["verdict"]["go"]
    # EM off by 8% in bytes: interpolation between E1500 (rho ~ +0.1%) ... use EM 1.08 and E1500 1.0 removed
    ex2 = dict(extra, EM=1.08, EM6=1.08, E1500=1.03)
    r = decide(build({"T24": 1.0004, "EM": 1.0002, "E2000": 1.0003, "E1500": 1.0002}, ex2), taus)
    kind, (a, b, lam) = r["match"]["sm91s2"]["T24"]
    assert kind == "interp" and a == "E1500" and b == "E2000" and 0 < lam < 1, r["match"]["sm91s2"]
    rows = build({}, extra)
    del rows[("sm99-big-inst-e2", elabel(1500))][966]
    assert not decide(rows, taus)["verdict"]["decided"]
    print("sm159_report selftest OK")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("select", "fallback", "labels", "report", "selftest"))
    p.add_argument("--group", default="all", help="labels: comma-separated model keys")
    a = p.parse_args()
    sys.exit({"select": select, "fallback": fallback, "report": report, "selftest": selftest,
              "labels": lambda: labels_cmd(a.group)}[a.mode]())


if __name__ == "__main__":
    main()
