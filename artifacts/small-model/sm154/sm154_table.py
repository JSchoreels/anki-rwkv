#!/usr/bin/env python
"""State-format rate-distortion table (SM154): CPU only, existing rows only, no scoring.

Reads the committed per-user benchmark rows of the state-format experiments (worktrees under WT), builds for each of the six
release models and each head the cost of every scored state format against (a) the model's own int8 rows (w8+s8hx8) and
(b) for Small and Tiny the matching Big model at int8 (two-head -> Big two-head (SM91s2), Instant-Only -> Big Instant-Only
(SM99)), then the lower convex hull of state bytes against cost.

  python sm154_table.py        -> sm154-tables.json and sm154-tables.md (next to this file); prints a short summary

Statistic (same as every 732-set read-out, showtime_report.compare): % = 100 x (sum LogLoss new / sum LogLoss ref - 1),
review-weighted over users, paired user bootstrap 2,000 draws, seed 0, 95% CI. Only users present with finite rows in EVERY
row set used here are scored (one common user set for all six models), so the numbers can differ in the third decimal from
the single-experiment reports (their pools had 718 to 723 users); the check block compares a few of them.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("C:/Users/jonat/workspace/wt")
E2S = "@e2s-query"
REFV = "w8+s8hx8"


def exp(n):
    return WT / f"sm{n}/artifacts/small-model/sm{n}"


ROWS = {  # experiment -> rows file
    "SM131": exp(131) / "gpu/sm131-rows.jsonl",
    "SM141": exp(141) / "gpu/sm141-rows-732.jsonl",
    "SM144": exp(144) / "gpu/sm144-rows-732.jsonl",
    "SM145": exp(145) / "gpu/sm145-rows-732.jsonl",
    "SM146": exp(146) / "gpu/sm146-rows-732.jsonl",
    "SM148": exp(148) / "gpu/sm148-rows-732.jsonl",
    "SM150": exp(150) / "gpu/sm150-rows-732.jsonl",
}
MODELS = {  # key: (tag, plain name, heads, size, Big match)
    "sm91s2": ("sm91s2-big-showtime", "Big two-head (SM91s2)", ("curve", "instant"), "Big", None),
    "sm99": ("sm99-big-inst-e2", "Big Instant-Only (SM99)", ("instant",), "Big", None),
    "sm118": ("sm118-e26-dec2ep32k-t91s2", "Small two-head (SM118)", ("curve", "instant"), "Small", "sm91s2"),
    "sm113": ("sm113-inst-e35st-st-dec2ep32k-t99", "Small Instant-Only (SM113)", ("instant",), "Small", "sm99"),
    "sm130th": ("sm130-th-dec2ep32k-t91s2", "six-layer Tiny two-head (SM130)", ("curve", "instant"), "Tiny", "sm91s2"),
    "sm130io": ("sm130-io-dec2ep32k-t99", "six-layer Tiny Instant-Only (SM130)", ("instant",), "Tiny", "sm99"),
}
KEY = {"curve": "ahead", "instant": "instant"}
BUDGET = {"Big": 0.15, "Small": 0.5, "Tiny": 1.0}
BUDGET_TEXT = {"Big": "at most +0.15% vs its own int8", "Small": "at most +0.5% total vs Big at int8",
               "Tiny": "at most +1.0% total vs Big at int8"}

# id -> (variant, plain description, source experiment, bytes source, sizes scored)
FORMATS = {
    "REF": (REFV, "int8 rows, float16 row scales, int8 shift vectors (the reference)", "SM139", "ref"),
    "C4": ("w8+s4c8nhx8", "compact int4 cards, int8 notes", "SM131", "calc:card4"),
    "CO4": ("w8+s4hgx4c8hx8n", "card-only int4 (one scale per head, int4 shifts), int8 notes", "SM131", "calc:cardonly"),
    "A": ("w8+t4.128.8c8nhx8", "cards int4 below 128 answers, int8 from 128; notes int8", "SM141", "tier"),
    "AN": ("w8+t4.128.8c4.128.8nhx8", "cards and notes int4 below 128 answers, int8 from 128", "SM141", "tier"),
    "B4": ("w8+t3.5.4.128.8c3.5.4.128.8nhx8", "AN with int3 for 1-4 answers", "SM141", "tier"),
    "B8": ("w8+t3.9.4.128.8c3.9.4.128.8nhx8", "AN with int3 for 1-8 answers", "SM141", "tier"),
    "D4": ("w8+t2.5.4.128.8c2.5.4.128.8nhx8", "AN with int2 (ternary) for 1-4 answers", "SM141", "tier"),
    "A32": ("w8+t4.32.8c8nhx8", "cards int4 below 32 answers, int8 from 32; notes int8", "SM144", "tier"),
    "A16": ("w8+t4.16.8c8nhx8", "cards int4 below 16 answers, int8 from 16; notes int8", "SM144", "tier"),
    "AN32": ("w8+t4.32.8c4.32.8nhx8", "cards and notes int4 below 32 answers, int8 from 32", "SM144", "tier"),
    "AN16": ("w8+t4.16.8c4.16.8nhx8", "cards and notes int4 below 16 answers, int8 from 16", "SM144", "tier"),
    "Y2N": ("w8+t3.3.4.16.8c3.3.4.16.8nhx8", "AN16 with int3 for 1-2 answers", "SM145", "tier"),
    "Y4C": ("w8+t3.5.4.16.8c4.16.8nhx8", "AN16 with int3 for 1-4 answers on cards only", "SM145", "tier"),
    "Y4N": ("w8+t3.5.4.16.8c3.5.4.16.8nhx8", "AN16 with int3 for 1-4 answers on cards and notes (also SM146 I3a)", "SM145", "tier"),
    "I3b": ("w8+t3.9.4.16.8c3.9.4.16.8nhx8", "AN16 with int3 for 1-8 answers on cards and notes (Tiny only)", "SM146", "tier"),
    "I2a": ("w8+t2.5.4.16.8c2.5.4.16.8nhx8", "AN16 with int2 for 1-4 answers on cards and notes (Tiny only)", "SM146", "tier"),
    "I2b": ("w8+t2.5.3.16.8c2.5.3.16.8nhx8", "int2 for 1-4 answers, int3 for 5-15, int8 from 16 (Tiny only)", "SM146", "tier"),
    "Q4": ("w8+lr4q8x8", "rank 4 int8 factors, every entity", "SM148", "lr"),
    "T24": ("w8+lr2.16.4q8x8", "rank 2 below 16 answers, rank 4 from 16 (the user's choice)", "SM148", "lr"),
    "Q2": ("w8+lr2q8x8", "rank 2 int8 factors, every entity", "SM148", "lr"),
    "R12": ("w8+lr1.16.2q8x8", "rank 1 below 16 answers, rank 2 from 16", "SM150", "lr"),
    "R1": ("w8+lr1q8x8", "rank 1 int8 factors, every entity", "SM150", "lr"),
}
LR_BYTES_KEY = {"Q4": "Q4", "T24": "T24", "Q2": "Q2", "R1": "Q1"}  # label in sm148-bytes.json 'sizes'
# Families left out of the hull (not comparable); reported in the json.
SHORT = {
    "REF": "int8 rows", "C4": "compact int4 cards", "CO4": "card-only int4 cards",
    "A": "int4 cards below 128 answers", "AN": "int4 cards and notes below 128 answers",
    "B4": "int3 at 1-4 answers, int4 below 128", "B8": "int3 at 1-8 answers, int4 below 128",
    "D4": "int2 at 1-4 answers, int4 below 128", "A32": "int4 cards below 32 answers", "A16": "int4 cards below 16 answers",
    "AN32": "int4 cards and notes below 32 answers", "AN16": "int4 cards and notes below 16 answers",
    "Y2N": "int3 at 1-2 answers, int4 below 16", "Y4C": "int3 cards at 1-4 answers, int4 below 16",
    "Y4N": "int3 at 1-4 answers, int4 below 16", "I3b": "int3 at 1-8 answers, int4 below 16",
    "I2a": "int2 at 1-4 answers, int4 below 16", "I2b": "int2 at 1-4 answers, int3 below 16",
    "Q4": "rank 4", "T24": "rank 2 then 4 from 16 answers", "Q2": "rank 2", "R12": "rank 1 then 2 from 16 answers", "R1": "rank 1",
}


def nm(fid):
    return f"{SHORT[fid]} ({fid})" + (" [the user's choice]" if fid == "T24" else "")


EXCLUDED = [
    {"what": "SM129 compact state for Tiny, rows w8+s4c8nhx8 and w8+s4hgx4c8hx8n",
     "why": "scored on earlier Tiny checkpoints (tags sm124 and sm115, one layer), not on the six-layer Tiny models (SM130); "
            "no int8 rows of the same checkpoint on the same pool. The six-layer Tiny models get the same compact int4 "
            "cards as scheme A of SM141."},
    {"what": "SM131 fork default w8+s4c8n (int4 cards with float32 row scales)",
     "why": "its int8 base is the older w8+s8 layout (float32 scales, larger shift vectors), not the release-candidate "
            "w8+s8hx8 layout, so its bytes are not on the same scale; and it overflows for Big user 3876."},
    {"what": "SM129 rows for Small (w8+s4c8n on SM118 and SM113)", "why": "same fork-default layout as above."},
    {"what": "SM146 report-only labels (t3.16.8, I2c, I2d) and SM145 bytes-only labels (Y8N, Y8C, Y2C)",
     "why": "bytes were computed but the formats were never scored."},
    {"what": "SM144 earlier int8 switch (SM144) rows", "why": "included (rows exist: A32, A16, AN32, AN16, same pool and inputs)."},
    {"what": "SM152 smaller shift vectors and factors on T24 states", "why": "still running when this table was built; no result rows."},
]


def read_rows(path):
    rows = {}
    for raw in open(path, "rb"):
        try:
            r = json.loads(raw.decode("utf-8", errors="replace"))
        except ValueError:
            continue
        if "benchmark" in r and r.get("inputs") == "e2s-query":
            rows.setdefault((r["tag"], r["variant"]), {})[r["user_id"]] = r["benchmark"]
    return rows


def loss(b):
    return {h: ((b[h]["LogLoss"], b[h]["size"]) if h in b else (float("nan"), 0)) for h in ("ahead", "instant")}


def compare(new, ref, users):
    """Review-weighted % of the reference LogLoss, paired user bootstrap 2,000 draws, seed 0 (showtime_report.compare)."""
    idx = np.random.default_rng(0).integers(0, len(users), size=(2000, len(users)))
    out = {}
    for h in ("ahead", "instant"):
        a = np.array([new[u][h][0] for u in users]); b = np.array([ref[u][h][0] for u in users])
        w = np.array([ref[u][h][1] for u in users], float)
        a, b = (np.where(w > 0, x, 0.0) for x in (a, b))
        boot = ((a - b)[idx] * w[idx]).sum(1) / (b[idx] * w[idx]).sum(1) * 100
        out[h] = {"value": round(float(((a - b) * w).sum() / (b * w).sum() * 100), 3),
                  "lo": round(float(np.percentile(boot, 2.5)), 3), "up": round(float(np.percentile(boot, 97.5)), 3)}
    return out


def lower_hull(pts):
    """Lower convex hull of (x, y, id) sorted by x; returns hull vertices from the smallest x up to the smallest y."""
    pts = sorted(pts, key=lambda p: (p[0], p[1]))
    hull = []
    for p in pts:
        while len(hull) >= 2 and (hull[-1][0] - hull[-2][0]) * (p[1] - hull[-2][1]) - (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0]) <= 0:
            hull.pop()
        hull.append(p)
    ymin = min(p[1] for p in hull)
    out = []
    for p in hull:
        out.append(p)
        if p[1] == ymin:
            break
    return out


def pareto_dominated(p, pts):
    return any(q is not p and q[0] <= p[0] and q[1] <= p[1] and (q[0] < p[0] or q[1] < p[1]) for q in pts)


def to_md(res):
    L = []
    for m, mm in res["models"].items():
        hs = mm["heads"]
        two = len(hs) == 2
        L.append(f"### {mm['name']}\n")
        L.append(f"Budget: {mm['budget_text']} (95% CI upper bound, every head). Cost columns: point estimate and upper bound in % of LogLoss, "
                 f"first against its own int8 rows" + ("" if mm["size"] == "Big" else f", then against {mm['budget_ref']}") + "." + "\n")
        k = 1 if mm["size"] == "Big" else 2
        hdr = "| format | state bytes vs int8 |" + "".join(f" {h} vs own int8 |" + ("" if k == 1 else f" {h} vs Big at int8 |") for h in hs) + " passes budget | frontier status |"
        L.append(hdr)
        L.append("|" + "---|" * (2 + k * len(hs) + 2))
        order = sorted(mm["formats"], key=lambda i: mm["formats"][i]["bytes_vs_int8_pct"], reverse=True)
        for fid in order:
            f = mm["formats"][fid]
            cells = []
            for h in hs:
                o, v = f["vs_own_int8"][h], f["vs_budget_ref"][h]
                cells.append(f"{o['value']:+.3f} / {o['up']:+.3f}")
                if k == 2:
                    cells.append(f"{v['value']:+.3f} / {v['up']:+.3f}")
            ok = all(f["vs_budget_ref"][h]["up"] <= mm["budget_pct"] for h in hs)
            st = []
            for h in hs:
                hu = mm["hull"][f"{h}_upper"]
                st.append("frontier" if fid in hu["frontier"] else "dominated" if fid in hu["pareto_dominated"] else "inside hull")
            stat = st[0] if len(set(st)) == 1 else "; ".join(f"{h}: {x}" for h, x in zip(hs, st))
            L.append(f"| {nm(fid)} | {f['bytes_vs_int8_pct']:+.2f}% | " + " | ".join(cells) + f" | {'yes' if ok else 'no'} | {stat} |")
        L.append("")
        for h in hs:
            hu = mm["hull"][f"{h}_upper"]
            L.append(f"Frontier on the {h} head (CI upper bound), from int8 toward fewer bytes: " +
                     "; ".join(f"{nm(s['from'])} to {nm(s['to'])} saves {s['bytes_saved_pct_of_int8']:.2f}% of the int8 bytes for +{s['cost_added_pct']:.3f}% of cost, "
                               f"which is {s['cost_per_saved_pct']:.4f}% per saved %" for s in hu["segments"]) + ".")
            c = mm["budget_crossing_on_upper_frontier"][h]
            L.append(f"The budget line is crossed on the {h} frontier at about {c:+.1f}% bytes vs int8." if c is not None else f"The {h} frontier never crosses the budget line.")
            L.append("")
    return "\n".join(L)


def main():
    raw = {n: read_rows(p) for n, p in ROWS.items()}
    # bytes ---------------------------------------------------------------------------------------------------------
    b141 = json.loads((exp(141) / "sm141-bytes.json").read_text())
    ent = json.loads((exp(146) / "sm146-bytes.json").read_text())["entities_732"]
    nc, nn = ent["answered_cards"], ent["notes_with_answers"]
    per_bits = b141["bytes"]
    sz_ref = {s: (per_bits[s]["card"]["8"], per_bits[s]["note"]["8"]) for s in ("Big", "Small", "Tiny")}
    tot_ref = {s: nc * c + nn * n for s, (c, n) in sz_ref.items()}
    bytes_pct = {}   # (format, size) -> vs int8 %
    for n in (141, 144, 145):
        d = json.loads((exp(n) / f"sm{n}-bytes.json").read_text())
        lab = {}
        for line in (exp(n) / "labels.sh").read_text().splitlines():
            if "=" in line and not line.startswith("#") and "\"" not in line.split("=")[0]:
                k, v = line.split("=", 1)
                lab[k.strip()] = v.split("#")[0].strip()
        for sc, x in d["scheme_bytes_732"].items():
            var = lab.get(sc)
            for fid, f in FORMATS.items():
                if f[0] == var:
                    for s, y in x.items():
                        bytes_pct.setdefault((fid, s), y["vs_REF_pct"])
    d146 = json.loads((exp(146) / "sm146-bytes.json").read_text())
    lab146 = {}
    for line in (exp(146) / "labels.sh").read_text().splitlines():
        if "=" in line and not line.startswith("#") and "\"" not in line.split("=")[0]:
            k, v = line.split("=", 1)
            lab146[k.strip()] = v.split("#")[0].strip()
    for sc, x in d146["labels"].items():
        for fid, f in FORMATS.items():
            if f[0] == lab146.get(sc):
                bytes_pct.setdefault((fid, "Tiny"), x["vs_REF_pct"])
    d148 = json.loads((exp(148) / "sm148-bytes.json").read_text())
    for s, x in d148["sizes"].items():
        for fid, k in LR_BYTES_KEY.items():
            bytes_pct[(fid, s)] = x[k]["vs_int8_pct"]
    r150 = json.loads((exp(150) / "sm150-result.json").read_text())
    for s, x in r150["report_only"]["bytes"].items():
        bytes_pct[("R12", s)] = x["R12"]["vs_int8_pct"]
        assert abs(x["R1"]["vs_int8_pct"] - bytes_pct[("R1", s)]) < 0.01
    b131 = json.loads((exp(131) / "sm131-bytes.json").read_text())["arms"]
    for s, arm in (("Big", "Big (fork)"), ("Small", "Small (w64)")):
        co = b131[arm]["w8+s4hgx4c8hx8n"]["card"]["bytes"]
        c4 = per_bits[s]["card"]["4"]
        for fid, cb in (("C4", c4), ("CO4", co)):
            bytes_pct[(fid, s)] = round((nc * cb + nn * sz_ref[s][1]) / tot_ref[s] * 100 - 100, 2)
    bytes_pct[("REF", "Big")] = bytes_pct[("REF", "Small")] = bytes_pct[("REF", "Tiny")] = 0.0
    # sanity: the calc path reproduces SM141's scheme A within 0.05 (A switches to int8 at 128 answers)
    assert abs(bytes_pct[("C4", "Big")] - bytes_pct[("A", "Big")]) < 0.1, (bytes_pct[("C4", "Big")], bytes_pct[("A", "Big")])
    # rows by (model, format) -------------------------------------------------------------------------------------
    rows, srcs, dup = {}, {}, {}
    for m, (tag, *_rest) in MODELS.items():
        for fid, (var, desc, srcexp, bsrc) in FORMATS.items():
            if fid != "REF" and (fid, MODELS[m][3]) not in bytes_pct:
                continue
            for n, rr in raw.items():
                r = rr.get((tag, var))
                if not r:
                    continue
                if (m, fid) not in rows:
                    rows[(m, fid)], srcs[(m, fid)] = r, n
                else:   # same label scored again by a later experiment: max |delta| of the per-user LogLoss
                    cm = set(r) & set(rows[(m, fid)])
                    dup[(m, fid, n)] = max((abs(r[u][h]["LogLoss"] - rows[(m, fid)][u][h]["LogLoss"]) for u in cm for h in ("ahead", "instant") if h in r[u]), default=0.0)
    # one common user set: users with a finite row in every row set used --------------------------------------------
    users = None
    for k, r in rows.items():
        ok = {u for u, b in r.items() if all(math.isfinite(b[h]["LogLoss"]) for h in ("ahead", "instant") if h in b)}
        users = ok if users is None else users & ok
    users = sorted(users)
    L = {k: {u: loss(r[u]) for u in users} for k, r in rows.items()}
    # costs ---------------------------------------------------------------------------------------------------------
    out_models = {}
    for m, (tag, name, heads, size, bigm) in MODELS.items():
        fm = {}
        for fid in FORMATS:
            if (m, fid) not in L:
                continue
            own = compare(L[(m, fid)], L[(m, "REF")], users) if fid != "REF" else {h: {"value": 0.0, "lo": 0.0, "up": 0.0} for h in ("ahead", "instant")}
            vb = compare(L[(m, fid)], L[(bigm, "REF")], users) if bigm else own
            fm[fid] = {"variant": FORMATS[fid][0], "description": FORMATS[fid][1], "source": FORMATS[fid][2] if fid != "REF" else "SM141",
                       "rows_from": srcs[(m, fid)], "bytes_vs_int8_pct": bytes_pct[(fid, size)],
                       "vs_own_int8": {h: own[KEY[h]] for h in heads},
                       "vs_budget_ref": {h: vb[KEY[h]] for h in heads}}
        out_models[m] = {"name": name, "size": size, "heads": list(heads), "budget_ref": "its own int8" if not bigm else MODELS[bigm][1] + " at int8",
                         "budget_pct": BUDGET[size], "budget_text": BUDGET_TEXT[size], "formats": fm}
    # hulls ---------------------------------------------------------------------------------------------------------
    for m, mm in out_models.items():
        mm["hull"] = {}
        for h in mm["heads"]:
            for metric, key in (("upper", "up"), ("point", "value")):
                pts = [(100.0 + f["bytes_vs_int8_pct"], f["vs_budget_ref"][h][key], fid) for fid, f in mm["formats"].items()]
                hull = lower_hull(pts)
                ids = [p[2] for p in hull]
                segs = []
                for a, b in zip(hull[::-1][1:], hull[::-1][:-1]):   # from the int8 side toward fewer bytes: a is the cheaper-bytes vertex
                    pass
                hr = hull[::-1]   # largest bytes first
                for p, q in zip(hr[:-1], hr[1:]):
                    saved = p[0] - q[0]
                    segs.append({"from": p[2], "to": q[2], "bytes_saved_pct_of_int8": round(saved, 2),
                                 "cost_added_pct": round(q[1] - p[1], 3),
                                 "cost_per_saved_pct": round((q[1] - p[1]) / saved, 4) if saved else None})
                pareto = {p[2] for p in pts if not pareto_dominated(p, pts)}
                mm["hull"][f"{h}_{metric}"] = {"frontier": [p[2] for p in hr], "segments": segs,
                                               "pareto_dominated": sorted(p[2] for p in pts if p[2] not in pareto),
                                               "inside_hull_only": sorted(pareto - set(ids))}
        # smallest format passing the budget on every head (CI upper bound), and per head
        passing = [fid for fid, f in mm["formats"].items() if all(f["vs_budget_ref"][h]["up"] <= mm["budget_pct"] for h in mm["heads"])]
        mm["passing_budget"] = sorted(passing, key=lambda i: mm["formats"][i]["bytes_vs_int8_pct"])
        mm["smallest_passing"] = mm["passing_budget"][0] if passing else None
        mm["budget_crossing_on_upper_frontier"] = {}
        for h in mm["heads"]:
            fr = mm["hull"][f"{h}_upper"]["frontier"]   # largest bytes first
            xy = [(100.0 + mm["formats"][i]["bytes_vs_int8_pct"], mm["formats"][i]["vs_budget_ref"][h]["up"]) for i in fr]
            cross = None
            for (x0, y0), (x1, y1) in zip(xy[:-1], xy[1:]):
                if y0 <= mm["budget_pct"] < y1:
                    cross = round(x0 + (mm["budget_pct"] - y0) * (x1 - x0) / (y1 - y0) - 100, 2)
            mm["budget_crossing_on_upper_frontier"][h] = cross
    # reproduction check against the published numbers --------------------------------------------------------------
    ver = json.loads((exp(148) / "sm148-verdict-1123.json").read_text())["table"]
    chk = []
    for r in ver:
        m = next(k for k, v in MODELS.items() if v[1] == r["model"])
        if r["label"] in out_models[m]["formats"]:
            mine = out_models[m]["formats"][r["label"]]["vs_budget_ref"][r["head"]]
            chk.append({"model": r["model"], "format": r["label"], "head": r["head"], "published_users": r["users"],
                        "published": [r["value"], r["up"]], "here": [mine["value"], mine["up"]]})
    res = {"note": "aggregates only; costs in % of the reference LogLoss (review-weighted, paired user bootstrap, 95% CI); bytes are "
                   "state bytes per answered card and note summed over the 732 set vs int8 rows (%, negative = smaller)",
           "users": len(users), "excluded": EXCLUDED, "models": out_models,
           "duplicate_label_check_max_abs_logloss_diff": {f"{k[0]}/{k[1]}/{k[2]}": v for k, v in dup.items()},
           "reproduction_vs_published_SM148_verdict": chk}
    (HERE / "sm154-tables.json").write_text(json.dumps(res, indent=1))
    (HERE / "sm154-tables.md").write_text(to_md(res))
    print("users", len(users), "dup max", max(dup.values()) if dup else None)
    for m, mm in out_models.items():
        print(mm["name"], "smallest passing:", mm["smallest_passing"], mm["hull"][f"{mm['heads'][-1]}_upper"]["frontier"])


if __name__ == "__main__":
    main()
