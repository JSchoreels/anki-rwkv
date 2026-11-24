#!/usr/bin/env python
"""Output-aware low-rank truncation (SM157): the pre-registered pointer rule (design.md "Rule", fixed before any probe
row was scored) and the read-out. CPU only.

  python sm157_report.py report     cpu/sm157-rows.jsonl + sm157-stats*.json + check-cpu-sm157.json
                                    -> sm157-result.{json,md}; exit 0 = decided, 2 = rows missing
  python sm157_report.py selftest   synthetic rows

Statistic: cost of a label = 100 x (sum over users of its benchmark LogLoss x size / the same sum for REF - 1), the
review-weighted % change vs int8 rows (REF, w8+s8hx8), on the 8 probe users of the low-rank structure probe (SM147);
paired user bootstrap 2,000 draws, seed 0, 95% CI (the ratio-of-sums bootstrap of the rank-1 int8 states test (SM150)).
Cells: (model, head), two heads (curve, instant) for the two-head models, instant only for the Instant-Only models: 9
cells. Rule per weight kind v (r, rk, g), on Q2 (rank 2 everywhere):
  reduction(cell) = 1 - cost(Q2 weighted v) / cost(Q2 plain), only for cells whose plain cost is >= 0.01% (informative)
  PASS(v) = at least 5 informative cells, median reduction over them >= 1/3, and no cell of the 9 worse: the direct
            paired % of Q2 weighted v vs Q2 plain is never above +0.01%
  pointer passes = PASS for at least one v; the GPU follow-up then scores the passing v with the largest median
  reduction (ties: r, then rk, then g). Fewer than 5 informative cells: inconclusive (no GPU run).
T24 (the shipped format) is reported with the same numbers, outside the rule.
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
KEY = {"curve": "ahead", "instant": "instant"}
TWO, IO = ("curve", "instant"), ("instant",)
MODELS = {   # key: (tag, plain name, heads)
    "sm91s2": ("sm91s2-big-showtime", "Big two-head (SM91s2)", TWO),
    "sm99": ("sm99-big-inst-e2", "Big Instant-Only (SM99)", IO),
    "sm118": ("sm118-e26-dec2ep32k-t91s2", "Small two-head (SM118)", TWO),
    "sm113": ("sm113-inst-e35st-st-dec2ep32k-t99", "Small Instant-Only (SM113)", IO),
    "sm130th": ("sm130-th-dec2ep32k-t91s2", "six-layer Tiny two-head (SM130)", TWO),
    "sm130io": ("sm130-io-dec2ep32k-t99", "six-layer Tiny Instant-Only (SM130)", IO),
}
LABELS = {"REF": "w8+s8hx8", "Q2": "w8+lr2q8x8", "T24": "w8+lr2.16.4q8x8",
          "Q2R": "w8+lr2q8x8wr", "Q2RK": "w8+lr2q8x8wrk", "Q2G": "w8+lr2q8x8wg",
          "T24R": "w8+lr2.16.4q8x8wr", "T24RK": "w8+lr2.16.4q8x8wrk", "T24G": "w8+lr2.16.4q8x8wg"}
KINDS = ("r", "rk", "g")
KIND_NAME = {"r": "receptance (a)", "rk": "receptance + removal key (b)", "g": "discounted Gramian (c)"}
WN = {"Q2": {"r": "Q2R", "rk": "Q2RK", "g": "Q2G"}, "T24": {"r": "T24R", "rk": "T24RK", "g": "T24G"}}
FLOOR, WORSE, MIN_CELLS, CUT = 0.01, 0.01, 5, 1 / 3
PROBE_USERS = [2388, 4676, 1866, 2883, 3882, 1215, 4407, 966]


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


def cmp(new, ref, h, users, draws=2000):
    a = np.array([new[u]["benchmark"][h]["LogLoss"] * new[u]["benchmark"][h]["size"] for u in users])
    b = np.array([ref[u]["benchmark"][h]["LogLoss"] * ref[u]["benchmark"][h]["size"] for u in users])
    idx = np.random.default_rng(0).integers(0, len(users), size=(draws, len(users)))
    boot = (a[idx].sum(1) / b[idx].sum(1) - 1) * 100
    return {"value": float((a.sum() / b.sum() - 1) * 100),
            "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], "users": len(users)}


def finite(r):
    return "nonfinite" not in r and all(math.isfinite(r["benchmark"][h]["LogLoss"]) for h in ("ahead", "instant")
                                        if isinstance(r["benchmark"].get(h), dict))


def decide(rows):
    """{cells, rule, verdict}; raises KeyError-free: missing rows -> 'missing' list."""
    cells, missing = {}, []
    for mk, (tag, name, heads) in MODELS.items():
        got = {n: rows.get((tag, lab), {}) for n, lab in LABELS.items()}
        users = [u for u in PROBE_USERS if all(u in got[n] for n in LABELS)]
        if len(users) != len(PROBE_USERS) or not all(finite(got[n][u]) for n in LABELS for u in users):
            missing.append([mk, sorted(set(PROBE_USERS) - set(users)),
                            [n for n in LABELS if any(u in got[n] and not finite(got[n][u]) for u in PROBE_USERS)]])
            continue
        for h in heads:
            c = {n: cmp(got[n], got["REF"], KEY[h], users) for n in LABELS if n != "REF"}
            vs_plain = {f"{fmt}{k}": cmp(got[WN[fmt][k]], got[fmt], KEY[h], users) for fmt in WN for k in KINDS}
            cells[f"{mk}|{h}"] = {"model": mk, "name": name, "head": h, "cost": c, "vs_plain": vs_plain,
                                  "reviews": int(sum(got["REF"][u]["benchmark"][KEY[h]]["size"] for u in users))}
    rule = {}
    for fmt in ("Q2", "T24"):
        for k in KINDS:
            red, worse, info = {}, [], 0
            for key, c in cells.items():
                p, w = c["cost"][fmt]["value"], c["cost"][WN[fmt][k]]["value"]
                if p >= FLOOR:
                    red[key] = 1 - w / p
                if c["vs_plain"][f"{fmt}{k}"]["value"] > WORSE:
                    worse.append(key)
            med = statistics.median(red.values()) if red else None
            ok = len(red) >= MIN_CELLS and med is not None and med >= CUT and not worse
            rule[f"{fmt}|{k}"] = {"informative_cells": len(red), "reductions": red, "median_reduction": med,
                                  "worse_cells": worse, "pass": bool(ok), "in_rule": fmt == "Q2"}
    if missing:
        verdict = {"decided": False, "missing": missing}
    else:
        inform = rule["Q2|r"]["informative_cells"]
        passing = [k for k in KINDS if rule[f"Q2|{k}"]["pass"]]
        best = max(passing, key=lambda k: (rule[f"Q2|{k}"]["median_reduction"], -KINDS.index(k))) if passing else None
        verdict = {"decided": True, "inconclusive": inform < MIN_CELLS, "pass": bool(passing), "passing_kinds": passing,
                   "gpu_kind": best, "gpu_recommended": bool(passing)}
    return {"cells": cells, "rule": rule, "verdict": verdict}


def stats_info():
    """M bytes per model (float16, full and symmetric storage), anisotropy and split-half stability."""
    sys.path.insert(0, str(ROOT))
    import torch
    from scripts import state_lowrank as s
    full = json.loads((HERE / "sm157-stats.json").read_text())
    halves = [json.loads((HERE / f"sm157-stats-{p}.json").read_text()) for p in ("a", "b")
              if (HERE / f"sm157-stats-{p}.json").exists()]
    out = {}
    for mk, (tag, name, _) in MODELS.items():
        layers = full["models"][tag]
        b_full = b_sym = 0
        aniso, stab, top2 = {k: [] for k in KINDS}, {k: [] for k in KINDS}, {k: [] for k in KINDS}
        for key, st in layers.items():
            H, K = len(st["rr"]), len(st["rr"][0])
            b_full += H * K * K * 2
            b_sym += H * K * (K + 1) // 2 * 2
            for k in KINDS:
                M = s.weight_matrix(st, k)
                lam = torch.linalg.eigvalsh(M).flip(-1)
                aniso[k] += (lam[:, :2].sum(-1) / lam.sum(-1)).tolist()
                if len(halves) == 2:
                    Ma = s.weight_matrix(halves[0]["models"][tag][key], k)
                    Mb = s.weight_matrix(halves[1]["models"][tag][key], k)
                    stab[k] += ((Ma - Mb).flatten(1).norm(dim=-1) / M.flatten(1).norm(dim=-1)).tolist()
                    va = torch.linalg.eigh(Ma)[1][..., -2:]
                    vb = torch.linalg.eigh(Mb)[1][..., -2:]
                    top2[k] += ((va.mT @ vb).pow(2).sum((-1, -2)) / 2).tolist()   # mean squared cosine, top-2 subspaces
        out[mk] = {"name": name, "layers": len(layers), "M_bytes_f16_full": b_full, "M_bytes_f16_symmetric": b_sym,
                   "top2_eigen_share_mean": {k: float(np.mean(v)) for k, v in aniso.items()},
                   "split_half_rel_diff_median": {k: float(np.median(v)) for k, v in stab.items() if v},
                   "split_half_rel_diff_max": {k: float(np.max(v)) for k, v in stab.items() if v},
                   "split_half_top2_overlap_min": {k: float(np.min(v)) for k, v in top2.items() if v},
                   "fit": full["provenance"].get(tag)}
    return out


def cpu_minutes(rows):
    sec = {}
    for (tag, lab), us in rows.items():
        sec[lab] = sec.get(lab, 0.0) + sum(r.get("model_seconds", 0) + r.get("prepare_seconds", 0) for r in us.values())
    speed = {}
    for mk, (tag, _, _) in MODELS.items():
        for n, lab in LABELS.items():
            us = rows.get((tag, lab), {})
            s_ = sum(r.get("model_seconds", 0) for r in us.values())
            rv = sum(r.get("reviews", 0) for r in us.values())
            speed[f"{mk} {n}"] = round(rv / s_) if s_ else None
    return {"probe_scoring_minutes": round(sum(sec.values()) / 60, 1), "review_evals_per_second": speed}


def pct(x):
    return "n/a" if x is None else f"{x:+.3f}%"


def write_md(res):
    L = ["# Output-aware low-rank truncation (SM157): pointer result", "",
         "Generated by `sm157_report.py report` from `cpu/sm157-rows.jsonl` (8 probe users of the low-rank structure "
         "probe (SM147), CPU, show-time inputs, leak-free). Cost = % change of the review-weighted benchmark LogLoss "
         "against int8 rows (REF, `w8+s8hx8`); 95% paired user bootstrap in sm157-result.json.", ""]
    v = res["verdict"]
    if not v.get("decided"):
        L += ["**Not decided: rows missing** " + json.dumps(v.get("missing")), ""]
    else:
        L += [f"**Verdict: {'PASS' if v['pass'] else ('INCONCLUSIVE' if v['inconclusive'] else 'NO GO')}.** "
              + (f"Passing weight kinds: {', '.join(KIND_NAME[k] for k in v['passing_kinds'])}; the GPU follow-up "
                 f"would score {KIND_NAME[v['gpu_kind']]}." if v["pass"] else "No weight kind meets the rule, so the "
                 "GPU follow-up is not recommended."), ""]
    L += ["## Cost vs int8 rows per model and head", "",
          "| Model | Head | Q2 plain | Q2 r | Q2 rk | Q2 g | T24 plain | T24 r | T24 rk | T24 g |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for c in res["cells"].values():
        L.append(f"| {c['name']} | {c['head']} | " + " | ".join(pct(c["cost"][n]["value"]) for n in
                 ("Q2", "Q2R", "Q2RK", "Q2G", "T24", "T24R", "T24RK", "T24G")) + " |")
    L += ["", "## Rule (Q2 decides; T24 for information)", "",
          "| Format | Weight | Informative cells | Median reduction | Worse cells | Pass |", "|---|---|---|---|---|---|"]
    for key, r in res["rule"].items():
        fmt, k = key.split("|")
        med = "n/a" if r["median_reduction"] is None else f"{100 * r['median_reduction']:.0f}%"
        L.append(f"| {fmt} | {KIND_NAME[k]} | {r['informative_cells']} | {med} | {', '.join(r['worse_cells']) or '-'} | "
                 f"{'yes' if r['pass'] else 'no'}{'' if r['in_rule'] else ' (info)'} |")
    if "stats" in res:
        L += ["", "## M per model", "",
              "| Model | Layers | M bytes (float16, full) | M bytes (float16, symmetric) | Top-2 share r / rk / g | "
              "Split-half rel. diff (median) r / rk / g |", "|---|---|---|---|---|---|"]
        for s in res["stats"].values():
            L.append(f"| {s['name']} | {s['layers']} | {s['M_bytes_f16_full']:,} | {s['M_bytes_f16_symmetric']:,} | "
                     + " / ".join(f"{s['top2_eigen_share_mean'][k]:.2f}" for k in KINDS) + " | "
                     + " / ".join(f"{s['split_half_rel_diff_median'].get(k, float('nan')):.3f}" for k in KINDS) + " |")
    if "cpu" in res:
        L += ["", f"Probe scoring CPU time (model + preparation, all processes): {res['cpu']['probe_scoring_minutes']} min."]
    (HERE / "sm157-result.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def report():
    path = HERE / "cpu" / "sm157-rows.jsonl"
    if not path.exists():
        print("rows missing")
        return 2
    rows = read_rows(path)
    res = decide(rows)
    res["cpu"] = cpu_minutes(rows)
    try:
        res["stats"] = stats_info()
    except FileNotFoundError as e:
        res["stats_error"] = repr(e)
    chk = HERE / "check-cpu-sm157.json"
    if chk.exists():
        res["codec_bench"] = json.loads(chk.read_text())["info"].get("bench")
    (HERE / "sm157-result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    write_md(res)
    print(json.dumps(res["verdict"]))
    return 0 if res["verdict"].get("decided") else 2


def selftest():
    def row(u, tag, lab, k):
        base = {h: 0.3 + 0.01 * ((u * 7 + i) % 5) for i, h in enumerate(("ahead", "instant"))}
        return {"user_id": u, "tag": tag, "variant": lab + "@e2s-query",
                "benchmark": {h: {"LogLoss": base[h] * k, "size": 1000 + u % 7} for h in base}}

    def build(mult):
        rows = {}
        for mk, (tag, _, _) in MODELS.items():
            for n, lab in LABELS.items():
                rows[(tag, lab)] = {u: row(u, tag, lab, mult.get(n, 1.0)) for u in PROBE_USERS}
        return rows
    base = {"Q2": 1.001, "T24": 1.0005}
    r = decide(build({**base, "Q2R": 1.0006, "Q2RK": 1.0008, "Q2G": 1.0013, "T24R": 1.0003}))
    assert r["verdict"]["pass"] and r["verdict"]["gpu_kind"] == "r", r["verdict"]
    assert abs(r["rule"]["Q2|r"]["median_reduction"] - 0.4) < 1e-6 and not r["rule"]["Q2|rk"]["pass"]
    assert r["rule"]["Q2|g"]["worse_cells"] and not r["rule"]["Q2|g"]["pass"]
    r = decide(build({**base, "Q2R": 1.0008, "Q2RK": 1.0009, "Q2G": 1.001}))
    assert not r["verdict"]["pass"]
    r = decide(build({"Q2": 1.00005, "Q2R": 1.0}))
    assert r["verdict"]["inconclusive"] and not r["verdict"]["pass"]
    rows = build(base)
    del rows[("sm99-big-inst-e2", LABELS["Q2G"])][966]
    assert not decide(rows)["verdict"]["decided"]
    print("sm157_report selftest OK")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("report", "selftest"))
    a = p.parse_args()
    sys.exit(report() if a.mode == "report" else selftest())


if __name__ == "__main__":
    main()
