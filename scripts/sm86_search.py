#!/usr/bin/env python
"""Wider simulator search with time-aware scheduling (SM86): an extension of response-time prediction (SM82) stage 2
(CPU only).

1. The stage 2 30-minute-limit search (stage 2's simulator: `sim` below is `sm82s2_analyze.simulate` with the same
   random draws, checked bit for bit, plus an optional new-card cap and a per-card target) with desired retention
   0.30-0.98 by 0.02 instead of 0.60-0.98, with an unlimited deck (stage 2) and with at most 10 new cards a day:
   optimum distribution per cost model (A (actual), A (head), B, C) and the cost of keeping the fixed-cost (A (head))
   optimum when B or C is the truth.
2. One simple time-aware per-card policy: after every review (and at learning) the card's next target recall is the
   one with the most expected stability gain per predicted second,
       r*(S, D) = argmax_r  G(r; S, D) / cost(r),   G = r E_pass[ln S'/S] + (1 - r) ln S_fail/S  (FSRS-6, user's fit),
   cost(r) = r E_pass[cost | r] + (1 - r) (cost_Again(r) + relearning step), from a cost version P; the target is then
   lowered by one per-user offset delta (-0.20 to 0.56, tuned per user like the single desired retention), clipped to
   0.30-0.98. Time-aware = P is the truth's own recall-dependent costs (B, or C with the grade mix over recall);
   time-unaware control = the same policy built with fixed per-grade costs (A (head)). Evaluated under the same
   30-minute daily limit and at most 10 new cards a day (with an unlimited deck the best single target sits at the
   0.30 bound), charged with the truth's costs: memorized cards (sum of recall) at day 365.

  run  --curves DIR --out-dir artifacts/small-model/sm84 --procs 6
"""
from argparse import ArgumentParser
import json
from multiprocessing import Pool
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sm82s2_analyze import (BUDGET_S, VARIANTS, cost_model, fsrs_w, load_user, simulate, user_fits, _low,  # noqa: E402
                                    md_table, fmt)

WIDE_DRS = [round(0.30 + 0.02 * i, 2) for i in range(35)]     # 0.30 .. 0.98
DELTAS = [round(-0.20 + 0.04 * i, 2) for i in range(20)]     # -0.20 .. 0.56 (target = r* - delta)
NEW_CAP = 10                                                   # new cards a day in the capped setup
SETUPS = {"unlimited": None, "cap10": NEW_CAP}
RG = np.round(np.arange(0.30, 0.9801, 0.01), 2)
LNS = np.linspace(np.log(0.01), np.log(1e5), 121)
DG = np.linspace(1, 10, 37)
PAIRS = [("B_recall_cost", "A"), ("B_recall_cost", "B"), ("C_recall_cost_and_mix", "A"), ("C_recall_cost_and_mix", "C")]
VNAME = {"A_fixed_actual": "A (actual)", "A_fixed_head": "A (head)", "B_recall_cost": "B", "C_recall_cost_and_mix": "C"}


# ------------------------------------------------------------------ the time-aware policy table
def policy_table(cm, w, version):
    """r*(ln S, D) on the LNS x DG grid: argmax over RG of expected ln-stability gain / expected predicted cost."""
    S = np.exp(LNS)[:, None, None]; D = DG[None, :, None]; r = RG[None, None, :]
    xs, cs, mr = cm["x"], cm["c"], cm["mix_r"]
    if version == "C" and len(xs):
        mix = np.stack([np.interp(RG, xs, mr[:, j]) for j in range(3)], -1); mix /= mix.sum(-1, keepdims=True)
    else:
        mix = np.broadcast_to(cm["mix"], (len(RG), 3))
    if version in ("B", "C") and len(xs):
        cg = np.stack([np.interp(RG, xs, cs[:, j]) for j in range(4)], -1)
    else:
        cg = np.broadcast_to(cm["fixed_head"], (len(RG), 4))
    cost = RG * (mix * cg[:, 1:]).sum(-1) + (1 - RG) * (cg[:, 0] + cm["relearn"])
    base = np.exp(w[8]) * (11 - D) * S ** (-w[9]) * (np.exp((1 - r) * w[10]) - 1)
    gpass = (mix[None, None, :, 0] * np.log1p(base * w[15]) + mix[None, None, :, 1] * np.log1p(base)
             + mix[None, None, :, 2] * np.log1p(base * w[16]))
    sf = np.minimum(w[11] * D ** (-w[12]) * ((S + 1) ** w[13] - 1) * np.exp((1 - r) * w[14]), S / np.exp(w[17] * w[18]))
    sf = sf * np.maximum(np.exp(w[17] * w[18]) * np.maximum(sf, 1e-3) ** (-w[19]), 1)
    G = r * gpass + (1 - r) * np.log(np.maximum(sf, 0.001) / S)
    ratio = G / cost[None, None, :]
    return RG[np.argmax(ratio, -1)]


def lookup(tab, s, d):
    i = np.clip(np.rint((np.log(np.maximum(s, 1e-6)) - LNS[0]) / (LNS[1] - LNS[0])).astype(np.int64), 0, len(LNS) - 1)
    j = np.clip(np.rint((d - DG[0]) / (DG[1] - DG[0])).astype(np.int64), 0, len(DG) - 1)
    return tab[i, j]


def sim(cm, variant, policy, new_cap=None, seed=0, days=365, w=None):
    """Stage 2's budget simulation (`sm82s2_analyze.simulate` with budget_variant; bit-for-bit the same draws): a
    30-minute daily limit charged with `variant`'s costs, due reviews first, then new cards while time is left, at most
    `new_cap` new cards a day (None = unlimited deck, stage 2). policy = ("dr", r) one desired retention, or
    ("tab", table, delta) the per-card target clip(r*(S, D) - delta, 0.30, 0.98)."""
    w = cm["w"] if w is None else w
    r_mix = variant == "C_recall_cost_and_mix"
    decay = -w[20]; factor = 0.9 ** (1 / decay) - 1
    rng = np.random.default_rng(seed)
    n = int(days * BUDGET_S / max(cm["learn"], 1.0)) + 1
    if new_cap is not None:
        n = min(n, days * new_cap)
    S = np.zeros(n); D = np.zeros(n); last = np.zeros(n); due = np.full(n, np.inf)
    learned = 0; tsum = 0.0; tn = 0; reviews = 0
    if policy[0] == "dr":
        target = lambda s, d: np.full(np.shape(s), policy[1])
    else:
        target = lambda s, d: np.clip(lookup(policy[1], s, d) - policy[2], 0.30, 0.98)
    ivl = lambda s, r: np.maximum(1, np.round(s / factor * (r ** (1 / decay) - 1)))
    init_d = lambda g: w[4] - np.exp(w[5] * (g - 1)) + 1

    def next_d(d, g):
        nd = d + (-w[6] * (g - 3)) * (10 - d) / 9
        nd = w[7] * init_d(4) + (1 - w[7]) * nd
        return np.clip(nd, 1, 10)

    xs, cs, mr = cm["x"], cm["c"], cm["mix_r"]
    use_r = len(xs) > 0
    for day in range(days):
        left = BUDGET_S
        idx = np.nonzero(due[:learned] <= day)[0]
        if idx.size:
            idx = idx[np.argsort(due[idx], kind="stable")]
            s, d = S[idx], D[idx]
            R = (1 + factor * (day - last[idx]) / s) ** decay
            u = rng.random(idx.size); v = rng.random(idx.size)
            forget = u > R
            if r_mix and use_r:
                m = np.stack([np.interp(R, xs, mr[:, j]) for j in range(3)], -1); m /= m.sum(-1, keepdims=True)
            else:
                m = np.broadcast_to(cm["mix"], (idx.size, 3))
            cum = np.cumsum(m, -1)
            g = np.where(forget, 1, 2 + (v[:, None] > cum[:, :2]).sum(-1))
            cg = np.stack([np.interp(R, xs, cs[:, j]) for j in range(4)], -1) if use_r else np.broadcast_to(cm["fixed_head"], (idx.size, 4))
            c_r = cg[np.arange(idx.size), g - 1]
            c_v = {"A_fixed_actual": cm["fixed_actual"][g - 1], "A_fixed_head": cm["fixed_head"][g - 1]}.get(variant, c_r)
            c_v = c_v + np.where(forget, cm["relearn"], 0.0)
            keep = np.cumsum(c_v) <= left
            left -= float(c_v[keep].sum())
            idx, s, d, R, forget, g = idx[keep], s[keep], d[keep], R[keep], forget[keep], g[keep]
            reviews += idx.size
            hard = np.where(g == 2, w[15], 1.0); easy = np.where(g == 4, w[16], 1.0)
            s_ok = s * (1 + np.exp(w[8]) * (11 - d) * s ** (-w[9]) * (np.exp((1 - R) * w[10]) - 1) * hard * easy)
            s_fail = np.minimum(w[11] * d ** (-w[12]) * ((s + 1) ** w[13] - 1) * np.exp((1 - R) * w[14]), s / np.exp(w[17] * w[18]))
            s_fail = s_fail * np.maximum(np.exp(w[17] * w[18]) * np.maximum(s_fail, 1e-3) ** (-w[19]), 1)
            nd = next_d(d, g)
            nd = np.where(forget, next_d(nd, 3), nd)
            ns = np.maximum(np.where(forget, s_fail, s_ok), 0.001)
            S[idx], D[idx], last[idx] = ns, nd, day
            t = target(ns, nd); tsum += float(t.sum()); tn += t.size
            due[idx] = day + ivl(ns, t)
        k = min(int(max(left, 0) // cm["learn"]), n - learned)
        if new_cap is not None:
            k = min(k, new_cap)
        if k > 0:
            new = np.arange(learned, learned + k)
            g0 = rng.choice(4, k, p=cm["first_probs"]) + 1
            S[new] = w[g0 - 1]; D[new] = np.clip(init_d(g0), 1, 10); last[new] = day
            t = target(S[new], D[new]); tsum += float(t.sum()); tn += t.size
            due[new] = day + ivl(S[new], t)
            learned += k
    R_end = (1 + factor * (days - last[:learned]) / S[:learned]) ** decay
    return {"memorized": float(R_end.sum()), "learned": learned, "reviews": reviews, "mean_target": tsum / max(tn, 1)}


def _user(args):
    path, data, w_fit = args
    _low()
    z = load_user(path, data)
    cm = cost_model(z)
    cm["w"] = w_fit
    w = w_fit if w_fit is not None else fsrs_w()
    out = {"user_id": int(z["user"]), "w_source": "user FSRS-6 fit" if w_fit is not None else "FSRS-6 default",
           "cost_x_min": float(cm["x"][0]) if len(cm["x"]) else None, "learn_s": cm["learn"]}
    for setup, cap in SETUPS.items():
        budget = []
        for dr in WIDE_DRS:
            rec = {"dr": dr}
            for v in VARIANTS:
                b = sim(cm, v, ("dr", dr), cap, w=w)
                rec[v] = b["memorized"]; rec[f"{v}_learned"] = b["learned"]; rec[f"{v}_reviews"] = b["reviews"]
            budget.append(rec)
        out[f"budget_{setup}"] = budget
    # bit-for-bit check against stage 2's simulator function on a few points
    out["check_vs_stage2_sim"] = max(abs(sim(cm, v, ("dr", dr), None, w=w)["memorized"] - simulate(cm, dr, v == "C_recall_cost_and_mix", budget_variant=v)["memorized"])
                                     for dr in (0.4, 0.7, 0.9) for v in ("A_fixed_head", "C_recall_cost_and_mix"))
    tabs = {p_: policy_table(cm, w, p_) for p_ in ("A", "B", "C")}
    ta = {}
    for truth, p_ in PAIRS:
        ta[f"{truth}|{p_}"] = [{"delta": dl, **sim(cm, truth, ("tab", tabs[p_], dl), NEW_CAP, w=w)} for dl in DELTAS]
    out["time_aware"] = ta
    out["probe"] = {p_: {f"S{sv:g}": float(lookup(tabs[p_], np.array([sv]), np.array([5.0]))[0]) for sv in (1, 10, 100)} for p_ in tabs}
    return out


# ------------------------------------------------------------------ aggregate + report
def opt_block(users, setup):
    key = f"budget_{setup}"
    for u in users:
        for v in VARIANTS:
            y = np.array([x[v] for x in u[key]]); k = int(np.argmax(y))
            u[f"{setup}_opt_{v}"] = WIDE_DRS[k]; u[f"{setup}_max_{v}"] = float(y[k])
        for v in ("B_recall_cost", "C_recall_cost_and_mix"):
            y = np.array([x[v] for x in u[key]])
            u[f"{setup}_loss_{v}_at_A_opt"] = float(1 - y[WIDE_DRS.index(u[f"{setup}_opt_A_fixed_head"])] / y.max())
            u[f"{setup}_loss_{v}_at_0.90"] = float(1 - y[WIDE_DRS.index(0.9)] / y.max())
    blk = {"opt": {}, "shift": {}, "loss": {}}
    for v in VARIANTS:
        o = np.array([u[f"{setup}_opt_{v}"] for u in users])
        blk["opt"][v] = {"median": float(np.median(o)), "mean": float(o.mean()), "quartiles": [float(x) for x in np.percentile(o, [10, 25, 75, 90])],
                         "at_lower": float((o == WIDE_DRS[0]).mean()), "at_upper": float((o == WIDE_DRS[-1]).mean()),
                         "below_0_60": float((o < 0.60 - 1e-9).mean()), "max_median": float(np.median([u[f"{setup}_max_{v}"] for u in users])),
                         "hist": {f"{lo:.2f}-{lo + 0.04:.2f}": float(((o >= lo - 1e-9) & (o < lo + 0.06 - 1e-9)).mean()) for lo in np.round(np.arange(0.30, 0.99, 0.06), 2)}}
    A = np.array([u[f"{setup}_opt_A_fixed_head"] for u in users])
    for v in ("A_fixed_actual", "B_recall_cost", "C_recall_cost_and_mix"):
        o = np.array([u[f"{setup}_opt_{v}"] for u in users])
        blk["shift"][v] = {"mean": float((o - A).mean()), "median": float(np.median(o - A)), "up": float((o > A + 1e-9).mean()), "down": float((o < A - 1e-9).mean()),
                           "same": float((np.abs(o - A) < 1e-9).mean())}
    for v in ("B_recall_cost", "C_recall_cost_and_mix"):
        x = np.array([u[f"{setup}_loss_{v}_at_A_opt"] for u in users]); y = np.array([u[f"{setup}_loss_{v}_at_0.90"] for u in users])
        blk["loss"][v] = {"mean": float(x.mean()), "median": float(np.median(x)), "p90": float(np.percentile(x, 90)), "share_zero": float((x == 0).mean()),
                          "at_0_90_mean": float(y.mean()), "at_0_90_median": float(np.median(y))}
    blk["curve"] = [{"dr": dr, **{v: float(np.median([u[key][i][v] for u in users])) for v in VARIANTS},
                     **{f"{v}_learned": float(np.median([u[key][i][f"{v}_learned"] for u in users])) for v in ("A_fixed_head", "C_recall_cost_and_mix")},
                     "C_reviews": float(np.median([u[key][i]["C_recall_cost_and_mix_reviews"] for u in users]))}
                    for i, dr in enumerate(WIDE_DRS)]
    return blk


def aggregate(users, old):
    a = {"users": len(users), "setups": {s_: opt_block(users, s_) for s_ in SETUPS}}
    if old:
        diffs = []
        for u in users:
            ou = old.get(u["user_id"])
            if not ou:
                continue
            for rec in ou["budget"]:
                if rec["dr"] in WIDE_DRS:
                    nr = u["budget_unlimited"][WIDE_DRS.index(rec["dr"])]
                    diffs += [abs(nr[v] - rec[v]) / max(abs(rec[v]), 1e-9) for v in VARIANTS]
        a["stage2_reproduction_max_rel_diff"] = float(max(diffs)) if diffs else None
    a["check_vs_stage2_sim_max_abs"] = float(max(u["check_vs_stage2_sim"] for u in users))
    ta = {}
    for truth, p_ in PAIRS:
        key = f"{truth}|{p_}"
        best, at0, dl, tgt, g_single, g_aopt, g_090 = [], [], [], [], [], [], []
        for u in users:
            ys = np.array([x["memorized"] for x in u["time_aware"][key]]); k = int(np.argmax(ys))
            best.append(ys[k]); at0.append(ys[DELTAS.index(0.0)]); dl.append(DELTAS[k]); tgt.append(u["time_aware"][key][k]["mean_target"])
            yb = np.array([x[truth] for x in u["budget_cap10"]])
            g_single.append(ys[k] / yb.max() - 1); g_aopt.append(ys[k] / yb[WIDE_DRS.index(u["cap10_opt_A_fixed_head"])] - 1)
            g_090.append(ys[k] / yb[WIDE_DRS.index(0.9)] - 1)
            u[f"ta_best_{key}"] = float(ys[k])
        ta[key] = {"truth": truth, "policy_costs": p_, "memorized_median": float(np.median(best)), "memorized_untuned_median": float(np.median(at0)),
                   "delta_median": float(np.median(dl)), "delta_at_bound": float(np.mean((np.array(dl) == DELTAS[0]) | (np.array(dl) == DELTAS[-1]))),
                   "mean_target_median": float(np.median(tgt)),
                   "vs_single_oracle_mean": float(np.mean(g_single)), "vs_single_oracle_median": float(np.median(g_single)),
                   "vs_single_oracle_share_better": float(np.mean(np.array(g_single) > 0)),
                   "vs_single_at_A_opt_mean": float(np.mean(g_aopt)), "vs_single_0_90_mean": float(np.mean(g_090))}
    for truth, pa, pt in (("B_recall_cost", "A", "B"), ("C_recall_cost_and_mix", "A", "C")):
        x = np.array([u[f"ta_best_{truth}|{pt}"] / u[f"ta_best_{truth}|{pa}"] - 1 for u in users])
        ta[f"{truth}|aware_vs_unaware"] = {"mean": float(x.mean()), "median": float(np.median(x)), "p10": float(np.percentile(x, 10)),
                                           "p90": float(np.percentile(x, 90)), "share_better": float((x > 1e-12).mean()), "share_worse": float((x < -1e-12).mean())}
    a["time_aware"] = ta
    a["probe"] = {p_: {k: float(np.median([u["probe"][p_][k] for u in users])) for k in users[0]["probe"][p_]} for p_ in ("A", "B", "C")}
    a["cost_x_min_median"] = float(np.median([u["cost_x_min"] for u in users if u["cost_x_min"] is not None]))
    a["w_sources"] = {s_: sum(u["w_source"] == s_ for u in users) for s_ in ("user FSRS-6 fit", "FSRS-6 default")}
    return a


def render_setup(blk, title, note):
    L = [title, "", note, ""]
    rows = [{"v": VNAME[v], **o, "q": f"{o['quartiles'][1]:.2f}-{o['quartiles'][2]:.2f}", "d": f"{o['quartiles'][0]:.2f}-{o['quartiles'][3]:.2f}"} for v, o in blk["opt"].items()]
    L += md_table(rows, ["v", "median", "mean", "q", "d", "at_lower", "at_upper", "below_0_60", "max_median"],
                  ["costs", "optimum, median", "mean", "quartiles", "10th-90th percentile", "users at 0.30", "users at 0.98", "users below 0.60 (outside stage 2's grid)", "memorized at day 365 at the optimum (median)"])
    L += ["", "Distribution of the optimum (share of users; bins of three grid points):", ""]
    bins = list(blk["opt"]["A_fixed_head"]["hist"])
    rows = [{"v": VNAME[v], **{b: o["hist"][b] for b in bins}} for v, o in blk["opt"].items()]
    L += md_table(rows, ["v"] + bins, ["costs"] + bins)
    L += [""]
    for v, s_ in blk["shift"].items():
        L += [f"- optimum {VNAME[v]} minus A (head): mean {s_['mean']:+.3f}, median {s_['median']:+.2f}; higher for {s_['up']:.0%} of users, lower {s_['down']:.0%}, same {s_['same']:.0%}."]
    for v, s_ in blk["loss"].items():
        L += [f"- if the costs are really {VNAME[v]}, keeping the fixed-cost (A (head)) optimum memorizes {s_['mean']:.2%} fewer cards on average (median {s_['median']:.2%}, 90th percentile {s_['p90']:.2%}; no loss for {s_['share_zero']:.0%} of users); desired retention 0.90 memorizes {s_['at_0_90_mean']:.1%} fewer than the optimum on average (median {s_['at_0_90_median']:.1%})."]
    L += ["", "Memorized cards at day 365 by desired retention (median over users):", ""]
    show = [c for c in blk["curve"] if round(c["dr"] * 100) % 6 == 0 or c["dr"] in (0.30, 0.98)]
    L += md_table(show, ["dr", "A_fixed_actual", "A_fixed_head", "B_recall_cost", "C_recall_cost_and_mix", "A_fixed_head_learned", "C_recall_cost_and_mix_learned", "C_reviews"],
                  ["desired retention", "A (actual)", "A (head)", "B", "C", "cards learned, A (head)", "cards learned, C", "reviews done, C"])
    return L + [""]


def render(r):
    a = r["agg"]
    rep = a.get("stage2_reproduction_max_rel_diff")
    L = ["# Wider simulator search with time-aware scheduling (SM86): the 30-minute limit with desired retention 0.30-0.98, and a per-card time-aware policy", ""]
    L += [f"Extension of response-time prediction (SM82) stage 2 (CPU only). Same {a['users']} users of the 732 set ({a['w_sources']['user FSRS-6 fit']} with their own FSRS-6 fit), same simulator and cost models (A (actual), A (head), B, C; see stage2-result.md section 3): daily limit 1,800 s charged with each cost model's own costs, due reviews first, then new cards while time is left, 365 days, objective = memorized cards (sum of recall) at day 365. Desired retention 0.30-0.98 by 0.02 (stage 2: 0.60-0.98). Stage 2 reproduced on the overlapping grid: max relative difference {fmt(rep, 8) if rep is not None else '—'} (and {fmt(a['check_vs_stage2_sim_max_abs'], 8)} against stage 2's simulator function). Run time {r['seconds']:.0f} s.", ""]
    L += ["## Takeaways", "", "TAKEAWAYS", ""]
    L += render_setup(a["setups"]["unlimited"], "## 1. Single desired retention, 30-minute limit, unlimited new cards (stage 2's setup)",
                      "With an unlimited deck, every second not spent on reviews learns a new card, and with the FSRS-6 power curve (the users' own fits) an unreviewed card keeps much of its recall within a year; so the sum of recall at day 365 rewards breadth.")
    L += render_setup(a["setups"]["cap10"], f"## 2. Single desired retention, 30-minute limit, at most {NEW_CAP} new cards a day",
                      f"Same limit, but at most {NEW_CAP} new cards a day ({NEW_CAP * 365:,} in the year, the deck of stage 2's no-limit runs): time left over is not spent, so high retention costs cards only when the reviews exceed 30 minutes.")
    L += [f"Extrapolation: the head's cost curve is binned over the curve-grid points (stage 2); below the lowest bin's mean recall (median over users {fmt(a['cost_x_min_median'])}) the cost and grade mix are held flat, and the lowest bins come mostly from the stretched-interval queries (up to 365 days), the least tested part of the counterfactual. Desired retention below about 0.6 is outside anything the users scheduled.", ""]
    L += [f"## 3. A time-aware per-card policy vs one desired retention ({NEW_CAP} new cards a day, 30-minute limit)", "",
          "Per card, after every review (and at learning): target recall r*(S, D) = the r (0.30-0.98) with the most expected FSRS-6 stability gain (in ln S) per predicted second (the review plus, for Again, the relearning step), then minus one per-user offset (-0.20 to 0.56 by 0.04) tuned on the evaluated simulation, as the single desired retention is: both are best cases. Time-aware: the policy's predicted costs are the truth's own recall-dependent costs (B, or C with the grade mix over recall). Time-unaware control: the same policy built with fixed per-grade costs (A (head)). Under B and C the predicted time depends on recall only, so cards differ only through their memory state (S, D); the time cost reshapes the target curve as a whole.", ""]
    pr = a["probe"]
    L += [f"Untuned target r* at difficulty 5 (median over users), S = 1 / 10 / 100 days: A (head) costs {pr['A']['S1']:.2f} / {pr['A']['S10']:.2f} / {pr['A']['S100']:.2f}; B {pr['B']['S1']:.2f} / {pr['B']['S10']:.2f} / {pr['B']['S100']:.2f}; C {pr['C']['S1']:.2f} / {pr['C']['S10']:.2f} / {pr['C']['S100']:.2f}.", ""]
    rows = []
    for key, t in a["time_aware"].items():
        if "|aware_vs_unaware" in key:
            continue
        rows.append({"truth": VNAME[t["truth"]], "p": "A (head) (time-unaware)" if t["policy_costs"] == "A" else t["policy_costs"] + " (time-aware)", **t})
    L += md_table(rows, ["truth", "p", "memorized_median", "memorized_untuned_median", "delta_median", "delta_at_bound", "mean_target_median", "vs_single_oracle_mean", "vs_single_oracle_median", "vs_single_oracle_share_better", "vs_single_at_A_opt_mean", "vs_single_0_90_mean"],
                  ["truth (charged costs)", "policy's predicted costs", "memorized at day 365, tuned offset (median)", "untuned (offset 0)", "offset, median", "users at an offset bound", "mean target recall", "vs the best single desired retention, mean", "median", "users better", "vs the single desired retention chosen with A (head), mean", "vs desired retention 0.90, mean"])
    L += [""]
    for truth in ("B_recall_cost", "C_recall_cost_and_mix"):
        s_ = a["time_aware"][f"{truth}|aware_vs_unaware"]
        L += [f"- truth {VNAME[truth]}: the time-aware policy vs the same policy with fixed costs (each with its tuned offset): mean {s_['mean']:+.2%}, median {s_['median']:+.2%} memorized cards (10th-90th percentile {s_['p10']:+.2%} to {s_['p90']:+.2%}; better for {s_['share_better']:.0%} of users, worse for {s_['share_worse']:.0%})."]
    L += ["", "## Limits", "",
          "- Same simulator as stage 2: the memory is the users' FSRS-6 fits (not the RWKV model), learning is a fixed cost per new card, one relearning step per lapse, 365 days, objective = sum of recall at day 365 (it rewards breadth: more cards at lower recall). The new-card cap of sections 2-3 is a modelling choice, not something the users set.",
          "- Costs: the answer-time head on richer frozen features (SM82c) as a per-user cost curve over recall (stage 2's B and C); below the lowest bin and at low desired retention the cost is extrapolated flat, and those bins come from the stretched-interval counterfactual queries. Desired retention 0.30-0.60 is outside the observed scheduling.",
          "- Policy: one hand-built heuristic (stability gain per predicted second + one tuned offset), not an optimal policy (no dynamic programming over the limit); the offset and the single desired retention are both tuned per user on the evaluated simulation (best cases for both). 150 users of the 732 set.", ""]
    return L


def main():
    ap = ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--curves", default="C:/Users/jonat/workspace/wt/sm82s2/artifacts/small-model/sm82/stage2/work/curves")
    ap.add_argument("--stage2-users", default=str(ROOT / "artifacts/small-model/sm82/stage2/stage2-workload-users.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts/small-model/sm84"))
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    from scripts import compress_common as cc
    _low()
    t0 = time.time()
    data = cc.data_path()
    files = sorted((Path(args.curves) / "users").glob("*.npz"))
    if args.limit:
        files = files[:args.limit]
    fits = user_fits([int(f.stem) for f in files])
    jobs = [(str(f), str(data), fits.get(int(f.stem))) for f in files]
    with Pool(min(args.procs, 6), initializer=_low) as pool:
        users = pool.map(_user, jobs, chunksize=1)
    old = {u["user_id"]: u for u in json.loads(Path(args.stage2_users).read_text())} if Path(args.stage2_users).exists() else {}
    res = {"name": "wider simulator search with time-aware scheduling (SM86)", "users": len(users), "agg": aggregate(users, old),
           "settings": {"drs": WIDE_DRS, "deltas": DELTAS, "budget_s": BUDGET_S}}
    res["seconds"] = round(time.time() - t0, 1)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    sfx = "" if not args.limit else f"-test{args.limit}"
    (out / f"simulator-search{sfx}.json").write_text(json.dumps(res, indent=1, default=float))
    (out / f"simulator-search-users{sfx}.json").write_text(json.dumps(users, default=float))
    md = out / f"simulator-search{sfx}.md"
    if md.exists() and "TAKEAWAYS" not in md.read_text(encoding="utf-8"):
        md = out / f"simulator-search{sfx}.new.md"
    md.write_text("\n".join(render(res)) + "\n", encoding="utf-8")
    print(f"[sm86] {len(users)} users -> {md} ({res['seconds']} s)", flush=True)


if __name__ == "__main__":
    main()
