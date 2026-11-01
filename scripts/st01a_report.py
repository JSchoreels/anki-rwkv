"""Curve consistency on real rapid Good pairs (ST01a), stage 1 report (CPU): jump and tail-calibration tables from the
per-user pair files of scripts/st01a_pairs.py, the small model next to the big model (1 leak-free epoch).

For each pair a -> b (consecutive reviews of one card, b Good): jump(H) = curve_b(H) - curve_a(gap + H), the change of
the recall predicted for the same date H days after b. After an easy confirmation, a plausible update moves the far
recall up or leaves it about flat; a large share of pairs with the far recall going DOWN, or much larger jumps than
the big model's, is the instability the curve stability track is about.
Tail calibration: where the card has a real next review c after b, LogLoss of curve_b (the model's prediction) and of
curve_a at the same date (the curve before b's answer) on c's outcome, by the time from b to c.
Stress slice ("mature card, then rapid repeats"): chains a, b1, b2, ... with a mature (interval before a >= 21 days),
each step Good after 1-3 days; the 365-day recall at each step.

usage: st01a_report.py --model small=DIR --model big=DIR --users-file F [--out PREFIX]
"""
import argparse
import json
from pathlib import Path

import numpy as np

HORIZONS = (7, 30, 90, 365)
DAY = 86400.0


def load(folder, users):
    parts, missing = [], []
    for u in users:
        f = Path(folder) / f"{u}.npz"
        if not f.exists():
            missing.append(u)
            continue
        d = dict(np.load(f))
        if len(d.get("review_th_a", [])) == 0 or "pa_7" not in d:
            continue
        d["user"] = np.full(len(d["review_th_a"]), u, dtype=np.int32)
        parts.append(d)
    keys = parts[0].keys()
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}, missing


def jump_stats(d, mask):
    out = {"pairs": int(mask.sum())}
    if out["pairs"] == 0:
        return out
    for h in HORIZONS:
        pa, pb = d[f"pa_{h}"][mask].astype(np.float64), d[f"pb_{h}"][mask].astype(np.float64)
        j = pb - pa
        out[h] = {"pa": pa.mean(), "pb": pb.mean(), "mean_jump": j.mean(), "median_jump": float(np.median(j)),
                  "median_ratio": float(np.median(pb / np.maximum(pa, 1e-6))),
                  "down": float((j < 0).mean()), "down_01": float((j < -0.1).mean()),
                  "abs_gt_01": float((np.abs(j) > 0.1).mean()), "p90_abs": float(np.quantile(np.abs(j), 0.9))}
    return out


def logloss(p, y):
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def calibration(d, mask):
    m = mask & d["has_c"].astype(bool)
    e = d["e_c_seconds"] / DAY
    rows = []
    for lo, hi in ((0, 7), (7, 30), (30, 90), (90, 180), (180, 1e9)):
        k = m & (e >= lo) & (e < hi)
        if k.sum() < 50:
            rows.append({"bin": f"{lo}-{hi if hi < 1e9 else ''}", "n": int(k.sum())})
            continue
        y = d["y_c"][k].astype(np.float64)
        rows.append({"bin": f"{lo}-{hi if hi < 1e9 else ''}", "n": int(k.sum()), "recall": y.mean(),
                     "mean_pb": float(d["pc_b"][k].mean()), "mean_pa": float(d["pc_a"][k].mean()),
                     "ll_b": logloss(d["pc_b"][k], y), "ll_a": logloss(d["pc_a"][k], y),
                     "b_closer": float((np.abs(d["pc_b"][k] - y) < np.abs(d["pc_a"][k] - y)).mean())})
    return rows


def chains(d, max_steps=4):
    """365-day recall along chains: mature a, then Goods 1-3 days apart."""
    rapid = (d["gap_days"] >= 1) & (d["gap_days"] <= 3) & (d["rating_a"] >= 2)
    nxt = {}
    for i in np.flatnonzero(rapid):
        nxt[(int(d["user"][i]), int(d["review_th_a"][i]))] = i
    starts = np.flatnonzero(rapid & (d["ivl_a"] >= 21))
    trajs = []
    for i in starts:
        traj = [float(d["pa_365"][i]), float(d["pb_365"][i])]
        u, b = int(d["user"][i]), int(d["review_th_b"][i])
        while len(traj) <= max_steps and (u, b) in nxt:
            k = nxt[(u, b)]
            traj.append(float(d["pb_365"][k]))
            b = int(d["review_th_b"][k])
        trajs.append(traj)
    out = {"chains": len(trajs)}
    for steps in range(1, max_steps + 1):
        t = np.array([x[:steps + 1] for x in trajs if len(x) >= steps + 1])
        if len(t) == 0:
            break
        out[f"{steps} Goods"] = {"chains": len(t), "p365": [float(v) for v in t.mean(0)],
                                 "down_total": float((t[:, -1] < t[:, 0]).mean()),
                                 "range_gt_01": float(((t.max(1) - t.min(1)) > 0.1).mean())}
    return out


def slices(d):
    gap, ivl, ra = d["gap_days"], d["ivl_a"], d["rating_a"]
    rapid = (gap >= 1) & (gap <= 3) & (ra >= 2)
    s = {"rapid (gap 1-3 d, a passed)": rapid,
         "rapid, Good expected (recall at b >= 0.85)": rapid & (d["p_a_at_b"] >= 0.85)}
    for g in (0, 1, 2, 3):
        s[f"gap {g} d"] = (gap == g) & (ra >= 2)
    s["gap 4-7 d"] = (gap >= 4) & (gap <= 7) & (ra >= 2)
    s["rapid, a = first review"] = rapid & (ivl < 0)
    s["rapid, interval before a 0 d"] = rapid & (ivl == 0)
    s["rapid, interval before a 1-6 d"] = rapid & (ivl >= 1) & (ivl < 7)
    s["rapid, interval before a 7-20 d"] = rapid & (ivl >= 7) & (ivl < 21)
    s["rapid, mature (interval before a >= 21 d)"] = rapid & (ivl >= 21)
    s["rapid, interval before a >= 90 d"] = rapid & (ivl >= 90)
    s["rapid, a = Again"] = (gap >= 1) & (gap <= 3) & (ra == 1)
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="NAME=DIR (pair files)")
    parser.add_argument("--users-file", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    users = json.loads(Path(args.users_file).read_text())["users"]
    data = {}
    for item in args.model:
        name, _, folder = item.partition("=")
        data[name], missing = load(folder, users)
        print(f"[st01a-report] {name}: {len(set(data[name]['user']))} users, {len(data[name]['user']):,} pairs, "
              f"{len(missing)} users missing")
    common = set.intersection(*(set(map(int, np.unique(d["user"]))) for d in data.values()))
    res = {"users": len(common), "models": {}}
    for name, d in data.items():
        keep = np.isin(d["user"], list(common))
        d = {k: v[keep] for k, v in d.items()}
        sl = slices(d)
        res["models"][name] = {"jumps": {k: jump_stats(d, m) for k, m in sl.items()},
                               "calibration": {k: calibration(d, sl[k]) for k in
                                               ("rapid (gap 1-3 d, a passed)", "rapid, mature (interval before a >= 21 d)")},
                               "chains": chains(d)}
    md = [f"# Curve consistency on real rapid Good pairs (ST01a), stage 1", "",
          f"{res['users']} users of the 732 set (both models), every review of their histories (held-out users). "
          "Pair = consecutive reviews a -> b of one card, b answered Good. jump(H) = curve_b(H days after b) - "
          "curve_a(same date). down = share of pairs where the recall at that date is lower after the Good.", ""]
    names = list(data)
    md += ["## Jumps by slice", "",
           "| slice | model | pairs | " + " | ".join(f"{h} d: before -> after, mean jump, down, abs>0.1" for h in HORIZONS) + " |",
           "|---|---|---|" + "---|" * len(HORIZONS)]
    for k in res["models"][names[0]]["jumps"]:
        for n in names:
            j = res["models"][n]["jumps"][k]
            if j["pairs"] == 0:
                continue
            cells = [f"{j[h]['pa']:.3f} -> {j[h]['pb']:.3f}, {j[h]['mean_jump']:+.3f}, {j[h]['down']:.0%}, "
                     f"{j[h]['abs_gt_01']:.0%}" for h in HORIZONS]
            md.append(f"| {k} | {n} | {j['pairs']:,} | " + " | ".join(cells) + " |")
    md += ["", "## 365-day jump detail (rapid pairs)", "",
           "| slice | model | median jump | median ratio | down | down by > 0.1 | abs > 0.1 | 90th pct abs |",
           "|---|---|---|---|---|---|---|---|"]
    for k in ("rapid (gap 1-3 d, a passed)", "rapid, Good expected (recall at b >= 0.85)",
              "rapid, mature (interval before a >= 21 d)", "rapid, interval before a >= 90 d"):
        for n in names:
            j = res["models"][n]["jumps"][k]
            if j["pairs"]:
                x = j[365]
                md.append(f"| {k} | {n} | {x['median_jump']:+.3f} | {x['median_ratio']:.3f} | {x['down']:.0%} | "
                          f"{x['down_01']:.0%} | {x['abs_gt_01']:.0%} | {x['p90_abs']:.3f} |")
    md += ["", "## Tail calibration: which curve was closer at the real next review", "",
           "LogLoss on the next review's outcome of curve_b (after the Good, the model's prediction) and of curve_a "
           "(before it) at the same date; by days from b to the next review.", "",
           "| slice | model | days to next review | reviews | recall | mean after | mean before | LogLoss after | "
           "LogLoss before | after closer |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k in ("rapid (gap 1-3 d, a passed)", "rapid, mature (interval before a >= 21 d)"):
        for n in names:
            for r in res["models"][n]["calibration"][k]:
                if "recall" in r:
                    md.append(f"| {k} | {n} | {r['bin']} | {r['n']:,} | {r['recall']:.3f} | {r['mean_pb']:.3f} | "
                              f"{r['mean_pa']:.3f} | {r['ll_b']:.4f} | {r['ll_a']:.4f} | {r['b_closer']:.0%} |")
    md += ["", "## Stress slice: mature card, then rapid Goods (365-day recall at each step)", "",
           "| model | Goods after the mature review | chains | 365-day recall, step by step | lower at the end | "
           "range > 0.1 |", "|---|---|---|---|---|---|"]
    for n in names:
        c = res["models"][n]["chains"]
        for k, v in c.items():
            if isinstance(v, dict):
                md.append(f"| {n} | {k} | {v['chains']:,} | {' -> '.join(f'{p:.3f}' for p in v['p365'])} | "
                          f"{v['down_total']:.0%} | {v['range_gt_01']:.0%} |")
    text = "\n".join(md) + "\n"
    print(text)
    if args.out:
        Path(args.out + ".md").write_text(text, encoding="utf-8")
        Path(args.out + ".json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
