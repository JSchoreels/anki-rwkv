#!/usr/bin/env python
"""SM72 step 3 launcher helper (CPU, seconds): per-epoch 464-set losses and the final summary of the leak-free retrain.

All rows here are scored leak-free (scripts/small_model_heldout.py --deleted-cards neighbour --split-existing, no
RWKV_DELETED_CARDS in the scorer), so "all" = every benchmark test review with the deleted-card signals removed.

  record K LABEL   L_K (review-weighted benchmark LogLoss, w8+s8, curve and instant) of LABEL over the 458 fit users of
                   the 464 set from gpu/small-lf-epochs-464.jsonl, appended once to epochs.jsonl; prints one line with
                   the gain vs epoch 1 and vs epoch K-1 and the gap to the big model leak-free (SM72 step 1) on the same
                   users; exit 2 if users are missing. Fixed epoch count: no stopping rule.
  summary          every finished cool-down (dec15 = main result, dec10 = side branch) vs the big model leak-free, the
                   shipped big model and the released small models (leak inputs, scored leak-free) on the 464 set and
                   the 732 pool: LogLoss on all / existing / deleted-card reviews, % vs the big model leak-free, paired
                   deltas (user bootstrap 2,000, seed 0, 95% CI); the epoch table; training times. sm72s-summary.{md,json}
"""
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
WT = HERE.parents[2]
sys.path.insert(0, str(WT))
from scripts.sm71_report import load as _load  # noqa: E402


def load(source, label, users):
    """{user: {subset: {head: (LogLoss, size)}}} restricted to users (sm71_report.load filters dump folders only)."""
    return {u: r for u, r in _load(source, label, users).items() if u in users}

G = HERE / "gpu"
EPOCHS = HERE / "epochs.jsonl"
OUT464 = G / "small-lf-epochs-464.jsonl"
OUTDEC = G / "small-lf-neighbour.jsonl"
OUTREF = G / "small-dec15-neighbour.jsonl"
USERS = {464: HERE / "eval-users-464-fit.json", 732: HERE / "eval-users-732-fit.json"}
LONG = HERE / "users-732-long.json"
SM72 = Path("C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72/gpu-big")      # read only
SM71 = Path("C:/Users/jonat/workspace/wt/sm71/artifacts/small-model/sm71/gpu")          # read only
LBL = "s-w64kd-final-sc-lanes-leakfree"
BIG_LF = (SM72 / "bigleak-neighbour.jsonl", "big-leakfree-1ep/w8+s8")
HEADS = {"ahead": "curve", "instant": "instant"}
SUBSETS = ("all", "existing", "deleted")


def users_of(pool):
    users = set(json.loads(USERS[int(str(pool).split("+")[0])].read_text())["users"])
    if pool == "732+long":  # plus the 3 long users scored at the 0.92 cap (726); "732" = the 723 users of the SM71/SM72 readouts
        users |= set(json.loads(LONG.read_text())["users"])
    return users


def wmean(rows, users, subset, head):
    v = np.array([rows[u][subset][head][0] for u in users]); w = np.array([rows[u][subset][head][1] for u in users], float)
    v = np.where(w > 0, v, 0.0)
    return float((v * w).sum() / w.sum())


def read_epochs():
    out = {}
    if EPOCHS.exists():
        for line in EPOCHS.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            out[r["epoch"]] = r
    return out


def record(k, label):
    users = users_of(464)
    mine = load(OUT464, label, users) if OUT464.exists() else {}
    if set(mine) != users:
        print(f"[sm72s-rule] {label}: {len(mine)} of {len(users)} users scored")
        return 2
    big = load(*BIG_LF, users)
    common = sorted(users & set(big))
    L = {h: wmean(mine, sorted(users), "all", head) for head, h in HEADS.items()}
    B = {h: wmean(big, common, "all", head) for head, h in HEADS.items()}
    M = {h: wmean(mine, common, "all", head) for head, h in HEADS.items()}
    epochs = read_epochs()
    if k not in epochs:
        with EPOCHS.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": k, "label": label, "users": len(users), **L,
                                "big_leakfree_users": len(common), **{f"pct_vs_big_leakfree_{h}": (M[h] / B[h] - 1) * 100 for h in M}}) + "\n")
        epochs = read_epochs()
    first, prev = epochs.get(1), epochs.get(k - 1)
    parts = [f"epoch {k}/20: curve {L['curve']:.5f} instant {L['instant']:.5f}"]
    if first and k > 1:
        parts.append(f"gain vs e1 curve {first['curve'] - L['curve']:+.5f} ({(first['curve'] - L['curve']) / first['curve']:+.2%}) "
                     f"instant {first['instant'] - L['instant']:+.5f} ({(first['instant'] - L['instant']) / first['instant']:+.2%})")
    if prev:
        parts.append(f"vs e{k - 1} curve {prev['curve'] - L['curve']:+.5f} instant {prev['instant'] - L['instant']:+.5f}")
    parts.append(f"vs big model leak-free ({len(common)} users) {(M['curve'] / B['curve'] - 1) * 100:+.2f}% / {(M['instant'] / B['instant'] - 1) * 100:+.2f}%")
    print("; ".join(parts))
    return 0


def summary():
    series = {
        "Big model leak-free (SM72 step 1)": BIG_LF,
        "Shipped big model (fork RWKV, leak inputs)": (SM71 / "big-neighbour.jsonl", "fork/w8+s8"),
        "Small leak-free, cool-down from epoch 20 (SM72)": (OUTDEC, f"{LBL}-dec20-1ep32k-muon/w8+s8"),
        "Small leak-free, cool-down from epoch 15 (SM72)": (OUTDEC, f"{LBL}-dec15-1ep32k-muon/w8+s8"),
        "Small leak-free, cool-down from epoch 10 (SM72)": (OUTDEC, f"{LBL}-dec10-1ep32k-muon/w8+s8"),
        "Released small, cool-down from epoch 10 (SM56, leak inputs)": (SM71 / "small-neighbour.jsonl", "sm56/w8+s8"),
        "Small, cool-down from epoch 15 (SM56, leak inputs)": (OUTREF, "sm56dec15/w8+s8"),
    }
    ref = "Big model leak-free (SM72 step 1)"
    pairs = [("Small leak-free, cool-down from epoch 20 (SM72)", "Small leak-free, cool-down from epoch 15 (SM72)"),
             ("Small leak-free, cool-down from epoch 20 (SM72)", "Released small, cool-down from epoch 10 (SM56, leak inputs)"),
             ("Small leak-free, cool-down from epoch 15 (SM72)", "Small, cool-down from epoch 15 (SM56, leak inputs)"),
             ("Small leak-free, cool-down from epoch 10 (SM72)", "Released small, cool-down from epoch 10 (SM56, leak inputs)"),
             ("Small leak-free, cool-down from epoch 15 (SM72)", "Released small, cool-down from epoch 10 (SM56, leak inputs)"),
             ("Small leak-free, cool-down from epoch 15 (SM72)", "Small leak-free, cool-down from epoch 10 (SM72)")]
    out, md = {}, []
    for pool in (464, 732, "732+long"):
        wanted = users_of(pool)
        rows = {n: load(f, l, wanted) for n, (f, l) in series.items() if Path(f).exists()}
        rows = {n: r for n, r in rows.items() if r}
        if ref not in rows or len(rows) < 2:
            md.append(f"{pool}: not enough series"); continue
        users = sorted(set.intersection(*(set(r) for r in rows.values())))
        idx = np.random.default_rng(0).integers(0, len(users), size=(2000, len(users)))
        res = {"users": len(users), "series": list(rows), "loss": {}, "pct_vs_big_leakfree": {}, "pairs": {}}
        md += [f"## {pool} set: {len(users)} users common to every series (leak-free scoring)", "",
               "| model | all curve | all instant | existing curve | existing instant | deleted curve | deleted instant | vs big leak-free curve | instant |",
               "|---|---|---|---|---|---|---|---|---|"]

        def arr(n, subset, head):
            v = np.array([rows[n][u][subset][head][0] for u in users]); w = np.array([rows[n][u][subset][head][1] for u in users], float)
            return np.where(w > 0, v, 0.0), w

        for n in rows:
            cells = []
            for subset in SUBSETS:
                for head, h in HEADS.items():
                    v, w = arr(n, subset, head)
                    res["loss"][f"{n}|{subset}|{h}"] = round(float((v * w).sum() / w.sum()), 6)
                    cells.append(f"{res['loss'][f'{n}|{subset}|{h}']:.5f}")
            pct = []
            for head, h in HEADS.items():
                (a, w), (b, _) = arr(n, "all", head), arr(ref, "all", head)
                base = (b * w).sum() / w.sum()
                boot = ((a - b)[idx] * w[idx]).sum(1) / (b[idx] * w[idx]).sum(1) * 100
                p = float(((a - b) * w).sum() / w.sum() / base * 100)
                res["pct_vs_big_leakfree"][f"{n}|{h}"] = {"pct": round(p, 3), "ci": [round(float(np.percentile(boot, q)), 3) for q in (2.5, 97.5)]}
                pct.append(f"{p:+.2f}% [{np.percentile(boot, 2.5):+.2f}, {np.percentile(boot, 97.5):+.2f}]" if n != ref else "-")
            md.append(f"| {n} | " + " | ".join(cells) + " | " + " | ".join(pct) + " |")
        md += ["", "| pair (A - B, all reviews) | curve delta [95% CI] | instant delta [95% CI] | users A better (curve / instant) |", "|---|---|---|---|"]
        for a_name, b_name in pairs:
            if a_name not in rows or b_name not in rows:
                continue
            cells, better = [], []
            for head, h in HEADS.items():
                (a, w), (b, _) = arr(a_name, "all", head), arr(b_name, "all", head)
                d = a - b
                boot = (d[idx] * w[idx]).sum(1) / w[idx].sum(1)
                point = float((d * w).sum() / w.sum())
                res["pairs"][f"{a_name} - {b_name}|{h}"] = {"delta": round(point, 6), "ci": [round(float(np.percentile(boot, q)), 6) for q in (2.5, 97.5)],
                                                            "users_a_better": int((a < b).sum())}
                cells.append(f"{point:+.6f} [{np.percentile(boot, 2.5):+.6f}, {np.percentile(boot, 97.5):+.6f}]")
                better.append(str(int((a < b).sum())))
            md.append(f"| {a_name} - {b_name} | " + " | ".join(cells) + f" | {' / '.join(better)} of {len(users)} |")
        md.append("")
        out[str(pool)] = res
    epochs = read_epochs()
    md += ["## Constant-LR epochs (464 set, 458 fit users, leak-free scoring)", "",
           "| epoch | curve | instant | gain vs e1 curve | instant | vs big leak-free curve | instant |", "|---|---|---|---|---|---|---|"]
    for k, r in sorted(epochs.items()):
        f = epochs.get(1, r)
        md.append(f"| {k} | {r['curve']:.5f} | {r['instant']:.5f} | {(f['curve'] - r['curve']) / f['curve']:+.2%} | "
                  f"{(f['instant'] - r['instant']) / f['instant']:+.2%} | {r['pct_vs_big_leakfree_curve']:+.2f}% | {r['pct_vs_big_leakfree_instant']:+.2f}% |")
    train = {}
    exp = Path(os.environ["EXP"])
    for line in (exp / "phases.jsonl").read_text().splitlines() if (exp / "phases.jsonl").exists() else []:
        r = json.loads(line)
        if r.get("status") == "complete":
            train[f"epoch {r['epoch']}"] = {k: r.get(k) for k in ("training_seconds", "wall_seconds", "reviews_per_second", "review_presentations")}
    for k in (20, 15, 10):
        result = WT / f"artifacts/small-model/scale-w64kd-final-sc-lanes-leakfree-dec{k}-1ep32k-muon/expertium/E89-F65e-F77a/result.json"
        if result.exists():
            r = json.loads(result.read_text())
            train[f"cool-down from e{k}"] = {x: r.get(x) for x in ("training_users", "review_presentations", "training_seconds", "reviews_per_second")}
    md += ["", "## Training", ""] + [f"- {k}: {v.get('review_presentations') or 0:,} reviews, {(v.get('training_seconds') or 0) / 60:.0f} min, "
                                     f"{v.get('reviews_per_second') or 0:,.0f} reviews/s" for k, v in train.items()]
    (HERE / "sm72s-summary.json").write_text(json.dumps({"pools": out, "epochs": epochs, "training": train}, indent=1) + "\n")
    (HERE / "sm72s-summary.md").write_text("# SM72 step 3: small model retrained leak-free (w8+s8, review-weighted, user bootstrap 2,000 "
                                           "seed 0, 95% CI)\n\n" + "\n".join(md) + "\n")
    print("[sm72s-summary]\n" + "\n".join(md))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "record":
        raise SystemExit(record(int(sys.argv[2]), sys.argv[3]))
    if cmd == "summary":
        raise SystemExit(summary())
    raise SystemExit(f"unknown command {cmd}")
