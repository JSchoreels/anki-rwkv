"""SM58 follow-up 3 (user 2026-10-23): does the small model change the ORDER of cards compared with the big model?
For each benchmark user and each day, take the cards reviewed that day after at least one day away (first review of the
card that day), rank them by predicted recall R under the small and the big model (w8+s8), and compare the two orders:
  - Kendall tau-b and Spearman between the two orders (1 = same order),
  - share of card pairs the two models put in opposite order, overall and by how far apart the big model puts the pair
    (|R_big_i - R_big_j| < 1%, 1-5%, 5-10%, >= 10%),
  - overlap of the "most at risk" cards (lowest 10% R of the day) between the two models,
  - for pairs the models order differently where one card was forgotten and the other recalled: which model put the
    forgotten card lower (= which order was right).
Days with fewer than MIN_CARDS cards are skipped; days with more than MAX_CARDS cards are subsampled for the pair counts
(seed 0). Same 722 users and dumps as summary.md.  python sm58_ranking.py  -> ranking.{json,md}"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

import sm58_umplus as U

HERE = Path(__file__).resolve().parent
MIN_CARDS, MAX_CARDS = 10, 400
BUCKETS = ((0, 0.01), (0.01, 0.05), (0.05, 0.10), (0.10, 2.0))
TAGS = {"sm55": "8k model (SM55)", "sm36": "Muon version (SM36)"}


def day_groups(u, s, b):
    r = pd.read_parquet(U.ORIG / "revlogs" / f"user_id={u}", columns=["card_id", "day_offset", "elapsed_seconds"])
    rows = s["review_th"] - 1
    d = pd.DataFrame({"card": r["card_id"].to_numpy()[rows], "day": r["day_offset"].to_numpy()[rows],
                      "el": r["elapsed_seconds"].to_numpy()[rows], "i": np.arange(len(rows))})
    d = d[d["el"] >= 86400].drop_duplicates(["day", "card"])
    for _, g in d.groupby("day"):
        if len(g) >= MIN_CARDS:
            yield g["i"].to_numpy()


def new_acc():
    return {"days": 0, "cards": 0, "tau": [], "rho": [], "pairs": 0, "flips": 0,
            "bucket_pairs": [0] * len(BUCKETS), "bucket_flips": [0] * len(BUCKETS),
            "risk_cards": 0, "risk_shared": 0, "decided_flips": 0, "small_right": 0}


def add_day(a, ps, pb, y, rng):
    n = len(ps)
    a["days"] += 1; a["cards"] += n
    a["tau"].append(kendalltau(ps, pb).statistic); a["rho"].append(spearmanr(ps, pb).statistic)
    k = max(1, n // 10)
    a["risk_cards"] += k
    a["risk_shared"] += len(set(np.argsort(ps, kind="stable")[:k]) & set(np.argsort(pb, kind="stable")[:k]))
    if n > MAX_CARDS:
        keep = rng.choice(n, MAX_CARDS, replace=False); ps, pb, y = ps[keep], pb[keep], y[keep]
    i, j = np.triu_indices(len(ps), 1)
    db, ds = pb[i] - pb[j], ps[i] - ps[j]
    ok = db != 0
    flip = ok & (db * ds < 0)
    a["pairs"] += int(ok.sum()); a["flips"] += int(flip.sum())
    gap = np.abs(db)
    for t, (lo, hi) in enumerate(BUCKETS):
        m = ok & (gap >= lo) & (gap < hi)
        a["bucket_pairs"][t] += int(m.sum()); a["bucket_flips"][t] += int((m & flip).sum())
    dec = flip & (y[i] != y[j])
    # the right order puts the forgotten card (y = 0) lower: sign(R_i - R_j) == sign(y_i - y_j)
    a["decided_flips"] += int(dec.sum())
    a["small_right"] += int((dec & (np.sign(ds) == np.sign(y[i] - y[j]))).sum())


def summarize(a):
    tau, rho = np.array(a["tau"]), np.array(a["rho"])
    tau, rho = tau[~np.isnan(tau)], rho[~np.isnan(rho)]
    return {"days": a["days"], "cards": a["cards"],
            "tau_median": float(np.median(tau)), "tau_mean": float(tau.mean()), "tau_p10": float(np.percentile(tau, 10)),
            "rho_median": float(np.median(rho)), "rho_mean": float(rho.mean()),
            "pairs": a["pairs"], "flip_share": a["flips"] / a["pairs"],
            "buckets": [{"gap": f"{lo:.0%}-{hi:.0%}" if hi < 1 else f">= {lo:.0%}", "pairs": p, "share_of_pairs": p / a["pairs"],
                         "flip_share": f / p if p else None, "share_of_all_flips": f / a["flips"] if a["flips"] else None}
                        for (lo, hi), p, f in zip(BUCKETS, a["bucket_pairs"], a["bucket_flips"])],
            "risk_overlap": a["risk_shared"] / a["risk_cards"],
            "decided_flips": a["decided_flips"], "small_right_share": a["small_right"] / a["decided_flips"] if a["decided_flips"] else None}


def main():
    users = json.loads((HERE / "users.json").read_text())["users"]
    big_dirs = [Path(U.SM24_DUMP), HERE / "dump"]
    res = {}
    for tag in TAGS:
        acc = {h: new_acc() for h in U.HEADS}
        used, rng = 0, np.random.default_rng(0)
        for u in users:
            sf = HERE / "dump" / f"{u}_{tag}_w8+s8.npz"
            bf = next((d / f"{u}_fork_w8+s8.npz" for d in big_dirs if (d / f"{u}_fork_w8+s8.npz").exists()), None)
            if not sf.exists() or bf is None:
                continue
            s, b = np.load(sf), np.load(bf)
            if not (np.array_equal(s["review_th"], b["review_th"]) and np.array_equal(s["y"], b["y"])):
                continue
            used += 1
            y = s["y"].astype(np.int8)
            cols = {h: (s[c].astype(np.float64), b[c].astype(np.float64)) for h, (c, _, _) in U.HEADS.items()}
            for idx in day_groups(u, s, b):
                for h, (ps, pb) in cols.items():
                    add_day(acc[h], ps[idx], pb[idx], y[idx], rng)
        res[tag] = {"users": used, **{h: summarize(a) for h, a in acc.items()}}
        print(tag, used, "users", {h: round(res[tag][h]["flip_share"], 4) for h in U.HEADS}, flush=True)
    (HERE / "ranking.json").write_text(json.dumps(res, indent=1) + "\n")
    lines = ["# SM58 follow-up 3: does the small model change the order of cards?", "",
             f"Per user and day: cards reviewed that day after >= 1 day away (first review of the card that day), days with >= {MIN_CARDS} cards; "
             f"pairs from at most {MAX_CARDS} cards per day (seed 0). R = predicted recall, w8+s8, small vs fork big model.", "",
             "| Model | Head | Users | Days | Cards | Kendall tau median (p10) | Spearman median | Pairs flipped | Risk top-10% overlap | Flipped pairs with one forgotten | Small model right |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for tag, name in TAGS.items():
        for h in U.HEADS:
            r = res[tag][h]
            lines.append(f"| {name} | {h} | {res[tag]['users']} | {r['days']:,} | {r['cards']:,} | {r['tau_median']:.3f} ({r['tau_p10']:.3f}) | "
                         f"{r['rho_median']:.3f} | {r['flip_share']:.1%} | {r['risk_overlap']:.1%} | {r['decided_flips']:,} | {r['small_right_share']:.1%} |")
    lines += ["", "## Flipped pairs by how far apart the big model puts the two cards", "",
              "| Model | Head | Big-model gap | Share of pairs | Flipped in that gap | Share of all flips |", "|---|---|---|---:|---:|---:|"]
    for tag, name in TAGS.items():
        for h in U.HEADS:
            for bk in res[tag][h]["buckets"]:
                lines.append(f"| {name} | {h} | {bk['gap']} | {bk['share_of_pairs']:.1%} | {bk['flip_share']:.1%} | {bk['share_of_all_flips']:.1%} |")
    (HERE / "ranking.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
