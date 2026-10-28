"""How much does the big model's predicted R for a card move with the hour the review happens? (CPU, read only)

Replays users through rwkv/run_as_rnn.RNNProcess (the shipped big model, fork RWKV). At sampled reviews of cards with
elapsed_days >= 1, before the review is applied, it asks for R with the review moved by DELTA hours on the same day
(day_offset, elapsed_days and the day's review counts unchanged):
  - query R (what the fork's predict_retrievability_many returns: the instant head on a query row), full shift:
    elapsed seconds and their daily sin/cos phase both move;
  - query R, phase only: just the sin/cos of elapsed seconds move (the pure clock-phase part);
  - curve R (what intervals are computed from): the card's stored forgetting curve at elapsed seconds + DELTA.
Reports, per measure, the spread (max - min over the shifts) and the change vs the actual time, in R points.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rwkv.run_as_rnn import RNNProcess

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/anki-revlogs-10k-original-20261011"
FORK = Path("C:/Users/jonat/workspace/anki/qt/aqt/rwkv_inference/RWKV_trained_on_5000_10000.pth")
SHIFTS = [-8, -6, -4, -2, 0, 2, 4, 6, 8]
DAY = 86400


class Probe(RNNProcess):
    phase_shift = None  # seconds added to the sin/cos phase only

    def add_same(self, row):
        row = super().add_same(row)
        if self.phase_shift is not None:
            s = (row["elapsed_seconds"] + self.phase_shift) % DAY
            row["elapsed_seconds_sin"] = np.sin(s * 2 * np.pi / DAY)
            row["elapsed_seconds_cos"] = np.cos(s * 2 * np.pi / DAY)
        return row

    def imm_at(self, info, shift, phase_only):
        q = info.copy()
        if phase_only:
            self.phase_shift = shift
        else:
            q["elapsed_seconds"] = max(1, int(q["elapsed_seconds"]) + shift)
        try:
            return float(self.imm_predict(q))
        finally:
            self.phase_shift = None


def load_user(u):
    df = pd.read_parquet(DATA / "revlogs" / f"user_id={u}")
    cards = pd.read_parquet(DATA / "cards", filters=[("user_id", "=", u)]).drop(columns=["user_id"])
    decks = pd.read_parquet(DATA / "decks", filters=[("user_id", "=", u)]).drop(columns=["user_id", "parent_id"])
    df = df.merge(cards, on="card_id", how="left", validate="many_to_one").merge(decks, on="deck_id", how="left", validate="many_to_one")
    df["review_th"] = range(1, len(df) + 1)
    return df


@torch.inference_mode()
def probe_user(u, every, max_samples, rng):
    df = load_user(u)
    p = Probe(path=str(FORK), device=torch.device("cpu"), dtype=torch.float32)
    curves, out = {}, []
    eligible = np.flatnonzero((df["elapsed_days"].to_numpy() >= 1))
    pick = set(rng.choice(eligible, min(max_samples, max(1, len(eligible) // every)), replace=False).tolist()) if len(eligible) else set()
    for i, row in df.iterrows():
        cid = row["card_id"]
        info = row.copy().drop(labels=["rating", "duration"])
        if i in pick and cid in curves:
            rec = {"elapsed_days": int(row["elapsed_days"]), "y": int(row["rating"] > 1)}
            rec["query_full"] = [p.imm_at(info, h * 3600, False) for h in SHIFTS]
            rec["query_phase"] = [p.imm_at(info, h * 3600, True) for h in SHIFTS]
            rec["curve"] = [float(p.predict_func(curves[cid], max(1, int(row["elapsed_seconds"]) + h * 3600))) for h in SHIFTS]
            out.append(rec)
        p.imm_predict(info)  # as run_as_rnn: the query before the review (no state change)
        curves[cid] = p.process_row(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", default="")
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--max-samples", type=int, default=400)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(7)
    users = [int(x) for x in a.users.split(",")]
    allrec = {}
    for u in users:
        recs = probe_user(u, a.every, a.max_samples, rng)
        allrec[u] = recs
        print(f"[probe] user {u}: {len(recs)} samples", flush=True)
    rows = [r for v in allrec.values() for r in v]
    z = SHIFTS.index(0)
    summary = {"users": users, "samples": len(rows), "shifts_h": SHIFTS, "measures": {}}
    for m in ("query_full", "query_phase", "curve"):
        a_ = np.array([r[m] for r in rows])
        spread = (a_.max(1) - a_.min(1)) * 100
        d = (a_ - a_[:, [z]]) * 100
        summary["measures"][m] = {
            "spread_points": {"median": float(np.median(spread)), "p90": float(np.percentile(spread, 90)), "max": float(spread.max())},
            "mean_change_points_by_shift": [float(x) for x in d.mean(0)],
            "mean_abs_change_points_by_shift": [float(x) for x in np.abs(d).mean(0)],
        }
        ed = np.array([r["elapsed_days"] for r in rows])
        summary["measures"][m]["spread_median_by_interval"] = {
            name: float(np.median(spread[sel])) if sel.any() else None
            for name, sel in (("1-3 d", ed <= 3), ("4-20 d", (ed > 3) & (ed <= 20)), ("21-100 d", (ed > 20) & (ed <= 100)), (">100 d", ed > 100))}
    Path(a.out).write_text(json.dumps({"summary": summary, "per_user": {str(k): v for k, v in allrec.items()}}) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
