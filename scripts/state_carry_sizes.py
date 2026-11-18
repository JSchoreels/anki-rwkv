"""SM21 sizing (CPU, data only, no model): how much carried state training would have to store, and how often a
training window sees a card cold (its previous review lies before the window start), per window length.

For each sampled gate-tier training user (artifacts/accuracy/experiments/<run>/run/epoch-01/protocol.json, train
range, same chunk boundaries as the R01 trainer: chunks of L reviews from the range start) and each L:
  occurrences  sum over windows of the distinct cards / notes / decks / presets (+1 user) in the window: the number of
               entity states option 3 must store per epoch (one end-of-window state per entity per window);
  warm         those whose entity also occurs before the window start (the ones a carried state changes);
  cold_reviews non-first reviews whose card's previous review lies before the window start (the card stream starts
               from a zero state at that review in training, never at deployment);
plus the per-user number of distinct entities (an in-memory "latest state per entity" dict, fresh carry) and the
fraction of non-first reviews whose previous card review is more than N reviews back (what a burn-in of N reaches).

Users are sampled every k-th in length order and scaled by k (review totals are checked against the protocol).
w64 state per entity (values: WKV state 4 heads x 16 x 16 = 1,024 per time-mixer layer, 64 for each token-shift
vector; channel mixers listed in expertium.variants.W80_STRIP_CMIX keep no state):
  card 2 layers (cmix of layer 1 stripped) 2,240; note 1 layer 1,152; deck 4 layers 4,480; preset 3,264; user 3,264.
Bytes: fp32 = 4 x values; int8 = 1 B per WKV value + one fp16 scale per state row (64 per layer) + fp16 shift vectors.

  python scripts/state_carry_sizes.py [--every 10] [--out artifacts/small-model/state-carry-sizes.json]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = Path("C:/Users/jonat/workspace/wt/small-model/artifacts/accuracy/experiments/scale-w64kd-gate-cached-ctx2048")
LENGTHS = (2048, 4096, 8192, 16384, 32768)
BURN = (2048, 4096, 8192, 32768)
STREAMS = ("card_id", "note_id", "deck_id", "preset_id")
# (WKV layers, token-shift vectors) per stream for w64
LAYOUT = {"card_id": (2, 3), "note_id": (1, 2), "deck_id": (4, 6), "preset_id": (3, 3), "user_id": (3, 3)}


def state_bytes(stream, dtype):
    layers, shifts = LAYOUT[stream]
    if dtype == "fp32":
        return 4 * (layers * 1024 + shifts * 64)
    return layers * (1024 + 2 * 64) + shifts * 64 * 2


def user_frame(data, user):
    rev = pd.read_parquet(data / "revlogs" / f"user_id={user}", columns=["card_id"])
    from rwkv.data_processing import read_user_metadata  # falls back to a filtered read for missing partitions
    cards = read_user_metadata(data / "cards", user)[["card_id", "note_id", "deck_id"]]
    decks = read_user_metadata(data / "decks", user)[["deck_id", "preset_id"]]
    df = rev.merge(cards, on="card_id", how="left").merge(decks, on="deck_id", how="left")
    for column in STREAMS:
        df[column] = pd.factorize(df[column].fillna(-1))[0]
    return df


def measure(df, start, end):
    section = df.iloc[start:end].reset_index(drop=True)
    n = len(section)
    position = np.arange(n)
    out = {"reviews": n, "entities": {s: int(section[s].nunique()) for s in STREAMS}}
    first_index = {s: pd.Series(position).groupby(section[s].to_numpy()).transform("min").to_numpy() for s in STREAMS}
    card = section["card_id"].to_numpy()
    previous = pd.Series(position).groupby(card).shift(1).to_numpy()  # previous review index of the same card
    repeat = ~np.isnan(previous)
    gap = np.where(repeat, position - np.nan_to_num(previous), 0)
    out["non_first"] = int(repeat.sum())
    for length in LENGTHS:
        window = position // length
        window_start = window * length
        row = {"windows": int(window.max()) + 1 if n else 0}
        cold = repeat & (np.nan_to_num(previous, nan=-1) < window_start)
        row["cold_reviews"] = int(cold.sum())
        for s in STREAMS:
            ids = section[s].to_numpy()
            pairs = pd.DataFrame({"w": window, "e": ids, "fi": first_index[s]}).drop_duplicates(["w", "e"])
            row[f"{s}_occ"] = int(len(pairs))
            row[f"{s}_warm"] = int((pairs["fi"].to_numpy() < pairs["w"].to_numpy() * length).sum())
        row["user_id_occ"] = row["windows"]
        row["user_id_warm"] = max(row["windows"] - 1, 0)
        out[str(length)] = row
    in_window = position % 2048
    for n_burn in BURN:
        out[f"burn{n_burn}_cold_2048"] = int((repeat & (gap > in_window + n_burn)).sum())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--out", default="artifacts/small-model/state-carry-sizes.json")
    args = parser.parse_args()
    protocol = json.loads((RUN / "run" / "epoch-01" / "protocol.json").read_text())
    data = Path(protocol["dataset_path"])
    rows = sorted(protocol["rows"], key=lambda r: (r["train_range"][1] - r["train_range"][0], r["user_id"]))
    sample = rows[args.every // 2::args.every]
    total_reviews = sum(r["train_range"][1] - r["train_range"][0] for r in protocol["rows"])
    started = time.perf_counter()
    agg, per_user = {}, []
    for index, row in enumerate(sample):
        df = user_frame(data, row["user_id"])
        start, end = row["train_range"]
        m = measure(df, start, end)
        per_user.append({"user_id": row["user_id"], "reviews": m["reviews"], "entities": m["entities"]})

        def add(prefix, value):
            if isinstance(value, dict):
                for key, item in value.items():
                    add(f"{prefix}{key}/", item)
            else:
                agg[prefix[:-1]] = agg.get(prefix[:-1], 0) + value * args.every
        add("", {k: v for k, v in m.items() if k != "entities"})
        if index % 50 == 0:
            print(f"[sizes] {index + 1}/{len(sample)} users, {time.perf_counter() - started:.0f}s", flush=True)
    scale = total_reviews / agg["reviews"]
    result = {"run": str(RUN), "users_sampled": len(sample), "every": args.every, "protocol_reviews": total_reviews,
              "sampled_reviews_scaled": agg["reviews"], "scale_to_protocol": scale,
              "non_first_reviews": agg["non_first"] * scale, "by_length": {}, "burn_in_cold_share_2048": {},
              "state_bytes": {s: {d: state_bytes(s, d) for d in ("fp32", "int8")} for s in LAYOUT}}
    for length in LENGTHS:
        key = str(length)
        row = {"cold_share_of_non_first": agg[f"{key}/cold_reviews"] / agg["non_first"]}
        for dtype in ("fp32", "int8"):
            occ = sum(agg[f"{key}/{s}_occ"] * state_bytes(s, dtype) for s in LAYOUT)
            warm = sum(agg[f"{key}/{s}_warm"] * state_bytes(s, dtype) for s in LAYOUT)
            row[f"store_all_GB_{dtype}"] = occ * scale / 1e9
            row[f"store_warm_GB_{dtype}"] = warm * scale / 1e9
            row[f"bytes_per_review_{dtype}"] = occ / agg["reviews"]
        for s in LAYOUT:
            row[f"{s}_occ_per_review"] = agg[f"{key}/{s}_occ"] / agg["reviews"]
            row[f"{s}_warm_per_review"] = agg[f"{key}/{s}_warm"] / agg["reviews"]
        result["by_length"][key] = row
    for n_burn in BURN:
        result["burn_in_cold_share_2048"][str(n_burn)] = agg[f"burn{n_burn}_cold_2048"] / agg["non_first"]
    dict_bytes = [sum(u["entities"][s] * state_bytes(s, "fp32") for s in STREAMS) for u in per_user]
    result["fresh_carry_dict_MB_fp32"] = {"max": max(dict_bytes) / 1e6, "median": float(np.median(dict_bytes)) / 1e6}
    result["largest_users"] = sorted(per_user, key=lambda u: -u["reviews"])[:3]
    result["seconds"] = round(time.perf_counter() - started, 1)
    out = ROOT / args.out
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("largest_users",)}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
