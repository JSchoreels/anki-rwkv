"""SM70: dataset inventory statistics on the probe users (raw tables only): how often the card / note / deck / preset
fields are present, sibling structure, review types. Usage: sm70_inventory.py --users-file F --count N --out J"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.sm70_features import DATA  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", required=True, help="comma list or @dump-dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.users.startswith("@"):
        users = sorted(int(p.name.split("_")[0]) for p in Path(a.users[1:]).glob("*_sm56_w8+s8.npz"))
    else:
        users = [int(u) for u in a.users.split(",")]
    tot = {"reviews": 0, "reviews_card_missing_from_cards_table": 0, "reviews_deck_missing_from_decks_table": 0,
           "reviews_on_multi_card_notes": 0, "cards_reviewed": 0, "cards_in_table_never_reviewed": 0,
           "state_counts": {}, "users_with_multiple_presets": 0, "users_with_deck_hierarchy": 0}
    for u in users:
        df = pd.read_parquet(DATA / "revlogs" / f"user_id={u}")
        cards = pd.read_parquet(DATA / "cards" / f"user_id={u}")
        decks = pd.read_parquet(DATA / "decks" / f"user_id={u}")
        m = df.merge(cards, on="card_id", how="left").merge(decks, on="deck_id", how="left")
        tot["reviews"] += len(df)
        tot["reviews_card_missing_from_cards_table"] += int(m["note_id"].isna().sum())
        tot["reviews_deck_missing_from_decks_table"] += int((m["note_id"].notna() & m["preset_id"].isna()).sum())
        ns = cards.groupby("note_id").size()
        tot["reviews_on_multi_card_notes"] += int(m["note_id"].map(ns).fillna(1).gt(1).sum())
        tot["cards_reviewed"] += int(df["card_id"].nunique())
        tot["cards_in_table_never_reviewed"] += int((~cards["card_id"].isin(df["card_id"])).sum())
        for k, v in df["state"].value_counts().items():
            tot["state_counts"][str(k)] = tot["state_counts"].get(str(k), 0) + int(v)
        tot["users_with_multiple_presets"] += int(decks["preset_id"].nunique() > 1)
        tot["users_with_deck_hierarchy"] += int(decks["parent_id"].isin(decks["deck_id"]).any())
    tot["users"] = len(users)
    Path(a.out).write_text(json.dumps(tot, indent=1))
    print(json.dumps(tot))


if __name__ == "__main__":
    main()
