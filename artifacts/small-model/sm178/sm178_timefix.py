"""Today's models on donor data (SM178): drop the impossible review timestamps from the frozen 218-donor snapshot.

The 11-26 build (`data/donations-20261126`, 218 donors) holds 106 reviews answered before 2006 (one donor's clock
set to 2000, one review with a zeroed id, 1970). The 1970 review gives a 55-year interval, past the model's interval
grid (exp(21) s, about 42 years): `interp`'s searchsorted index runs off the grid and the GPU scorer hits a
device-side assert (ticket 001790). The converter now drops reviews before 2006-01-01 UTC
(`donations_to_dataset.MIN_REVIEW_MS`) before the builder rules, but a fresh conversion reads today's donation
folders (more files, renumbered donors). So this script applies the same rule to the snapshot itself:

  every output row of the show-time layout gives back its raw revlog row (review id = review_time + duration, review
  type = state - 1, Learning start = state 0); the bad rows are the earliest rows of their cards, so the rows the
  builder had already dropped (history before the last Learning start, cram rows) cannot change the result; the
  converter's own `build_revlogs` then rebuilds both layouts. next_day_at comes from the stored day boundary
  (`sources.parquet`); only its phase matters.

  python sm178_timefix.py check   rebuild all 218 users WITHOUT dropping, must equal the 11-26 build bit for bit
  python sm178_timefix.py write   write data/donations-20261126-timefix (+ -answer-time) with the bad rows dropped
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MAIN = Path(r"C:\Users\jonat\workspace\anki-rwkv")
sys.path.insert(0, str(MAIN / "scripts"))
import donations_to_dataset as conv  # noqa: E402

OLD = MAIN / "data" / "donations-20261126"
OLD_AT = MAIN / "data" / "donations-20261126-answer-time"
NEW = MAIN / "data" / "donations-20261126-timefix"
NEW_AT = MAIN / "data" / "donations-20261126-timefix-answer-time"


def raw_rows(show):
    rev = np.zeros(len(show), conv.REV_DT)
    rev["cid"] = show["card_id"].to_numpy()
    rev["time"] = show["duration"].to_numpy()
    rev["id"] = show["review_time"].to_numpy() + rev["time"]
    state = show["state"].to_numpy()
    rev["type"] = np.where(state == 0, 0, state - 1)
    rev["ease"] = show["rating"].to_numpy()
    rev["factor"] = 1  # filtered rows without rescheduling are already gone
    return rev


def nda_of(src, rev):
    day = {"rollover": int(src["rollover_hour"]), "schedVer": 2}
    if src["offset_source"] != "fallback-utc":
        day[src["offset_source"]] = int(src["utc_offset_minutes_west"])
    assert src["day_rule"] == "v2"
    return conv.next_day_at(day, int(rev["id"].max()))[0]


def same(a, b):
    return a.shape == b.shape and list(a.columns) == list(b.columns) and a.index.equals(b.index) and a.equals(b) \
        and all(a[c].dtype == b[c].dtype for c in a.columns)


def main():
    mode = sys.argv[1]
    sources = pd.read_parquet(OLD / "sources.parquet").set_index("user_id")
    if mode == "write":
        for d in (NEW, NEW_AT):
            if d.exists() and any(d.iterdir()):
                sys.exit(f"{d} exists and is not empty")
    bad_users, report = 0, {}
    for user, src in sources.iterrows():
        show_old = pd.read_parquet(OLD / "revlogs" / f"user_id={user}" / "data.parquet")
        at_old = pd.read_parquet(OLD_AT / "revlogs" / f"user_id={user}" / "data.parquet")
        rev = raw_rows(show_old)
        nda = nda_of(src, rev)
        bad = rev["id"] < conv.MIN_REVIEW_MS
        if mode == "write":
            rev = rev[~bad]
        show = conv.build_revlogs(rev, nda, show_time=True)
        at = conv.build_revlogs(rev, nda, show_time=False)
        if mode == "check":
            if not (same(show, show_old) and same(at, at_old)):
                bad_users += 1
                print("DIFF", user, flush=True)
            continue
        cards = pd.read_parquet(OLD / "cards" / f"user_id={user}" / "data.parquet")
        decks = pd.read_parquet(OLD / "decks" / f"user_id={user}" / "data.parquet")
        conv.write_user(NEW, user, show, cards, decks)
        conv.write_user(NEW_AT, user, at, cards, decks)
        if bad.any():
            report[str(user)] = {"bad_rows_in_snapshot": int(bad.sum()), "reviews_before": len(show_old),
                                 "reviews_after": len(show), "cards_lost": int(show_old["card_id"].nunique()
                                                                              - show["card_id"].nunique())}
            sources.loc[user, "reviews"] = len(show)
            sources.loc[user, "cards_reviewed"] = int(show["card_id"].nunique())
            sources.loc[user, "reviews_deleted_cards"] = int((~show["card_id"].isin(cards["card_id"])).sum())
        elif not (same(show, show_old) and same(at, at_old)):
            sys.exit(f"user {user} changed without a bad row")
    if mode == "check":
        print(f"check: {len(sources) - bad_users} of {len(sources)} users rebuilt bit for bit")
        sys.exit(1 if bad_users else 0)
    out = sources.reset_index()
    for d in (NEW, NEW_AT):
        out.to_parquet(d / "sources.parquet", index=False)
    summary = json.loads((OLD / "summary.json").read_text(encoding="utf-8"))
    summary["timefix"] = {"rule": "reviews answered before 2006-01-01 UTC dropped before the builder rules",
                          "from": OLD.name, "users": report,
                          "reviews_before": int(pd.read_parquet(OLD / "sources.parquet")["reviews"].sum()),
                          "reviews_after": int(out["reviews"].sum())}
    summary["reviews"] = int(out["reviews"].sum())
    (NEW / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary["timefix"], indent=1))


if __name__ == "__main__":
    main()
