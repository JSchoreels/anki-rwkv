"""Deleted-card leak check (SM71): the dataset marks reviews of cards that were deleted before the export.

rwkv/data_processing.get_rwkv_data merges the revlogs with the export's cards table (card_id -> note_id, deck_id) and
decks table (deck_id -> preset_id). A card deleted before the export is missing from the cards table, so its reviews get
  note_id_is_nan = deck_id_is_nan = preset_id_is_nan = 1   (three input columns of every review row and query),
  note_id  = ID_PLACEHOLDER + card_id                     (a note of its own: no note state shared with siblings),
  deck_id  = preset_id = ID_PLACEHOLDER                   (ONE shared deck and preset lane for all deleted cards).
In live Anki a card is never deleted at review time, so all of this is future information.
(A card whose deck is missing from the decks table, 117 of 10.1 M reviews on SM70's 200 users, gets preset_id_is_nan
= 1 and the placeholder preset only.)

Modes for scoring (scripts/small_model_heldout.py --deleted-cards):
  keep      : today's inputs (default; the scorer does not call this module).
  flags     : the three *_is_nan flags set to 0 on every row; ids unchanged, so the random id codes are bit-identical
              and only the deleted-card rows' inputs change.
  neighbour : flags 0, and each deleted card is put in the deck of the existing card (cards table, any card of the
              user, reviewed or not) whose card_id (creation time, ms) is nearest; preset = that deck's preset (the
              placeholder if the deck is missing from the decks table). The note stays a note of its own (the card's
              real note and siblings cannot be restored; most notes have one card). Removing the shared placeholder
              deck changes the set of deck ids, which re-draws the random deck/preset/day codes of the whole user, i.e.
              a reseed of those codes; scripts/small_model_heldout.py --seed-offset measures that noise floor.
"""
import numpy as np


def card_tables(data, user):
    from rwkv.data_processing import read_user_metadata
    cards = read_user_metadata(data / "cards", user)
    decks = read_user_metadata(data / "decks", user)
    return cards, decks


def deleted_review_ths(data, user):
    """Sorted 1-based review_th of the user's reviews whose card is missing from the cards table."""
    import pandas as pd
    rev = pd.read_parquet(data / "revlogs" / f"user_id={user}", columns=["card_id"])
    cards, _ = card_tables(data, user)
    missing = ~rev["card_id"].isin(set(cards["card_id"].tolist())).to_numpy()
    return np.flatnonzero(missing) + 1


def neutralise(frame, mode, data, user):
    """The get_rwkv_data frame with the deleted-card signals replaced (see the module docstring); returns (frame,
    stats). Rows are matched by the note_id_is_nan flag, which get_rwkv_data sets exactly for missing cards."""
    from rwkv.data_processing import ID_PLACEHOLDER
    frame = frame.copy()
    deleted = frame["note_id_is_nan"].to_numpy() == 1
    stats = {"deleted_reviews": int(deleted.sum()), "deleted_cards": int(frame.loc[deleted, "card_id"].nunique()),
             "preset_only_reviews": int(((frame["preset_id_is_nan"] == 1) & ~deleted).sum())}
    for col in ("note_id_is_nan", "deck_id_is_nan", "preset_id_is_nan"):
        frame[col] = 0
    if mode == "flags":
        return frame, stats
    if mode != "neighbour":
        raise ValueError(mode)
    if deleted.any():
        cards, decks = card_tables(data, user)
        if len(cards):
            order = np.argsort(cards["card_id"].to_numpy(), kind="stable")
            ids = cards["card_id"].to_numpy()[order]
            deck_of = cards["deck_id"].to_numpy()[order]
            want = frame.loc[deleted, "card_id"].to_numpy()
            right = np.clip(np.searchsorted(ids, want), 0, len(ids) - 1)
            left = np.clip(right - 1, 0, len(ids) - 1)
            pick = np.where(np.abs(ids[left] - want) <= np.abs(ids[right] - want), left, right)
            new_deck = deck_of[pick]
            preset = dict(zip(decks["deck_id"].tolist(), decks["preset_id"].tolist()))
            new_preset = np.array([preset.get(d, ID_PLACEHOLDER) for d in new_deck.tolist()], dtype=np.int64)
            frame.loc[deleted, "deck_id"] = new_deck.astype(frame["deck_id"].dtype)
            frame.loc[deleted, "preset_id"] = new_preset.astype(frame["preset_id"].dtype)
            stats["neighbour_decks"] = int(len(set(new_deck.tolist())))
    return frame, stats
