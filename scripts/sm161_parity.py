"""Three-way input parity for the fork's feature builder (SM161), CPU only.

Stage `build`: for each benchmark user, turn the dataset review log (anki-revlogs-10k) into a throwaway Anki
collection spec (synthetic millisecond revlog ids that reproduce every dataset `elapsed_seconds`, `day_offset` and the
review order exactly), written under work/ only. The fork's ignored Rust test `rwkv_sm161_input_parity_dump`
(fork branch parity-sm161, rslib/src/rwkv/parity_sm161.rs) then creates the collection with rslib, runs the fork's
own canonical history builder (`rwkv_historical_review_inputs`, the production request: deck-config first-review
policy, no dynamic presets), maps each row to the engine input exactly as qt/aqt/rwkv_history.review_input_from_row
does, and dumps `FeatureState::features_for` for every answer row and for an emulated show-time query before it.

Stage `compare`: the reference rows are the scorer's own samples with show-time inputs
(scripts/sm31_inputs.samples(data, user, "e2s-query"), full layout, the inputs the released models are scored on);
every one of the 24 card-feature columns plus the two day-offset inputs and the entity grouping is compared row by row.

  python scripts/sm161_parity.py build   --users 260 330 ...  [--variant present|export]
  (run the fork test, see artifacts/small-model/sm161/RESULT.md)
  python scripts/sm161_parity.py compare --users 260 330 ...  --out work/compare.json

Variants: `present` = cards deleted before the export are kept as cards (one shared placeholder deck and preset, a
note of their own), i.e. the collection as it was at review time; `export` = they are absent from the cards table, as
in the exported collection (their reviews drop out of the fork's history).
Nothing under work/ is ever committed (dataset-derived).
"""
import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path(r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011")
WORK = ROOT / "work"
ROLLOVER = 4
FEATURES = 92
DELETED_CARDS = "neighbour"   # the scorer's standard leak-free inputs: --inputs e2s-query --deleted-cards neighbour
DAY_PERIODS = (3.0, 7.0, 30.0, 100.0, 365.0, 3650.0, 36500.0)


def next_day_at(now=None):
    """Anki's next local rollover after now (v2/v3 timing, rollover hour 4), in seconds."""
    now = (now or datetime.now()).astimezone()
    nd = now.replace(hour=ROLLOVER, minute=0, second=0, microsecond=0)
    if nd <= now:
        nd += timedelta(days=1)
    return int(nd.timestamp())


def solve_times(rev, anchor):
    """Millisecond revlog ids T (strictly increasing in row order) with floor((T[k] - T[prev of card]) / 1000) ==
    elapsed_seconds and fixed 86,400 s days anchored at `anchor` reproducing day_offset (relative). Difference
    constraints solved for the least solution by alternating forward / backward relaxation."""
    n = len(rev)
    cards = rev["card_id"].to_numpy()
    es = rev["elapsed_seconds"].to_numpy().astype(np.int64)
    day = rev["day_offset"].to_numpy().astype(np.int64)
    prev = pd.Series(np.arange(n)).groupby(cards).shift(1).fillna(-1).to_numpy().astype(np.int64)
    dmax = int(day.max())
    base = anchor - (dmax + 2) * 86400        # start of dataset day 0; the last dataset day is yesterday
    lo = (base + day * 86400) * 1000
    hi = (base + (day + 1) * 86400) * 1000 - 1
    step_lo = es * 1000
    step_hi = es * 1000 + 999
    T = lo.copy()
    passes = 0
    while True:
        passes += 1
        changed = 0
        for r in range(n):                    # lower bounds: day start, order, card step
            v = T[r]
            if r and T[r - 1] + 1 > v:
                v = T[r - 1] + 1
            j = prev[r]
            if j >= 0 and T[j] + step_lo[r] > v:
                v = T[j] + step_lo[r]
            if v != T[r]:
                T[r] = v
                changed += 1
        for r in range(n - 1, -1, -1):        # a later answer pulls its card's previous answer up (step upper bound)
            j = prev[r]
            if j >= 0 and T[r] - step_hi[r] > T[j]:
                T[j] = T[r] - step_hi[r]
                changed += 1
        if (T > hi).any():
            raise RuntimeError(f"infeasible: {(T > hi).sum()} rows past their day after {passes} passes")
        if not changed:
            break
        if passes > 500:
            raise RuntimeError("relaxation did not converge")
    return T, passes, base


def build(user, variant):
    from rwkv.data_processing import ID_PLACEHOLDER, read_user_metadata
    rev = pd.read_parquet(DATA / "revlogs" / f"user_id={user}")
    cards = read_user_metadata(DATA / "cards", user)
    decks = read_user_metadata(DATA / "decks", user)
    anchor = next_day_at()
    ids, passes, base = solve_times(rev, anchor)
    assert (np.diff(ids) > 0).all()
    prev_id = pd.Series(ids).groupby(rev["card_id"].to_numpy()).shift(1)
    first = prev_id.isna().to_numpy()
    es_rebuilt = np.where(first, -1, (ids - prev_id.fillna(0).to_numpy().astype(np.int64)) // 1000)
    es_bad = int((es_rebuilt != rev["elapsed_seconds"].to_numpy()).sum())
    day_rebuilt = (ids // 1000 - base) // 86400
    day_bad = int((day_rebuilt != rev["day_offset"].to_numpy()).sum())

    card_meta = cards.set_index("card_id")
    known = set(card_meta.index)
    deck_preset = decks.set_index("deck_id")["preset_id"] if len(decks) else pd.Series(dtype=float)
    user_cards = rev["card_id"].drop_duplicates().to_numpy()
    deleted = [c for c in user_cards if c not in known]
    deck_keys, preset_keys = {}, {}
    neighbour = {}
    if deleted and variant == "present":
        from rwkv.data_processing import get_rwkv_data
        from scripts import sm71_leak
        frame, _ = sm71_leak.neutralise(get_rwkv_data(DATA, user), "neighbour", DATA, user)
        gone = frame[frame["card_id"].isin(set(deleted))].drop_duplicates("card_id")
        neighbour = {c: (d, p) for c, d, p in zip(gone["card_id"], gone["deck_id"], gone["preset_id"])}

    def deck_index(key, preset_key):
        if key not in deck_keys:
            if preset_key not in preset_keys:
                preset_keys[preset_key] = len(preset_keys)
            deck_keys[key] = (len(deck_keys), preset_keys[preset_key])
        return deck_keys[key][0]

    card_rows = []
    cid_of = {}
    note_ids = {}
    first_ms = pd.Series(ids).groupby(rev["card_id"].to_numpy()).min()
    for k, c in enumerate(user_cards):
        cid = 1_000_000_000_000 + k   # unique; the creation time only feeds new-card queries, which have no dataset row
        cid_of[c] = cid
        if c in known:
            m = card_meta.loc[c]
            d = int(m["deck_id"])
            p = deck_preset.get(d, np.nan)
            preset_key = f"p{int(p)}" if pd.notna(p) else "p-placeholder"
            di = deck_index(f"d{d}", preset_key)
            nk = f"n{int(m['note_id'])}"
        elif variant == "present":
            # the leak-free scorer's placement (scripts/sm71_leak.py neighbour): the deck of the existing card with
            # the nearest card_id, that deck's preset; a note of its own
            d, p = neighbour[c]
            preset_key = f"p{int(p)}" if p != ID_PLACEHOLDER else "p-placeholder"
            di = deck_index(f"d{int(d)}", preset_key)
            nk = f"deleted{c}"
        else:
            continue
        nid = note_ids.setdefault(nk, 1_000_000 + len(note_ids))
        card_rows.append([cid, nid, di])
    state = rev["state"].to_numpy()
    rtype = np.select([state == 0, state == 1, state == 2, state == 3, state == 4], [0, 0, 1, 2, 3], -1)
    assert (rtype >= 0).all()
    revlog = [[int(i), cid_of[c], int(e), int(d), int(ty), 1, 2500]
              for i, c, e, d, ty in zip(ids, rev["card_id"].to_numpy(), rev["rating"].to_numpy(),
                                         rev["duration"].to_numpy(), rtype)]
    folder = WORK / f"u{user}-{variant}"
    folder.mkdir(parents=True, exist_ok=True)
    col = folder / "collection.anki2"
    for path in folder.glob("collection.anki2*"):
        path.unlink()
    out = folder / "fork"
    spec = {"collection": str(col), "output": str(out), "next_day_at": anchor, "crt": int(base - 30 * 86400),
            "decks": [[key, v[1]] for key, v in sorted(deck_keys.items(), key=lambda kv: kv[1][0])],
            "presets": len(preset_keys), "cards": card_rows, "revlog": revlog}
    (folder / "spec.json").write_text(json.dumps(spec))
    np.save(folder / "review_ids.npy", ids)
    info = {"user": user, "variant": variant, "reviews": len(rev), "cards": len(user_cards), "deleted_cards": len(deleted),
            "deleted_reviews": int(rev["card_id"].isin(set(deleted)).sum()), "decks": len(deck_keys),
            "presets": len(preset_keys), "relax_passes": passes, "construction_elapsed_mismatch": es_bad,
            "construction_day_mismatch": day_bad, "anchor": anchor}
    (folder / "build.json").write_text(json.dumps(info))
    return info


def reference(user):
    """The scorer's e2s-query sample (full layout) as numpy arrays keyed by (review_th, is_query)."""
    import torch
    from scripts import sm31_inputs
    torch.manual_seed(0)
    samples = sm31_inputs.samples(DATA, user, "e2s-query", deleted_cards=DELETED_CARDS)
    assert len(samples) == 1
    s = samples[0]
    return {"x": s.card_features.numpy().astype(np.float64), "th": s.review_ths.numpy().astype(np.int64),
            "day": s.day_offsets.numpy().astype(np.int64), "day_first": s.day_offsets_first.numpy().astype(np.int64),
            "ids": {k: v.numpy().astype(np.int64) for k, v in s.ids.items()}}


COLUMNS = ["scaled_elapsed_days", "scaled_elapsed_days_cumulative", "scaled_elapsed_seconds", "elapsed_seconds_sin",
           "elapsed_seconds_cos", "scaled_elapsed_seconds_cumulative", "elapsed_seconds_cumulative_sin",
           "elapsed_seconds_cumulative_cos", "scaled_duration", "rating_1", "rating_2", "rating_3", "rating_4",
           "note_id_is_nan", "deck_id_is_nan", "preset_id_is_nan", "day_offset_diff", "day_of_week", "diff_new_cards",
           "diff_reviews", "cum_new_cards_today", "cum_reviews_today", "scaled_state", "is_query"]


def day_encoding(day, first):
    cols = []
    for period in DAY_PERIODS:
        ph = 2 * np.pi / period
        d, f = np.mod(day, period), np.mod(first, period)
        cols += [np.sin(ph * d), np.cos(ph * d), np.sin(ph * f), np.cos(ph * f)]
    return np.stack(cols, 1)


def grouping_ok(a, b):
    """True when the two id columns induce the same partition of the rows (a bijection)."""
    pairs = pd.DataFrame({"a": a, "b": b}).drop_duplicates()
    return pairs["a"].is_unique and pairs["b"].is_unique


KINDS = ((0, "answer_canonical_replay"), (1, "query_show_time_reviewer_rule"), (2, "query_show_time_batch_rule"),
         (3, "answer_live_rule"))


def compare(user, variant, tol):
    """Row-by-row comparison. A row whose features differ is attributed to a raw-input difference (seconds off by one,
    a different interval rule, a different day, the deleted-card flags); a differing row with identical raw inputs is
    `unexplained`, i.e. a drift of the feature builder itself."""
    from rwkv.data_processing import get_rwkv_data
    folder = WORK / f"u{user}-{variant}"
    meta = json.loads((folder / "fork" / "rows.json").read_text())
    feats = np.fromfile(folder / "fork" / "features.f32", dtype="<f4").reshape(-1, FEATURES).astype(np.float64)
    rows = pd.DataFrame(meta["rows"], columns=meta["columns"])
    ids = np.load(folder / "review_ids.npy")
    th_of_id = pd.Series(np.arange(1, len(ids) + 1), index=ids)
    rows["th"] = th_of_id.reindex(rows["review_id"].to_numpy()).to_numpy().astype(np.int64)
    ref = reference(user)
    from scripts import sm71_leak
    frame, _ = sm71_leak.neutralise(get_rwkv_data(DATA, user), DELETED_CARDS, DATA, user)
    frame = frame.set_index("review_th")
    is_q = ref["x"][:, 23] == 1
    key = pd.Series(np.arange(len(ref["th"])), index=pd.MultiIndex.from_arrays([ref["th"], is_q.astype(int)]))
    answer_rows = rows[rows["kind"] == 0]
    fork_day0 = answer_rows["day_offset"].min()
    fork_answer_day = pd.Series(answer_rows["day_offset"].to_numpy() - fork_day0, index=answer_rows["th"].to_numpy())
    result = {"user": user, "variant": variant, "reviews": int(len(frame)), "fork_history_reviews": int(len(answer_rows)),
              "reference_answer_rows": int((~is_q).sum()), "reference_query_rows": int(is_q.sum()),
              "reference_answer_rows_without_fork_row":
                  int(len(set(ref["th"][~is_q].tolist()) - set(answer_rows["th"].tolist()))), "kinds": {}}
    for kind, name in KINDS:
        sel = (rows["kind"] == kind).to_numpy()
        if not sel.any():
            continue
        sub = rows[sel]
        want_q = 1 if kind in (1, 2) else 0
        idx = key.reindex(pd.MultiIndex.from_arrays([sub["th"].to_numpy(), np.full(len(sub), want_q)]))
        ok = idx.notna().to_numpy()
        r = idx.to_numpy()[ok].astype(np.int64)
        sub = sub[ok]
        th = sub["th"].to_numpy()
        f = feats[sel][ok]
        x = ref["x"][r]
        diff = np.abs(f[:, :24] - x)
        enc_diff = np.abs(f[:, FEATURES - 28:] - day_encoding(ref["day"][r], ref["day_first"][r]))
        bad_col = diff > tol
        any_bad = bad_col.any(1) | (enc_diff > tol).any(1)
        # raw inputs on both sides
        fr = frame.loc[th]
        ref_es = fr["elapsed_seconds"].to_numpy()
        if want_q:
            ref_es = np.floor(np.maximum(ref_es - fr["duration"].to_numpy() / 1000.0, 0)).astype(np.int64)
        ref_ed = fr["elapsed_days"].to_numpy()
        f_es = sub["elapsed_seconds"].to_numpy().astype(np.int64)
        f_ed_raw = sub["elapsed_days"].to_numpy().astype(object)
        if not want_q:   # the engine's creation_elapsed_is_query_only: an answered learning start has no interval
            start = sub["card_type"].to_numpy() == 0
            f_es = np.where(start, -1, f_es)
            f_ed_raw[start] = -1
        f_ed = np.array([e if e is not None and not (isinstance(e, float) and np.isnan(e)) else (s // 86400 if s >= 0 else -1)
                         for e, s in zip(f_ed_raw, f_es)], dtype=np.int64)
        f_day = sub["day_offset"].to_numpy().astype(np.int64) - fork_day0
        d_sec = f_es - ref_es
        day_moved = f_day != fork_answer_day.reindex(th).to_numpy()
        deleted = x[:, 13] == 1
        causes = {
            "deleted_card_flags (cards present at review time)": deleted & bad_col[:, 13:16].any(1),
            "show_time_day_differs_from_answer_day": day_moved,
            "elapsed_seconds_rounding (1 or 2 s)": (np.abs(d_sec) >= 1) & (np.abs(d_sec) <= 2),
            "elapsed_seconds_rule (diff > 2 s)": np.abs(d_sec) > 2,
            "elapsed_days_rule": f_ed != ref_ed,
        }
        explained = np.zeros(len(sub), bool)
        cause_counts = {}
        for cname, mask in causes.items():
            hit = any_bad & mask
            cause_counts[cname] = int(hit.sum())
            explained |= hit
        # a deleted-card row whose only difference is the flags is fully explained; any other column must be too
        flags_only = deleted & ~np.delete(bad_col, [13, 14, 15], 1).any(1) & ~(enc_diff > tol).any(1)
        unexplained = any_bad & ~(explained & (~deleted | flags_only | (np.abs(d_sec) > 0) | day_moved | (f_ed != ref_ed)))
        clean_inputs = (d_sec == 0) & ~day_moved & (f_ed == ref_ed) & ~deleted
        cols = {}
        for j, cname in enumerate(COLUMNS):
            cols[cname] = {"max_abs_diff": float(diff[:, j].max()), "mismatch_rows": int(bad_col[:, j].sum()),
                           "max_abs_diff_same_raw_inputs": float(diff[clean_inputs, j].max()) if clean_inputs.any() else 0.0}
        cols["day_offset_inputs(28 encodings)"] = {
            "max_abs_diff": float(enc_diff.max()), "mismatch_rows": int((enc_diff > tol).any(1).sum()),
            "max_abs_diff_same_raw_inputs": float(enc_diff[clean_inputs].max()) if clean_inputs.any() else 0.0}
        group = {}
        for ent in ("card_id", "note_id", "deck_id", "preset_id"):
            group[ent] = bool(grouping_ok(sub[ent].to_numpy(), fr[ent].to_numpy()))
        state = fr["state"].to_numpy()
        result["kinds"][name] = {
            "rows": int(len(sub)), "unmatched_rows": int((~ok).sum()), "mismatching_rows": int(any_bad.sum()),
            "unexplained_rows": int(unexplained.sum()), "rows_with_identical_raw_inputs": int(clean_inputs.sum()),
            "mismatch_causes": cause_counts, "columns": cols, "grouping_bijective_vs_frame_ids": group,
            "mismatch_by_state": {int(k): int(v) for k, v in zip(*np.unique(state[any_bad], return_counts=True))},
            "seconds_diff_hist": {int(k): int(v) for k, v in zip(*np.unique(np.clip(d_sec[any_bad], -3, 3), return_counts=True))},
            "_unexplained_th": th[unexplained][:20].tolist()}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "compare"])
    ap.add_argument("--users", type=int, nargs="+", required=True)
    ap.add_argument("--variant", default="present", choices=["present", "export"])
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = []
    for user in args.users:
        started = time.time()
        res = build(user, args.variant) if args.stage == "build" else compare(user, args.variant, args.tol)
        res["seconds"] = round(time.time() - started, 1)
        print(json.dumps({k: v for k, v in res.items() if k != "kinds"}), flush=True)
        out.append(res)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
