"""Explain a selected review component using exact events and card/deck metadata.

Private output contains source identifiers and dates; do not publish it as data.
"""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance
from scripts.review_relations import _matches, canonical_events, compare_histories, load_events


def digest(path):
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def date(value):
    return datetime.fromtimestamp(int(value) / 1000, timezone.utc).isoformat()


def span(events):
    if not len(events):
        return None
    return {"first": date(events["revlog_id"].min()), "last": date(events["revlog_id"].max())}


def year_counts(events):
    years = events["revlog_id"].astype("datetime64[ms]").astype("datetime64[Y]").astype(int) + 1970
    values, counts = np.unique(years, return_counts=True)
    return {str(v): int(c) for v, c in zip(values, counts)}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--component", type=int, default=0)
    parser.add_argument("--reference", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    provenance = process_provenance()
    manifest = json.loads((args.audit / "input_manifest.json").read_text())
    component = json.loads(args.components.read_text())["components"][args.component]
    histories = {h["user_id"]: h for h in manifest["histories"]}
    users = component["users"]
    if args.reference not in users:
        raise ValueError("Reference must belong to component")
    events = {u: load_events(histories[u]) for u in users}
    # Each canonical history contains distinct tuples: counts are user ownership.
    union, counts = np.unique(np.concatenate(list(events.values())), return_counts=True)
    all_core = union[counts == len(users)]
    majority = union[counts >= int(np.ceil(.9 * len(users)))]
    repeated = union[counts >= 2]
    ref = events[args.reference]
    ref_cards = set(map(int, ref["card_id"]))
    sources = []
    review_tables = {}
    cards, decks = {}, {}
    for user in users:
        review_paths = [Path(s["path"]) for s in histories[user]["sources"]]
        tables = []
        for path, recorded in zip(review_paths, histories[user]["sources"]):
            actual = digest(path)
            if actual != recorded["sha256"]:
                raise ValueError(f"Review source changed: {path}")
            tables.append(pq.ParquetFile(path).read())
        review_tables[user] = pa.concat_tables(tables)
        rebuilt = canonical_events(review_tables[user], time_kind=manifest["time_kind"])
        if not np.array_equal(rebuilt, events[user]):
            raise ValueError(f"Canonical cache differs from source: {user}")
        for kind, target in (("cards", cards), ("decks", decks)):
            paths = sorted((Path(manifest["data_path"]) / kind / f"user_id={user}").glob("*.parquet"))
            tables = []
            for path in paths:
                sources.append({"path": str(path.resolve()), "sha256": digest(path)})
                tables.append(pq.ParquetFile(path).read())
            target[user] = pa.concat_tables(tables).to_pylist() if tables else []
    card_maps = {u: {r["card_id"]: r for r in rows} for u, rows in cards.items()}
    ref_meta = card_maps[args.reference]
    ref_card_counts = Counter(map(int, ref["card_id"]))
    per_user = []
    for user in users:
        arr = events[user]
        _, ref_hit = _matches(ref, arr)
        _, is_ref = _matches(arr, ref)
        non_reference_positions = np.flatnonzero(~is_ref)
        overlap = min(len(arr), len(ref))
        prefix_mismatches = np.flatnonzero(arr[:overlap] != ref[:overlap])
        _, is_repeated = _matches(arr, repeated)
        extras = arr[~is_ref]
        private = arr[~is_repeated]
        positions = np.searchsorted(union, arr)
        assert np.all(union[positions] == arr)
        old = arr[arr["revlog_id"] <= ref["revlog_id"][-1]]
        missing = ref[~ref_hit]
        missing_counts = Counter(map(int, missing["card_id"]))
        entirely_missing = {cid for cid, n in missing_counts.items() if n == ref_card_counts[cid]}
        raw = review_tables[user]
        raw_card = raw["card_id"].combine_chunks().to_numpy()
        raw_state = raw["state"].combine_chunks().to_numpy()
        raw_answer = raw["review_time"].combine_chunks().to_numpy() + raw["duration"].combine_chunks().to_numpy()
        later_learn_cards = set(map(int, raw_card[(raw_state == 0) & (raw_answer > ref["revlog_id"][-1])]))
        meta = card_maps[user]
        relevant = [r for cid, r in meta.items() if cid in ref_cards]
        by_deck = Counter(r["deck_id"] for r in relevant)
        by_review_deck = Counter(meta[int(cid)]["deck_id"] for cid in arr["card_id"] if int(cid) in ref_cards and int(cid) in meta)
        comparison = compare_histories(ref, arr)
        per_user.append({
            "user": user, "reviews": len(arr), "shared_reference": int(ref_hit.sum()),
            "first_event_is_reference_first": bool(len(arr) and arr[0] == ref[0]),
            "initial_reference_only_reviews": int(non_reference_positions[0]) if len(non_reference_positions) else len(arr),
            "initial_exact_reference_prefix_reviews": int(prefix_mismatches[0]) if len(prefix_mismatches) else overlap,
            "non_reference_before_first_reference": int(np.sum(extras["revlog_id"] < ref["revlog_id"][0])),
            "missing_reference": int((~ref_hit).sum()), "extra_reference": len(extras),
            "extra_before_or_during_reference": int(np.sum(extras["revlog_id"] <= ref["revlog_id"][-1])),
            "extra_after_reference": int(np.sum(extras["revlog_id"] > ref["revlog_id"][-1])),
            "extra_span": span(extras), "extra_years": year_counts(extras),
            "private_reviews": len(private), "private_span": span(private),
            "old_reviews": len(old), "old_non_reference": int(np.sum(~is_ref[:len(old)])),
            "reference_card_count": len(relevant), "total_cards": len(meta),
            "reference_cards_same_note": sum(r["note_id"] == ref_meta[r["card_id"]]["note_id"] for r in relevant if r["card_id"] in ref_meta),
            "reference_cards_same_deck": sum(r["deck_id"] == ref_meta[r["card_id"]]["deck_id"] for r in relevant if r["card_id"] in ref_meta),
            "reference_card_decks": dict(by_deck), "reference_review_decks": dict(by_review_deck),
            "extra_reviews_on_reference_cards": int(np.isin(extras["card_id"], list(ref_cards)).sum()),
            "missing_reference_cards": len(missing_counts),
            "entirely_missing_reference_cards": len(entirely_missing),
            "partially_missing_reference_cards": len(missing_counts) - len(entirely_missing),
            "missing_reference_cards_with_later_learning_start": len(set(missing_counts) & later_learn_cards),
            "later_learning_reference_cards_without_missing_old_reviews": len((later_learn_cards & ref_cards) - set(missing_counts)),
            "missing_reviews_on_cards_with_later_learning_start": sum(n for cid, n in missing_counts.items() if cid in later_learn_cards),
            "reference_comparison": comparison,
            "ownership_histogram": dict(Counter(map(int, counts[positions]))),
        })
    owner_hist = dict(Counter(map(int, counts)))
    # Inspect shared card identities beyond retained reference review rows.
    card_sets = [set(r["card_id"] for r in cards[u]) for u in users]
    common_cards = set.intersection(*card_sets)
    note_sets = [set(r["note_id"] for r in cards[u]) for u in users]
    common_notes = set.intersection(*note_sets)
    core_cards = set(map(int, all_core["card_id"]))
    per_card = []
    for cid in sorted(ref_cards):
        rows = [card_maps[u].get(cid) for u in users]
        present = [r for r in rows if r is not None]
        per_card.append({"card_id": cid, "users": len(present),
                         "distinct_note_ids": len(set(r["note_id"] for r in present)),
                         "distinct_deck_ids": len(set(r["deck_id"] for r in present)),
                         "reference_metadata": ref_meta.get(cid)})
    result = {"provenance": provenance, "input_manifest_sha256": digest(args.audit / "input_manifest.json"),
              "components_sha256": digest(args.components), "metadata_sources": sources,
              "filter": json.loads(args.components.read_text())["filter"],
              "groups": component["groups"], "users": users, "links": component["edge_count"],
              "event_identity": ["revlog_id", "card_id", "rating", "duration"],
              "canonical_inputs": [{k: histories[u][k] for k in ("user_id", "event_sha256", "reviews", "events_path", "sources")} for u in users],
              "union_distinct_events": len(union), "ownership_histogram": owner_hist,
              "shared_distinct_events": len(repeated), "shared_event_years": year_counts(repeated),
              "private_event_years": year_counts(union[counts == 1]),
              "shared_presentations": int(counts[counts >= 2].sum()),
              "all_user_core": {"reviews": len(all_core), "cards": len(core_cards), "span": span(all_core), "years": year_counts(all_core)},
              "majority_core": {"minimum_users": int(np.ceil(.9 * len(users))), "reviews": len(majority), "span": span(majority), "years": year_counts(majority)},
              "reference": {"user": args.reference, "reviews": len(ref), "cards": len(ref_cards), "span": span(ref), "years": year_counts(ref),
                            "rating_counts": dict(Counter(map(int, ref["rating"]))),
                            "duration_percentiles_ms": {str(p): float(np.percentile(ref["duration"], p)) for p in (0, 25, 50, 75, 100)}},
              "common_metadata_cards": len(common_cards), "common_metadata_notes": len(common_notes),
              "source_review_checks": "All selected source SHA256s agree with audit; canonical arrays rebuilt and exactly match caches.",
              "reference_cards_metadata": per_card, "decks": decks, "per_user": per_user}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("union_distinct_events", "ownership_histogram", "all_user_core", "majority_core", "reference", "common_metadata_cards", "common_metadata_notes")}, indent=2))
    print(json.dumps([{k: r[k] for k in ("user", "missing_reference", "missing_reference_cards", "partially_missing_reference_cards", "missing_reviews_on_cards_with_later_learning_start")} for r in per_user if r["missing_reference"]], indent=2))


if __name__ == "__main__":
    main()
