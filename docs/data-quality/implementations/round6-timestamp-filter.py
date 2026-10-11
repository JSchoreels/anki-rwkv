"""Exact history containment and uninterrupted review blocks; CPU only.

Event identity is (revlog_id, card_id, rating, duration). Canonical histories
contain distinct events ordered by that tuple, with revlog_id first. Hashes
only discover candidates: every reported relation is verified on all fields.
"""

from argparse import ArgumentParser
from collections import defaultdict
from contextlib import ExitStack, closing
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance

EVENT_DTYPE = np.dtype([(name, "<i8") for name in ("revlog_id", "card_id", "rating", "duration")])
POSTING_DTYPE = np.dtype([("hash", "<u8"), ("owner", "<u4")])
CACHE_VERSION = 1
BATCH_ROWS = 65_536


def _canonical_array(events):
    """Sort integer answer IDs first; resolve full tuples only at timestamp ties."""
    if len(events) < 2:
        return events
    timestamps = events["revlog_id"]
    if np.any(timestamps[1:] < timestamps[:-1]):
        events = events[np.argsort(timestamps, kind="stable")]
        timestamps = events["revlog_id"]
    ties = timestamps[1:] == timestamps[:-1]
    if not ties.any():
        return events
    tied_rows = np.r_[False, ties] | np.r_[ties, False]
    events[tied_rows] = np.sort(events[tied_rows])
    return events[np.r_[True, events[1:] != events[:-1]]]


def canonical_events(table, time_kind="show"):
    """Normalize an Arrow table without narrowing IDs or using derived labels."""
    if time_kind not in ("show", "answer"):
        raise ValueError("time_kind must be show or answer")
    clock = next((name for name in ("revlog_id", "id", "review_time") if name in table.column_names), None)
    if clock is None:
        raise ValueError("Timestamp dataset required: missing revlog_id, id, and review_time")
    columns = {}
    for name in (clock, "card_id", "rating", "duration"):
        column = table[name]
        if column.null_count or not pa.types.is_integer(column.type):
            raise ValueError(f"{name} must contain non-null integers")
        values = column.combine_chunks().to_numpy(zero_copy_only=False)
        if len(values) and (int(values.min()) < np.iinfo(np.int64).min or int(values.max()) > np.iinfo(np.int64).max):
            raise ValueError(f"{name} is outside signed int64")
        columns[name] = values.astype(np.int64, copy=False)
    duration = columns["duration"]
    if np.any(duration < 0):
        raise ValueError("Negative answer duration")
    timestamp = columns[clock]
    if clock == "review_time" and time_kind == "show":
        if np.any(timestamp > np.iinfo(np.int64).max - duration):
            raise ValueError("review_time + duration overflows int64")
        timestamp = timestamp + duration
    events = np.empty(len(table), dtype=EVENT_DTYPE)
    events["revlog_id"] = timestamp
    for name in ("card_id", "rating", "duration"):
        events[name] = columns[name]
    return _canonical_array(events)


def _matches(query, target, timestamps=None):
    """Check primary timestamp equality before fetching complete event tuples."""
    if timestamps is None:
        timestamps = target["revlog_id"]
    positions = np.searchsorted(timestamps, query["revlog_id"])
    hit = np.zeros(len(query), dtype=bool)
    indexes = np.flatnonzero(positions < len(target))
    indexes = indexes[timestamps[positions[indexes]] == query["revlog_id"][indexes]]
    if len(target) > 1:
        within = indexes[positions[indexes] < len(target) - 1]
        tied = within[timestamps[positions[within]] == timestamps[positions[within] + 1]]
        if len(tied):
            positions[tied] = np.searchsorted(target, query[tied])
    valid = indexes[positions[indexes] < len(target)]
    hit[valid] = target[positions[valid]] == query[valid]
    return positions, hit


def is_history_subset(left, right, batch_rows=BATCH_ROWS):
    """Return left <= right for canonical arrays, including the empty set.

    O(len(left) * log(len(right))) time, O(batch_rows) temporary memory.
    Stops at the first missing event. Use canonical_events() on raw tables.
    """
    if batch_rows < 1:
        raise ValueError("batch_rows must be positive")
    if len(left) > len(right):
        return False
    for start in range(0, len(left), batch_rows):
        _, hit = _matches(left[start:start + batch_rows], right)
        if not hit.all():
            return False
    return True


def compare_histories(left, right, min_segment=20, batch_rows=BATCH_ROWS):
    """Count exact shared events and every maximal uninterrupted matching block.

    Block coordinates are zero-based, half-open indexes in canonical histories.
    Matching blocks require consecutive positions in BOTH histories.
    """
    if min_segment < 1 or batch_rows < 1:
        raise ValueError("min_segment and batch_rows must be positive")
    swapped = len(left) > len(right)
    query, target = (right, left) if swapped else (left, right)
    query_hits, target_hits = [], []
    target_timestamps = np.ascontiguousarray(target["revlog_id"])
    for start in range(0, len(query), batch_rows):
        positions, hit = _matches(query[start:start + batch_rows], target, target_timestamps)
        query_hits.append(np.flatnonzero(hit) + start)
        target_hits.append(positions[hit])
    q = np.concatenate(query_hits) if query_hits else np.empty(0, dtype=np.int64)
    t = np.concatenate(target_hits) if target_hits else np.empty(0, dtype=np.int64)
    a, b = (t, q) if swapped else (q, t)
    shared = len(a)
    segments, longest = [], 0
    if shared:
        cuts = np.r_[0, np.flatnonzero((np.diff(a) != 1) | (np.diff(b) != 1)) + 1, shared]
        lengths = np.diff(cuts)
        longest = int(lengths.max())
        for index in np.flatnonzero(lengths >= min_segment):
            lo, hi = int(cuts[index]), int(cuts[index + 1])
            a_start, b_start, length = int(a[lo]), int(b[lo]), hi - lo
            segments.append({"a_start": a_start, "a_end": a_start + length,
                             "b_start": b_start, "b_end": b_start + length, "reviews": length,
                             "first_revlog_id": str(int(left[a_start]["revlog_id"])),
                             "last_revlog_id": str(int(left[int(a[hi - 1])]["revlog_id"]))})
    return {"shared_reviews": shared, "a_reviews": len(left), "b_reviews": len(right),
            "a_is_subset": shared == len(left), "b_is_subset": shared == len(right),
            "coverage_a": shared / len(left) if len(left) else None,
            "coverage_b": shared / len(right) if len(right) else None,
            "longest_segment": longest, "segments": segments}


def _mix_inplace(values, temporary):
    np.right_shift(values, np.uint64(30), out=temporary)
    np.bitwise_xor(values, temporary, out=values)
    values *= np.uint64(0xBF58476D1CE4E5B9)
    np.right_shift(values, np.uint64(27), out=temporary)
    np.bitwise_xor(values, temporary, out=values)
    values *= np.uint64(0x94D049BB133111EB)
    np.right_shift(values, np.uint64(31), out=temporary)
    np.bitwise_xor(values, temporary, out=values)
    return values


def _mix(values):
    values = np.array(values, dtype=np.uint64, copy=True)
    return _mix_inplace(values, np.empty_like(values))


def event_hashes(events):
    """Fast deterministic uint64 candidate hashes; collisions are harmless."""
    hashes = np.full(len(events), np.uint64(0x9E3779B97F4A7C15), dtype=np.uint64)
    temporary = np.empty_like(hashes)
    for name in EVENT_DTYPE.names:
        np.bitwise_xor(hashes, events[name].view(np.uint64), out=hashes)
        _mix_inplace(hashes, temporary)
    return hashes


def window_minimizers(hashes, window):
    """O(n) sliding-window minima using NumPy prefix/suffix block minima.

    Any identical window selects the same minimum hash in both histories.
    Store unique selected hashes: collisions only add candidates.
    """
    if window < 1:
        raise ValueError("window must be positive")
    n = len(hashes)
    if n < window:
        return np.empty(0, dtype=np.uint64)
    padding = (-n) % window
    padded = np.pad(hashes, (0, padding), constant_values=np.iinfo(np.uint64).max).reshape(-1, window)
    prefix = np.minimum.accumulate(padded, axis=1).ravel()
    suffix = np.minimum.accumulate(padded[:, ::-1], axis=1)[:, ::-1].ravel()
    minima = np.minimum(suffix[:n - window + 1], prefix[window - 1:n])
    # Adjacent windows usually select the same event. Drop these repeats before
    # sorting the remaining hashes; the selected set and recall are unchanged.
    return np.unique(minima[np.r_[True, minima[1:] != minima[:-1]]])


def _hash_file(path):
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stamp(path):
    value = path.stat()
    return [value.st_size, value.st_mtime_ns, value.st_ino]


def prepare_history(user, files, cache, time_kind="show"):
    """Checksum source files; reuse a canonical on-disk cache only on a match."""
    sources = []
    for path in files:
        stamp = _stamp(path)
        sources.append({"path": str(path.resolve()), "stamp": stamp, "sha256": _hash_file(path)})
        if _stamp(path) != stamp:
            raise RuntimeError(f"Input changed while reading: {path}")
    cache_key = sha256(json.dumps({"version": CACHE_VERSION, "time_kind": time_kind,
                                  "sources": sources}, sort_keys=True).encode()).hexdigest()
    folder = cache / cache_key
    metadata_path = folder / "metadata.json"
    if metadata_path.is_file() and (folder / "events.npy").is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return {**metadata, "user_id": user, "cache_hit": True, "sources": sources}
    arrays, rows = [], 0
    for path in files:
        with pq.ParquetFile(path) as parquet:
            names = parquet.schema_arrow.names
            clock = next((name for name in ("revlog_id", "id", "review_time") if name in names), None)
            if clock is None:
                raise ValueError(f"{path}: missing absolute review timestamp; use the timestamp dataset")
            table = parquet.read(columns=[clock, "card_id", "rating", "duration"], use_threads=False)
        rows += len(table)
        arrays.append(canonical_events(table, time_kind))
    events = _canonical_array(np.concatenate(arrays)) if len(arrays) > 1 else arrays[0]
    for source in sources:
        if _stamp(Path(source["path"])) != source["stamp"]:
            raise RuntimeError(f"Input changed while preparing: {source['path']}")
    identity_repeats = ((events["revlog_id"][1:] == events["revlog_id"][:-1]) &
                        (events["card_id"][1:] == events["card_id"][:-1]))
    metadata = {"user_id": user, "rows": rows, "reviews": len(events),
                "duplicate_rows": rows - len(events), "extra_identity_payloads": int(identity_repeats.sum()),
                "first_revlog_id": int(events[0]["revlog_id"]) if len(events) else None,
                "last_revlog_id": int(events[-1]["revlog_id"]) if len(events) else None,
                "event_sha256": sha256(memoryview(events)).hexdigest(),
                "events_path": str((folder / "events.npy").resolve()), "sources": sources}
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / f"events-{uuid4().hex}.npy"
    np.save(temporary, events, allow_pickle=False)
    temporary.replace(folder / "events.npy")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {**metadata, "cache_hit": False}


def load_events(history):
    return np.load(history["events_path"], mmap_mode="r", allow_pickle=False)


def _equal(left, right):
    return len(left) == len(right) and all(np.array_equal(left[start:start + BATCH_ROWS], right[start:start + BATCH_ROWS])
                                          for start in range(0, len(left), BATCH_ROWS))


def equal_groups(histories):
    """Digests shortlist equality; verify original tuples before collapsing."""
    buckets, groups = defaultdict(list), []
    for history in histories:
        bucket = buckets[(history["reviews"], history["event_sha256"])]
        for ordinal in bucket:
            if _equal(load_events(history), load_events(groups[ordinal])):
                groups[ordinal]["users"].append(history["user_id"])
                break
        else:
            bucket.append(len(groups))
            groups.append({**history, "id": history["user_id"], "users": [history["user_id"]]})
    return groups


def _add_candidates(db, pairs, flag):
    db.executemany(f"INSERT INTO candidates(a,b,{flag}) VALUES(?,?,1) "
                   f"ON CONFLICT(a,b) DO UPDATE SET {flag}=1", pairs)


def subset_candidates(groups, db, anchor_count=3):
    selections = []
    for group in groups:
        events = load_events(group)
        count = min(anchor_count, len(events))
        chosen = events[np.argpartition(event_hashes(events), count - 1)[:count]] if count else events[:0]
        selections.append(np.array(chosen))
    anchors = np.unique(np.concatenate(selections)) if selections else np.empty(0, dtype=EVENT_DTYPE)
    postings = [0] * len(anchors)
    for owner, group in enumerate(groups):
        _, hit = _matches(anchors, load_events(group))
        for index in np.flatnonzero(hit):
            postings[int(index)] |= 1 << owner
    for owner, chosen in enumerate(selections):
        if not len(chosen):
            continue  # Do not fill the graph with vacuous empty-set relations.
        owners = (1 << len(groups)) - 1
        for index in np.searchsorted(anchors, chosen):
            owners &= postings[int(index)]
        owners &= ~(1 << owner)
        batch = []
        while owners:
            bit = owners & -owners
            other = bit.bit_length() - 1
            owners ^= bit
            a, b = groups[owner], groups[other]
            if b["reviews"] >= a["reviews"] and b["first_revlog_id"] <= a["first_revlog_id"] and b["last_revlog_id"] >= a["last_revlog_id"]:
                batch.append(tuple(sorted((owner, other))))
            if len(batch) >= 10_000:
                _add_candidates(db, batch, "subset_candidate")
                batch.clear()
        _add_candidates(db, batch, "subset_candidate")
    db.commit()


def segment_candidates(groups, db, work, min_segment=20, partitions=256, index_memory_mib=256):
    """Sparse winnowing index on disk, with an enforced per-shard read limit."""
    if partitions < 1 or partitions & (partitions - 1) or index_memory_mib < 1:
        raise ValueError("partitions must be a power of two; index_memory_mib must be positive")
    paths = {}
    with ExitStack() as handles:
        opened = {}
        for owner, group in enumerate(groups):
            selected = window_minimizers(event_hashes(load_events(group)), min_segment)
            codes = _mix(selected) & np.uint64(partitions - 1)
            order = np.argsort(codes, kind="stable")
            codes, selected = codes[order], selected[order]
            cuts = np.r_[0, np.flatnonzero(np.diff(codes) != 0) + 1, len(selected)]
            for lo, hi in zip(cuts[:-1], cuts[1:]):
                if lo == hi:
                    continue
                shard = int(codes[lo])
                if shard not in opened:
                    paths[shard] = work / f"postings-{shard}.bin"
                    opened[shard] = handles.enter_context(paths[shard].open("wb"))
                records = np.empty(hi - lo, dtype=POSTING_DTYPE)
                records["hash"], records["owner"] = selected[lo:hi], owner
                opened[shard].write(memoryview(records))
    postings_count, pairs = 0, []
    # Leave headroom for sort copies and group-boundary arrays.
    max_bytes = index_memory_mib * 2**20 // 4
    for path in paths.values():
        if path.stat().st_size > max_bytes:
            raise MemoryError("Index shard exceeds memory budget; increase --partitions or --index-memory-mib")
        records = np.fromfile(path, dtype=POSTING_DTYPE)
        postings_count += len(records)
        # Writers append owners in ascending order. Stable integer hash sorting
        # consequently orders both columns, without a slower structured sort.
        records = records[np.argsort(records["hash"], kind="stable")]
        cuts = np.r_[0, np.flatnonzero(records["hash"][1:] != records["hash"][:-1]) + 1, len(records)]
        lengths = np.diff(cuts)
        # Most hashes have only one owner and cannot connect users. Handle the
        # common two-owner case in arrays, deduplicating pairs within each shard.
        starts = cuts[:-1][lengths == 2]
        packed = (records["owner"][starts].astype(np.uint64) << np.uint64(32)) | records["owner"][starts + 1]
        packed = np.unique(packed)
        _add_candidates(db, ((int(key >> np.uint64(32)), int(key & np.uint64(0xFFFFFFFF))) for key in packed), "segment_candidate")
        for index in np.flatnonzero(lengths > 2):
            lo, hi = int(cuts[index]), int(cuts[index + 1])
            for a, b in combinations(records["owner"][lo:hi], 2):
                pairs.append((int(a), int(b)))
                if len(pairs) >= 10_000:
                    _add_candidates(db, pairs, "segment_candidate")
                    pairs.clear()
        _add_candidates(db, pairs, "segment_candidate")
        pairs.clear()
        db.commit()
    return postings_count


def find_relations(histories, output, min_segment=20, partitions=256, index_memory_mib=256, progress=None):
    """Build complete exact-containment and >=min_segment block relations.

    Histories point to sorted NumPy caches. SQLite stores candidates and verified
    relations on disk; the function returns small summary counts, not all edges.
    """
    if min_segment < 1:
        raise ValueError("min_segment must be positive")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    database = output / "relations.sqlite"
    if database.exists():
        raise FileExistsError(f"Use a fresh output directory: {database}")
    groups = equal_groups(histories)
    if progress:
        progress(f"{len(histories)} histories -> {len(groups)} distinct history groups; finding candidates")
    with closing(sqlite3.connect(database)) as db:
        db.execute("PRAGMA temp_store=FILE")
        db.execute("PRAGMA cache_size=-16384")
        db.execute("CREATE TABLE candidates(a INTEGER,b INTEGER,subset_candidate INTEGER DEFAULT 0,segment_candidate INTEGER DEFAULT 0,PRIMARY KEY(a,b)) WITHOUT ROWID")
        db.execute("CREATE TABLE relations(a INTEGER,b INTEGER,kind TEXT,subset_id INTEGER,superset_id INTEGER,shared_reviews INTEGER,coverage_a REAL,coverage_b REAL,longest_segment INTEGER,segments TEXT)")
        subset_candidates(groups, db)
        with TemporaryDirectory(prefix="segment-index-", dir=output) as folder:
            work = Path(folder).resolve()
            if not work.is_relative_to(output):
                raise ValueError("Index workspace must remain inside output")
            postings_count = segment_candidates(groups, db, work, min_segment, partitions, index_memory_mib)
        candidates = db.execute("SELECT count(*) FROM candidates").fetchone()[0]
        if progress:
            progress(f"{candidates:,} candidate pairs; verifying original review tuples")
        related_ids, counts = set(), defaultdict(int)
        with (output / "relations.jsonl").open("w", encoding="utf-8") as stream:
            for a, b in db.execute("SELECT a,b FROM candidates ORDER BY a,b"):
                left, right = groups[a], groups[b]
                comparison = compare_histories(load_events(left), load_events(right), min_segment)
                subset_id = superset_id = None
                if comparison["a_is_subset"] or comparison["b_is_subset"]:
                    smaller, larger = (left, right) if comparison["a_is_subset"] else (right, left)
                    subset_id, superset_id = smaller["id"], larger["id"]
                    kind = "subset"
                elif comparison["longest_segment"] >= min_segment:
                    kind = "common_segment"
                else:
                    continue
                relation = {"a": left["id"], "b": right["id"], "kind": kind,
                            "subset_id": subset_id, "superset_id": superset_id, **comparison}
                stream.write(json.dumps(relation) + "\n")
                db.execute("INSERT INTO relations VALUES(?,?,?,?,?,?,?,?,?,?)",
                           (left["id"], right["id"], kind, subset_id, superset_id,
                            comparison["shared_reviews"], comparison["coverage_a"], comparison["coverage_b"],
                            comparison["longest_segment"], json.dumps(comparison["segments"])))
                related_ids.update((left["id"], right["id"]))
                counts[kind] += 1
            db.commit()
        nodes = [{"id": g["id"], "users": g["users"], "reviews": g["reviews"],
                  "rows": g["rows"], "first_revlog_id": str(g["first_revlog_id"]) if g["reviews"] else None,
                  "last_revlog_id": str(g["last_revlog_id"]) if g["reviews"] else None}
                 for g in groups if g["id"] in related_ids or (g["reviews"] and len(g["users"]) > 1)]
    (output / "nodes.json").write_text(json.dumps(nodes, indent=2) + "\n", encoding="utf-8")
    return {"users": len(histories), "distinct_groups": len(groups),
            "empty_users": [h["user_id"] for h in histories if not h["reviews"]],
            "equal_groups": sum(g["reviews"] > 0 and len(g["users"]) > 1 for g in groups),
            "subset_edges": counts["subset"], "common_segment_edges": counts["common_segment"],
            "related_groups": len(nodes), "candidate_pairs": candidates, "minimizer_postings": postings_count,
            "min_segment": min_segment, "possible_group_pairs": len(groups) * (len(groups) - 1) // 2}


def write_graph(output, summary, edge_limit=5000):
    output = Path(output)
    with closing(sqlite3.connect(output / "relations.sqlite")) as db:
        edges = [{"a": a, "b": b, "kind": kind, "subset_id": sub, "superset_id": sup,
                  "shared_reviews": shared, "coverage_a": ca, "coverage_b": cb,
                  "longest_segment": longest, "segments": json.loads(segments)}
                 for a, b, kind, sub, sup, shared, ca, cb, longest, segments in db.execute(
                     "SELECT * FROM relations ORDER BY longest_segment DESC,shared_reviews DESC,a,b LIMIT ?", (edge_limit,))]
    graph = {"summary": summary, "nodes": json.loads((output / "nodes.json").read_text()), "edges": edges,
             "edge_limit": edge_limit, "edges_truncated": len(edges) < summary["subset_edges"] + summary["common_segment_edges"]}
    (output / "graph.json").write_text(json.dumps(graph) + "\n", encoding="utf-8")
    encoded = json.dumps(graph).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    template = (Path(__file__).with_name("review_relations_graph.html")).read_text(encoding="utf-8")
    vendor = Path(__file__).parent / "vendor"
    license_text = (vendor / "D3-LICENSE.txt").read_text(encoding="utf-8")
    d3 = ("/*\n" + license_text + "\n*/\n" + (vendor / "d3-7.9.0.min.js").read_text(encoding="utf-8")).replace("</script", "<\\/script")
    (output / "graph.html").write_text(template.replace("__REVIEW_RELATIONS_DATA__", encoded).replace("__D3_SOURCE__", d3), encoding="utf-8")
    (output / "D3-LICENSE.txt").write_bytes((vendor / "D3-LICENSE.txt").read_bytes())


def run(args):
    started = perf_counter()
    data, output, cache = args.data.resolve(), args.output.resolve(), args.cache.resolve()
    if output.is_relative_to(data) or cache.is_relative_to(data):
        raise ValueError("Output/cache must be outside the input dataset")
    if (output / "summary.json").exists() or (output / "relations.sqlite").exists():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    available = {}
    for partition in (data / "revlogs").glob("user_id=*"):
        if partition.is_dir() and partition.name[8:].isdigit():
            files = sorted(partition.glob("*.parquet"))
            if files:
                available[int(partition.name[8:])] = files
    users = sorted(available)
    if args.users:
        users = sorted(set(int(x) for x in args.users.split(",")))
        missing = set(users) - available.keys()
        if missing:
            raise ValueError(f"Missing review partitions: {sorted(missing)}")
    if args.limit_users is not None:
        if args.limit_users < 1:
            raise ValueError("limit_users must be positive")
        users = users[:args.limit_users]
    if not users:
        raise ValueError(f"No review histories in {data / 'revlogs'}")
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    provenance = process_provenance()
    manifest_file = data / "download_manifest.json"
    manifest = json.loads(manifest_file.read_text()) if manifest_file.is_file() else None
    log = lambda message: print(message, flush=True)
    histories = []
    for index, user in enumerate(users):
        histories.append(prepare_history(user, available[user], cache, args.time_kind))
        if index == 0 or (index + 1) % 20 == 0 or index + 1 == len(users):
            log(f"Prepared {index + 1}/{len(users)} users ({sum(h['reviews'] for h in histories):,} distinct reviews)")
    (output / "input_manifest.json").write_text(json.dumps({"data_path": str(data), "available_users_at_start": len(available),
        "selected_users": users, "time_kind": args.time_kind, "dataset_manifest": manifest,
        "histories": histories}, indent=2) + "\n", encoding="utf-8")
    summary = find_relations(histories, output, args.min_segment, args.partitions, args.index_memory_mib, log)
    for history in histories:
        for source in history["sources"]:
            if _stamp(Path(source["path"])) != source["stamp"]:
                raise RuntimeError(f"Input changed during audit: {source['path']}")
    summary.update(status="complete", data_path=str(data), selected_user_ids=users,
                   available_users_at_start=len(available), distinct_reviews=sum(h["reviews"] for h in histories),
                   raw_rows=sum(h["rows"] for h in histories), cache_hits=sum(h["cache_hit"] for h in histories),
                   timestamp_kind=args.time_kind, elapsed_seconds=perf_counter() - started,
                   provenance=provenance, dataset_download_manifest_present=manifest is not None,
                   scope="Selected exported histories only; canonical answer-ID order; no data exclusion.")
    write_graph(output, summary, args.graph_edge_limit)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("users", "distinct_reviews", "equal_groups", "subset_edges",
                                                   "common_segment_edges", "candidate_pairs", "elapsed_seconds")}, indent=2))
    print(f"Graph: {output / 'graph.html'}")
    return summary


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/anki-revlogs-10k")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/data-quality" / f"relations-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid4().hex[:6]}")
    parser.add_argument("--cache", type=Path, default=ROOT / "artifacts/data-quality/history-cache")
    parser.add_argument("--users", help="Comma-separated user IDs; otherwise freeze all currently available partitions")
    parser.add_argument("--limit-users", type=int, help="Bound the audit to the first N selected IDs")
    parser.add_argument("--time-kind", choices=["show", "answer"], default="show", help="Meaning of review_time; explicit revlog_id/id columns are always answer IDs")
    parser.add_argument("--min-segment", type=int, default=20, help="Minimum uninterrupted shared block; exact containment is found at any nonzero size")
    parser.add_argument("--partitions", type=int, default=256)
    parser.add_argument("--index-memory-mib", type=int, default=256)
    parser.add_argument("--graph-edge-limit", type=int, default=5000, help="Viewer edge cap; SQLite/JSONL retain every verified relation")
    args = parser.parse_args()
    if args.min_segment < 1 or args.graph_edge_limit < 1:
        parser.error("min-segment and graph-edge-limit must be positive")
    try:
        run(args)
    except (ValueError, RuntimeError, FileExistsError, MemoryError) as error:
        parser.exit(1, f"Review audit failed: {error}\n")


if __name__ == "__main__":
    main()
