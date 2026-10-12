"""Check exact containment, block discovery, caching and graph output; no GPU."""

from argparse import Namespace
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import review_relations as relations
from scripts.experiment_provenance import process_provenance


def event(index):
    return (1_600_000_000_000 + index * 1000, 1_500_000_000_000 + index % 13, index % 4 + 1, 1000 + index)


def canonical(items):
    return np.unique(np.array(items, dtype=relations.EVENT_DTYPE))


def brute(left, right, minimum):
    a, b = [tuple(row) for row in left.tolist()], [tuple(row) for row in right.tolist()]
    shared = len(set(a) & set(b))
    blocks, longest = [], 0
    for i, first in enumerate(a):
        for j, second in enumerate(b):
            if first != second or (i and j and a[i - 1] == b[j - 1]):
                continue
            length = 0
            while i + length < len(a) and j + length < len(b) and a[i + length] == b[j + length]:
                length += 1
            longest = max(longest, length)
            if length >= minimum:
                blocks.append((i, j, length))
    return shared, longest, blocks


def fixtures():
    base = [event(i) for i in range(100)]
    interleaved = base + [(e[0] + 500, e[1], e[2], e[3]) for e in base]
    return {101: canonical(base), 102: canonical(base), 103: canonical([event(i) for i in range(-10, 120)]),
            104: canonical(base[:40]), 105: canonical(base[40:80] + [event(i) for i in range(200, 206)]),
            106: canonical([event(i) for i in range(300, 340)]), 107: canonical(interleaved),
            108: canonical(base[10:13]), 109: canonical([]),
            110: canonical([(e[0], e[1], e[2] % 4 + 1, e[3]) for e in base])}


def persist(histories, root):
    root.mkdir(parents=True, exist_ok=True)
    result = []
    for user, events in histories.items():
        path = root / f"{user}.npy"
        np.save(path, events, allow_pickle=False)
        result.append({"user_id": user, "events_path": str(path), "reviews": len(events), "rows": len(events),
                       "event_sha256": sha256(memoryview(events)).hexdigest(),
                       "first_revlog_id": int(events[0]["revlog_id"]) if len(events) else None,
                       "last_revlog_id": int(events[-1]["revlog_id"]) if len(events) else None})
    return result


def expected_graph(histories, minimum):
    # This reference deliberately compares every original pair and uses Python sets.
    representatives = []
    for user, events in histories.items():
        keys = set(events.tolist())
        if not any(keys == other for _, other in representatives):
            representatives.append((user, keys))
    expected = set()
    for i, (a, keys_a) in enumerate(representatives):
        for b, keys_b in representatives[i + 1:]:
            if not keys_a or not keys_b:
                continue
            if keys_a <= keys_b or keys_b <= keys_a:
                expected.add((a, b, "subset"))
            elif brute(histories[a], histories[b], minimum)[1] >= minimum:
                expected.add((a, b, "common_segment"))
    return expected


class ReviewRelationChecks(unittest.TestCase):
    def setUp(self):
        temp_root = ROOT / "artifacts/diagnostics/check-temp"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temporary = TemporaryDirectory(prefix="review-relations-", dir=temp_root)
        self.root = Path(self.temporary.name).resolve()
        assert self.root.is_relative_to(temp_root.resolve())

    def tearDown(self):
        self.temporary.cleanup()

    def test_pair_api_against_independent_reference(self):
        rng = np.random.default_rng(2025)
        for trial in range(100):
            a = canonical([event(int(i)) for i in rng.choice(45, int(rng.integers(0, 25)), replace=False)])
            b = canonical([event(int(i)) for i in rng.choice(45, int(rng.integers(0, 25)), replace=False)])
            minimum = int(rng.integers(1, 8))
            shared, longest, blocks = brute(a, b, minimum)
            result = relations.compare_histories(a, b, minimum, batch_rows=3)
            self.assertEqual(result["shared_reviews"], shared)
            self.assertEqual(result["longest_segment"], longest)
            self.assertEqual([(s["a_start"], s["b_start"], s["reviews"]) for s in result["segments"]], blocks)
            self.assertEqual(relations.is_history_subset(a, b, batch_rows=3), set(a.tolist()) <= set(b.tolist()))
        empty = canonical([])
        self.assertTrue(relations.is_history_subset(empty, empty))
        self.assertTrue(relations.is_history_subset(empty, canonical([event(0)])))

    def test_window_minima_against_naive_and_shifted_blocks(self):
        rng = np.random.default_rng(91)
        for length in (0, 1, 2, 7, 20, 127):
            hashes = rng.integers(0, 100, length, dtype=np.uint64)
            for window in (1, 2, 3, 20, 128):
                expected = sorted({int(hashes[i:i + window].min()) for i in range(length - window + 1)})
                self.assertEqual(relations.window_minimizers(hashes, window).tolist(), expected)
        shared = relations.event_hashes(canonical([event(i) for i in range(12, 32)]))
        a = np.r_[rng.integers(0, 2**63, 5, dtype=np.uint64), shared, rng.integers(0, 2**63, 7, dtype=np.uint64)]
        b = np.r_[rng.integers(0, 2**63, 9, dtype=np.uint64), shared, rng.integers(0, 2**63, 3, dtype=np.uint64)]
        self.assertTrue(set(relations.window_minimizers(a, 20)) & set(relations.window_minimizers(b, 20)))

    def test_tied_timestamps_canonical_order_and_exact_search(self):
        rng = np.random.default_rng(772)
        for trial in range(30):
            rows = [(int(rng.integers(-5, 6)), int(rng.integers(-9, 10)),
                     int(rng.integers(1, 5)), int(rng.integers(0, 100))) for _ in range(80)]
            rows += rows[:10]
            rng.shuffle(rows)
            table = pa.table({name: pa.array([row[index] for row in rows], type=pa.int64())
                              for index, name in enumerate(relations.EVENT_DTYPE.names)})
            expected = canonical(rows)
            actual = relations.canonical_events(table)
            self.assertTrue(np.array_equal(actual, expected))
            query = np.array(rows[:30] + [(0, 999, 3, 0)], dtype=relations.EVENT_DTYPE)
            positions, hits = relations._matches(query, actual)
            reference = np.searchsorted(actual, query)
            expected_hits = np.array([tuple(row) in set(expected.tolist()) for row in query.tolist()])
            self.assertTrue(np.array_equal(hits, expected_hits))
            self.assertTrue(np.array_equal(positions[hits], reference[hits]))
        bounds = canonical([(-2**63, 2**63 - 1, 1, 0), (2**63 - 1, -2**63, 4, 0)])
        positions, hits = relations._matches(bounds, bounds)
        self.assertTrue(hits.all())
        self.assertEqual(positions.tolist(), [0, 1])

    def test_candidate_hashes_against_integer_reference(self):
        mask = (1 << 64) - 1
        rows = [event(i) for i in range(32)] + [(-2**63, 2**63 - 1, 4, 0)]
        expected = []
        for row in rows:
            value = 0x9E3779B97F4A7C15
            for field in row:
                value ^= field & mask
                value ^= value >> 30
                value = value * 0xBF58476D1CE4E5B9 & mask
                value ^= value >> 27
                value = value * 0x94D049BB133111EB & mask
                value ^= value >> 31
            expected.append(value)
        self.assertEqual(relations.event_hashes(np.array(rows, dtype=relations.EVENT_DTYPE)).tolist(), expected)

    def test_graph_against_brute_force_and_forced_hash_collisions(self):
        histories = fixtures()
        stored = persist(histories, self.root / "histories")
        for collision in (False, True):
            output = self.root / ("collision" if collision else "normal")
            context = patch.object(relations, "event_hashes", lambda array: np.zeros(len(array), dtype=np.uint64)) if collision else patch.object(relations, "CACHE_VERSION", relations.CACHE_VERSION)
            with context:
                summary = relations.find_relations(stored, output, min_segment=4, partitions=8)
            edges = [json.loads(line) for line in (output / "relations.jsonl").read_text().splitlines()]
            self.assertEqual({(e["a"], e["b"], e["kind"]) for e in edges}, expected_graph(histories, 4))
            self.assertEqual(summary["equal_groups"], 1)
            self.assertEqual(summary["empty_users"], [109])
            nodes = json.loads((output / "nodes.json").read_text())
            self.assertEqual(next(n for n in nodes if n["id"] == 101)["users"], [101, 102])
            # Full containment survives interleaving and histories below the block minimum.
            self.assertTrue(any(e["subset_id"] == 101 and e["superset_id"] == 107 and e["longest_segment"] == 1 for e in edges))
            self.assertTrue(any(e["subset_id"] == 108 and e["superset_id"] == 101 for e in edges))
            self.assertFalse(any(110 in (e["a"], e["b"]) for e in edges))

    def test_random_cohort_candidate_recall(self):
        rng = np.random.default_rng(44)
        histories = {}
        for user in range(1, 26):
            chosen = rng.choice(70, int(rng.integers(1, 35)), replace=False)
            histories[user] = canonical([event(int(index)) for index in chosen])
        histories[26] = canonical([event(i) for i in range(70)])
        histories[27] = canonical([event(i) for i in range(1000)])
        stored = persist(histories, self.root / "histories")
        output = self.root / "random"
        relations.find_relations(stored, output, min_segment=3, partitions=16)
        edges = [json.loads(line) for line in (output / "relations.jsonl").read_text().splitlines()]
        self.assertEqual({(e["a"], e["b"], e["kind"]) for e in edges}, expected_graph(histories, 3))

    def test_schema_timestamp_precision_duplicates_and_cache(self):
        original = event(4)
        high_id = (original[0], 2**63 - 100, original[2], original[3])
        rows = [high_id, event(3), high_id]
        table = pa.table({"card_id": pa.array([r[1] for r in rows], type=pa.int64()),
                          "review_time": pa.array([r[0] - r[3] for r in rows], type=pa.int64()),
                          "rating": pa.array([r[2] for r in rows], type=pa.int64()),
                          "duration": pa.array([r[3] for r in rows], type=pa.int64()),
                          "day_offset": [123, 0, 99], "state": [0, 2, 3]})
        self.assertTrue(np.array_equal(relations.canonical_events(table), canonical(rows)))
        answer = table.set_column(table.schema.get_field_index("review_time"), "review_time", pa.array([r[0] for r in rows]))
        self.assertTrue(np.array_equal(relations.canonical_events(answer, "answer"), canonical(rows)))
        direct = answer.rename_columns(["card_id", "revlog_id", "rating", "duration", "day_offset", "state"])
        self.assertTrue(np.array_equal(relations.canonical_events(direct), canonical(rows)))
        path = self.root / "reviews.parquet"
        pq.write_table(table, path)
        first = relations.prepare_history(1, [path], self.root / "cache")
        second = relations.prepare_history(1, [path], self.root / "cache")
        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        self.assertEqual(first["duplicate_rows"], 1)
        self.assertEqual(int(relations.load_events(first)[-1]["card_id"]), 2**63 - 100)
        changed = table.set_column(table.schema.get_field_index("rating"), "rating", pa.array([4, 4, 4]))
        pq.write_table(changed, path)
        third = relations.prepare_history(1, [path], self.root / "cache")
        self.assertFalse(third["cache_hit"])
        self.assertNotEqual(third["event_sha256"], first["event_sha256"])
        with self.assertRaises(ValueError):
            relations.canonical_events(table.drop(["review_time"]))
        with self.assertRaises(ValueError):
            relations.canonical_events(pa.table({"card_id": [1], "review_time": [2**63 - 1], "rating": [3], "duration": [1]}))
        with self.assertRaises(ValueError):
            relations.canonical_events(pa.table({"card_id": pa.array([2**63], type=pa.uint64()), "review_time": [1], "rating": [3], "duration": [1]}))

    def test_multi_file_user_and_offline_graph(self):
        files = []
        for index, items in enumerate(([event(0), event(1)], [event(1), event(2)])):
            table = pa.table({"card_id": [e[1] for e in items], "revlog_id": [e[0] for e in items],
                              "rating": [e[2] for e in items], "duration": [e[3] for e in items]})
            path = self.root / f"{index}.parquet"
            pq.write_table(table, path)
            files.append(path)
        history = relations.prepare_history(1, files, self.root / "cache")
        self.assertEqual(history["reviews"], 3)
        self.assertEqual(history["duplicate_rows"], 1)
        stored = persist(fixtures(), self.root / "histories")
        output = self.root / "graph"
        summary = relations.find_relations(stored, output, min_segment=4, partitions=8)
        relations.write_graph(output, {**summary, "fixture": True}, edge_limit=2)
        graph = json.loads((output / "graph.json").read_text())
        self.assertEqual(len(graph["edges"]), 2)
        self.assertTrue(graph["edges_truncated"])
        html = (output / "graph.html").read_text()
        self.assertNotIn("__D3_SOURCE__", html)
        self.assertNotIn("__REVIEW_RELATIONS_DATA__", html)
        self.assertNotIn('<script src=', html)
        self.assertIn("v7.9.0", html)
        self.assertTrue((output / "D3-LICENSE.txt").is_file())


def main():
    provenance = process_provenance()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ReviewRelationChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    demo = ROOT / "artifacts/data/data-quality/relations-demo"
    if (demo / "relations.sqlite").exists():
        from uuid import uuid4
        demo = demo.with_name("relations-demo-" + uuid4().hex[:6])
    histories = fixtures()
    stored = persist(histories, demo / "fixture-histories")
    summary = relations.find_relations(stored, demo, min_segment=4, partitions=8)
    summary.update(status="complete", fixture=True, distinct_reviews=sum(len(a) for a in histories.values()), provenance=provenance)
    relations.write_graph(demo, summary)
    (demo / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = {"status": "passed", "test_methods": result.testsRun, "random_pair_cases": 100,
              "random_cohort_users": 27, "forced_hash_collisions_checked": True,
              "provenance": provenance, "demo": str(demo.relative_to(ROOT))}
    (ROOT / "artifacts/data/data-quality/relation-checks.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Checks passed. Synthetic graph: {demo / 'graph.html'}")


if __name__ == "__main__":
    main()
