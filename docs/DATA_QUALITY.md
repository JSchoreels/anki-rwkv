# Data quality

Track dataset anomalies, audit methods, and proposed filtering rules here. Keep
pending and completed investigations visible, with supporting evidence and next
steps. GPU and training optimization tracks belong in
[GPU_OPTIMIZATIONS.md](GPU_OPTIMIZATIONS.md).

## Investigation tracks

| ID | Investigation | Status | Evidence | Next step |
|---|---|---|---|---|
| DQ01 | Audit repeated review patterns | Pending; feasibility inspection only | Raw reviews contain chronological card order, day offsets, ratings, learning states, answer durations and per-card elapsed times. No detector or thresholds have been tested. | Rank repeated long card sequences and ordering of shared cards across days, with rating/duration evidence. Inspect examples before choosing anomaly thresholds; report supporting review ranges. |
| DQ02 | Exact history containment, uninterrupted common blocks, and user graph | Implemented; functional checks and full 10,000-user audit complete | [Detector](../scripts/review_relations.py), [checks](data-quality/checks.json), [full-cohort measurements](data-quality/benchmarks/optimized-full-cold.json), and [D3 results](data-quality/index.html). 727,738,650 events; two equal-history groups, 18 strict containment edges and 14,880 common-block edges. | Inspect related components before proposing any filtering. Training impact remains unmeasured. |
| DQ03 | Improve exact relation detector speed | Three rounds retained; R04/R05 inconclusive, R06–R08 screened, further rounds pending | [Matched measurements](data-quality/performance.json): 1,000-user cache-hit median fell from 31.27 to 12.42 seconds (2.52×); retained full-cohort run 417.67 seconds, 656.5 MiB sampled peak process RSS. [Continuation](data-quality/performance-continuation.json) establishes no additional end-to-end gain above 10%. Exact outputs agree across completed trials. | Retain round 3. Test shared hash scanning and controlled full-cohort verification before concluding the optimization stop audit. |
| DQ04 | Freeze benchmark dataset identity during active-folder replacement | Resolved for GPU batch F01–F08; no filtering | The first short runner check stopped before updates with `review_time not found` after the timestamped export replaced the active folder. [Failed check](../artifacts/gpu-batch1-smoke/baseline/control/result.json), [replacement record](../artifacts/dataset-inspection/anki-revlogs-10k-id/replacement.json). The original export and count artifact share revision `75299740cff05894ef42d7ad990666691efdd2da`. | Pin the 500-user GPU batch to `data/anki-revlogs-10k-original-20261011` and record that path/revision. Timestamp-export training compatibility remains a separate, untested change. |

DQ01 was originally recorded as F12 in `GPU_OPTIMIZATIONS.md` and moved here
because it concerns data quality.

For this audit, use raw chronological reviews before feature processing sorts or
filters them. Account for deck and note structure, learning states, legitimate
scheduler ordering, and capped answer durations. Repeated order alone is
insufficient evidence of suspicious activity. In the original dataset, elapsed
times are per-card intervals. The timestamp variant below adds absolute timing.

The first deliverable is a ranked audit report with examples. No users have been
classified or excluded. Any later training filter needs a separately requested
matched quality/time comparison under the validation requirements in
[AGENT.md](../AGENT.md); its training benefit is unmeasured.

## DQ02: timestamped review identity and containment

### Verified timestamp semantics

The active dataset is `data/anki-revlogs-10k`, from
`open-spaced-repetition/anki-revlogs-10k-id`. Its README records builder commit
`e29946556796443283bf9c8e9d21f4502cb2e9c1` with
`--raw-ids --show-time --emit-review-time`. The complete download is pinned to
revision `93a7e8242690022d75943f67943901f81c556b0e`: 29,937 source files,
21,363,150,385 bytes and 727,738,650 reviews across all 10,000 users. Integrity
checks covered all source sizes, all 29,934 Parquet SHA-256 hashes, all Parquet
schemas and footer row counts. Cards metadata is absent for 66 users, as in the
source inventory; all users have reviews and decks metadata.
[Verification](../artifacts/dataset-inspection/anki-revlogs-10k-id/download_verification.json),
[per-file sizes and checksums](../artifacts/dataset-inspection/anki-revlogs-10k-id/remote_files.json).
Both the local copy and `\\syno-marauder\torrent\anki-revlogs-10k` were replaced
on 2026-10-11 after full verification. The NAS read-back checked every source
file and the download manifest against SHA-256; the original datasets are
preserved as sibling `anki-revlogs-10k-original-20261011` folders.
[Replacement record](../artifacts/dataset-inspection/anki-revlogs-10k-id/replacement.json),
[NAS verification](../artifacts/dataset-inspection/anki-revlogs-10k-id/nas_verification.json),
[five-user CPU loader smoke check](../artifacts/dataset-inspection/anki-revlogs-10k-id/loader_compatibility_full_download.json).
The full manifest was unavailable during the earlier five-user probe. Complete
file integrity does not establish the full dataset's duplicate rate or model
quality.

The verified timestamp variant was subsequently promoted to the active
`data/anki-revlogs-10k` directory (also on the NAS), with the original dataset
preserved in a dated backup. The performance work below uses that active local
path. [Replacement record](../artifacts/dataset-inspection/anki-revlogs-10k-id/replacement.json).

Anki normally assigns the revlog ID from the answer timestamp in epoch
milliseconds. IDs are unique within a collection, not globally across users;
the storage code can adjust an ID on a collision.
[Anki manual](https://docs.ankiweb.net/manual/stats#manual-analysis),
[answer logging](https://github.com/ankitects/anki/blob/main/rslib/src/scheduler/answering/revlog.rs),
[collision handling](https://github.com/ankitects/anki/blob/main/rslib/src/storage/revlog/add.sql).

This variant writes `review_time = revlog.id - taken_millis` and preserves
`duration = taken_millis`. Consequently, recover the original event ID with
`revlog_id = review_time + duration`. This arithmetic is reversible even when
the recorded duration was capped: capping affects the estimated show time,
but not recovery of the stored ID.
[Pinned builder source](https://github.com/open-spaced-repetition/anki-revlogs-dataset-builder/blob/e29946556796443283bf9c8e9d21f4502cb2e9c1/build_parquet.py).

Use signed int64 values throughout. Compare event identity `(card_id, revlog_id)`
and verify the stored rating and duration. For strict event-set containment, use
the tuple `(card_id, revlog_id, rating, duration)`. Record identical identities
with conflicting payloads separately. Neither a timestamp nor a shared card ID
alone establishes a duplicated history.

Exclude `day_offset`, elapsed intervals, transformed `state`, and current
deck/note metadata from this key: they can depend on the export's origin,
retained predecessors, learning resets, or later collection edits. The builder
filters some raw events and retains only certain learning histories, so exact
containment proves a relationship between the **retained exported events**.
It does not recover all reviews in the original collections or establish that
two collection IDs belong to the same person.

### Exact containment algorithm

1. Freeze the input cohort and manifest. For each user's distinct event set,
   record its size, time bounds, duplicate rows and identity/payload conflicts.
   Select a few deterministic anchor events, for example the three lowest
   deterministic 64-bit event hashes. Keep the original tuples as anchors;
   hashes only select them. Keep empty histories separate.
2. Scan the histories again to index which users contain each selected anchor.
   Only anchors need postings; use compact user-ID bitmaps and bounded batches
   rather than a Python dictionary containing every event in the dataset.
3. For each nonempty history A, intersect the owner lists for all of A's anchors.
   Candidate supersets B must also have at least A's distinct event count and
   time bounds covering A's. Every exact superset contains every anchor, so this
   candidate step has **no false negatives for exact containment**, provided the
   input manifest and event-key semantics are consistent.
4. Verify candidates against all original event tuples, using timestamp binary
   searches followed by full-tuple checks (full-tuple searches at timestamp ties) and reusable disk caches. Anchor matches or hash matches alone
   never establish an edge. Count shared events and record both directional
   coverage values: `shared / size(A)` and `shared / size(B)`.
5. Collapse identical sets into equivalence groups. Add a directed edge
   `A -> B` only when `A` is a proper subset of `B`. These groups form a directed
   acyclic graph. The implementation retains all verified containment edges;
   transitive reduction is a possible future display option. Multiple supersets
   are allowed, including diverging snapshots.

The candidate step avoids exhaustive intersections of every pair of full
histories. Read cost requires at least two passes; verification cost depends on
candidate count and history size. A dense relationship graph can still have
quadratically many edges. The full-cohort runtime and memory measurements are recorded below.

The reusable pair API `is_history_subset(A, B)` uses vectorized binary searches
on canonical arrays in bounded batches, stops at the first missing event, and
takes `O(|A| log |B|)` time. The empty set is a subset in this API; empty
histories are recorded separately and omitted from the graph. Equality groups
use SHA-256 to shortlist candidates and exact tuple comparison before collapsing.

### Uninterrupted common blocks

Per the user's definition, a common block requires matching events at
**consecutive positions in both histories**. Matching reviews separated by
other reviews contribute to shared coverage, but cannot bridge a block.
Canonical histories contain distinct event tuples, sorted by recovered answer
ID, with the other key fields breaking ties. Input row order and redundant
copies of the same event do not define additional reviews. Every block range
uses zero-based, half-open indexes in that canonical sequence.

Candidate discovery selects the minimum event hash from every sliding window
of `L` reviews and stores each selected hash once per history. Any identical
block of at least `L` reviews contains an identical window; that window selects
the same hash in both histories. This has **no false negatives for exact blocks
of length at least L**, regardless of where they start. A linear-time
prefix/suffix minimum calculation generates the window minima, followed by
deduplication. Hash collisions add candidates and never establish a match.

Hash/user postings are sharded on disk. SQLite deduplicates candidate pairs,
then `compare_histories(A, B)` compares the original tuples and splits shared
position pairs wherever either history's index jumps. It reports all maximal
blocks at least `L`, their timestamps, shared review counts, and coverage of
both histories. This costs `O(min(|A|, |B|) log max(|A|, |B|))` per candidate.
The longest-block count includes runs shorter than the reporting minimum.
Full containment remains detectable below `L` and with interleaved extra reviews
because it has its own exact-anchor candidate path.

The default minimum is 20 reviews and can be changed. This block graph does not
enumerate pairs whose only overlap consists of isolated reviews or shorter
blocks. Such pairs require the separate overlap investigation below.

### Run the detector

From the repository root, with the timestamp variant downloaded:

```powershell
# Functional checks and a synthetic graph showing each relation type.
.\.venv\Scripts\python.exe scripts/check_review_relations.py

# Bounded audit of users 1–100.
.\.venv\Scripts\python.exe scripts/review_relations.py --data data/anki-revlogs-10k --limit-users 100 --min-segment 20

# Complete verified cohort.
.\.venv\Scripts\python.exe scripts/review_relations.py --data data/anki-revlogs-10k --min-segment 20
```

Equivalent recipes are `just check-review-relations` and
`just review-relations --data data/anki-revlogs-10k --min-segment 20`.
Use `--users 32,41` for selected users. The active path is now `data/anki-revlogs-10k`. Default `review_time`
interpretation is show time; `--time-kind answer` supports exports already
containing answer time. Explicit `revlog_id` or `id` columns are always used
directly. Missing absolute timestamps, invalid integers, and overflow fail
explicitly.

Each run uses a fresh artifact directory and records the frozen user/file
selection, SHA-256 of every selected source file, source stats, cache metadata,
and process-start provenance. It refuses changed inputs during the run or an
existing result directory. A snapshot taken during a download covers only the
selected files; it does not claim the incoming files were included.

Outputs include:

- `input_manifest.json`: inputs and per-user duplicate-row/payload counts.
- `relations.sqlite` and `relations.jsonl`: every verified relation and block.
- `nodes.json`: related groups and all their equivalent user IDs.
- `summary.json`: completed audit counts, configuration, timing, and provenance.
- `graph.html` and `graph.json`: offline D3 viewer and its embedded graph data.

The viewer groups equal histories, draws arrows from subset to superset, and
uses dashed links for partial histories with common blocks. User, relation,
block-length, and visible-group filters narrow the view; selecting a link shows
coverage and exact review ranges. D3 and its license are bundled locally. The
minimum coverage filter applies to the shared-review fraction for both users
by default, or either user when selected. It filters drawn links and the table;
it does not change the detector's complete audit. For example, a 5% threshold
with "Both users" hides a relation whose directional coverage is 5.2% / 3.4%,
while "Either user" retains it. Reset restores a 0% threshold and "Both users".
The
default viewer cap is 5,000 strongest edges and explicitly reports truncation;
the SQLite/JSONL outputs retain all edges. Increase `--graph-edge-limit` if needed.

Canonical caches are memory mapped and use 32 bytes per distinct event, plus
NumPy headers. Preparation and window selection still need working memory
proportional to the largest single user's history. `--index-memory-mib` bounds
per-shard index processing with headroom; it is not a whole-process RAM limit.
An oversized shard fails explicitly with instructions to increase `--partitions`
(a power of two) or the budget. No candidates are silently discarded. Cache
reuse still checks each source file's checksum. Dense related cohorts can
generate quadratically many candidates and edges.

### Implementation checks and 100-user evidence

The eight current functional check methods compare 100 seeded random history pairs and a
27-user random cohort against independent brute-force references. Fixtures
cover equality, nested containment, large size imbalance, partial common
blocks, interleaved supersets, histories below the minimum, empty histories,
changed payloads, timestamp recovery, 64-bit precision, multi-file users, and
cache invalidation. Extra cases check repeated timestamps, signed-int64 bounds,
and the candidate hash calculation against an independent integer reference.
Forcing all candidate hashes to collide still produces the
same exact graph. The checks also exercise the offline graph and explicit
viewer truncation. [Raw check results](../artifacts/data-quality/relation-checks.json).

The first real audit selected users **1–100**, with **5,226,500** distinct review
events, no redundant event rows, and no cache hits. It examined **one candidate
pair out of 4,950** possible pairs. Users **32 and 41** share **761 exact events**
(1.2636% and 0.8776% of their histories), including nine maximal blocks of at
least 20 reviews. The longest contains **145** consecutive reviews: user 32
range `[5379, 5524)` and user 41 range `[14346, 14491)`. There were no equal
histories or full-containment edges in this cohort. This is evidence of shared
exported history, not a classification of suspicious users.

Measured audit time was **6.69 seconds** on this workspace, including input
checksums, provenance capture, canonical preparation, candidate indexing, exact
verification, and the final source-stat check. HTML rendering and final summary
writing are outside that timing. It is a single bounded data audit; it does not
establish full-dataset throughput or model quality. No training was run.

Historical planning estimate, superseded by the measured full-cohort run below: the complete cohort has 727,738,650 reviews,
139.24 times the measured 100-user sample. Linear review-count extrapolation
gives **15.53 minutes**. Allow roughly **20–40 minutes** for a first local run
with the same minimum block length (20), assuming sparse candidate relations.
This is an estimate, not a full-cohort measurement or upper bound: the anchor
lookup work grows with both users and selected anchors, and large shared-history
clusters increase exact pair verification. The canonical cache alone would use
about 23.29 GB before deduplication, plus temporary indexes and results.

The original 100-user process recorded Git HEAD
`020e2fac1e62fe8e69c2d8daa36a81c881919a95`, a dirty working tree, and preserved
source snapshot `b63d6348c6cba56f61a0d8aa3a8045e4f937e6bc43451b86bd4bf6c763f92fb5`.
[Summary](../artifacts/data-quality/relations-100-users/summary.json),
[exact relation/block evidence](../artifacts/data-quality/relations-100-users/relations.jsonl),
[interactive graph](../artifacts/data-quality/relations-100-users/graph.html).
An [independent Python-tuple check](../artifacts/data-quality/relations-100-users/verify_source_pair.py)
also recovered the pair's keys directly from the original Parquet files and
verified all nine maximal blocks, shared count, longest length, and endpoint
timestamps. Its [raw result](../artifacts/data-quality/relations-100-users/source_verification.json)
records the separate verification and updated viewer-render provenance.

### DQ03: measured optimization rounds

Further rounds are in progress under the user's stop rule: continue while a
round reduces matched end-to-end audit time by more than 10%, with unchanged
canonical, node, and complete relation digests. Compare sequential controls and
trials on users 1–1000, then verify substantial gains on the complete cohort.
The first new trial (R05) indexes only the lowest-hash anchor, then checks the
other two anchors only for possible supersets. Its 1,000-user median was
16.25 seconds versus a sequential round-3 control of 15.79 seconds (three
repetitions each). Full-cohort times were 451.05 versus 494.40 seconds,
with identical canonical, node and relation digests. The full comparison was
confounded by competing CPU preparation workers in the control; it does not
establish a speedup. R05 is not retained.

R06 filters timestamp matches before fetching complete tuples and reuses a
contiguous timestamp column during exact pair verification. It retains
complete tuple verification and the same candidate-recall guarantees.

R07 restricts exact searches to overlapping timestamp bounds. R08 tests a native
Arrow timestamp hash join for large, comparable-sized histories; all fields
are verified afterwards, and repeated timestamps use the binary-search
fallback to avoid a Cartesian explosion. Both have passed the existing eight
functional checks; R08 also passed an independent 72,000-event join/payload and
timestamp-tie check. End-to-end gains are pending. A separate GPU training run
had four CPU workers active during the first new full-cohort control; that run
is preserved but cannot by itself establish a matched speedup.

An interleaved diagnostic on 140 real candidate pairs (seed 51273, three
repetitions, rotated execution order) returned identical exact results for
every pair. Median pair-comparison totals were 0.967 seconds for the original
verification path, 0.744 for R06, 0.696 for R07, and 2.029 for R08. This excludes
preparation, discovery, indexing, mapping and graph generation; it is not an
end-to-end gain. R08 is slower in this diagnostic and is not retained. R07's
three 1,000-user end-to-end trials took 26.30, 15.39 and 15.17 seconds;
the median is only 2.5% below the earlier control, with identical outputs.
It is not retained. Full-cohort verification and further indexing ideas remain
pending. These candidates are based on retained round 3 rather than the
rejected R05 anchor experiment. [Continuation records](data-quality/performance-continuation.json)
preserve reports and immutable sources; the original performance report remains
the record of the retained implementation.

R09 (pending): reuse the event-hash scan for both anchor selection and block
minimizers, rather than reading and hashing every canonical event twice.
Preserve all three actual anchors and complete minimizer postings, then verify
unchanged candidate pairs and exact outputs. End-to-end gain is unmeasured.

The retained implementation audited **all 10,000 users and 727,738,650 reviews
in 417.67 seconds (6 minutes 58 seconds)**, including graph generation, from an
empty canonical cache. Sampled peak process RSS was **656.5 MiB**; memory-mapped
file pages in the OS cache are outside that figure. The on-disk canonical cache
occupies approximately **23.29 GB**, plus temporary postings and audit outputs.
No training or data filtering was run.

Matched users 1–1000 contain 71,400,715 reviews. Each cache-hit variant ran three
times sequentially, with the same canonical cache and settings: 20-review block
minimum, three anchors, 256 posting partitions, and a 256 MiB shard-processing
budget. Medians and observed ranges include the complete detector and graph
writing, but exclude benchmark setup, initial source snapshot capture, and
subsequent result hashing. OS file caches and other system work were uncontrolled.
These are measurements on this computer, not a full-dataset baseline speedup.

| Variant | Change | Median seconds | Observed range | Baseline / variant |
|---|---|---:|---:|---:|
| Baseline | Full-tuple searches and sorts, general posting groups | 31.27 | 25.03–31.76 | 1.00× |
| Round 1 | Search signed-int64 answer timestamps, then check all fields; full-tuple fallback at ties | 23.90 | 22.28–24.29 | 1.31× |
| Round 2 | Sort full tuples only at timestamp ties; discard adjacent repeated window minima before final deduplication | 19.62 | 18.93–20.45 | 1.59× |
| Round 3, retained | Reuse hash scratch buffers; stable integer posting sort; skip singleton groups and batch two-owner pairs | 12.42 | 12.31–12.44 | **2.52×** |
| Round 4, experimental | 4 MiB timestamp bucket filter for anchor lookup | 14.15 | 13.37–22.80 | 2.21× |

Separate single runs on users 1–1000 with empty canonical caches took **54.64
seconds baseline** and **23.80 seconds round 3**, a measured **2.30×** difference.
The original baseline was not run over all 10,000 users.

The fourth experiment initially slowed the 1,000-user case, then used an
adaptive threshold of 2,048 users to preserve direct lookup on small cohorts.
Its full run took **468.20 seconds (7 minutes 48 seconds)**. Anchor discovery
fell from 133.04 to 104.76 seconds, but unchanged preparation, block indexing,
and verification phases also slowed. Its end-to-end benefit is **inconclusive**;
round 3 remains the default, and the trial source and results are preserved.

| Full-cohort stage | Retained round 3 seconds | Experimental round 4 seconds |
|---|---:|---:|
| Canonical preparation, input checksums and cache creation | 114.78 | 135.35 |
| Equality grouping | 0.05 | 0.05 |
| Subset candidates | 133.04 | 104.76 |
| Common-block candidates | 55.69 | 74.70 |
| Exact pair verification | 85.62 | 111.99 |
| Graph generation | 0.54 | 0.70 |
| Complete detector, including other discovery/report overhead | **417.67** | **468.20** |

Both full-cohort runs generated identical canonical-event, node, and relation
digests. All 1,000-user trials also agree on those three outputs. The eight
functional checks pass, including brute-force comparisons, forced hash
collisions, timestamp ties, and signed-int64 boundary cases. The full-cohort
graph has **9,996 distinct history groups**, **two equality groups**, **18 strict
subset edges**, **14,880 common-block edges**, and **1,685 related groups**.
Only **25,109 candidate pairs** were verified, from 49,955,010 possible group
pairs; 69,300,344 minimizer postings were indexed. Equality groups contain users
449/2066/6812/7872 and 8590/9576. These are exported-history relationships;
the results do not establish account ownership or suspicious behavior.

The [published D3 viewer](data-quality/index.html) embeds **all 14,898 computed
relations**, with shared counts, directional coverage, and longest-block
lengths. Review timestamps and detailed block ranges stay in the local audit,
in accordance with the dataset's restriction on public data redistribution.
The local full-evidence viewer retains its explicit 5,000-edge display cap;
its JSONL and SQLite reports contain every edge. The published graph can be
viewed from another computer through this
[GitHub Pages viewer](https://jschoreels.github.io/anki-rwkv/docs/data-quality/).
GitHub Pages is enabled; subsequent result commits update this view after deployment.

[Comparison and provenance](data-quality/performance.json),
[individual benchmark reports](data-quality/benchmarks),
[immutable detector implementations](data-quality/implementations), and
[functional check report](data-quality/checks.json) are committed research
results. Raw logs, complete review-range evidence, and complete workspace source
archives remain under ignored `artifacts/data-quality/relations-performance`
and `artifacts/experiment-sources` in this workspace. Every measured process
recorded full Git HEAD **020e2fac1e62fe8e69c2d8daa36a81c881919a95** and a dirty
working tree; this later implementation commit is separate from that tested
revision. The retained detector SHA-256 is
**4427c42f6821f7739d43e7231fe8b25c6f0403cbe1b878dd96c6ad929c1a5287**.
The full run's preserved workspace snapshot is
**8de576ecbd5d50230a0dab5b192302f7f0e51078d38f19b067b4d65d7eaed4e3**.
Its recorded environment was Python 3.12.14, NumPy 2.5.4, PyArrow 26.0.0,
eight physical / sixteen logical CPU cores, and approximately 31.1 GiB RAM.

Implementation commit: [ef1d345c3d6b7db02e70e5b695b8db8193a62b4c](https://github.com/JSchoreels/anki-rwkv/commit/ef1d345c3d6b7db02e70e5b695b8db8193a62b4c)
contains DQ02/DQ03, the retained round-3 detector, archived trial implementations,
benchmark records, and the public D3 graph. The committed detector's SHA-256
matches the preserved full-run implementation. This commit records the code
after measurement; the earlier experiment HEAD and dirty source snapshots above
remain the identities of the measured runs.

To reproduce the measurements or publish a later completed audit:

```powershell
# Three sequential cache-hit trials on users 1–1000.
.\.venv\Scripts\python.exe scripts/benchmark_review_relations.py --label round3-repeat --repeats 3 --limit-users 1000

# A full audit with a fresh canonical cache (do not reuse an existing path).
.\.venv\Scripts\python.exe scripts/benchmark_review_relations.py --label full-repeat --limit-users 10000 --cache artifacts/data-quality/fresh-full-cache

# Aggregate research-results viewer for publication; exact evidence stays local.
.\.venv\Scripts\python.exe scripts/publish_review_relations.py --audit artifacts/data-quality/relations-performance/optimized-full-cold-1379b1/repeat-01 --benchmark artifacts/data-quality/relations-performance/optimized-full-cold-1379b1/benchmark.json
```

### Partial overlap and edited histories

Keep partial and near-containment relations separate from strict subset edges.
An ordinary Jaccard similarity cutoff can miss exact subsets: a 10,000-event
history inside a 100,000-event history has containment 1.0 but Jaccard 0.1.
Likewise, fixed anchors can miss approximate subsets when an anchor was deleted
or edited; the exact-containment guarantee does not extend to these cases.

For a complete exact-overlap graph, an event-to-users inverted index can count
shared events for every owner pair, using external sorting/partitioning for
the full dataset. For approximate candidate discovery at lower cost, investigate
containment-aware sketches or size buckets and disclose their recall limits.
Verify discovered overlaps against original tuples and store shared event/card
counts, both coverage fractions, matched time spans and payload conflicts.

Use confirmed overlap groups to inform train/test grouping before interpreting
model generalization. Diverging histories can each contain unique reviews;
selecting one largest history per connected component can discard those reviews.
Filtering rules remain a separate decision and require measured training impact.

### Bounded probe evidence

On 2026-10-11, the probe inspected all five complete review files available at
the initial inspection: users 1, 17, 1710, 1711 and 1712. Their 224,712 rows had
no identical rows within a user, no identity/payload conflicts, and no shared
card IDs or strict review events across users. This small, unrepresentative
snapshot does not establish the duplicate rate of the full dataset.

Seven synthetic histories in the earlier probe exercise equality, strict and very unequal-size
containment, partial overlap, empty histories and conflicting ratings. Exact
anchor verification agrees with brute-force containment. This is a prototype
check; the implementation and subsequent functional checks above supersede its
prototype algorithm. It was not a model-quality comparison.

[Probe source](../artifacts/data-quality/timestamp-overlap-probe/probe.py) and
[raw results](../artifacts/data-quality/timestamp-overlap-probe/results.json)
retain input checksums, source provenance and every pair's counts. No training
data was modified or excluded.
