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
| DQ03 | Improve exact relation detector speed | R13 retained; R14 full comparison running | [Matched measurements](data-quality/performance.json): R13 full audit **317.00 seconds**, **21.07%** below the faster of bracketing R12 controls (410.72 / 401.60 seconds); 452.9 MiB sampled peak process RSS. R14's 1,000-user screen gains **5.89%**, below the 10% acceptance threshold. Complete candidates and exact outputs agree. Earlier rounds and inconclusive trials remain in [continuation records](data-quality/performance-continuation.json). | Retain the two-worker detector. Complete the full parallel-preparation comparison before the stopping decision. |
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
preserve reports and immutable sources; the performance report links the
historical measurements and the current retained implementation.

R09 (exact full-cohort outputs confirmed, speed gain provisional): reuse the event-hash scan for both anchor selection and block
minimizers, rather than reading and hashing every canonical event twice.
Preserve all three actual anchors and complete minimizer postings, then verify
unchanged candidate pairs and exact outputs. The eight independent functional
checks pass, including forced hash collisions and empty histories. End-to-end
three-repeat 1,000-user median is 15.59 seconds versus 18.20 seconds for its
preceding control (14.3% less elapsed time). Canonical, node and complete
relation digests agree. Its subset stage falls from a 4.07-second median to
1.32 seconds; common-block indexing includes both shared hash computation and
anchor selection. System work remains uncontrolled, so a following control and
full-cohort trials are needed before promotion. The following control's median
was 18.26 seconds, also yielding more than 10% improvement (14.6%). All nine
control/trial outputs have identical complete candidate rows and flags, in
addition to canonical, node and relation digests. The full-cohort control and
R09 trial ran sequentially from the same existing canonical cache. Their
phase order differs because R09 obtains anchors during block-index creation;
this is part of its implementation rather than a discarded timing cost.

The new full-cohort control completed in **496.36 seconds (8 minutes 16
seconds)** using 10,000 canonical-cache hits. Its stages were 81.36 seconds
for input checksums/preparation, 172.47 for subset candidates, 79.63 for common
blocks, 118.58 for exact verification and 0.73 for graph generation (plus
other audit overhead). Complete candidate flags and all three output digests
match the original full audit. The immediately following R09 trial completed
in **446.61 seconds (7 minutes 27 seconds)**: preparation 73.52 seconds, common
blocks plus shared anchor selection 112.24, subset candidate membership 98.80,
verification 117.83, and graph generation 0.79 (plus other audit overhead).
It used 10,000 cache hits and 394.1 MiB sampled peak process RSS. Complete
candidate rows/flags, canonical identities, nodes and relations are identical.
The **10.02%** elapsed reduction is barely above the requested cutoff, and
preparation was also 7.83 seconds faster. This single pair is provisional;
R09 is not promoted yet. R10/R11 full trials and a following R09 control are
next. These timings do not replace the earlier fresh-cache measurements.
The sequential follow-up queue has now started the R10 native-anchor full
trial, with R11 verification and an R09 control queued behind it; they share
the same canonical cache and do not overlap one another.

Archived R05–R08 trial implementation commit:
[a19266cfff7d83efb45750691cce2b50bd6b337c](https://github.com/JSchoreels/anki-rwkv/commit/a19266cfff7d83efb45750691cce2b50bd6b337c).
This is separate from their measured process HEADs and dirty source snapshots.
Archived R09/R10 trial implementation commit:
[25179f4bce9efa7b61735da626f6fd86571191f5](https://github.com/JSchoreels/anki-rwkv/commit/25179f4bce9efa7b61735da626f6fd86571191f5).
Their 1,000-user measurements started at full HEAD
`a19266cfff7d83efb45750691cce2b50bd6b337c` with a dirty workspace; the reports
preserve exact per-process source archives and immutable implementation hashes.
The later archive commit does not replace those measured revisions.

R10 (exact full-cohort outputs confirmed, incremental speed below 10%): use a single-threaded native Arrow hash join on all four exact
event fields to locate global anchors in each history. Anchors and canonical
histories are distinct full tuples, so there is no timestamp-tie cross product.
Preserve the three-anchor intersections and complete verification. Its benefit
and memory cost on the full cohort are unmeasured; this differs from R08's pair
timestamp join. The 1,000-user median was 17.00 seconds versus R09's 15.59,
with identical outputs. Sampled peak process RSS reached 343.6 MiB versus
254.2 MiB for R09. It is not retained; full-cohort anchor lookup remains a
possible separate diagnostic, without an established end-to-end gain.
The full trial completed in **425.51 seconds (7 minutes 6 seconds)** with
identical complete candidate flags and exact outputs. Its subset-membership
stage took 63.85 seconds versus R09's 98.80, but whole-audit time was only
4.72% lower than R09's preceding full trial. Sampled peak process RSS was
451.7 MiB. Four unrelated CPU workers started at 14:48:44 local time,
about 14 seconds into this trial, and remained active when it ended; disclose
that load difference before interpreting either the incremental result or
the 14.27% reduction versus the earlier round-3 control. It is not promoted
yet. R11 and the following R09 control results are recorded below.

R11 (exact full-cohort outputs confirmed, not retained): combine R09's shared hash scan with R07's timestamp-bound
restriction and reusable timestamp column for pair verification. The isolated
R07 diagnostic improves large real candidate pairs, but the earlier 1,000-user
end-to-end screen did not exceed 10%. Measure the combination against R09 on
the full cohort rather than projecting a combined gain from separate tests.
The combination passes the eight independent functional checks, including
random-cohort candidate recall and forced hash collisions. Its preserved
implementation SHA-256 is
`c7611386a787b7da47cbb13b780c80a5eca3cefb961ab40f2add36f1487b3ad6`.
The full trial took **471.68 seconds**, with unchanged complete candidate
flags and exact outputs. Exact verification was 117.30 seconds versus R09's
117.83 (about 0.5% less), despite the promising pair diagnostic. The small
sample did not predict the full workload. Preparation was 82.50 seconds,
common blocks 112.01, subset candidates 102.07 and graph generation 0.82,
plus other audit overhead. Four unrelated CPU workers remained active.
R11 is not retained. The following control completed in **417.46 seconds**, with
identical outputs and candidate flags. Its preparation was 84.78 seconds,
common blocks 103.05, subset membership 91.79 and verification 100.43.
The unchanged R09 implementation is faster than R10's 425.51-second trial in
this later observation; neither R10 nor R11 establishes a robust incremental
gain above 10%. The CPU preparation workers had disappeared by the post-queue
process inventory. Preserve this drift rather than averaging unlike load
conditions into a claimed algorithm speedup.
Archived R11 trial implementation commit:
[ebdfc083d9518ae3db750693ab58dcd2beecf34a](https://github.com/JSchoreels/anki-rwkv/commit/ebdfc083d9518ae3db750693ab58dcd2beecf34a).
The test report and later full benchmark retain their own measured process
HEADs, dirty-workspace source snapshots and implementation hashes.

R12 (correctness checked, timing pending): use native full-tuple anchor lookup
only when there are at least 12,000 global anchors. Below that, retain direct
exact lookup to avoid the native join's observed small-cohort overhead. This
crossover is a heuristic, not a measured optimum. The initially proposed
combination with R11 verification is deferred after R11's full trial showed
no material verification gain. Its untested prototype is preserved separately;
R12's current source retains the original pair verifier.
Both the default branch and forced native lookup pass the eight independent
checks, including brute-force cohort comparisons and forced hash collisions.

R13 (retained; correctness and full bracketing comparisons passed): process two histories at a time during hash/minimizer creation
and two candidate pairs at a time during verification, using bounded ordered
thread queues. Keep owner-ordered shard writes and sorted relation output in
the main thread; workers read immutable arrays and do not access SQLite.
Measure the memory increase and complete elapsed time. Never overlap separate
benchmark processes, and preserve complete candidate flags and output digests.
For its benchmark, distinguish summed concurrent function-call durations from
phase elapsed time; use synchronized counters and an ordered-map phase timer.
Complete end-to-end time remains the acceptance metric.
The default and forced-native configurations both pass the eight independent
checks. The updated benchmark runner synchronizes concurrent timing sums and
records ordered-map phase elapsed separately. Sequential before/trial/after
controls on users 1–1000 (three repetitions each), followed by the same order
on all 10,000 users, are now queued from the same corresponding canonical
caches. Only the trial uses two indexing/verification workers.
The completed 1,000-user medians are **13.36 seconds** before, **11.94 seconds**
for R13, and **13.32 seconds** after (three repetitions each): a **10.34%**
reduction against the faster control median. Every complete candidate row/flag,
canonical digest, node digest and relation digest agrees across all nine runs.
R13's common-block stage median is 5.12 seconds versus 6.17 before and 6.23
after. Parallel pair-call time sums are not the pair phase wall time; the latter
is recorded as `verify_pair_phase_elapsed`. The completed full-cohort
before/trial/after measurements are recorded below.
Archived R12/R13 implementations and synchronized benchmark instrumentation:
[67b3df0003459ede4a21ec2e507d7c955c999d11](https://github.com/JSchoreels/anki-rwkv/commit/67b3df0003459ede4a21ec2e507d7c955c999d11).
This later archive commit is separate from the measured initial HEADs and
dirty-workspace snapshots in each report. Unmeasured combined-verifier
prototypes remain archived with explicit `unmeasured` names.

The first full-cohort R12 control completed in **410.72 seconds**. R13's
following full trial completed in **317.00 seconds (5 minutes 17 seconds)**,
an observed **22.82% reduction**. All complete candidate rows/flags and the
canonical, node and relation digests agree. Both used 10,000 canonical-cache
hits. Sampled process peak RSS was 452.6 versus 452.9 MiB. R13 stages included
84.40 seconds of preparation, 75.50 for common blocks (62.08 for ordered
index creation/writing), 65.71 for subset membership and **87.43 seconds**
for the complete ordered verification phase, including main-thread result
emission. Its concurrent pair-call duration sum is 108.15 seconds and must
not be read as phase elapsed. Promotion waited for the bracketing comparison.
The following control
completed in **401.60 seconds**, confirming a **21.07% reduction** against the
faster full control. All complete candidate flags and actual output-file
digests agree across the three full runs. R13 is now the default detector;
the published graph retains all 14,898 derived relations and shows the 317.00
second measurement with explicit canonical-cache reuse and integrity-check
scope. The preserved and production detector SHA-256 is
`753c93b6e5f2f74b983e3b48eb958775698ae5c7e747216047edb75366b86a68`.
Production promotion and updated viewer implementation commit:
[f81dfd120a8a3de4435669235b394339108ab651](https://github.com/JSchoreels/anki-rwkv/commit/f81dfd120a8a3de4435669235b394339108ab651).
This later delivery commit is separate from the preserved experiment revisions.

R14 (correctness checked; full bracketing comparison running): use the same bounded two-worker ordered queue for history
preparation and source checksums. R13's full trial still spends 84.40 seconds
there. Preserve complete checksum/stamp checks, sorted user ordering and
atomic per-history cache writes; do not substitute metadata-only integrity
checks. Measure the whole audit against R13 before deciding on the gain. Both
default and forced-native configurations pass the eight independent checks.
The completed 1,000-user screen has median **10.28 seconds**, versus R13
controls **10.93 / 11.39 seconds** (three repetitions each), a **5.89%**
reduction against the faster control. Recomputed canonical manifests,
actual node/relation files and every candidate row/flag agree across all nine
runs. Full-cohort controls and trial are running sequentially on the same
cache, with all source checksums preserved. Concurrent GPU training and its
CPU workers remain external timing influences; the initial control repetition
was 21.42 seconds, followed by 10.93 / 10.54 seconds. Use the recorded median,
not that outlier, and wait for both full controls before deciding retention.
[Screen reports and parity evidence](data-quality/performance-continuation.json).
The full R13 preceding control completed in **308.58 seconds** (83.27 seconds
preparation, 68.94 common-block candidates, 61.97 subset candidates and 90.43
ordered verification). Complete candidate rows/flags and actual output files
also match the previously retained R13 full trial. The R14 full trial and
following control are pending; this preceding control alone implies a 277.72
second upper bound for clearing 10%, before considering the following control.

The previously retained round-3 implementation audited **all 10,000 users and 727,738,650 reviews
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
and `artifacts/experiment-sources` in this workspace. The initial measured processes
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
