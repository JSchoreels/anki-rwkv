# Rank-2 to rank-4 low-rank states in the fork (SM151): format design

Written Mon 11-23, before any end-to-end scoring. The user approved low-rank states for Anki on 11-23 ("I think we can implement Rank 2 for anki") and at about 22:20 chose the mixed schedule for every size: rank 2 below 16 answers, rank 4 from 16 answers, int8 factors (the training repository's label `w8+lr2.16.4q8x8`, called T24 in the low-rank int8 states experiment (SM148)). Plain rank 2 (`w8+lr2q8x8`, called Q2) stays available as a configuration of the same codec.

## What is stored

The codec is the training repository's `LowRankCodec` (`scripts/state_lowrank.py`, branch `sm148`). It applies to the card and the note modules (the scorer's streams 0 and 2); deck, preset and global states stay float32 as before.

- **Placement:** the answer's own output uses the float update S' = S G + v k^T; the stored state, read by the entity's next review, is the codec's reconstruction (the same placement as every int codec).
- **Rank:** an entity with n answers (this write included) stores rank min(n, r), where r is the schedule's rank for n: r = 2 for n < 16 and r = 4 for n >= 16 under T24, r = 2 always under Q2.
- **Truncation:** the previous stored state is L R, so the columns of S' lie in span(L, v). The basis vectors [L_1 .. L_c, v] are normalised, their Gram matrix is eigen-decomposed, directions with an eigenvalue at or below 1e-5 x the largest are dropped, and Q is the orthonormal basis left. B = Q^T S' is c + 1 rows by K; the eigenvectors of B B^T, by decreasing eigenvalue, give the components: left_i = Q u_i, right_i = u_i^T B, sigma_i = |right_i|, L_i = left_i sqrt(sigma_i), R_i = right_i / sqrt(sigma_i). This equals the SVD truncation of S' up to rounding.
- **Factor rounding:** each factor vector (each column of L, each row of R) is rounded to int8 with its own float16 scale: abs-max / 127, clamped to 65,504, 1.0 when it is 0, codes round-half-even in [-127, 127]. This is the arithmetic of the compact int8 rows already in the release candidate. The stored matrix is the product of the decoded factors (float32).
- **Shift vectors:** the compact int8 shifts of the release candidate (one float16 scale per vector), unchanged.
- **The switch at 16 answers:** at the 16th answer the basis has 3 vectors (2 stored components plus v), so the state grows to rank 3 with a fourth zero component stored; from the 17th answer it is a full rank 4. The scorer does the same (its left factors are zero-padded to rank 4).

## Byte layout in the fork

Precision byte `0x28` (`LowRankInt8`) under the existing magic `ARWKVMODSTATE2`. Per layer: the x shift block (as compact int8), the u32 value count of the matrix (unchanged, so empty layers keep their meaning), then the low-rank block, then the channel shift block. The low-rank block is: the answer count n (u8, saturating at 255), the rank c (u8), then per head c left vectors and c right vectors, each a float16 scale and K int8 codes. One head at rank c costs 2c (K + 2) bytes; the scorer's byte estimate counted one byte per entity for the rank, the fork stores 2 bytes per layer (count and rank), which costs 4 to 6 bytes more per card.

The answer count is stored because the schedule needs it (rank 2 and rank 4 tiers), and the rank because the bytes must decode without knowing the schedule. The stored bytes do not depend on the schedule, so a later schedule change keeps every stored state decodable.

## Fork wiring

- `StatePrecision::LowRankInt8(RankSchedule)` (module precision) and `StatePrecision::CompactLowRank(RankSchedule)` (runtime policy; card and note modules both LowRankInt8). Policy strings: `compact_lowrank_int8` (T24, the new default) and `compact_lowrank_int8:<spec>` (for example `compact_lowrank_int8:2` for Q2) for benchmarks and tests.
- The write happens where the previous factors and the value vector are at hand: in the per-answer path inside the time mixer (`TimeMixer::run`), in bulk replay inside the per-stream recurrence sweep (`bulk.rs`). Both call the same encoder and decode the same bytes, so the per-answer and the bulk paths stay bit-identical.
- A matrix without factors (a state held at float32, or stored in another layout) is factored on the full basis (a K x K eigen problem, rare) with the answer count unknown (saturated at 255); an all-zero matrix is stored as count 0, rank 0.
- Non-finite values (never produced by a finite model) are replaced before the solve (NaN as 0, infinities as plus or minus the largest float32), so every stored value is finite and decodable; the scorer instead keeps such a matrix as it is and flags it.
- The solve runs in float64 (the scorer: float32 with LAPACK on the CPU); the parity tests bound the difference.

## Upgrade

As the int8 card rows did (fork branch `rc-state-int8`): the profile default moves to the new policy string in one place (`RWKV_STATE_PRECISION_DEFAULT` in `qt/aqt/rwkv_precision.py`, mirrored by the TypeScript fallback), the replay key hashes the precision string, and `_RWKV_STATE_CACHE_VERSION` goes from 14 to 15, so every state cache rebuilds once from the revlog. Stored states are not converted. The decoder keeps reading every older layout, so undo states, spills and old checkpoints stay readable until the rebuild replaces them.

## Parity plan

1. Codec vectors (`qt/tools/rwkv_lowrank_state_codec_vectors.py`): the scorer's `LowRankCodec.encode` on synthetic seeded recurrences (head size 16 and 32; T24 and Q2 with int8 factors; T24 with float factors), 24 writes per entity across the switch at 16 answers. The Rust test rebuilds every float update from the scorer's previous state and compares the stored state; a second test runs the Rust chain on its own.
2. End to end on benchmark users (per-user benchmark files only, never the user's own collection): fork-Rust per-user LogLoss at T24 against the low-rank int8 states experiment (SM148) T24 rows, and the int8 base against its REF rows.
3. Speed and bytes: per-review time of the state write and read for each size against the int8 base, single thread, release build; bytes per card and note.

## Note on a factored update

A factored form of the recurrence (S' = [U | v] [...]^T, two thin QR factorisations and a small SVD, about K (r+1)^2 work) would avoid the dense K x K update. The fork computes the dense float update anyway, because the answer's output must use it (the scorer's placement); the codec's own extra work is B = Q^T S' (K^2 (c + 1) multiply-adds, about 3,000 to 5,000 for K = 32) plus eigen problems of size 3 to 5, small next to the dense update and the time mixer's projections. The factored update is therefore not adopted; the measured speed is in RESULT.md.
