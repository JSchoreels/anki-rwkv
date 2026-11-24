# Faster low-rank codec in the fork (SM156): result

Written Tue 11-24 (night). Fork branch `rc-lowrank-speed` (worktree `wt/fork-lowrank-speed`, from `origin/rc-2026-11`
at 514f3a41d, the merge of the low-rank states in the fork (SM151)), commit 47163a997. **Not merged into
`rc-2026-11`** (that needs the user's OK). No GPU job was used.

## Verdict

The codec stores **bit-identical bytes** (and bit-identical stored and decoded matrices) and a layer write takes about
half the time. On the long replay (1,000 cards x 24 answers, one thread) the low-rank default's extra cost over the
int8 rows drops from +31 to +13 µs per review for Small and from +23 to +8 µs for Tiny; Big's T24 replay is no longer
slower than its int8 rows. Measured under load (a GPU job and other CPU work), medians of 5 (Small, Tiny) or 3 (Big)
alternated runs.

## What changed (`rslib/src/rwkv/lowrank.rs`)

1. **No per-head heap allocation.** The basis, the normalised basis, Q, B, the Gram and B B^T matrices, the factors and
   the eigenvectors live in fixed-size stack arrays (basis at most 9 vectors = rank 8 + v; head size a compile-time
   constant); the small eigen problems run at a compile-time size (n = 1 to 9, dispatched once per solve).
2. **The four heads' Jacobi solves run in lockstep** (`jacobi_lanes`). After the allocations, about 75% of a write was
   the two 5 x 5 Jacobi solves per head, and they are latency-bound: each rotation waits on a chain of two divisions,
   two square roots and a reciprocal (about 110 cycles) that depends on the previous rotation. The heads are
   independent problems, so the solver now computes the rotation of every head for one (p, q) pair, then applies them:
   the CPU overlaps the four chains. Each head still runs exactly the old sequence (same stop rule, same 60-sweep cap,
   same cyclic order, same operations in the same order); a head that has stopped is skipped. The B B^T solves group
   the heads by size (a head may keep fewer directions after the drop rule).
3. **Smaller savings:** the stored matrix is the product of the rounded factors as written (the old code decoded the
   block again; the values are the same), no zero fill before the product (every value is written), the finite checks
   run without early exit, and the decoder's reconstruction uses stack arrays (read 1.7 times faster at head size 16,
   5 times at 32).
4. Unchanged: the full-basis solve (a state without factors, `Previous::Matrix` or an unreadable block) keeps the
   general allocating solve; the stored format, the rank schedule, the default and the float64 arithmetic.

Tried and dropped: the same lockstep solver in a lane-minor layout compiled four lanes wide (AVX, same IEEE
operations, no fused multiply-add, so the same bits). No measurable gain over the scalar lanes (the lockstep chain is
now the limit, not the arithmetic), so the simpler portable version stays.

## Bit identity and parity

- `rslib/src/rwkv/lowrank_reference.rs` (tests only) is a frozen copy of the codec as merged in `rc-2026-11`
  (514f3a41d).
- `rwkv_state_lowrank_codec_matches_the_frozen_codec_bit_for_bit`: the current and the frozen codec each run their
  own chain of 300 answers for 11 schedules (T24, `1.3.8`, `8.20.2`, uniform rank 1 to 8) x 3 starts (fresh, a dense
  matrix without factors on the full basis, an unreadable previous block at answer 120), with huge (1e30), zero, tiny
  (1e-33), non-finite (NaN, +-inf) and repeated value vectors, a zero key, an all-zero matrix and the 255-answer
  saturation. Every write: identical stored bytes, identical stored matrix bits, identical decoded matrix bits (both
  decoders). 9,900 layer writes per head size, both head sizes (16 and 32, both engines).
- `rwkv_state_lowrank_solve_matches_the_frozen_solve_bit_for_bit`: identical factor bits (before rounding) on 1,332
  head solves: every write of the scorer's vectors (`lowrank-state-codec.json.gz`, head size 16 and 32) and random
  bases of 1 to 9 vectors at every rank with zero, duplicated and nearly dependent vectors, alone and as the four
  heads of one layer (lanes that keep different numbers of directions and stop after different sweeps).
- A deliberate one-ulp change (multiplying by the reciprocal of the norm instead of dividing) fails both tests, so
  they do detect rounding differences.
- The existing scorer parity tests are unchanged: int8 factors identical on 252 of 252 head writes at head size 32
  and 249 of 252 at head size 16 (125 + 124), same errors as before; the chain test gives the same means and maxima.
- End to end: no rerun was needed, since the bytes do not change; the replay benchmark's checksums (all model
  outputs) are identical between the frozen and the current binary for every size, both precisions and both
  histories.

## Speed

Release build, one thread (`RAYON_NUM_THREADS=1`), `rwkv_model_size_benchmark` (full state replay through the bulk
path, synthetic answers, the whole model; 3 samples after a warm-up), the frozen binary (built from 514f3a41d) and the
current binary back to back, int8 rows (`compact_cards_int8`, the same code in both binaries) and T24
(`compact_lowrank_int8`). Four configurations rotated in a different order on each pass (`speed_run_alternating.sh`):
Small and Tiny 5 passes, Big 3 passes. Median reviews/s and its µs per review.

**Load:** the GPU job "Smaller shift vectors and factors on T24 states (SM152)" ran throughout (queue tickets 001540,
then 001543), with other CPU work: CPU load 46% to 79% (median 65%) during the long history, 61% to 96% (median 86%)
during the short one, 60% to 86% for Big. The absolute numbers are therefore "under load"; a first pass with all
three sizes at once (`speed-first-pass-heavy-load.log`, load 44% to 97%) varied by up to 40% on the identical int8
rows and is not used. A clean re-time without the GPU job is a follow-up.

Codec alone (`rwkv_state_lowrank_codec_micro_benchmark`, one layer of 4 heads, frozen and current codec in the same
binary, two runs, load 74%; rank 2 = writes at 2 to 9 answers, rank 4 = writes from answer 41):

| Head size (sizes) | Write at rank 2: frozen / current | Write at rank 4: frozen / current | Read at rank 4: frozen / current |
|---|---:|---:|---:|
| 16 (Small, Tiny) | 7.7 / 3.3 µs (2.3 times faster) | 21.6 / 11.0 µs (2.0 times) | 0.84 / 0.50 µs |
| 32 (Big) | 17.2 / 6.4 µs (2.7 times) | 33.2 / 16.0 µs (2.1 times) | 6.9 / 1.4 µs |

Replay, 1,000 cards x 24 answers (24,000 answers; a third of the card writes and half of the note writes at rank 4):

| Size | int8 rows: reviews/s (µs per review), frozen / current binary | T24 frozen: reviews/s (µs per review) | T24 current: reviews/s (µs per review) | T24 extra µs per review vs int8 rows: frozen -> current | T24 time, current vs frozen |
|---|---|---:|---:|---:|---:|
| Big | 3,987 (250.8) / 3,939 (253.9) | 3,316 (301.6) | 4,219 (237.0) | +49.3 -> -15.3 | -21% |
| Small | 16,793 (59.5) / 17,039 (58.7) | 11,075 (90.3) | 13,804 (72.4) | +31.2 -> +13.3 | -20% |
| Tiny | 32,128 (31.1) / 32,222 (31.0) | 18,334 (54.5) | 25,541 (39.2) | +23.4 -> +8.1 | -28% |

Replay, 4,000 cards x 8 answers (32,000 answers; cards below 16 answers, at rank 1 to 2), Small and Tiny only, heavier
load:

| Size | int8 rows: reviews/s (µs per review), frozen / current binary | T24 frozen | T24 current | T24 extra µs per review vs int8 rows: frozen -> current | T24 time, current vs frozen |
|---|---|---:|---:|---:|---:|
| Small | 14,797 (67.6) / 13,751 (72.7) | 11,993 (83.4) | 14,506 (68.9) | +13.3 -> -1.2 | -17% |
| Tiny | 28,490 (35.1) / 26,961 (37.1) | 21,416 (46.7) | 27,801 (36.0) | +10.6 -> -0.1 | -23% |

(The extra is T24 minus the mean of the two int8 medians. Big's T24 below its int8 rows and the short history's
negative extras are within the load noise; the low-rank states also move 2.5 (Small) to 4.4 (Big) times fewer
bytes than the int8 rows. Best-of-runs gives the same picture: long history, Small +32.0 -> +14.7 µs, Tiny +24.2 -> +7.9 µs.)

Before / after summary as the low-rank states in the fork (SM151) quoted it (long history, T24 vs int8 rows): Small +59% -> about +23%, Tiny +74% ->
about +26%, Big +15% -> no measurable extra. Full state cache rebuilds on long histories take about 1.2 times
(Small) and 1.25 times (Tiny) the int8 time instead of 1.6 and 1.75 times.

## Checks

- `cargo test -p anki --lib` (default features): 1,564 passed, 0 failed, 46 ignored (base `rc-2026-11` 514f3a41d:
  1,560 run, 46 ignored; the branch adds the 2 bit-identity tests in both engines = 4 tests).
- `cargo test -p anki --lib --no-default-features`: 1,564 passed, 0 failed, 45 ignored (base: 1,560 run, 45 ignored).
- `cargo clippy -p anki --lib --tests`, with and without default features: no warnings.
- `cargo fmt -p anki --check`: clean (the nightly formatter only reports comment wrapping in three files this branch
  does not touch).
- The timed current binary is the committed code before one `debug_assert` and formatting (no change in a release
  build).

## Ready to merge?

Yes from this side: bit-identical output (no cache version change, no parity or end-to-end rerun needed), all suites
green, clippy and fmt clean. It needs the user's OK to merge `rc-lowrank-speed` into `rc-2026-11` (a fast-forward of
one commit on top of 514f3a41d). The Deck Options replay-savings table (72% / 84%, measured at four threads with the
old codec) is now slightly pessimistic for the low-rank default; re-measuring it is optional.

## Follow-ups

- Clean re-time without a GPU job (absolute reviews/s; the ratios here come from alternated runs under load).
- Further gains need more independent problems in flight than the four heads of one write: batching the codec of
  several streams (cards) or of the card and note writes of one answer in the bulk replay would let 8 or more lockstep
  lanes share one chain. That is a restructuring of the bulk sweep, not of the codec.
- Optional: re-measure the Deck Options replay savings at four threads with this codec.
