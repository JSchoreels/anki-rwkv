# Rank-2 to rank-4 low-rank states in the fork (SM151): result

Written Tue 11-24. Fork branch `rc-lowrank` (worktree `wt/fork-lowrank`), merged into `rc-2026-11` as approved by the
user on 11-23 ("ok let's integrate in anki and push when finished"). Format and decisions: `design.md` here and the
fork's `docs/rwkv-lowrank-state-2026-11-23.MD`. No GPU job was used.

## What was built

- **Codec** (`rslib/src/rwkv/lowrank.rs`): the training repository's `LowRankCodec` (low-rank int8 states experiment
  (SM148)) in Rust. Rank 2 below 16 answers and rank 4 from the 16th (T24, label `w8+lr2.16.4q8x8`) by default, any
  schedule as `compact_lowrank_int8:<spec>` (plain rank 2, Q2, is `compact_lowrank_int8:2`). Factors are int8 with one
  float16 scale per vector; shift vectors stay compact int8. Precision byte `0x28` under the existing state magic; the
  decoder still reads every older layout.
- **Integration**: per-answer path (inside the time mixer) and bulk replay (inside the per-stream sweep) call the same
  encoder, so both paths stay bit-identical; queries, undo, checkpoints and spills carry the new layout.
- **Default switch**: `compact_lowrank_int8` is the profile default for Big, Small and Tiny; saved
  `compact_cards_int8` profiles move to it (decision kept as built by the main session); the state cache version goes
  14 -> 15, so every cache rebuilds once from the revlog; int8 rows stay readable and writable natively.
- **This run**: end-to-end tests on benchmark users (ignored, inputs from an environment variable, never bundled),
  a codec micro-benchmark, configurable benchmark sizes (`ANKI_RWKV_SIZE_BENCH_CARDS` / `_ROUNDS`), a Jacobi stopping
  fix (below), and the Deck Options tooltip with the byte range (rank 1 to rank 4) beside the mean.

## Parity

Synthetic codec vectors from the scorer's `LowRankCodec.encode` (`qt/tools/rwkv_lowrank_state_codec_vectors.py`; head
size 16 and 32; T24 and Q2 with int8 factors; T24 with float factors), unchanged by this run's Jacobi stop (identical
counts and errors before and after):

- each write from the scorer's own previous state: int8 factors identical on 625 of 630 head writes (head size 32:
  252 of 252; head size 16: 249 of 252, the rest one code step apart, relative error at most 3.5e-3); float factors at
  most 2.2e-6;
- the Rust chain on its own: mean relative difference at most 1.3e-3 (int8, head size 16), 0 (int8, head size 32),
  at most 7.3e-6 (float factors);
- per-answer and bulk replay bit-identical at the new precision (the existing precision tests, now also run at
  `compact_lowrank_int8`).

## End to end on benchmark users

Benchmark users 2388, 4676 and 1866 of the low-rank int8 states experiment (SM148) smoke set (1866 has a card with
128+ answers), show-time inputs (`e2s-query`), deleted cards `neighbour`, the scorer's own data pipeline (seed
2025 + user; `e2e_big_vectors.py`). The fork runtime ran the release models with float32 weights (as Anki loads them)
through the ignored tests `rwkv_big_lowrank_state_end_to_end` (big engine) and `rwkv_small_lowrank_state_end_to_end`
(small engine); `e2e_compare.py` / `e2e_compare_small.py` compute the LogLoss (aggregates in `e2e/`). Curve LogLoss
is the raw forgetting curve, as the fork predicts it (the scorer's `curve_raw_log_loss`); instant LogLoss is
1 - P(again) on the query rows.

Two checks, both under the discrete-codec rule (per user |delta LogLoss| <= 3e-4 and mean signed delta <= 1e-4):

1. **Rust vs the scorer on the same rows (Big only):** the scorer run on the same features with the same state formats
   (float32 weights, the fork's placement: the first row of a stream keeps its own shift input). The big models run
   the fork architecture in the scorer, so its per-row outputs are directly comparable.
2. **Rust T24 minus Rust int8 rows vs the same difference in the low-rank int8 states experiment (SM148)**, per user (all six release models). That experiment scored int8
   weights (`w8`), the fork runs float32 weights, so the absolute LogLoss differs slightly and only the deltas are
   compared. For Small and Tiny the scorer path is the eager training model with state hooks; reproducing its per-row
   outputs was not needed because check 2 plus the codec parity on head size 16 cover the same arithmetic.

Every check passes. Largest per-user |Rust - scorer| LogLoss at T24: 1.8e-5 (Big); largest |Rust delta - delta of the low-rank int8 states experiment (SM148)|:
5.5e-5 (Big), 5.2e-5 (Small and Tiny). In % of the user's LogLoss, T24 vs int8 rows on the Big curve model (SM91s2):
-0.07% (2388), +0.02% (4676), +0.07% (1866), matching the experiment's -0.10%, +0.01%, +0.06%.


show-time big model (SM91s2):

| User | rows (curve / instant labels) | int8 rows: Rust curve / instant | T24: Rust curve / instant | T24 minus int8, Rust (curve / instant) | same in low-rank int8 states (SM148) | Q2 minus int8, Rust | same in low-rank int8 states (SM148) | max per-user abs(Rust - scorer) LogLoss, T24 |
|---|---|---|---|---|---|---|---|---|
| 2388 | 2,096 / 2,096 | 0.162069 / 0.079711 | 0.161962 / 0.079717 | -0.000107 / +0.000006 | -0.000155 / +0.000000 | -0.000107 / +0.000006 | -0.000156 / +0.000000 | 4.4e-06 / 1.9e-07 |
| 4676 | 2,838 / 2,838 | 0.083439 / 0.063303 | 0.083459 / 0.063344 | +0.000019 / +0.000041 | +0.000010 / +0.000053 | +0.000017 / +0.000064 | +0.000045 / +0.000080 | 1.1e-05 / 7.7e-06 |
| 1866 | 3,547 / 3,547 | 0.166796 / 0.114542 | 0.166913 / 0.114632 | +0.000116 / +0.000090 | +0.000096 / +0.000034 | +0.000626 / +0.000050 | +0.000595 / +0.000013 | 5.8e-06 / 1.8e-05 |

Gates (per user abs <= 3e-4, mean signed <= 1e-4):

| Check | max abs | mean signed | pass |
|---|---:|---:|---|
| fp32 rust-scorer curve | 2.15e-08 | +4.88e-09 | yes |
| fp32 rust-scorer instant | 3.78e-09 | -1.54e-09 | yes |
| compact_int8 rust-scorer curve | 3.63e-06 | +1.77e-06 | yes |
| compact_int8 rust-scorer instant | 2.97e-06 | +1.23e-06 | yes |
| t24 rust-scorer curve | 1.14e-05 | +4.22e-07 | yes |
| t24 rust-scorer instant | 1.80e-05 | +8.49e-06 | yes |
| q2 rust-scorer curve | 4.17e-06 | -1.39e-06 | yes |
| q2 rust-scorer instant | 1.12e-05 | +2.01e-06 | yes |
| t24-int8 Rust vs low-rank int8 states (SM148) curve | 4.80e-05 | +2.60e-05 | yes |
| t24-int8 Rust vs low-rank int8 states (SM148) instant | 5.51e-05 | +1.64e-05 | yes |
| q2-int8 Rust vs low-rank int8 states (SM148) curve | 4.90e-05 | +1.75e-05 | yes |
| q2-int8 Rust vs low-rank int8 states (SM148) instant | 3.69e-05 | +8.71e-06 | yes |

show-time instant-only big model (SM99):

| User | rows (curve / instant labels) | int8 rows: Rust curve / instant | T24: Rust curve / instant | T24 minus int8, Rust (curve / instant) | same in low-rank int8 states (SM148) | Q2 minus int8, Rust | same in low-rank int8 states (SM148) | max per-user abs(Rust - scorer) LogLoss, T24 |
|---|---|---|---|---|---|---|---|---|
| 2388 | 2,096 / 2,096 | 0.145835 / 0.079163 | 0.145940 / 0.079222 | +0.000106 / +0.000059 | +0.000090 / +0.000063 | +0.000106 / +0.000059 | +0.000091 / +0.000063 | 7.4e-07 / 1.0e-06 |
| 4676 | 2,838 / 2,838 | 0.089361 / 0.063037 | 0.089156 / 0.063045 | -0.000205 / +0.000008 | -0.000183 / -0.000002 | -0.000185 / +0.000066 | -0.000193 / +0.000063 | 4.4e-06 / 1.1e-05 |
| 1866 | 3,547 / 3,547 | 0.190651 / 0.113346 | 0.190888 / 0.113336 | +0.000237 / -0.000010 | +0.000234 / +0.000039 | +0.000409 / -0.000131 | +0.000478 / -0.000098 | 1.3e-05 / 1.3e-06 |

Gates (per user abs <= 3e-4, mean signed <= 1e-4):

| Check | max abs | mean signed | pass |
|---|---:|---:|---|
| fp32 rust-scorer curve | 7.60e-09 | +2.02e-10 | yes |
| fp32 rust-scorer instant | 2.39e-09 | +7.84e-10 | yes |
| compact_int8 rust-scorer curve | 2.13e-06 | +1.43e-06 | yes |
| compact_int8 rust-scorer instant | 5.88e-06 | -1.33e-06 | yes |
| t24 rust-scorer curve | 1.27e-05 | +2.53e-06 | yes |
| t24 rust-scorer instant | 1.10e-05 | +4.43e-06 | yes |
| q2 rust-scorer curve | 1.67e-05 | -1.81e-06 | yes |
| q2 rust-scorer instant | 5.87e-06 | +2.67e-06 | yes |
| t24-int8 Rust vs low-rank int8 states (SM148) curve | 2.17e-05 | -1.06e-06 | yes |
| t24-int8 Rust vs low-rank int8 states (SM148) instant | 4.90e-05 | -1.43e-05 | yes |
| q2-int8 Rust vs low-rank int8 states (SM148) curve | 6.93e-05 | -1.57e-05 | yes |
| q2-int8 Rust vs low-rank int8 states (SM148) instant | 3.32e-05 | -1.13e-05 | yes |

Small and Tiny (fork runtime deltas vs the low-rank int8 states (SM148) deltas; T24 minus int8 rows, curve / instant):

| Model | User | Rust T24 - int8 | low-rank int8 states (SM148) T24 - REF | Rust Q2 - int8 | low-rank int8 states (SM148) Q2 - REF |
|---|---|---|---|---|---|
| Small curve model (SM118) | 2388 | +0.000174 / -0.000015 | +0.000173 / +0.000010 | +0.000174 / -0.000015 | +0.000174 / +0.000010 |
| Small curve model (SM118) | 4676 | -0.000002 / -0.000047 | -0.000004 / -0.000037 | -0.000059 / -0.000008 | -0.000046 / +0.000010 |
| Small curve model (SM118) | 1866 | -0.000099 / +0.000088 | -0.000128 / +0.000068 | -0.000184 / +0.000065 | -0.000227 / +0.000075 |
| Small instant-only model (SM113) | 2388 | +0.000001 / -0.000003 | +0.000001 / -0.000002 | +0.000001 / -0.000003 | +0.000001 / -0.000002 |
| Small instant-only model (SM113) | 4676 | +0.000009 / +0.000007 | +0.000010 / +0.000015 | +0.000009 / -0.000024 | +0.000010 / -0.000024 |
| Small instant-only model (SM113) | 1866 | +0.000001 / -0.000015 | +0.000001 / -0.000036 | -0.000004 / -0.000026 | -0.000012 / -0.000081 |
| six-layer Tiny two-head model (SM130) | 2388 | +0.000229 / -0.000014 | +0.000236 / +0.000007 | +0.000229 / -0.000014 | +0.000236 / +0.000007 |
| six-layer Tiny two-head model (SM130) | 4676 | +0.000041 / +0.000080 | +0.000014 / +0.000118 | +0.000103 / +0.000291 | +0.000093 / +0.000337 |
| six-layer Tiny two-head model (SM130) | 1866 | -0.000097 / +0.000096 | -0.000081 / +0.000044 | -0.000420 / +0.000202 | -0.000443 / +0.000159 |
| six-layer Tiny instant-only model (SM130) | 2388 | +0.000000 / +0.000065 | +0.000000 / +0.000037 | +0.000000 / +0.000065 | +0.000000 / +0.000037 |
| six-layer Tiny instant-only model (SM130) | 4676 | -0.000000 / +0.000208 | -0.000000 / +0.000187 | -0.000000 / +0.000114 | -0.000000 / +0.000104 |
| six-layer Tiny instant-only model (SM130) | 1866 | -0.000000 / +0.000017 | -0.000001 / +0.000025 | -0.000001 / +0.000150 | -0.000000 / +0.000157 |

| Check | max abs | mean signed | pass |
|---|---:|---:|---|
| sm113 t24-int8 Rust vs low-rank int8 states (SM148) curve | 7.99e-07 | -8.97e-08 | yes |
| sm113 t24-int8 Rust vs low-rank int8 states (SM148) instant | 2.10e-05 | +4.30e-06 | yes |
| sm113 q2-int8 Rust vs low-rank int8 states (SM148) curve | 7.63e-06 | +2.45e-06 | yes |
| sm113 q2-int8 Rust vs low-rank int8 states (SM148) instant | 5.46e-05 | +1.77e-05 | yes |
| sm118 t24-int8 Rust vs low-rank int8 states (SM148) curve | 2.92e-05 | +1.07e-05 | yes |
| sm118 t24-int8 Rust vs low-rank int8 states (SM148) instant | 2.51e-05 | -5.07e-06 | yes |
| sm118 q2-int8 Rust vs low-rank int8 states (SM148) curve | 4.23e-05 | +1.00e-05 | yes |
| sm118 q2-int8 Rust vs low-rank int8 states (SM148) instant | 2.50e-05 | -1.78e-05 | yes |
| sm130io t24-int8 Rust vs low-rank int8 states (SM148) curve | 1.87e-07 | +4.76e-08 | yes |
| sm130io t24-int8 Rust vs low-rank int8 states (SM148) instant | 2.80e-05 | +1.38e-05 | yes |
| sm130io q2-int8 Rust vs low-rank int8 states (SM148) curve | 9.67e-08 | -3.39e-08 | yes |
| sm130io q2-int8 Rust vs low-rank int8 states (SM148) instant | 2.80e-05 | +1.00e-05 | yes |
| sm130th t24-int8 Rust vs low-rank int8 states (SM148) curve | 2.70e-05 | +1.38e-06 | yes |
| sm130th t24-int8 Rust vs low-rank int8 states (SM148) instant | 5.20e-05 | -2.46e-06 | yes |
| sm130th q2-int8 Rust vs low-rank int8 states (SM148) curve | 2.30e-05 | +8.81e-06 | yes |
| sm130th q2-int8 Rust vs low-rank int8 states (SM148) instant | 4.65e-05 | -8.40e-06 | yes |


## Speed

Release build, one thread (`RAYON_NUM_THREADS=1`), `rwkv_model_size_benchmark` (full state replay through the bulk
path, synthetic answers; the time includes the whole model, not only the state write and read), int8 rows
(`compact_cards_int8`) and T24 (`compact_lowrank_int8`) timed back to back, twice each, mean of the two medians of 3
samples. Two histories: the default 4,000 cards x 8 answers (cards stay below 16 answers, so at rank 1 to 2; notes,
two cards each, reach 16 in the last round) and 1,000 cards x 24 answers (a third of the card writes and half of the
note writes at rank 4).

**Load:** the GPU job "Smaller shift vectors and factors on T24 states (SM152)" ran during every timing (queue ticket
001537, its CPU data loading included) and other CPU work was active: CPU load 19% to 71% at the block starts. The
absolute reviews/s are therefore "under load" (the int8 rows of the same history varied by up to 25% between runs);
the ratios come from back-to-back pairs. A clean re-time is a follow-up.

**Finding and fix:** the first run showed T24 much slower than expected (up to 2.6 times the int8 replay time for Tiny
on long cards). The cause was the cyclic Jacobi eigen solver: its stop rule (off-diagonal mass at or below 1e-32 of the
diagonal) is below what float64 rounding reaches for many 3 x 3 and 5 x 5 problems, so a third of the 5 x 5 solves ran
all 60 sweeps (a Python replica of the solver: mean 22 sweeps, 37% at the cap). The solver now also stops when a sweep
no longer reduces the off-diagonal mass (rotations below one ulp): mean 5.4 sweeps, identical eigenvalues and vectors
on the replica's 300 random problems, and identical codec parity counts and errors in the Rust tests. Codec alone
after the fix (`rwkv_state_lowrank_codec_micro_benchmark`, one layer of 4 heads, mostly rank 4): write 19 to 21 µs
(head size 16) and 33 µs (head size 32), read 0.7 µs (16) and 6 µs (32).

Before the Jacobi stop (first run):

| History | Size | int8 rows: reviews/s | µs per review | T24: reviews/s | µs per review | T24 extra µs per review | T24 time vs int8 |
|---|---|---:|---:|---:|---:|---:|---:|
| 4,000 cards x 8 answers (32,000) | Big | 4,123 | 242.5 | 3,842 | 260.3 | +17.8 | +7% |
| 4,000 cards x 8 answers (32,000) | Small | 18,271 | 54.7 | 12,843 | 77.9 | +23.1 | +42% |
| 4,000 cards x 8 answers (32,000) | Tiny | 34,157 | 29.3 | 21,930 | 45.6 | +16.3 | +56% |
| 1,000 cards x 24 answers (24,000) | Big | 4,034 | 247.9 | 2,846 | 351.4 | +103.4 | +42% |
| 1,000 cards x 24 answers (24,000) | Small | 17,853 | 56.0 | 7,561 | 132.3 | +76.2 | +136% |
| 1,000 cards x 24 answers (24,000) | Tiny | 31,786 | 31.5 | 11,991 | 83.4 | +51.9 | +165% |

With the Jacobi stop (as merged):

| History | Size | int8 rows: reviews/s | µs per review | T24: reviews/s | µs per review | T24 extra µs per review | T24 time vs int8 |
|---|---|---:|---:|---:|---:|---:|---:|
| 4,000 cards x 8 answers (32,000) | Big | 3,455 | 289.5 | 3,614 | 276.7 | -12.8 | -4% |
| 4,000 cards x 8 answers (32,000) | Small | 16,436 | 60.8 | 13,330 | 75.0 | +14.2 | +23% |
| 4,000 cards x 8 answers (32,000) | Tiny | 30,703 | 32.6 | 23,438 | 42.7 | +10.1 | +31% |
| 1,000 cards x 24 answers (24,000) | Big | 3,653 | 273.8 | 3,169 | 315.5 | +41.8 | +15% |
| 1,000 cards x 24 answers (24,000) | Small | 16,397 | 61.0 | 10,285 | 97.2 | +36.2 | +59% |
| 1,000 cards x 24 answers (24,000) | Tiny | 30,644 | 32.6 | 17,600 | 56.8 | +24.2 | +74% |

With the fix, the low-rank state costs about 10 to 14 µs per review on short histories (Small and Tiny; Big within the
noise) and 24 to 42 µs per review once cards pass 16 answers. In the app this is the per-answer cost (well under a
millisecond); the visible effect is the one-time state cache rebuild after the upgrade (cache version 15) and later
full rebuilds, which take up to about 1.6 times (Small) and 1.75 times (Tiny) as long on long histories, about 1.15
times for Big.

**Factored update (ideas note "Idea 1"):** not adopted. The dense float update is computed anyway because the answer's
own output must use it (the scorer's placement); the measured extra cost is in the codec's small solves (Jacobi, the
B = Q^T S' product and the reconstruction), not in the K x K update, so a factored recurrence would not remove it.
Cheaper exact options for later: fixed-size arrays instead of per-head allocations, and float32 small solves checked
against the parity tests.

**Deck Options replay savings** (4 threads, 32,000 answers, the benchmark the page quotes; at the default only): at
the low-rank default, Small and Tiny save 72% and 84% of Big's replay time as pairs (70% and 84% alone, instant-only
70% and 83%), against 77% and 88% (76% and 87%) at the int8 rows. Medians of two back-to-back runs under the same load
(Big pair 5,740 ms, Small pair 1,617 ms, Tiny pair 902 ms; Big alone 3,136 ms, Small 946 ms, Tiny 505 ms). The page's
table, its unit test and the deck-options end-to-end test now use these values (the end-to-end test also moves to the
low-rank default; its shared-model byte line was stale before this change).


## Bytes per card and note

Fork layout (one model; a pair stores two states plus a 19-byte wrapper). The mean is over the answer counts of the
723 fit benchmark users of the 732 set (the byte estimate of the low-rank int8 states experiment (SM148) plus the fork's 2 bytes per layer for the count and the
rank). The Deck Options page shows the mean; its tooltip now shows the rank 1 to rank 4 range.

| Size | card / note at rank 1 | rank 2 | rank 4 | mean (card / note) | int8 rows (`compact_cards_int8`) | saving vs int8 rows (card + note, mean) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Big | 1,657 / 1,111 | 2,473 / 1,655 | 4,105 / 2,743 | 2,524 / 1,720 | 13,891 / 9,267 | -81.7% |
| Small | 533 / 309 | 821 / 453 | 1,397 / 741 | 839 / 470 | 2,545 / 1,315 | -66.1% |
| Tiny | 309 / 309 | 453 / 453 | 741 / 741 | 462 / 470 | 1,315 / 1,315 | -64.6% |

## Long cards

`cargo test --release -p anki --lib rwkv_long_card_full -- --ignored --nocapture` (`RWKV_LONG_CARD_THREADS=8`,
20,000 synthetic answers on one card; rows in `longcard-rows.jsonl`): 10 models x 3 patterns x 3 policies x 2 paths =
180 runs, every run complete and finite, no failure. The low-rank default never saturates a scale (int8 rows: 2 of
30 per-answer runs, int4 cards: 22 of 30). Mean |recall - float32 recall| over the per-answer runs:

| Policy | answers 1-500 | 500-5,000 | 5,000-20,000 | largest late-window drift |
| --- | ---: | ---: | ---: | ---: |
| `compact_cards_int8` (int8 rows) | 9.2e-4 | 2.0e-3 | 3.4e-3 | 1.9e-2 |
| `compact_lowrank_int8` (T24, new default) | 2.2e-3 | 3.1e-3 | 4.6e-3 | 2.6e-2 (show-time big model (SM91s2), mixed pattern) |
| `compact_cards_int4` | 7.4e-3 | 9.2e-3 | 1.2e-2 | 7.8e-2 |

The low-rank drift sits between the int8 rows and the int4 cards, as the benchmark LogLoss does; it stays bounded over
20,000 answers.

## Checks

- `cargo test -p anki --lib` (default features): 1,560 passed, 0 failed, 46 ignored on `rc-lowrank` (42 ignored
  before this run plus the 4 new manual tests). Base `rc-2026-11` (aecc5b09d), same command: 1,546 passed, 0 failed, 42 ignored (the branch adds 14 tests that run by default and 4 manual ones). The brief's
  402/402 is a different count (not this command).
- `cargo test -p anki --lib --no-default-features rwkv::`: 416 passed, 0 failed, 38 ignored.
- `cargo clippy -p anki --lib --tests` with and without default features: no warnings. `cargo fmt --check`: clean
  after formatting the two new test blocks.
- Merged tree (`rc-2026-11` after the merge): merge commit 514f3a41d (parents aecc5b09d and `rc-lowrank` 316111fc1, the same tree as `rc-lowrank`): `cargo test -p anki --lib` 1,560 passed, 0 failed, 46 ignored; `--no-default-features rwkv::` 416 passed, 38 ignored; `cargo fmt --check` clean. Pushed to `origin/rc-2026-11` (aecc5b09d..514f3a41d, fast-forward, no force).

## Tests that must run on the Mac (full `just build`)

- Python: `qt/tests/test_rwkv_precision.py`, `qt/tests/test_rwkv_model.py`, `qt/tests/test_rwkv_inference_process.py`,
  `qt/tests/test_rwkv_scheduler.py`, `qt/tests/test_rwkv_srs_benchmark.py` (new default, the moved
  `compact_cards_int8` profile value, cache version 15, the new policy string).
- TypeScript: `ts/routes/deck-options/rwkv-model-estimates.test.ts` (new default bytes, the byte range, replay savings
  72% / 84%); `ts/tests/e2e/deck-options.test.ts` (now at `compact_lowrank_int8`: Small 2.7 KB per card + note and
  2.2 MB as a pair, 1.3 KB and 1.1 MB shared, 72% / 70% less replay time). Its shared-model line expected 3.5 KB /
  2.7 MB before this run, which matched no precision of the release candidate, so it was likely failing already.
- The new ftl string `deck-config-rwkv-model-per-card-note-range` (tooltip) and `just check` (formatting of the
  Svelte and TypeScript changes, which were not formatted here: no Node on this PC).

## Open risks and follow-ups

- **Speed:** the low-rank state makes replay slower: +10 to 14 µs per review for Small and Tiny on short histories,
  +24 to 42 µs on long ones (full rebuilds up to about 1.6 times (Small) and 1.75 times (Tiny) as long, about 1.15
  times for Big), measured under load. Follow-ups: a clean re-time without the GPU job, and cheaper exact codec code
  (no per-head allocations, fixed-size arrays; float32 small solves only with the parity tests).
- The Deck Options byte estimate is a mean; long-studied collections store more (up to the rank 4 size). The tooltip
  says so.
- Saved `compact_cards_int8` profiles move to the new default (as built); int8 rows stay a native precision.
- End to end used three benchmark users; the 732-user accuracy comes from the low-rank int8 states experiment (SM148)
  in the scorer, which this run reproduces per user.

