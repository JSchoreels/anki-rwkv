# Batched low-rank codec in the fork (SM168): result

Written Wed 11-25, 08:10. Fork branch `codec-batch`, commit **39e8b308f** (one commit on fork main eda153ddb).
Post-release work: not merged into fork main or `rc-2026-11`; the weekend release stays at eda153ddb.
**Before** = fork main eda153ddb (the state-building speed re-measure (SM163) "after" build), **after** = 39e8b308f.
Release builds in `wt/fork-lowrank-speed` (checked out detached for the builds, then back on `rc-lowrank-speed`).
Synthetic data only.

## Verdict

- **The one-thread long-history gap is mostly closed.** On the one-thread replay of 1,000 cards x 24 answers, the
  default low-rank state now trails int8 rows by **3.2%** for Small (was 17.9%) and **5.0%** for Tiny (was 24.2%).
  Before vs after on the default: Small +18.9%, Tiny +25.3%, Big +13.1%.
- **The full rebuild got faster, not slower.** On the realistic first-launch rebuild (pairs, all threads, 205,785
  synthetic reviews), the default runs +5.4% (Big), +4.0% (Small) and +5.0% (Tiny) faster than before. Small and Tiny
  are now level with int8 rows (+0.2% and +0.5%, was -3.4% and -4.8%); Big is +48.7% vs int8 rows (was +38.9%).
- **Same bytes.** Every write stores the bytes and the matrix that one write at a time stores, bit for bit (new test,
  both head sizes, int8 and int6 factors, 1, 2 and 4 cards per solve), and the frozen-codec tests still pass. The
  whole suite passes with the vector width capped at baseline, AVX2 and AVX-512.
- **Int8 rows unchanged:** before vs after within -1.6% to +0.9% everywhere (inside the run-to-run spread).
- **Per-review latency improved a little too** (the per-review write uses the same solve): Review show-time p50
  -5% (Big), -7% (Small), -12% (Tiny) on the default format.

## What changed

- `lowrank::encode_layers` encodes several layer writes at once: the heads of up to four writes (four cards) are the
  lanes of one solve. The small Jacobi eigen problems store the lanes side by side (`a[i][j][lane]`), so the
  rotations of all lanes run in SIMD registers; a lane that would not rotate keeps its values through a select.
  `encode_layer` (the per-review write) is the batch of one.
- On x86-64 the whole solve is also compiled for AVX2 and AVX-512 and picked at run time. No operation is fused or
  reordered, so every width gives the same bits.
- B = Q^T S runs row by row (each value still sums the rows in order), and the largest factor magnitude uses eight
  running maxima (a maximum gives the same value in any order).
- The bulk replay steps the low-rank streams of a chunk in groups of four (longest first) and factors each step's
  writes in one batched solve; per stream every operation and its order are unchanged.
- With 128-bit vectors only (CPUs without AVX2, and Apple Silicon), the replay keeps one card per solve: more lanes
  there only added work in the codec benchmark. These machines get the restructured code at one card per solve,
  about as fast as before (codec benchmark capped at baseline: -9% to +2% time per write vs before). They were not
  measured end to end here.
- The decode (`decode_block`, queue scoring) is unchanged: it is not on the replay's hot path, and queue scoring was
  already as fast as int8 rows (SM163).

## Quiet machine and method

- Same harness and method as the state-building speed re-measure (SM163): `sm163_speed.rs` (temporary untracked
  example, copied into `rslib/examples/` for the two builds and removed afterwards) and `rwkv_model_size_benchmark`.
  Four configurations (before int8 rows, after int8 rows, before default, after default) in alternating
  before/after order (ABAB), reversed on pass 2; 3 passes; every number is the median of the 3. "Spread" is
  (max - min) / median.
- Runtimes as in SM163: Big = show-time big model (SM91s2) + show-time instant-only big model (SM99); Small = Small
  two-head (SM118) + Small Instant-Only (SM113); Tiny = six-layer Tiny two-head (SM130) + six-layer Tiny Instant-Only
  (SM130). Default format = `compact_lowrank` (Big and Small int6 factors, Tiny int8 factors).
- CPU: AMD Ryzen 7 9850X3D (16 threads, AVX-512). Run 07:42 to 08:01. A GPU quiet window covered 07:44 to 09:15
  ("SM168 batched low-rank codec speed measure"); it was removed at 08:02, once the timing was done. Before every
  step: CPU load 0% to 7% (median 2%), GPU 0% (twice 4%) at 598 MiB, no compute process, empty queue.
- Tests: `cargo test -p anki --lib` 1,605 passed, `--no-default-features` 1,605 passed (baseline 1,603 each, plus the
  new batched-codec test on both engines).

## One-thread replay (reviews/s; models alone)

1,000 cards x 24 answers (every card reaches the rank-4 tier):

| Size | int8 rows before / after | default before (spread) | default after (spread) | after vs before | vs int8 rows before | vs int8 rows after |
|---|---:|---:|---:|---:|---:|---:|
| Big | 4,557 / 4,560 | 4,940 (0.4%) | **5,589** (1.6%) | +13.1% | +8.4% | **+22.6%** |
| Small | 20,968 / 21,160 | 17,225 (0.4%) | **20,475** (1.5%) | +18.9% | -17.9% | **-3.2%** |
| Tiny | 39,784 / 39,765 | 30,165 (1.6%) | **37,793** (1.7%) | +25.3% | -24.2% | **-5.0%** |

4,000 cards x 8 answers (cards below 16 answers, rank 1 to 2):

| Size | int8 rows before / after | default before | default after | after vs before | vs int8 rows after |
|---|---:|---:|---:|---:|---:|
| Small | 20,596 / 20,480 | 20,087 | 21,550 | +7.3% | +5.2% |
| Tiny | 38,243 / 37,630 | 37,111 | 39,712 | +7.0% | +5.5% |

The before numbers match SM163 within 1% to 5%.

## Full first-launch rebuild (reviews/s of the replay; pairs, all threads, 205,785 reviews)

| Size | int8 rows before / after | default before (spread) | default after (spread) | after vs before | vs int8 rows before | vs int8 rows after | whole rebuild, default before -> after (int8 rows) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Big | 10,457 / 10,303 | 14,530 (0.3%) | **15,320** (1.7%) | +5.4% | +38.9% | **+48.7%** | 14.39 -> 13.65 s (20.44 s) |
| Small | 48,913 / 49,030 | 47,227 (1.0%) | **49,121** (0.3%) | +4.0% | -3.4% | **+0.2%** | 4.47 -> 4.30 s (4.35 s) |
| Tiny | 87,017 / 86,544 | 82,844 (0.7%) | **86,959** (0.7%) | +5.0% | -4.8% | **+0.5%** | 2.58 -> 2.47 s (2.49 s) |

## Per-review latency (pairs, µs, median over runs of p50; show-time / answer)

| Size | Format | Review before | Review after | Learning before | Learning after |
|---|---|---:|---:|---:|---:|
| Big | int8 rows | 3,126 / 763 | 3,079 / 754 | 3,100 / 740 | 3,055 / 726 |
| Big | default | 3,019 / 767 | 2,869 / 717 | 2,820 / 686 | 2,749 / 657 |
| Small | int8 rows | 572 / 187 | 562 / 184 | 567 / 188 | 560 / 184 |
| Small | default | 640 / 200 | 594 / 191 | 577 / 188 | 552 / 182 |
| Tiny | int8 rows | 287 / 82 | 282 / 82 | 315 / 108 | 307 / 106 |
| Tiny | default | 351 / 103 | 310 / 91 | 312 / 106 | 303 / 104 |

The Small and Tiny show-time extra cost vs int8 rows on Review cards drops from +68 and +64 µs to +32 and +28 µs.

## Codec alone (one layer of 4 heads, one thread, per layer write; end of `speed-run.log`)

| Head size (sizes) | rank | before (one card) | after, 1 card | after, 2 cards per solve | after, 4 cards per solve |
|---|---:|---:|---:|---:|---:|
| 16 (Small, Tiny), int8 | 2 | 3,019 ns | 2,222 ns | 2,121 ns | 1,914 ns |
| 16, int8 | 4 | 9,837 ns | 6,281 ns | 4,752 ns | 4,177 ns |
| 32 (Big), int8 | 2 | 5,422 ns | 3,531 ns | 3,161 ns | 3,372 ns |
| 32, int8 | 4 | 14,597 ns | 8,422 ns | 6,876 ns | 6,633 ns |

Int6 factors are within 1% to 8% of int8 at every point. A rank-4 write costs 58% less for Small and Tiny and 55%
less for Big with four cards per solve.

## Files

- `speed-run.log`: the whole run (load probes, replay lines, codec benchmark).
- `app-rebuild-latency.jsonl`: the 36 rebuild and latency records.
- `summary.json`: medians and every run (`python summarize.py .` prints the tables' numbers).
- `run.sh` (the driver), `summarize.py`. The example is the SM163 one, unchanged (`../sm163/sm163_speed.rs`).
