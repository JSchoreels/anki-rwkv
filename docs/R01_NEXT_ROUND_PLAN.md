# R01 next-round order: iterations 0046–0055 (Expertium)

This is a proposed order, not an authorization. It was written on 2026-10-12 from the [0041–0045 review](../artifacts/diagnostics/r01-round-0041-0045-review/summary.md) and the [ledger](GPU_OPTIMIZATIONS.md). A round starts only on a quoted human request passed with `--round-iterations 10`; counted rounds run to completion through misses. Alex/Default and broader validation stay paused unless the human says otherwise. Track details live in the ledger rows named below.

## What the order rests on

- **Gate.** Since 52d81c4, a candidate must beat the frozen noise band with one full run. For Expertium the band is 7.045%, from 0033 vs 0040. Non-exact candidates must also meet the four loss ceilings. Most single exact ideas cannot clear 7% alone, so the order stacks exact pieces (see the rules).
- **Noise.** 0033 and 0040 differ by 7–8% in every work segment, a whole-run slow state. 0041's loss is a late transient. Paired segments of 0042 and 0045 differ by about ±4%. Slow runs tend to show lower power per utilization point (about 1.7 vs 2.5 W/%) or more VRAM held by other processes. Neither is proven causal.
- **Time budget in 0042.** Training is 38.8 s of an 88.5 s case. Held-out evaluation takes 29.6 s and the primer 15.3 s. GPU activity is 66.5%. Large updates are GPU-bound: the recurrence backward is about 99 of 237 GPU ms on the diagnostic large update. Medium updates are host-bound. Queue wait totals 0.47 s, all in the first eighth of work, so feeding-side ideas (F47, F53) have about a 1% ceiling.
- **VRAM.** Accepted runs use 14.2–14.9 GB of the 16 GB card. F49 (15.46 GB reserved) and the 640-chunk run collapsed near the cap. Any candidate that adds memory must report `peak_reserved_mib`.

## Decisions for the human before 0046

1. **Gate estimator.** Keep single-run totals against 7.045%, or adopt the paired per-update estimator in R01-M01, recomputing the band with the same estimator. If paired timing is adopted, run slots 1–5 singly instead of stacking.
2. **Turnaround.** May evaluation and primer savings (F59–F61) be accepted on complete process time and identical scores? They cannot change reviews/s.
3. **Non-exact work.** May slots 9–10 go to F51/F52 ragged execution under the loss ceilings, or must the round stay exact?
4. **Instrumented reference.** Authorize an uncounted, instrumented 0042 repeat (like 0040) as the paired reference? Under the max/min rule it can only widen the band.

## Before 0046: no training, CPU or few-batch checks

| Step | Work | Done when |
|---|---|---|
| P1 | R01-M01 instrumentation. Record host wall and CUDA-event GPU ms per update, read late as in F33. Add SM/memory clocks, P-state and throttle reasons to the GPU monitor. Record allocator `num_alloc_retries`/`num_ooms` and other-process VRAM at launch in the result. | Accounting only. A smoke shows identical updates and scores. |
| P2 | Launch hygiene. `background-load.json` records other-process VRAM and flags runs above about 2.5 GB. The human keeps the same Windows power plan and NVIDIA power mode for every run and closes VRAM-heavy apps. System settings stay the human's to change. | Field present in the guard output. |
| P3 | Recurrence microbenchmarks on the nine real WKV shapes, using the `wkv-shapes.json` method. Test F58 next-step load prefetch, with and without F54 compact state, and F55 rows per row count. Run the bitwise set (216 synthetic cases, five real batches) on the chosen variant. | Kernel time per shape and bitwise pass recorded. |
| P4 | F49 memory diagnostic on five real batches. Collect peak reserved memory and allocator retries for: current streams; recurrence-only overlap with outputs allocated on the main stream; two streams only. | Shows whether overlap is possible without VRAM growth. |
| P5 | F62 feasibility. Bitwise check of flat-buffer AdamW, clipping and master/child copies, including signed zeros, on five batches. | Pass, or a recorded reason to drop it. |
| T | If decision 2 allows: F59 (reuse the training pool for evaluation and prepare held-out batches early), F60 (fewer evaluation syncs) and F61 (primer breakdown), as a harness change outside the iteration count. | `validation_input_sha256` and all four scores identical. Every later iteration finishes sooner. |

## Iteration order

| Slot | Iteration | Candidate (stacked on the last non-negative candidate) | Kind | Why at this point |
|---|---|---|---|---|
| 1 | 0046 | F58 recurrence next-step prefetch, plus F54 if P3 favours it | Exact kernel | The dominant GPU kernel on GPU-bound large updates. |
| 2 | 0047 | + F55 wave-aware backward rows per row count | Exact option | Variants are already bitwise equal. A 512-row backward runs 3 waves for 2.03 waves of work. |
| 3 | 0048 | + F62 flat parameter, gradient and Adam buffers (if P5 passes) | Exact host | Host-bound updates. About 30 ms host per update in old profiles. |
| 4 | 0049 | + F64 allocator pre-reserve in the primer (only if P1 shows a first-segment allocator cost) + F57 bookkeeping | Exact host | The first eighth runs 257 ms/update on small updates. Disclose any cost moved into primer time. |
| 5 | 0050 | + bounded F49 streams (only if P4 shows no VRAM growth); otherwise F34 slice 1 | Exact concurrency | Up to about 23 ms per large update if recurrence overlap works within memory. |
| 6 | 0051 | Flex: repeat a stacked run that P1/P2 flagged as a slow state (concrete repeat reason), or the next F34 slice | Repeat or exact | Separates a machine-state miss from a real one. |
| 7 | 0052 | F34 slice 1: bitwise fused time-mixer lerp/glue, forward and backward | Exact kernel | About 40,000 dispatches per large update dominate host-bound updates. |
| 8 | 0053 | F34 slice 2: channel mixer, or GroupNorm/bonus/gate glue | Exact kernel | Same mechanism, next-largest op group. |
| 9 | 0054 | F51 ragged single-pass streams, opt-in (decision 3); else F34 slice 3 | Non-exact | 34.1 layer instances per update for 14 layers. |
| 10 | 0055 | F52 padding-free positions on F51, or an F51 retune; else F34 slice 4 or F63 groundwork | Non-exact | 24.4% of hierarchy positions are padding. |

## Rules for running the order

- **Stack, don't isolate.** While the gate is single-run 7.045%, each exact slot adds its piece to the previous candidate. A piece stays in the stack unless the full run or its paired per-update effect is negative. Before stacking, every piece needs its own exactness check (option checker, five real batches, kernel synthetic cases for new kernels). Once a stack is promoted, it becomes the reference.
- **Attribution.** Use the R01-M01 in-run A/B (twin updates, RNG restored) on a few real batches for components inside a promoted stack. Do not spend iterations on single-component runs.
- **Read each run before the decision.** Check telemetry against the slow-state signals (power per utilization point, other-process VRAM, peak reserved) and record what you find in the iteration summary.
- **Out of this round:** F47 (feeding ceiling about 1%), F53 alone, F56, F22 fastmode, Alex, and broader validation.
