# R01 next-round order: iterations 0047–0056 (Expertium)

This is a proposed order, not an authorization. It was written on 2026-10-12 from the [0041–0045 review](../artifacts/diagnostics/r01-round-0041-0045-review/summary.md) and the [ledger](GPU_OPTIMIZATIONS.md), then revised with the human's decisions below. A round starts only on a quoted human request passed with `--round-iterations 10 --loss-ceiling 0.001`; counted rounds run to completion through misses. Iteration 0046 is the uncounted, human-authorized instrumented repeat of 0042. Alex/Default and broader validation stay paused unless the human says otherwise. Track details live in the ledger rows named below.

## Human decisions (2026-10-12)

1. **Smaller gains are fine.** Every run now logs per-update host and device timing (commit e3b15e8). `scripts/r01_paired_timing.py` compares exact runs update by update; with several runs per side it uses the per-update minimum (best-of). The promotion rule for gains below the frozen band is proposed below. It needs an instrumented A/A pair to calibrate.
2. **Turnaround counts only if it matters on millions of reviews.** Fixed per-run costs vanish at that scale, so primer work (F61), pool spawn and first-segment warm-up (F64) are out. Evaluation runs at about 29k reviews/s against training's 197k, and that cost scales with held-out reviews, so F60 (fewer evaluation syncs) and the overlap part of F59 (prepare held-out batches during training) stay in.
3. **Loss differences: below 0.1% for now.** `--loss-ceiling 0.001` freezes the ceiling at round start; continuations inherit it. Regressions within it may be promoted, checked against both the previous accepted run and the round's starting reference, so small regressions cannot add up past 0.1% within a round. A later round can raise the ceiling. At 0.1%, anything that changes dropout masks or update order will likely miss: seed spread was 0.36–0.77%. Non-exact candidates should keep the RNG stream and change only floating-point order.
4. **Instrumented repeat authorized; keep small gains.** 0046 is that repeat. `decide()` marks positive gains that miss the promotion rule as `stack_candidate`, and iteration summaries name them. The last slot combines them.

## What the order rests on

- **Gate.** A single full run must beat the frozen noise band: 7.045% for Expertium (0033 vs 0040). Until the small-gain rule is implemented, the order stacks exact pieces so a stack can clear it.
- **Noise.** 0033 and 0040 differ by 7–8% in every work segment, a whole-run slow state; 0041's loss is a late transient. Paired segments of 0042 and 0045 differ by about ±4%. Slow runs tend to show lower power per utilization point or more VRAM held by other processes; neither is proven causal. Clocks, P-state and other-process VRAM are now recorded.
- **Time budget in 0042.** Training is 38.8 s of an 88.5 s case. GPU activity is 66.5%. Large updates are GPU-bound: the recurrence backward is about 99 of 237 GPU ms on the diagnostic large update. Medium updates are host-bound. Queue wait totals 0.47 s, so feeding-side ideas (F47, F53) have about a 1% ceiling.
- **VRAM.** Accepted runs use 14.2–14.9 GB of the 16 GB card. F49 (15.46 GB reserved) and the 640-chunk run collapsed near the cap. Allocator retries are now recorded per run.

## Proposed small-gain promotion rule (implement after calibration)

An exact candidate whose total gain is inside the band is promoted only when all of these hold:
- the paired per-update wall gain against the instrumented reference is positive, with its update-resampling 95% lower bound above the A/A noise measured between two instrumented repeats;
- neither run carries a machine-state flag: no allocator retries, other-process VRAM at most about 2.5 GB, busy SM clock median within 3% of the reference;
- the total gain is not below the negative band.

Twin A/B timing on real batches (P3b) gives attribution inside promoted stacks. Calibration needs one more uncounted instrumented repeat of 0042 to form the A/A pair, because 0042 itself has no per-update timing. That repeat needs the human's OK.

## Before 0047: no counted iterations

| Step | Work | Status |
|---|---|---|
| P1 | Per-update timing, allocator counters, device memory at start, GPU clocks/P-state/event reasons; paired comparison tool; round loss ceiling; stack candidates. | Done, e3b15e8: 143 policy and 10 paired-timing CPU checks; instrumented smoke bitwise equal to 0042's. |
| P1b | 0046 instrumented repeat of 0042 (uncounted). | Done ([summary](../artifacts/retraining/gpu-retrain-r01/iteration-0046/summary.md)): exact parity, but −18.13% in a slow state (pre-launch background CPU 22.4% vs 7%, GPU at maximum clocks with lower power). Under max/min it would raise the Expertium band to 22.149%. Human decision needed on the band before 0047 is prepared. |
| P1c | A/A calibration repeat, then implement the small-gain rule in `decide()` with policy checks. | Needs the human's OK for the second repeat. |
| P2 | Launch hygiene: `r01_iterate.py` records `memory.used`, SM clock and P-state with background load, flags other-process VRAM above about 2.5 GB, and waits or refuses when pre-launch background CPU exceeds about 10% (0042/0045: 7%, slow 0046: 22.4%). Add per-update main-thread CPU time (`time.thread_time`) to separate a descheduled dispatch thread from a slower one. The human keeps the same Windows power plan and NVIDIA power mode for every run and closes VRAM-heavy apps; system settings stay the human's. | Pending. |
| P3 | Recurrence microbenchmarks on the nine real WKV shapes: F58 prefetch with and without F54, F55 rows per row count; bitwise set (216 synthetic cases, five real batches) on the chosen variant. | Pending. |
| P3b | Twin A/B timing: extend `check_r01_exact_options.py` to run each sampled real batch under A and B back to back, alternating order, with RNG restored; report the paired gain with a CI. | Pending. |
| P4 | F49 memory diagnostic on five real batches: peak reserved and allocator retries for current streams, recurrence-only overlap with main-stream outputs, and two streams only. | Pending. |
| P5 | F62 feasibility: bitwise check of flat-buffer AdamW, clipping and master/child copies, including signed zeros, on five batches. | Pending. |
| T | F60 and the F59 held-out overlap, as a harness change outside the count; `validation_input_sha256` and all four scores identical. | Pending. |

## Iteration order

| Slot | Iteration | Candidate (stacked on the last non-negative candidate) | Kind | Why at this point |
|---|---|---|---|---|
| 1 | 0047 | F58 recurrence next-step prefetch, plus F54 if P3 favours it | Exact kernel | The dominant GPU kernel on GPU-bound large updates. |
| 2 | 0048 | + F55 wave-aware backward rows per row count | Exact option | Variants are already bitwise equal; a 512-row backward runs 3 waves for 2.03 waves of work. |
| 3 | 0049 | + F62 flat parameter, gradient and Adam buffers (if P5 passes) | Exact host | Host-bound updates; about 30 ms host per update in old profiles. Scales with updates. |
| 4 | 0050 | + bounded F49 streams if P4 shows no VRAM growth; otherwise F34 slice 1 | Exact concurrency | Up to about 23 ms per large update if recurrence overlap works within memory. |
| 5 | 0051 | F34 slice 1: bitwise fused time-mixer lerp/glue, forward and backward | Exact kernel | About 40,000 dispatches per large update dominate host-bound updates. |
| 6 | 0052 | Flex: repeat a stacked run that the state flags marked slow (concrete repeat reason), or F34 slice 2 | Repeat or exact | Separates a machine-state miss from a real one. |
| 7 | 0053 | F34 slice 2 or 3: channel mixer, or GroupNorm/bonus/gate glue | Exact kernel | Same mechanism, next op group. |
| 8 | 0054 | F51 RNG-preserving hybrid: batch per-token ops across a stream's splits, keep dropout calls and per-split weight-gradient reductions in the original order | Non-exact (float order only) | 34.1 layer instances per update for 14 layers; built to stay within 0.1%. |
| 9 | 0055 | F51 continued or F52 padding-free positions under the same RNG rule | Non-exact | 24.4% of hierarchy positions are padding. |
| 10 | 0056 | Stack consolidation: combine every remaining stack candidate of the round | Bundle | The human's point 4: several small gains together may clear the noise. |

## Rules for running the order

- **Stack, don't isolate.** While the single-run band applies, each exact slot adds its piece to the previous candidate. A piece stays unless the full run or its paired per-update effect is negative. Before stacking, every piece needs its own exactness check (option checker, five real batches, kernel synthetic cases for new kernels). A promoted stack becomes the reference.
- **Record small gains.** Read `stack_candidate` in each comparison and carry those options into the round handoff for slot 10.
- **Attribution.** Use twin A/B timing (P3b) on real batches for components inside a promoted stack. Do not spend iterations on single-component runs.
- **Read each run before the decision.** Run `scripts/r01_paired_timing.py` against the instrumented reference and record the state flags in the iteration summary.
- **Out of this round:** F47 (feeding ceiling about 1%), F53 alone, F56, F22 fastmode, F61, F64, Alex, and broader validation.
