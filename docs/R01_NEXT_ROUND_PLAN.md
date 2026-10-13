# R01 next-round order: ten counted iterations from 0048 (Expertium)

This is a proposed order, not an authorization. It was written on 2026-10-12 from the [0041–0045 review](../artifacts/diagnostics/r01-round-0041-0045-review/summary.md) and the [ledger](GPU_OPTIMIZATIONS.md), then revised with the human's decisions below. A round starts only on a quoted human request passed with `--round-iterations 10 --loss-ceiling 0.001`; counted rounds run to completion through misses. Iterations 0046 (instrumented repeat of 0042) and 0047 (calibration replicates) are uncounted and human-authorized; the counted round starts at 0048. Alex/Default and broader validation stay paused unless the human says otherwise. Track details live in the ledger rows named below.

## Active counted campaign (2026-10-13)

The human authorized: "I want you to follow R01 cycles, spawn each time a new GPT 6.1 Sol medium thread for 3 iterations, until you reach iteration number 70. Go up to 70 and then shutdown the computer." The frozen campaign budget is 23 counted candidates, 0048-0070, with loss ceiling 0.001. The existing slot order below governs the first ten candidates; later slots must be chosen from preserved pending tracks with the same gates. Workers execute successive three-candidate batches; the parent chat creates later fresh chats and performs final shutdown. Continuations inherit the campaign budget and use the previous counted protocol without resetting it. Alex and broader validation remain paused.

Completed first batch: 0048 F58 is an exact unproven stack candidate (+0.110% mean, six-block cap); 0049 F58+F55 and 0050 F58+F62 are negative and rejected. Accepted reference remains 0042 E33-F48-F50. Next is `--iteration 0051 --previous 0050`, inheriting 20 remaining authorized candidates. Slot 4 requires P4 bounded F49 memory diagnostics; use F34 slice 1 if P4 does not support bounded overlap. Preserve F58 as the last nonnegative candidate (`recurrence_prefetch=true`) for stacking; omit F55/F62 unless a new mechanism is checked.

## Human decisions (2026-10-12)

1. **Smaller gains are fine.** Every run now logs per-update host and device timing (commit e3b15e8). `scripts/r01_paired_timing.py` compares exact runs update by update; with several runs per side it uses the per-update minimum (best-of). The promotion rule for gains below the frozen band is proposed below. It needs an instrumented A/A pair to calibrate.
2. **Turnaround counts only if it matters on millions of reviews.** Fixed per-run costs vanish at that scale, so primer work (F61), pool spawn and first-segment warm-up (F64) are out. Evaluation runs at about 29k reviews/s against training's 197k, and that cost scales with held-out reviews, so F60 (fewer evaluation syncs) and the overlap part of F59 (prepare held-out batches during training) stay in.
3. **Loss differences: below 0.1% for now.** `--loss-ceiling 0.001` freezes the ceiling at round start; continuations inherit it. Regressions within it may be promoted, checked against both the previous accepted run and the round's starting reference, so small regressions cannot add up past 0.1% within a round. A later round can raise the ceiling. At 0.1%, anything that changes dropout masks or update order will likely miss: seed spread was 0.36–0.77%. Non-exact candidates should keep the RNG stream and change only floating-point order.
4. **Instrumented repeat authorized; keep small gains.** 0046 is that repeat. `decide()` marks positive gains that miss the promotion rule as `stack_candidate`, and iteration summaries name them. The last slot combines them.
5. **Measure with replicates, not against the best run ever seen.** The human's direction: "better do more runs and have a better statistical understanding of our results than just merely doing 7% from the best we ever saw". A single-run comparison against a promoted reference suffers from the winner's curse, because that reference was promoted for measuring high. Each iteration therefore re-measures the reference alongside the candidate (see the measurement design).

## What the order rests on

- **Gate.** A single full run must beat the frozen noise band. Since the human's option C (2026-10-12) the band is the median spread over all identical-configuration repeat pairs: 6.364% for Expertium (4 pairs; the 0042/0046 pair alone is 22.149%), 3.833% for Alex. A neutral change beats the median about one time in four under symmetric noise, so stack candidates and paired timing provide the follow-up. Until the small-gain rule is implemented, the order still stacks exact pieces.
- **Noise.** 0033 and 0040 differ by 7–8% in every work segment, a whole-run slow state; 0041's loss is a late transient. Paired segments of 0042 and 0045 differ by about ±4%. Slow runs tend to show lower power per utilization point or more VRAM held by other processes; neither is proven causal. Clocks, P-state and other-process VRAM are now recorded.
- **Time budget in 0042.** Training is 38.8 s of an 88.5 s case. GPU activity is 66.5%. Large updates are GPU-bound: the recurrence backward is about 99 of 237 GPU ms on the diagnostic large update. Medium updates are host-bound. Queue wait totals 0.47 s, so feeding-side ideas (F47, F53) have about a 1% ceiling.
- **VRAM.** Accepted runs use 14.2–14.9 GB of the 16 GB card. F49 (15.46 GB reserved) and the 640-chunk run collapsed near the cap. Allocator retries are now recorded per run.

## Measurement design (human direction 2026-10-12)

- **Interleaved blocks.** Each counted iteration runs the reference and the candidate as full-history replicates in alternating order (R C R C …) with `scripts/r01_replicates.py`. Each block's effect is log(candidate/reference) of its adjacent pair, so slow drift in machine state cancels within a block.
- **Decision.** `scripts/r01_replicate_stats.py` averages block effects with a 95% t interval. Promote when the lower bound is above zero (and parity is exact, or losses are within the 0.1% ceiling). A positive mean whose interval includes zero is a stack candidate. Otherwise reject.
- **Block count.** Start with the number calibration says is needed for a 3% gain. Add blocks, up to a cap, while the interval straddles zero and the mean is positive. Report the reference speed as the mean of its replicates, not as one promoted run.
- **Quiet machine.** No replicate launches unless system CPU is at most 10%. The runner waits, then refuses and records the refusal.
- **Calibration (done, 0047).** Six exact replicates: 180,236 reviews/s mean, 2.57% run SD. Across 15 identical-run pairs, total time has SD 3.74%, while the trimmed time-weighted per-update gain (5 MAD) has SD 1.01% and recovers heterogeneous effects without bias ([analysis](../artifacts/diagnostics/r01-calibration-0047/summary.md)). At about 3–4 minutes per block: 2 blocks resolve a 2% gain, 7 blocks a 1% gain.
- **Gate (approved 2026-10-13, implemented in `3914aed`).** Block effect: 5-MAD trimmed per-update gain of the candidate against its adjacent reference run. Looks after 2, 4 and 6 blocks against one Pocock boundary z = 1.9927, standard error from max(calibrated 1.01%, observed block SD). Promote if the lower bound is above zero; reject if the mean is not positive; extend otherwise; a positive result unproven at six blocks is a stack candidate. Simulated power: 2% gain 79% at 2 blocks, 99.8% by 6 (2.5 blocks on average); 1% gain 72% by 6; neutral change 5%.
- **Harness.** Done: `decide()` reads the block decision; new protocols freeze the rule and its parameters; `r01_replicates.py` runs blocks inside a counted iteration. The median pairwise band stays only as descriptive history.

Twin A/B timing on real batches (P3b) stays useful for attribution inside promoted stacks and for screening small exact options cheaply before spending full-run blocks.

## Before the counted round (0048): no counted iterations

| Step | Work | Status |
|---|---|---|
| P1 | Per-update timing, allocator counters, device memory at start, GPU clocks/P-state/event reasons; paired comparison tool; round loss ceiling; stack candidates. | Done, e3b15e8: 143 policy and 10 paired-timing CPU checks; instrumented smoke bitwise equal to 0042's. |
| P1b | 0046 instrumented repeat of 0042 (uncounted). | Done ([summary](../artifacts/retraining/gpu-retrain-r01/iteration-0046/summary.md)): exact parity, but −18.13% in a slow state (pre-launch background CPU 22.4% vs 7%, GPU at maximum clocks with lower power). Under max/min it would have raised the Expertium band to 22.149%; the human chose option C, a median pairwise band (6.364%). |
| P1c | Calibration and gate. | Done: 0047 calibration (six exact replicates; [analysis](../artifacts/diagnostics/r01-calibration-0047/summary.md)); gate approved 2026-10-13 and implemented in `3914aed` (162 policy checks). |
| P2 | Launch hygiene: `r01_iterate.py` records `memory.used`, SM clock and P-state with background load, flags other-process VRAM above about 2.5 GB, and waits or refuses when pre-launch background CPU exceeds about 10% (0042/0045: 7%, slow 0046: 22.4%). Add per-update main-thread CPU time (`time.thread_time`) to separate a descheduled dispatch thread from a slower one. The human keeps the same Windows power plan and NVIDIA power mode for every run and closes VRAM-heavy apps; system settings stay the human's. | Pending. |
| P3 | Recurrence microbenchmarks on the nine real WKV shapes: F58 prefetch with and without F54, F55 rows per row count; bitwise set (216 synthetic cases, five real batches) on the chosen variant. | Done for 0048/0049: 216 synthetic cases, nine real shapes and five real batches; see [candidate preparation](../artifacts/diagnostics/r01-round-0048-0050/preparation.md). F54 not consistently favorable. |
| P3b | Twin A/B timing: extend `check_r01_exact_options.py` to run each sampled real batch under A and B back to back, alternating order, with RNG restored; report the paired gain with a CI. | Pending. |
| P4 | F49 memory diagnostic on five real batches: peak reserved and allocator retries for current streams, recurrence-only overlap with main-stream outputs, and two streams only. | Pending. |
| P5 | F62 feasibility: bitwise check of flat-buffer AdamW, clipping and master/child copies, including signed zeros, on five batches. | Done for aligned flat-storage slice: five batches, three optimizer steps, signed-zero/copy/norm/moment checks pass. Per-parameter AdamW/None-gradient semantics retained; whole-group single-tensor Adam remains pending. |
| T | F60 and the F59 held-out overlap, as a harness change outside the count; `validation_input_sha256` and all four scores identical. | Pending. |

## Iteration order

| Slot | Iteration | Candidate (stacked on the last non-negative candidate) | Kind | Why at this point |
|---|---|---|---|---|
| 1 | 0048 | F58 recurrence next-step prefetch, plus F54 if P3 favours it | Exact kernel | The dominant GPU kernel on GPU-bound large updates. |
| 2 | 0049 | + F55 wave-aware backward rows per row count | Exact option | Variants are already bitwise equal; a 512-row backward runs 3 waves for 2.03 waves of work. |
| 3 | 0050 | + F62 flat parameter, gradient and Adam buffers (if P5 passes) | Exact host | Host-bound updates; about 30 ms host per update in old profiles. Scales with updates. |
| 4 | 0051 | + bounded F49 streams if P4 shows no VRAM growth; otherwise F34 slice 1 | Exact concurrency | Up to about 23 ms per large update if recurrence overlap works within memory. |
| 5 | 0052 | F34 slice 1: bitwise fused time-mixer lerp/glue, forward and backward | Exact kernel | About 40,000 dispatches per large update dominate host-bound updates. |
| 6 | 0053 | Flex: repeat a stacked run that the state flags marked slow (concrete repeat reason), or F34 slice 2 | Repeat or exact | Separates a machine-state miss from a real one. |
| 7 | 0054 | F34 slice 2 or 3: channel mixer, or GroupNorm/bonus/gate glue | Exact kernel | Same mechanism, next op group. |
| 8 | 0055 | F51 RNG-preserving hybrid: batch per-token ops across a stream's splits, keep dropout calls and per-split weight-gradient reductions in the original order | Non-exact (float order only) | 34.1 layer instances per update for 14 layers; built to stay within 0.1%. |
| 9 | 0056 | F51 continued or F52 padding-free positions under the same RNG rule | Non-exact | 24.4% of hierarchy positions are padding. |
| 10 | 0057 | Stack consolidation: combine every remaining stack candidate of the round | Bundle | The human's point 4: several small gains together may clear the noise. |

## Rules for running the order

- **Measure single pieces when blocks can resolve them; stack the rest.** With interleaved blocks, a piece whose expected gain is at least the calibrated detectable effect is measured alone. Smaller pieces are stacked onto the previous candidate and stay unless their block effect is negative. Every piece first needs its own exactness check (option checker, five real batches, kernel synthetic cases for new kernels). A promoted candidate becomes the reference, re-measured in the next iteration's blocks.
- **Record small gains.** Read `stack_candidate` in each comparison and carry those options into the round handoff for slot 10.
- **Attribution.** Use twin A/B timing (P3b) on real batches for components inside a promoted stack. Do not spend iterations on single-component runs.
- **Read each run before the decision.** Run `scripts/r01_paired_timing.py` against the instrumented reference and record the state flags in the iteration summary.
- **Out of this round:** F47 (feeding ceiling about 1%), F53 alone, F56, F22 fastmode, F61, F64, Alex, and broader validation.

## For the agent running the round

- **Start.** Quote the human's request: `scripts/r01_iterate.py --iteration 0048 --previous 0045 --round-authorization "<quoted request>" --round-iterations 10 --loss-ceiling 0.001 …`. Use `--previous 0045`: iterations 0046 and 0047 are uncounted diagnostics without `comparison.json`, as 0040 was. The accepted reference stays 0042 (E33-F48-F50), and the round-start reference for the loss ceiling is frozen to it.
- **Per iteration.** Plan, prepare, select, smoke and one candidate full run (parity, losses, profile) as in AGENT.md. Then step 5b: `scripts/r01_replicates.py run --output artifacts/retraining/gpu-retrain-r01/iteration-NNNN --sequence <control>,<candidate>,<control>,<candidate> --reason "<why>"` as one ordinary background job. Then the report. If the report says to run blocks to the next look, run two more blocks and rerun it.
- **Known pitfalls.**
  - The isolated runtime copies the iteration folder when it is first built, so never write live logs or records inside the folder while runs are pending; `r01_replicates.py` and `r01_iterate.py` already write beside it.
  - A detached runner process disappeared during calibration, so use ordinary background jobs.
  - Replicates wait for system CPU at most 10% and refuse after `--max-wait-minutes`; a game (AOW4.exe) was the cause before.
- **First use.** The report's replicate-block path (`report_gpu_r01_next.py` with `replicated_blocks`) is CPU-checked but untested on a real iteration. If it fails on 0048, repair it, rerun the report, and record the repair; a repaired report does not consume an iteration.
- **Candidate preparation.** P3 (recurrence microbenchmarks), P4 (F49 memory) and P5 (F62 bitwise feasibility) belong to their slots' candidate preparation. P2 (CPU gate inside `r01_iterate.py`) matters less now, because only gated replicate blocks decide speed.

## Campaign handoff after 0053

0051–0053 completed under the inherited campaign authorization. P4 failed memory-neutral independent-overlap prerequisites, so 0051 used exact gate-forward F34 glue; 0052 fused its backward and 0053 tested channel ReLU-square without gates. All were exact but rejected at two blocks (−1.654%, −1.288%, −1.207%). Accepted 0042 and retained unproven 0048 F58 are unchanged. [Preparation](../artifacts/diagnostics/r01-round-0051-0053/preparation.md).

Continue `--iteration 0054 --previous 0053` without resetting budget or loss ceiling: 6/23 completed, 17 remaining through 0070. For slot 7, establish exact feasibility for another F34 group (eight-way shift/lerp and its original reductions, bonus/GroupNorm glue or vectorized loading); do not repeat negative bundles unchanged. If none is viable, advance to F51 RNG-preserving split batching preparation, retaining dropout-call and per-split weight-gradient order. F58 alone remains for stacking; omit rejected F55/F62 and these glue slices without a concrete changed mechanism. Parent manages fresh chats and shutdown.

## Campaign handoff after 0056

0054–0056 completed exactly: distinct F34 bonus products, F51 input-gather hybrid and F52 padding-index slice were exact but rejected at 2/2/4 blocks (−1.226%, −1.103%, −0.365%). 0056 extended after its first positive look, then rejected at four; no sixth look required. [Preparation](../artifacts/diagnostics/r01-round-0054-0056/preparation.md), [compact handoff](../artifacts/retraining/gpu-retrain-r01/codex-round-0054-0056-handoff.json).

Continue `--iteration 0057 --previous 0056` without resetting budget or ceiling: 9/23 complete, 14 remaining, loss ceiling 0.001. Accepted 0042 remains unchanged; only prior 0048 F58 survives as an unproven stack. Omit all rejected 0049–0056 slices. Slot 10 has no new component to consolidate: an identical 0048 repeat is not a useful counted candidate. Select a concretely changed pending mechanism (F58 recompute/per-row prefetch, genuinely deferred F49 overlap, whole-group F62 Adam) with its prerequisites, or establish the larger F51/F52 RNG/reduction-preserving hybrid. Keep pending P3b attribution and exact eight-way lerp feasibility visible. Alex and broader validation remain paused; parent handles fresh chats and shutdown.


## Extended campaign order from 0057

No component from 0051–0056 survived; slot 10 consolidation is replaced with changed F58 mechanisms. 0057 adds recomputation scalar/per-row v prefetch, 0058 adds per-row backward gradient prefetch, 0059 adds forward scalar prefetch. Each is checked independently against accepted 0042 and starts with retained F58 only; carry any newly nonnegative component only when a compatible combined check supports it. These remain unmeasured until their own frozen blocks complete.

Remaining slots 0060–0070 are preparation-dependent: 0060 P3b twin A/B attribution followed by a changed recurrence load schedule; 0061 whole-group F62 AdamW respecting None gradients, per-parameter steps and signed zeros; 0062 exact eight-way shift/lerp forward; 0063 its backward with original reduction order; 0064 independent deferred F49 overlap only if memory-neutral preparation passes; 0065 F51 per-token normalization hybrid; 0066 F51 linear scheduling retaining split weight-gradient order; 0067 F52 compact valid-position mixer work; 0068 F52 padding-free recurrence with exact skipped-state/output semantics; 0069 consolidate surviving changed components after combined checks; 0070 next changed pending mechanism if consolidation has no surviving component, otherwise a distinct checked extension. This is a pending order, not a claim of feasibility or new authorization. Do not count identical repeats or no-op candidates; failed prerequisites require a concretely changed pending replacement. Preserve original campaign budget/baseline and cumulative 0.001 ceiling.


## Campaign handoff after 0059

0057-0059 completed: changed F58 recomputation, per-row gradient and forward scalar pipelines all exact but rejected at 4/2/2 blocks. Omit recurrence_load_pipeline modes 1/2/3. Only prior 0048 F58 remains unproven; accepted 0042/options unchanged. Continue --iteration 0060 --previous 0059, original budget 23/index 12 complete, 11 remaining and cumulative ceiling 0.001. [Handoff](../artifacts/retraining/gpu-retrain-r01/codex-round-0057-0059-handoff.json).

Slot 13 (0060): establish P3b twin A/B attribution for inherited F58, then prepare a concretely changed mechanism (forward per-row v prefetch or shape-selective load schedule justified by shape evidence), or advance to whole-group F62 AdamW. Diagnostics do not replace counted measurements. Whole-group AdamW, exact eight-way lerp, deferred memory-neutral F49 and substantive F51/F52 remain pending. Preserve the extended order with feasibility-based substitutions; do not count identical replays, empty consolidations or no-op candidates. Parent handles chats/shutdown; Alex and broader validation paused.
