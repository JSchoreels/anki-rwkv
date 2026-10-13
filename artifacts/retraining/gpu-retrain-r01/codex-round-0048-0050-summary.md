# Codex R01 round: iterations 0048–0050

The accepted preliminary expertium reference is **0042: 197,499 reviews/s**, +0.00% versus the round’s starting accepted reference and +558.22% versus the fixed original baseline. Implementation commit `e8b906614639edf1e091f050c82541a9025d878c` is recorded separately from experiment revisions: 0048/0049 froze source before it, and 0050 originated from it. The archived experiment revisions below remain authoritative.

Authorization: I want you to follow R01 cycles, spawn each time a new GPT 6.1 Sol medium thread for 3 iterations, until you reach iteration number 70. Go up to 70 and then shutdown the computer.. Explicit count: 3 of 23 iterations completed; 20 remain authorized. Consecutive misses do not end this counted round.

## Full-history results

Training seconds and raw review presentations/s include user transitions; complete process seconds also include setup, discarded primer, checkpoint saving and evaluation. Gains below use each candidate’s own recorded previous accepted reference, which can differ between rows. One full timing per candidate does not establish repeatability; broader validation remains pending.

| Iteration | Bundle | Training seconds | Reviews/s | Gain vs prior accepted | Complete process seconds | Queue wait s | Summed worker preparation s | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [0042](iteration-0042/summary.md) | E33-F48-F50 | 38.759 | 197,499 | — | 90.735 | 0.47 | 118.3 | Starting accepted reference |
| [0048](iteration-0048/summary.md) | E33-F48-F50-F58 | 39.422 | 194,178 | -1.68% | 90.074 | 0.50 | 116.3 | retain previous accepted run; speed target missed |
| [0049](iteration-0049/summary.md) | E33-F48-F50-F58-F55 | 45.271 | 169,093 | -14.38% | 99.417 | 0.82 | 155.6 | retain previous accepted run; speed target missed |
| [0050](iteration-0050/summary.md) | E33-F48-F50-F58-F62 | 44.715 | 171,194 | -13.32% | 98.788 | 0.75 | 154.0 | retain previous accepted run; speed target missed |

**0048, E33-F48-F50-F58.** Prefetch next backward-step scalar inputs before current recurrence barriers; retain accepted arithmetic and four-row layout. P3 did not favor compact previous-state storage consistently, so omit F54. Recorded decision: retain previous accepted run; speed target missed; exact full recorded parity: True. Training throughput changed -1.68% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.50 versus 0.47; summed preparation s: 116.30 versus 118.28; validation s: 28.44 versus 29.63; primer s: 15.31 versus 15.31; worker CPU %: 52.12 versus 53.23; GPU activity %: 62.49 versus 66.54. These observations include machine-state variation and do not independently attribute a component’s effect.

**0049, E33-F48-F50-F58-F55.** Stack nonnegative F58 with independent wave-aware backward rows. Keep four-row forward; use eight rows for measured long partial-wave shapes and large short histories. Recorded decision: retain previous accepted run; speed target missed; exact full recorded parity: True. Training throughput changed -14.38% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.82 versus 0.47; summed preparation s: 155.63 versus 118.28; validation s: 31.75 versus 29.63; primer s: 15.26 versus 15.31; worker CPU %: 59.92 versus 53.23; GPU activity %: 62.24 versus 66.54. These observations include machine-state variation and do not independently attribute a component’s effect.

**0050, E33-F48-F50-F58-F62.** Add aligned flat master/child, gradient and Adam moment storage to the last nonnegative F58 candidate. Retain per-parameter norm reductions, AdamW arithmetic and None-gradient skips; F55 was negative and is omitted. Recorded decision: retain previous accepted run; speed target missed; exact full recorded parity: True. Training throughput changed -13.32% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.75 versus 0.47; summed preparation s: 154.01 versus 118.28; validation s: 31.77 versus 29.63; primer s: 15.15 versus 15.31; worker CPU %: 59.92 versus 53.23; GPU activity %: 66.42 versus 66.54. These observations include machine-state variation and do not independently attribute a component’s effect.

## Replicated-block decisions

The frozen gate uses 5-MAD trimmed per-update effects, looks at 2/4/6 blocks, Pocock boundary 1.9927, and SD at least the calibrated 1.01%. Raw reviews/s means below describe contemporaneous reference runs; decisions use the paired block effects, rather than the historical 0042 speed. Every reference and candidate replicate retained full recorded exactness; all four held-out losses remained equal, satisfying the 0.001 ceiling.

| Iteration | Blocks | Reference mean reviews/s | Candidate mean reviews/s | Block mean gain | Lower bound | Decision |
|---|---:|---:|---:|---:|---:|---|
| 0048 | 6 | 173,507 | 173,437 | +0.110% | -1.650% | Unproven stack candidate |
| 0049 | 2 | 169,244 | 170,193 | -0.041% | -1.446% | reject |
| 0050 | 2 | 169,693 | 168,733 | -0.220% | -1.623% | reject |

F58 reached the six-block cap with a positive unproven mean; preserve `recurrence_prefetch=true` for stacking. F58+F55 and F58+F62 had negative means at their first looks and are rejected. These stacked results do not isolate independent F55/F62 effects. Best-of paired wall diagnostics were -1.00%, +0.52%, and -0.10% respectively; they are attribution evidence and do not replace the frozen gate. [Paired files and candidate preparation](../../diagnostics/r01-round-0048-0050/preparation.md) retain the machine-state fields. Replicate CPU gates were at most 10%; the standalone 0050 run recorded 11.0%, so its timing is descriptive only. No allocator retries/OOMs were recorded in the paired-run state data; similar high clocks do not establish a cause for the slow-state throughput shift.

Candidate preparation included synthetic recurrence edge/skip tests, nine real WKV shapes and five real model batches. The F55 dtype dispatch failure was repaired before freeze. F62 unpadded packing changed two linear gradients on the 37-review batch; padding each weight slice to preserve 256-byte alignment restored all five batches. Alignment is an inference; no cuBLAS algorithm trace was captured. Aligned transfer/clipping/norm/Adam/moment/step/master-child copies and signed-zero edges passed three steps per batch. Whole-group single-tensor Adam and F58 recompute-loop/per-row prefetch remain pending.

## Separate held-out binary losses

| Iteration | Metric | Loss | Delta vs prior accepted | Delta vs original baseline |
|---|---|---:|---:|---:|
| 0048 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0048 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0048 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0048 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |
| 0049 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0049 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0049 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0049 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |
| 0050 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0050 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0050 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0050 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |

## Checks and provenance

Full recorded parity is exact in 3/3 cases. The recorded acceptance gates and decisions are reused; this report does not change them or substitute parity for broader quality validation.

- 0048: experiment `6a73d011393b4f3c8e24e3a2e7f91d4392886900`, origin `2ee2b4eb16b41030fa93eed50fc7426db9d3788f`, dirty at process start: True; [source archive](../../sources/experiment-sources/537f8a6b311a2a67bd03a36d264dc396581dcfab549362ecad13504291923429/source.zip), [protocol](iteration-0048/protocol.json), [comparison](iteration-0048/comparison.json).
- 0048 pre-run background: 8.2% total CPU; 0 other Python GPU processes detected.
- 0049: experiment `4f675d2790d09d468d9cfa7cf5e7bc7194147a1a`, origin `2ee2b4eb16b41030fa93eed50fc7426db9d3788f`, dirty at process start: True; [source archive](../../sources/experiment-sources/78568fcb38959a91c0149d12bfb640ff9a10fc8ee8c2ea69fcae641540f90ccb/source.zip), [protocol](iteration-0049/protocol.json), [comparison](iteration-0049/comparison.json).
- 0049 pre-run background: 8.5% total CPU; 0 other Python GPU processes detected.
- 0050: experiment `9482526e5b5a66b994dcfa77d3dd1b5db1644d42`, origin `e8b906614639edf1e091f050c82541a9025d878c`, dirty at process start: True; [source archive](../../sources/experiment-sources/a4952d971185594a6d31ac35a7843a45621be94fd52e15911f636c5dd4410666/source.zip), [protocol](iteration-0050/protocol.json), [comparison](iteration-0050/comparison.json).
- 0050 pre-run background: 11.0% total CPU; 0 other Python GPU processes detected.

## Pending next steps

- Continue campaign at --iteration 0051 --previous 0050; inherit budget 23/index 3 and 20 remaining candidates with loss ceiling 0.001. No new round authorization or budget reset; parent creates follow-on chats and final shutdown.
- Accepted reference stays 0042 E33-F48-F50 with exact recorded options. F58 0048 remains the last nonnegative unproven stack candidate: add recurrence_prefetch=true when following the stack order. Omit rejected F55/F62.
- Slot 4: P4 five-real-batch memory/allocator diagnostics for bounded recurrence-only F49 overlap; if no viable memory-neutral overlap, implement F34 slice 1 exact time-mixer glue. Keep GPU replicas gated to CPU at most 10 percent.
- P3b twin A/B attribution remains pending. F58 recompute-loop/per-row load prefetch and F54 latency refinement remain unimplemented; F55 new shape selection needs a concrete changed mechanism before another trial.
- F62 aligned flat-storage slice is exact but measured negative in stack. Whole-group single-tensor AdamW and gradient/clipping dispatch consolidation remain pending; preserve None-gradient/step and signed-zero semantics. Alex and broader validation remain paused.
