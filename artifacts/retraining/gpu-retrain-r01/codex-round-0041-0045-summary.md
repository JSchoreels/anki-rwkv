# Codex R01 round: iterations 0041–0045

The accepted preliminary expertium reference is **0042: 197,499 reviews/s**, +15.69% versus the round’s starting accepted reference and +558.22% versus the fixed original baseline. Implementation commit `66e0bf6e7c70b4580b9078fc8fe9747744ba400b` was made after the experiments.

Authorization: Ok, here is its feedback. Pop now a new thread for 9 iterations.. The authorized maximum was nine new candidate iterations. LOOP stopped this round after five completed iterations because 0043–0045 missed the applicable speed gate consecutively; four budget slots remain unused. Another round requires a new human request.

The accepted result is a measured preliminary bundle improvement, with no independent attribution to F48 or F50. Diagnostic-only historical E33 repeat 0040 reached 182,745 reviews/s (+7.05% vs 0033) with exact recorded parity. F48-only 0041 also coincided with slower preparation/validation. These timing-state differences limit causality and repeatability; all acceptance accounting below uses the original accepted references. Alex and broader validation remained paused.

F53 received a consumer-stream metadata lifetime repair after its 0044 experiment. A CPU tensor-enumeration assertion passed; repaired-path full timing remains unmeasured. The implementation commit includes this repair, while 0044 retains its original source archive. The 0041 frozen plan filename retained the helper default agent label `claude`; this round was performed by Codex, and the plan was preserved unchanged.

## Full-history results

Training seconds and raw review presentations/s include user transitions; complete process seconds also include setup, discarded primer, checkpoint saving and evaluation. Gains below use each candidate’s own recorded previous accepted reference, which can differ between rows. One full timing per candidate does not establish repeatability; broader validation remains pending.

| Iteration | Bundle | Training seconds | Reviews/s | Gain vs prior accepted | Complete process seconds | Queue wait s | Summed worker preparation s | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [0033](iteration-0033/summary.md) | E30-F37-F42 | 44.840 | 170,718 | — | 96.242 | 0.59 | 110.3 | Starting accepted reference |
| [0041](iteration-0041/summary.md) | E33-F48 | 58.617 | 130,593 | -23.50% | 116.298 | 0.81 | 173.7 | retain previous accepted run; speed target missed |
| [0042](iteration-0042/summary.md) | E33-F48-F50 | 38.759 | 197,499 | +15.69% | 90.735 | 0.47 | 118.3 | retain main improvement; preliminary only |
| [0043](iteration-0043/summary.md) | E33-F48-F50-F49 | 203.178 | 37,676 | -80.92% | 255.819 | 0.56 | 115.9 | retain previous accepted run; speed target missed |
| [0044](iteration-0044/summary.md) | E33-F48-F50-F53 | 40.705 | 188,060 | -4.78% | 91.635 | 0.66 | 135.1 | investigate retuning freed resources |
| [0045](iteration-0045/summary.md) | E33-F48-F50-F54 | 38.894 | 196,818 | -0.34% | 89.688 | 0.54 | 120.1 | retain previous accepted run; speed target missed |

**0041, E33-F48.** Disable only deterministic uninitialized-memory fill, keeping deterministic algorithms and the complete E33 training exposure and schedule. Recorded decision: retain previous accepted run; speed target missed; exact full recorded parity: True. Training throughput changed -23.50% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.81 versus 0.59; summed preparation s: 173.73 versus 110.32; validation s: 34.73 versus 29.19; primer s: 15.17 versus 15.03; worker CPU %: 49.97 versus 42.43; GPU activity %: 64.89 versus 69.09. These observations include machine-state variation and do not independently attribute a component’s effect.

**0042, E33-F48-F50.** Combine F48 with an exact row-selection backward that excludes zero hierarchy padding sources while preserving sequential real-source order and the unchanged clamped forward. Recorded decision: retain main improvement; preliminary only; exact full recorded parity: True. Training throughput changed +15.69% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.47 versus 0.59; summed preparation s: 118.28 versus 110.32; validation s: 29.63 versus 29.19; primer s: 15.31 versus 15.03; worker CPU %: 53.23 versus 42.43; GPU activity %: 66.54 versus 69.09. These observations include machine-state variation and do not independently attribute a component’s effect.

**0043, E33-F48-F50-F49.** Run independent hierarchy splits on CUDA streams with unchanged host issue order, parent waits, recorded input/output lifetimes and a backward join before gradient transfer. Recorded decision: retain previous accepted run; speed target missed; exact full recorded parity: True. Training throughput changed -80.92% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.56 versus 0.47; summed preparation s: 115.88 versus 118.28; validation s: 30.27 versus 29.63; primer s: 15.20 versus 15.31; worker CPU %: 11.04 versus 53.23; GPU activity %: 92.40 versus 66.54. These observations include machine-state variation and do not independently attribute a component’s effect.

**0044, E33-F48-F50-F53.** Build stable shift and row-scatter orders in preparation workers, ship immutable int64 metadata in the existing packed/bulk transfer, and install it at update entry; keep all samples and RNG unchanged. Recorded decision: investigate retuning freed resources; exact full recorded parity: True. Training throughput changed -4.78% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.66 versus 0.47; summed preparation s: 135.06 versus 118.28; validation s: 28.53 versus 29.63; primer s: 15.31 versus 15.31; worker CPU %: 57.50 versus 53.23; GPU activity %: 58.75 versus 66.54. These observations include machine-state variation and do not independently attribute a component’s effect.

**0045, E33-F48-F50-F54.** Eliminate the recurrence backward previous-state float buffer by retrieving the last unskipped state from the unchanged chunk buffer or checkpoint; keep all pinned float operations and checkpoint/update schedule. Recorded decision: retain previous accepted run; speed target missed; exact full recorded parity: True. Training throughput changed -0.34% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.54 versus 0.47; summed preparation s: 120.13 versus 118.28; validation s: 28.49 versus 29.63; primer s: 15.32 versus 15.31; worker CPU %: 53.68 versus 53.23; GPU activity %: 64.12 versus 66.54. These observations include machine-state variation and do not independently attribute a component’s effect.

## Separate held-out binary losses

| Iteration | Metric | Loss | Delta vs prior accepted | Delta vs original baseline |
|---|---|---:|---:|---:|
| 0041 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0041 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0041 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0041 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |
| 0042 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0042 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0042 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0042 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |
| 0043 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0043 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0043 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0043 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |
| 0044 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0044 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0044 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0044 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |
| 0045 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0045 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0045 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0045 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |

## Checks and provenance

Full recorded parity is exact in 5/5 cases. The recorded acceptance gates and decisions are reused; this report does not change them or substitute parity for broader quality validation.

- 0041: experiment `b65bb85aa82869be3b8b72e892c3f5bcf0ce6dcb`, origin `722f0427032a542ae0f6b4dd553c791fa1cd2f93`, dirty at process start: True; [source archive](../../sources/experiment-sources/73dc691bb80fda5969edf9cb837b1330aa6d07c09387c66c33c4732b22c715b1/source.zip), [protocol](iteration-0041/protocol.json), [comparison](iteration-0041/comparison.json).
- 0041 pre-run background: 9.5% total CPU; 0 other Python GPU processes detected.
- 0042: experiment `32ad65319fbabc00871ab6a797177f227d5e5fa6`, origin `722f0427032a542ae0f6b4dd553c791fa1cd2f93`, dirty at process start: True; [source archive](../../sources/experiment-sources/b6f8db5fef06815f95bd7423d3a273db130ec3c6a62c18ad6d53f3d1d5dca804/source.zip), [protocol](iteration-0042/protocol.json), [comparison](iteration-0042/comparison.json).
- 0042 pre-run background: 7.4% total CPU; 0 other Python GPU processes detected.
- 0043: experiment `e3f9034c53026cce75dd29424c036cc5ee1d6924`, origin `722f0427032a542ae0f6b4dd553c791fa1cd2f93`, dirty at process start: True; [source archive](../../sources/experiment-sources/a7e0a4363921f544065448c96a76c579592ea2bb4b2a9da6ae932e794d17e474/source.zip), [protocol](iteration-0043/protocol.json), [comparison](iteration-0043/comparison.json).
- 0043 pre-run background: 10.9% total CPU; 0 other Python GPU processes detected.
- 0044: experiment `2c058c7fb407a8eae7b1b7908f46d7c019dcfd28`, origin `722f0427032a542ae0f6b4dd553c791fa1cd2f93`, dirty at process start: True; [source archive](../../sources/experiment-sources/301857ae77d0d5f079439100a93ab8714004bcb5cc2fa7c457bb92ceaef05ba9/source.zip), [protocol](iteration-0044/protocol.json), [comparison](iteration-0044/comparison.json).
- 0044 pre-run background: 8.4% total CPU; 0 other Python GPU processes detected.
- 0045: experiment `cfb4940bee8edb0bae4c7ecf6cbbcf959854fcb7`, origin `722f0427032a542ae0f6b4dd553c791fa1cd2f93`, dirty at process start: True; [source archive](../../sources/experiment-sources/d151ed1b7e44b3007dc766bcd42f51035d080d07ccfb42c5d2265fe39b566525/source.zip), [protocol](iteration-0045/protocol.json), [comparison](iteration-0045/comparison.json).
- 0045 pre-run background: 7.0% total CPU; 0 other Python GPU processes detected.

## Pending next steps

- No new round or broader validation is authorized; Alex remains paused. Three consecutive misses stop the nine-iteration budget after five candidates.
- Accepted reference is 0042 E33-F48-F50, 197499 reviews/s. Keep original E33 0033 accounting and disclose diagnostic 0040 +7.05 percent drift; no routine control repeats.
- F53 consumer-stream metadata lifetime repair is implemented and CPU enumeration-checked after 0044; repaired-path training timing is pending. Review transfer cost and worker retuning only under fresh authorization.
- F54 compact previous-state storage halves local memory but missed the speed gate. Further latency/barrier refinements or F55 independent forward/backward layouts need a new concrete mechanism and checks.
- F49 split streams were exact but 80.92 percent slower; revisit only recurrence-only overlap or another mechanism with explicit RNG/accumulation/lifetime evidence.
- F47 grouping amortization remains unmeasured. F51/F52 ragged/padding-free behavior changes remain high-risk nonexact ideas with 5 percent gate and four separate loss ceilings. F56/F57 remain screened-negative or too small.
