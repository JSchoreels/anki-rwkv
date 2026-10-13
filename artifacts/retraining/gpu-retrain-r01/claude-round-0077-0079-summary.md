# Claude R01 round: iterations 0077–0079

The accepted preliminary expertium reference is **0079: 182,206 reviews/s**, -1.54% versus the round’s starting accepted reference and +507.25% versus the fixed original baseline. Implementation commit `187c81ad028aa78da1b18c348a1ae23b8ac61228` was made after the experiments.

Authorization: Ten full R01 training cycles. Explicit count: 8 of 10 iterations completed; 2 remain authorized. Consecutive misses do not end this counted round.

## Full-history results

Training seconds and raw review presentations/s include user transitions; complete process seconds also include setup, discarded primer, checkpoint saving and evaluation. Gains below use each candidate’s own recorded previous accepted reference, which can differ between rows. One full timing per candidate does not establish repeatability; broader validation remains pending.

| Iteration | Bundle | Training seconds | Reviews/s | Gain vs prior accepted | Complete process seconds | Queue wait s | Summed worker preparation s | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [0076](iteration-0076/summary.md) | E72-F71-F73a | 41.366 | 185,056 | — | 84.273 | 0.50 | 108.1 | Starting accepted reference |
| [0077](iteration-0077/summary.md) | E72-F71-F73b | 39.559 | 193,507 | +4.57% | 83.021 | 0.58 | 96.8 | retain previous accepted run; speed target missed |
| [0078](iteration-0078/summary.md) | E72-F71-F73a-F86 | 39.104 | 195,758 | +5.78% | 81.841 | 0.53 | 107.3 | retain previous accepted run; speed target missed |
| [0079](iteration-0079/summary.md) | E76-F86-F65c | 42.013 | 182,206 | -1.54% | 86.216 | 0.26 | 108.8 | retain main improvement; preliminary only |

**0077, E72-F71-F73b.** F73b drops to four preparation workers on F71 physical-core placement, giving the main process four dedicated cores instead of three. Batch bytes and RNG are identical across 6/5/4 workers. Recorded decision: retain previous accepted run; speed target missed; exact full recorded parity: True. Training throughput changed +4.57% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.58 versus 0.50; summed preparation s: 96.76 versus 108.12; validation s: 28.65 versus 28.22; primer s: 9.44 versus 9.65; worker CPU %: 64.59 versus 51.60; GPU activity %: 76.58 versus 64.40. These observations include machine-state variation and do not independently attribute a component’s effect.

**0078, E72-F71-F73a-F86.** F86 opts the main process and the five pinned preparation workers out of Windows power throttling (EcoQoS) and raises them to above-normal priority, recording the prior state and timer resolution. Scheduling only; arithmetic and order are unchanged. Recorded decision: retain previous accepted run; speed target missed; exact full recorded parity: True. Training throughput changed +5.78% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.53 versus 0.50; summed preparation s: 107.31 versus 108.12; validation s: 27.86 versus 28.22; primer s: 9.50 versus 9.65; worker CPU %: 53.46 versus 51.60; GPU activity %: 70.72 versus 64.40. These observations include machine-state variation and do not independently attribute a component’s effect.

**0079, E76-F86-F65c.** Stacks the two exact stack candidates on accepted 0076 (E76): F86 power-throttling opt-out at above-normal priority for the main process and pinned workers, and F65c native GroupNorm dispatcher/autograd with unchanged reduction kernels. E76 aliases the accepted 0076 options for Windows path length. Recorded decision: retain main improvement; preliminary only; exact full recorded parity: True. Training throughput changed -1.54% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.26 versus 0.50; summed preparation s: 108.76 versus 108.12; validation s: 27.05 versus 28.22; primer s: 12.02 versus 9.65; worker CPU %: 50.52 versus 51.60; GPU activity %: 66.38 versus 64.40. These observations include machine-state variation and do not independently attribute a component’s effect.

## Separate held-out binary losses

| Iteration | Metric | Loss | Delta vs prior accepted | Delta vs original baseline |
|---|---|---:|---:|---:|
| 0077 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0077 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0077 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0077 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |
| 0078 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0078 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0078 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0078 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |
| 0079 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0079 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0079 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0079 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |

## Checks and provenance

Full recorded parity is exact in 3/3 cases. The recorded acceptance gates and decisions are reused; this report does not change them or substitute parity for broader quality validation.

- 0077: experiment `ba1253fe612ca2bb98982f39c54f4fe70dc1ed82`, origin `fd3266cb7370ea14f14506bc750e6f89bc75850d`, dirty at process start: True; [source archive](../../sources/experiment-sources/fe85a3a9db950de47be1660442f29dd699570599bcb17b67d32a390472e0d9b4/source.zip), [protocol](iteration-0077/protocol.json), [comparison](iteration-0077/comparison.json).
- 0077 pre-run background: 9.3% total CPU; 0 other Python GPU processes detected.
- 0078: experiment `0c937dd3328529336c0967b95633706522eaaca7`, origin `dae7c7eb486e994d131e2124371c540a017a6af9`, dirty at process start: True; [source archive](../../sources/experiment-sources/19b087d1d97d277c3ef8b2a89c93757823bc58919dff554664f9ac1e5a243409/source.zip), [protocol](iteration-0078/protocol.json), [comparison](iteration-0078/comparison.json).
- 0078 pre-run background: 8.8% total CPU; 0 other Python GPU processes detected.
- 0079: experiment `72b00c62f6997e2e546a756dc6e3572a37b9e8b3`, origin `187c81ad028aa78da1b18c348a1ae23b8ac61228`, dirty at process start: True; [source archive](../../sources/experiment-sources/273fdb2f8251ec037d3ea2d1f349b9c5ac0552ed52bc4919cb0f6213f8b4adbb/source.zip), [protocol](iteration-0079/protocol.json), [comparison](iteration-0079/comparison.json).
- 0079 pre-run background: 5.7% total CPU; 0 other Python GPU processes detected.

## Interpretation and caveats

- 0079 was promoted by the frozen replicated-block rule at the first look: per-update trimmed gains +5.99% and +6.33% in the two blocks (mean +6.16%, lower bound +4.67%). Its headline single-run figure (182,206 reviews/s, −1.54%) is lower than the block gains because a few slow updates in that run are trimmed by the 5-MAD estimator; total-throughput comparisons across runs remain dominated by host state (the 0078 reference ranged 177,389–210,660 reviews/s over its six runs).
- The combined +6.16% exceeds the separate estimates for F86 (0078: +1.74% at six blocks) and F65c (0073: +0.97% at six blocks, on the earlier unpinned reference). How the gain splits between the two is not established; component-removal blocks on the 0079 reference are needed before either is called independently proven.
- F86 recorded the starting state on every run: main and worker processes had throttling control mask 0 (system-managed) at normal priority with a 1.0 ms system timer, so the option changes Windows policy rather than undoing an explicit throttle.
- F80 needs no run: the accepted 0076 path makes 644 device allocations and 39 frees over the whole run.

## Pending next steps

- Attribute the 0079 gain between F86 and F65c (component-removal blocks on the new reference) before treating either as independently proven
- F72 main-thread budget audit (per-phase thread_time) on the 0079 reference
- F77 WDDM flush-placement A/B and F88 L2 persisting window microbenchmark as exact host/GPU candidates
