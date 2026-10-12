# Codex R01 round: iterations 0037–0039

The accepted preliminary expertium reference is **0033: 170,718 reviews/s**, +0.00% versus the round’s starting accepted reference and +468.96% versus the fixed original baseline. Implementation commit `a8da8767391a005d21c344cd3898905f94de61a1` was made after the experiments.

Authorization: 2026-10-12 human request: pop a new worker for 3 new. The selected iteration budget is complete; another round requires a new human request.

## Full-history results

Training seconds and raw review presentations/s include user transitions; complete process seconds also include setup, discarded primer, checkpoint saving and evaluation. Gains below use each candidate’s own recorded previous accepted reference, which can differ between rows. One full timing per candidate does not establish repeatability; broader validation remains pending.

| Iteration | Bundle | Training seconds | Reviews/s | Gain vs prior accepted | Complete process seconds | Queue wait s | Summed worker preparation s | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [0033](iteration-0033/summary.md) | E30-F37-F42 | 44.840 | 170,718 | — | 96.242 | 0.59 | 110.3 | Starting accepted reference |
| [0037](iteration-0037/summary.md) | E33-F44 | 50.861 | 150,508 | -11.84% | 107.521 | 0.57 | 130.8 | retain previous accepted run; speed target missed |
| [0038](iteration-0038/summary.md) | E33-F45 | 52.745 | 145,133 | -14.99% | 111.346 | 0.80 | 164.0 | retain previous accepted run; speed target missed |
| [0039](iteration-0039/summary.md) | E33-F46 | 49.597 | 154,344 | -9.59% | 106.543 | 0.91 | 164.1 | retain previous accepted run; speed target missed |

**0037, E33-F44.** Hoist chunk-local day minima, card-first offsets and weekday transforms into one batch-wide pandas pass. Preserve six workers, four recurrence rows and all E33 sample arithmetic and boundaries. Recorded decision: retain previous accepted run; speed target missed; exact full recorded parity: True. Training throughput changed -11.84% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.57 versus 0.59; summed preparation s: 130.75 versus 110.32; validation s: 33.86 versus 29.19; primer s: 14.98 versus 15.03; worker CPU %: 44.37 versus 42.43; GPU activity %: 65.62 versus 69.09. These observations include machine-state variation and do not independently attribute a component’s effect.

**0038, E33-F45.** Build each feature and label float64 matrix directly in its final array instead of converting every column into a temporary and then stacking. Preserve E33 columnar samples, six workers and all seeded arithmetic. Recorded decision: retain previous accepted run; speed target missed; exact full recorded parity: True. Training throughput changed -14.99% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.80 versus 0.59; summed preparation s: 164.01 versus 110.32; validation s: 35.32 versus 29.19; primer s: 15.03 versus 15.03; worker CPU %: 52.74 versus 42.43; GPU activity %: 61.51 versus 69.09. These observations include machine-state variation and do not independently attribute a component’s effect.

**0039, E33-F46.** Pipeline BF16 shift-backward source-index and gradient loads across 32-source passes. Preserve each lane accumulation order, warp-tree reduction, sequential tail and final rounding; all E33 host and recurrence options stay fixed. Recorded decision: retain previous accepted run; speed target missed; exact full recorded parity: True. Training throughput changed -9.59% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.91 versus 0.59; summed preparation s: 164.12 versus 110.32; validation s: 33.77 versus 29.19; primer s: 15.17 versus 15.03; worker CPU %: 56.48 versus 42.43; GPU activity %: 58.11 versus 69.09. These observations include machine-state variation and do not independently attribute a component’s effect.

## Separate held-out binary losses

| Iteration | Metric | Loss | Delta vs prior accepted | Delta vs original baseline |
|---|---|---:|---:|---:|
| 0037 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0037 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0037 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0037 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |
| 0038 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0038 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0038 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0038 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |
| 0039 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0039 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0039 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0039 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |

## Checks and provenance

Full recorded parity is exact in 3/3 cases. The recorded acceptance gates and decisions are reused; this report does not change them or substitute parity for broader quality validation.

- 0037: experiment `f9d314fa6cda6179ae547f4659fc2062da94e8ec`, origin `1155e2a1a4b430eb6740162b1d9bd736156c3075`, dirty at process start: True; [source archive](../../sources/experiment-sources/f517e78cf67ed1b2ca3a59e96f80e7db19b3a96c34b0989e7651545733a18958/source.zip), [protocol](iteration-0037/protocol.json), [comparison](iteration-0037/comparison.json).
- 0037 pre-run background: 7.7% total CPU; 0 other Python GPU processes detected.
- 0038: experiment `52b76064dadd779df1913dde6a718fdb4f946a04`, origin `1155e2a1a4b430eb6740162b1d9bd736156c3075`, dirty at process start: True; [source archive](../../sources/experiment-sources/b8c12c699f122c39d508d536253c24ce59578f53a73d6d4757546c296c6be48c/source.zip), [protocol](iteration-0038/protocol.json), [comparison](iteration-0038/comparison.json).
- 0038 pre-run background: 11.7% total CPU; 0 other Python GPU processes detected.
- 0039: experiment `206c75fab46613f982f832cd9e87abb93169db50`, origin `1155e2a1a4b430eb6740162b1d9bd736156c3075`, dirty at process start: True; [source archive](../../sources/experiment-sources/d81407b4401a7a2978177afd90d5223f3edbdb5734e11ec6134d9ffbd89685bb/source.zip), [protocol](iteration-0039/protocol.json), [comparison](iteration-0039/comparison.json).
- 0039 pre-run background: 10.2% total CPU; 0 other Python GPU processes detected.

## Pending next steps

- New human authorization required: three exact speed misses; retain E33 at 170,718 reviews/s and keep Alex paused.
- Preserve F44 diagnostic sample savings as a pending mechanism; full-run preparation and validation drift prevent attributing the speed miss. Any new retest requires a concrete comparability rationale.
- Investigate F47 chunk/ID grouping and permutation amortization using bounded CPU evidence before implementation; no measured benefit yet.
- F46 exact load pipelining missed full-run speed; inspect register-pressure evidence before proposing a changed kernel, preserving lane sums and warp-tree arithmetic.
- Token usage comparisons remain descriptive across unlike workloads; final post-cutoff usage can be refreshed from the saved collector.

## Interpretation and token audit

The CPU checks measure sample construction separately from full training. F44 cut selected-window sample time 0.842 to 0.416 s, F45 reached 0.808 s, and their untrained combination reached 0.426 s. Each variant and the combination matched all sample fields, packed bytes and RNG on five edge windows. The ordered single-process diagnostic is not a throughput estimate. Full runs had slower preparation and held-out validation than E33, so these speed misses do not establish independent component costs or gains. No candidate was promoted; all stay opt-in for reproduction.

R01-T01 — implemented controls, measured 0037–0039 usage; causal savings — (inconclusive). At the [latest audit cutoff](../../diagnostics/r01-round-0037-0039/token-audit.json), root uncached input/output were 108,806/21,701 tokens, cached input 3,771,008, 50 tool calls, 43 shell operations, 10 process polls and 1 tool-wait resumption. Delegated Luna reading added 83,596/5,946 uncached input/output and 10 calls; attributed approval reviewers added 13,756/375 and 0 calls. The [previous round audit](../../diagnostics/r01-round-0034-0036-options/token-audit.json) recorded root 67,255/14,374 uncached input/output, cached input 2,613,376, 43 calls, 38 shell operations, 9 polls and 1 wait resumption. Unlike workloads (new CPU paths and kernel/checks versus option retuning), different cutoffs, regex-derived call classification and unavailable comparable tool-output characters prevent causal claims. Post-cutoff closeout/response usage is excluded. One compatible preflight was reused for three focused checks and all three iterations; five sample/packed/RNG windows, 72 synthetic cases, five real-model batches and all full/smoke parity checks passed. [Collector and method](../../diagnostics/r01-round-0037-0039/collect_token_audit.py), [round report](codex-round-0037-0039-summary.md). Further iterations require new human authorization.
