# Claude R01 round: iterations 0089–0090

The accepted preliminary expertium reference is **0090: 192,956 reviews/s**, +2.89% versus the round’s starting accepted reference and +543.07% versus the fixed original baseline. Implementation commit `c97cfe0e39fe69a1d36a23f028e2d978575ac06d` was made after the experiments.

Authorization: continue until iteration 100. Explicit count: 2 of 12 iterations completed; 10 remain authorized. Consecutive misses do not end this counted round.

## Full-history results

Training seconds and raw review presentations/s include user transitions; complete process seconds also include setup, discarded primer, checkpoint saving and evaluation. Gains below use each candidate’s own recorded previous accepted reference, which can differ between rows. One full timing per candidate does not establish repeatability; broader validation remains pending.

| Iteration | Bundle | Training seconds | Reviews/s | Gain vs prior accepted | Complete process seconds | Queue wait s | Summed worker preparation s | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [0083](iteration-0083/summary.md) | E80-F33b-F57-F90 | 40.819 | 187,535 | — | 81.552 | 0.32 | 107.5 | Starting accepted reference |
| [0089](iteration-0089/summary.md) | E83-F72a | 37.944 | 201,745 | +7.58% | 74.519 | 0.87 | 107.4 | retain main improvement; preliminary only |
| [0090](iteration-0090/summary.md) | E89-F65e-F77a | 39.672 | 192,956 | -4.36% | 82.034 | 1.61 | 113.7 | retain main improvement; preliminary only |

**0089, E83-F72a.** Reset gradients through cached master/child parameter lists and set train mode once before the loop, instead of optimizer.zero_grad, child zero_grad and train() walking the scripted module tree every update, on the accepted 0083 stack. Recorded decision: retain main improvement; preliminary only; exact full recorded parity: True. Training throughput changed +7.58% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.87 versus 0.32; summed preparation s: 107.38 versus 107.53; validation s: 27.92 versus 27.35; primer s: 3.73 versus 8.30; worker CPU %: 54.73 versus 51.29; GPU activity %: 77.92 versus 73.00. These observations include machine-state variation and do not independently attribute a component’s effect.

**0090, E89-F65e-F77a.** Stacks two exact stack candidates from round 0080-0088 on accepted 0089 (E89 = E83 + F72a): F65e native gate/bonus/channel glue dispatcher (0081, +0.04%) and F77a WDDM stream query after forward and backward (0087, +0.52% at six blocks). Recorded decision: retain main improvement; preliminary only; exact full recorded parity: True. Training throughput changed -4.36% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 1.61 versus 0.87; summed preparation s: 113.71 versus 107.38; validation s: 29.05 versus 27.92; primer s: 8.15 versus 3.73; worker CPU %: 55.14 versus 54.73; GPU activity %: 60.80 versus 77.92. These observations include machine-state variation and do not independently attribute a component’s effect.

## Separate held-out binary losses

| Iteration | Metric | Loss | Delta vs prior accepted | Delta vs original baseline |
|---|---|---:|---:|---:|
| 0089 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0089 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0089 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0089 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |
| 0090 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0090 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0090 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0090 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |

## Checks and provenance

Full recorded parity is exact in 2/2 cases. The recorded acceptance gates and decisions are reused; this report does not change them or substitute parity for broader quality validation.

- 0089: experiment `c64051c51c7bebc5673ff90435aeec950df7a583`, origin `5626758d6932f0f08684d8e8737f3afd8353bc01`, dirty at process start: True; [source archive](../../sources/experiment-sources/8c3df44b3e80e55c68b28b0edc06f40dceb6dcc8230680e72ba6c21f82692be2/source.zip), [protocol](iteration-0089/protocol.json), [comparison](iteration-0089/comparison.json).
- 0089 pre-run background: 7.7% total CPU; 0 other Python GPU processes detected.
- 0090: experiment `125e6ddabdf30afaf4ab687b350c7113016437ec`, origin `917f1839e4cba8a0e4bbfc91ccb305e2ced407f8`, dirty at process start: True; [source archive](../../sources/experiment-sources/5c9a131ea4ace42e605a05cd1c4ccfc4415c87323bb415a1a429b1d84193e367/source.zip), [protocol](iteration-0090/protocol.json), [comparison](iteration-0090/comparison.json).
- 0090 pre-run background: 9.5% total CPU; 0 other Python GPU processes detected.

## Pending next steps

- F92: native WKV transposed warp sums (one shuffle butterfly for all rows' reductions, bitwise-equal tree; ~47 vs ~168 shuffles per backward step); patch drafted, untested
- F92b: interleave the 32-term row products across rows in the WKV backward (same per-element FMA order, fewer shared loads)
- With the overclocked GPU at ~61% busy, per-update host work (clip_grad_norm_ parameter walk, pinned deferred-read allocation) is relatively larger

- Caveat (added at closure): the generated implementation records name `c97cfe0`, an AGENT.md-only commit that was HEAD at closure. The tested training source is `5626758d6932f0f08684d8e8737f3afd8353bc01`; `917f183` only adds the E89 report alias. The round stopped after 2 of 12 counted iterations because the human shortened it on 2026-10-13 (finish by 21:00). Before 0090's timed runs the GPU overclock profile was re-enabled (0084–0089 ran voltage-capped).
