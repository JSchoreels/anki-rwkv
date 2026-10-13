# R01 iteration 0073

Status: **complete preliminary screen**. Authorization: Ten full R01 training cycles.

Same fixed **80 training / 20 unseen held-out users**, **7,654,975 training / 868,370 evaluation reviews**, seed 2025, one epoch, BF16 compute/FP32 master. Chunks contain 128 reviews; prefetch is bounded at twice the worker count. Review exposure stays fixed; declared batch-size or cross-user packing changes may alter update membership, counts, batch seeds, dropout and the learning-rate trajectory. Exact worker/chunk settings and explanations are preserved in the protocol and raw results.

Effective rules: [loop definition](loop-definition.md).

[Frozen protocol](protocol.json), [selection from recorded profiles](selection.json), [loop definition](loop-definition.md), [raw comparison](comparison.json). References retain their original process provenance. Timing includes complete training loads, worker startup/shutdown, preparation, transfers, updates and logs. Setup/primer, save and evaluation are separate; process wall includes them.

Acceptance: measured relative regression ≤1% for all four binary losses against the previous accepted run; ≤2% versus the fixed original baseline is preferred. Confidence bounds are reported separately. Faster but less accurate variants remain separate opt-in fastmode. Accurate resource-saving candidates without a speed gain remain pending retuning. Target ≥5% reviews/s gain versus the previous accepted run; complete the explicit 10-iteration count despite consecutive misses. This remains a preliminary screen; broader validation is pending.

## Full-history measurements

| Model / IDs | Training s | Reviews/s | Gain vs previous / original baseline | Process wall s | Weighted Curve / Instant | User-average Curve / Instant |
|---|---:|---:|---|---:|---|---|
| expertium: E33 (accepted iteration-0033 Expertium configuration: E31, F42), F48 (deterministic algorithms without uninitialized-memory fill), F50 (skip zero hierarchy padding sources in exact row backward), F65b (native WKV dispatcher/autograd using unchanged NVRTC kernels), F65c (native exact GroupNorm dispatcher/autograd using unchanged reduction kernels) | 45.152 | 169,538 | -0.23% / +465.03% | 100.231 | 0.387472684 / 0.398428596 | 0.370387972 / 0.382592464 |

## Resource evidence and decisions

expertium: Recorded E33 (accepted iteration-0033 Expertium configuration: E31, F42), F48 (deterministic algorithms without uninitialized-memory fill), F50 (skip zero hierarchy padding sources in exact row backward), F65b (native WKV dispatcher/autograd using unchanged NVRTC kernels) GPU activity is 63.55% and worker-pool CPU 62.81%. Main process CPU averaged 110% of one core; system load averaged 6.2 busy logical threads on 8 physical cores. Replace accepted exact GroupNorm Python autograd and CuPy gamma/beta dispatch with C++ autograd, identical ATen forward/input gradients and retained BF16 gamma/beta kernels, preserving reduction order and current stream.

F28 starts worker imports during setup: **6 distinct workers** acknowledged initializer imports before training. Setup-to-ready elapsed **15.344 s**, final ready wait **0.000424 s**. These costs are outside the training timer and inside the reported process wall; shutdown remains inside training.

E33 (accepted iteration-0033 Expertium configuration: E31, F42), F48 (deterministic algorithms without uninitialized-memory fill), F50 (skip zero hierarchy padding sources in exact row backward), F65b (native WKV dispatcher/autograd using unchanged NVRTC kernels), F65c (native exact GroupNorm dispatcher/autograd using unchanged reduction kernels): measured GPU input-stage time **3.041 s**, with **0.010 s** overlapping the preceding training batch. The input-stage event covers copying and GPU feature materialization; this is separate device timing, not a wall-time saving estimate.

E33 (accepted iteration-0033 Expertium configuration: E31, F42), F48 (deterministic algorithms without uninitialized-memory fill), F50 (skip zero hierarchy padding sources in exact row backward), F65b (native WKV dispatcher/autograd using unchanged NVRTC kernels), F65c (native exact GroupNorm dispatcher/autograd using unchanged reduction kernels): GPU activity **64.8%**, worker-pool CPU **63.5%**, peak reserved VRAM **11.90 GiB**, minimum available system RAM **9.46 GiB**. [Resource profile](profiles/expertium-E33-F48-F50-F65b-F65c/profile.md). Full-history sampling-window sufficiency: `False`; GPU activity is not occupancy.

Speed change: **same within recorded repeat spread** under the applicable rule. Gains within ±0.81% (median spread over pairs of repeats of an identical configuration) count as the same speed. Retuning requires a material resource reduction.

Decision: **retain previous accepted run; speed target missed**. Exact recorded parity versus previous accepted run: **True**; logged update mismatches: **0**. Consecutive iterations without a 5% gain: **1**. Three-miss threshold reached: **False**. Round stopping rule: complete the explicit 10-iteration count despite consecutive misses.

| Binary loss | Candidate | Δ vs previous / original | Familywise upper vs previous / original |
|---|---:|---|---|
| curve_log_loss_label_weighted | 0.387472684 | +0.00% / -3.25% | +0.00% / -0.49% |
| curve_log_loss_user_average | 0.370387972 | +0.00% / -4.48% | +0.00% / -0.62% |
| instant_log_loss_label_weighted | 0.398428596 | +0.00% / -0.76% | +0.00% / +0.59% |
| instant_log_loss_user_average | 0.382592464 | +0.00% / -0.99% | +0.00% / +0.86% |

Replicate blocks: **6** reference/candidate pairs; trimmed per-update gain mean **+0.97%**, lower bound **-0.22%** at z = 1.9927 (block SD used 1.46%); block decision **stack candidate**. Replicates exact: reference True, candidate True.

Stack candidate: **positive -0.23% below the promotion rule**; keep the option for later stacking with other small gains (2026-10-12 human request).

## Evidence and provenance

Each process captures its own dirty source snapshot and full commit at start; the shared training-implementation fingerprint is checked at every user boundary. Historical references retain their original revisions. [Recorded ordered-prefetch contracts](../iteration-0004/prefetch-checks-v1/result.json); functional smoke results are separate from full-history timing.

- [expertium: E33 (accepted iteration-0033 Expertium configuration: E31, F42), F48 (deterministic algorithms without uninitialized-memory fill), F50 (skip zero hierarchy padding sources in exact row backward), F65b (native WKV dispatcher/autograd using unchanged NVRTC kernels), F65c (native exact GroupNorm dispatcher/autograd using unchanged reduction kernels)](expertium/E33-F48-F50-F65b-F65c/result.json): complete; training start UTC 2026-10-13T11:30:33.911230+00:00; commit `e4ebdabab770d7c6541f7947b0ad3f6f3ecaa16e`, dirty `True`; [source snapshot](../../../sources/experiment-sources/54b7f2582ad4243df9f30196558442e162491f362f1141c801f694d89f723313/source.zip), [updates](expertium/E33-F48-F50-F65b-F65c/updates.jsonl), [checkpoint](expertium/E33-F48-F50-F65b-F65c/master.pth).
- [expertium: E33 (accepted iteration-0033 Expertium configuration: E31, F42), F48 (deterministic algorithms without uninitialized-memory fill), F50 (skip zero hierarchy padding sources in exact row backward), F65b (native WKV dispatcher/autograd using unchanged NVRTC kernels), F65c (native exact GroupNorm dispatcher/autograd using unchanged reduction kernels) smoke](functional-smoke-v1/expertium/E33-F48-F50-F65b-F65c/result.json): complete; training start UTC 2026-10-13T11:30:02.389795+00:00; commit `e4ebdabab770d7c6541f7947b0ad3f6f3ecaa16e`, dirty `True`; [source snapshot](../../../sources/experiment-sources/54b7f2582ad4243df9f30196558442e162491f362f1141c801f694d89f723313/source.zip), [updates](functional-smoke-v1/expertium/E33-F48-F50-F65b-F65c/updates.jsonl), [checkpoint](functional-smoke-v1/expertium/E33-F48-F50-F65b-F65c/master.pth).

Implementation commit: `7dfa19dbe3627971543d9c8204d4bba805ef7421` (after experiments).

20 distinct held-out users; one training seed and one timing/configuration; no broader validation or timing-repeatability claim. This iteration is complete. Recorded scope: Explicit human count: iteration 3 of 10; complete the count despite consecutive misses; reuse accepted references; fixed cohort; no broader validation; no broader validation is authorized.
