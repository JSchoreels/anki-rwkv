# R01 iteration 0081

Status: **complete preliminary screen**. Authorization: Go for 9 iterations of R01 !.

Same fixed **80 training / 20 unseen held-out users**, **7,654,975 training / 868,370 evaluation reviews**, seed 2025, one epoch, BF16 compute/FP32 master. Chunks contain 128 reviews; prefetch is bounded at twice the worker count. Review exposure stays fixed; declared batch-size or cross-user packing changes may alter update membership, counts, batch seeds, dropout and the learning-rate trajectory. Exact worker/chunk settings and explanations are preserved in the protocol and raw results.

Effective rules: [loop definition](loop-definition.md).

[Frozen protocol](protocol.json), [selection from recorded profiles](selection.json), [loop definition](loop-definition.md), [raw comparison](comparison.json). References retain their original process provenance. Timing includes complete training loads, worker startup/shutdown, preparation, transfers, updates and logs. Setup/primer, save and evaluation are separate; process wall includes them.

Acceptance: measured relative regression ≤1% for all four binary losses against the previous accepted run; ≤2% versus the fixed original baseline is preferred. Confidence bounds are reported separately. Faster but less accurate variants remain separate opt-in fastmode. Accurate resource-saving candidates without a speed gain remain pending retuning. Target ≥5% reviews/s gain versus the previous accepted run; complete the explicit 9-iteration count despite consecutive misses. This remains a preliminary screen; broader validation is pending.

## Full-history measurements

| Model / IDs | Training s | Reviews/s | Gain vs previous / original baseline | Process wall s | Weighted Curve / Instant | User-average Curve / Instant |
|---|---:|---:|---|---:|---|---|
| expertium: E80 (accepted iteration-0080 Expertium configuration: E79, F65d (alias for Windows path length)), F65e (native exact gate, bonus-product and channel glue dispatcher/autograd using the F34 kernels) | 38.292 | 199,913 | -0.22% / +566.26% | 83.841 | 0.387472684 / 0.398428596 | 0.370387972 / 0.382592464 |

## Resource evidence and decisions

expertium: Recorded E79 (accepted iteration-0079 Expertium configuration: E76, F86, F65c (alias for Windows path length)), F65d (native exact shift-gather and padded row-select dispatcher/autograd using unchanged F26/F27 kernels) GPU activity is 74.53% and worker-pool CPU 55.35%. Main process CPU averaged 95% of one core; system load averaged 5.1 busy logical threads on 8 physical cores. F65e replaces three unfused pointwise chains per layer instance (time-mixer bonus products r*bonus*k and gate g*(out+bonus), channel-mixer relu squared) with native single-launch ops whose backward is native too, using the exact F34 kernels whose Python wrappers previously cost more than they saved. E80 aliases the accepted 0080 options.

F28 starts worker imports during setup: **5 distinct workers** acknowledged initializer imports before training. Setup-to-ready elapsed **11.557 s**, final ready wait **0.000297 s**. These costs are outside the training timer and inside the reported process wall; shutdown remains inside training.

E80 (accepted iteration-0080 Expertium configuration: E79, F65d (alias for Windows path length)), F65e (native exact gate, bonus-product and channel glue dispatcher/autograd using the F34 kernels): measured GPU input-stage time **1.630 s**, with **0.009 s** overlapping the preceding training batch. The input-stage event covers copying and GPU feature materialization; this is separate device timing, not a wall-time saving estimate.

E80 (accepted iteration-0080 Expertium configuration: E79, F65d (alias for Windows path length)), F65e (native exact gate, bonus-product and channel glue dispatcher/autograd using the F34 kernels): GPU activity **74.7%**, worker-pool CPU **53.6%**, peak reserved VRAM **11.95 GiB**, minimum available system RAM **8.52 GiB**. [Resource profile](profiles/expertium-E80-F65e/profile.md). Full-history sampling-window sufficiency: `False`; GPU activity is not occupancy.

Speed change: **same within recorded repeat spread** under the applicable rule. Gains within ±0.86% (median spread over pairs of repeats of an identical configuration) count as the same speed. Retuning requires a material resource reduction.

Decision: **retain previous accepted run; speed target missed**. Exact recorded parity versus previous accepted run: **True**; logged update mismatches: **0**. Consecutive iterations without a 5% gain: **1**. Three-miss threshold reached: **False**. Round stopping rule: complete the explicit 9-iteration count despite consecutive misses.

| Binary loss | Candidate | Δ vs previous / original | Familywise upper vs previous / original |
|---|---:|---|---|
| curve_log_loss_label_weighted | 0.387472684 | +0.00% / -3.25% | +0.00% / -0.49% |
| curve_log_loss_user_average | 0.370387972 | +0.00% / -4.48% | +0.00% / -0.62% |
| instant_log_loss_label_weighted | 0.398428596 | +0.00% / -0.76% | +0.00% / +0.59% |
| instant_log_loss_user_average | 0.382592464 | +0.00% / -0.99% | +0.00% / +0.86% |

Replicate blocks: **6** reference/candidate pairs; trimmed per-update gain mean **+0.04%**, lower bound **-0.78%** at z = 1.9927 (block SD used 1.01%); block decision **stack candidate**. Replicates exact: reference True, candidate True.

Stack candidate: **positive -0.22% below the promotion rule**; keep the option for later stacking with other small gains (2026-10-12 human request).

## Evidence and provenance

Each process captures its own dirty source snapshot and full commit at start; the shared training-implementation fingerprint is checked at every user boundary. Historical references retain their original revisions. [Recorded ordered-prefetch contracts](../iteration-0004/prefetch-checks-v1/result.json); functional smoke results are separate from full-history timing.

- [expertium: E80 (accepted iteration-0080 Expertium configuration: E79, F65d (alias for Windows path length)), F65e (native exact gate, bonus-product and channel glue dispatcher/autograd using the F34 kernels)](expertium/E80-F65e/result.json): complete; training start UTC 2026-10-13T15:49:34.776884+00:00; commit `e96f138072e27b3bf133006d9b9c6a056e5f9626`, dirty `True`; [source snapshot](../../../sources/experiment-sources/eedf7e9a23fb6e432709d9df58cc4f5f942a534bd05293d80a70eee534eb27fc/source.zip), [updates](expertium/E80-F65e/updates.jsonl), [checkpoint](expertium/E80-F65e/master.pth).
- [expertium: E80 (accepted iteration-0080 Expertium configuration: E79, F65d (alias for Windows path length)), F65e (native exact gate, bonus-product and channel glue dispatcher/autograd using the F34 kernels) smoke](functional-smoke-v1/expertium/E80-F65e/result.json): complete; training start UTC 2026-10-13T15:47:05.476268+00:00; commit `e96f138072e27b3bf133006d9b9c6a056e5f9626`, dirty `True`; [source snapshot](../../../sources/experiment-sources/eedf7e9a23fb6e432709d9df58cc4f5f942a534bd05293d80a70eee534eb27fc/source.zip), [updates](functional-smoke-v1/expertium/E80-F65e/updates.jsonl), [checkpoint](functional-smoke-v1/expertium/E80-F65e/master.pth).

Implementation commit: `7fced6223659f08303718bd8e14055f3f4c956a4` (after experiments).

20 distinct held-out users; one training seed and one timing/configuration; no broader validation or timing-repeatability claim. This iteration is complete. Recorded scope: Explicit human count: iteration 2 of 9; complete the count despite consecutive misses; reuse accepted references; fixed cohort; no broader validation; no broader validation is authorized.
