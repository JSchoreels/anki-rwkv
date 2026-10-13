# R01 iteration 0072

Status: **complete preliminary screen**. Authorization: Ten full R01 training cycles.

Same fixed **80 training / 20 unseen held-out users**, **7,654,975 training / 868,370 evaluation reviews**, seed 2025, one epoch, BF16 compute/FP32 master. Chunks contain 128 reviews; prefetch is bounded at twice the worker count. Review exposure stays fixed; declared batch-size or cross-user packing changes may alter update membership, counts, batch seeds, dropout and the learning-rate trajectory. Exact worker/chunk settings and explanations are preserved in the protocol and raw results.

Effective rules: [loop definition](loop-definition.md).

[Frozen protocol](protocol.json), [selection from recorded profiles](selection.json), [loop definition](loop-definition.md), [raw comparison](comparison.json). References retain their original process provenance. Timing includes complete training loads, worker startup/shutdown, preparation, transfers, updates and logs. Setup/primer, save and evaluation are separate; process wall includes them.

Acceptance: measured relative regression ≤1% for all four binary losses against the previous accepted run; ≤2% versus the fixed original baseline is preferred. Confidence bounds are reported separately. Faster but less accurate variants remain separate opt-in fastmode. Accurate resource-saving candidates without a speed gain remain pending retuning. Target ≥5% reviews/s gain versus the previous accepted run; complete the explicit 10-iteration count despite consecutive misses. This remains a preliminary screen; broader validation is pending.

## Full-history measurements

| Model / IDs | Training s | Reviews/s | Gain vs previous / original baseline | Process wall s | Weighted Curve / Instant | User-average Curve / Instant |
|---|---:|---:|---|---:|---|---|
| expertium: E33 (accepted iteration-0033 Expertium configuration: E31, F42), F48 (deterministic algorithms without uninitialized-memory fill), F50 (skip zero hierarchy padding sources in exact row backward), F65b (native WKV dispatcher/autograd using unchanged NVRTC kernels) | 45.047 | 169,934 | -13.96% / +466.35% | 99.676 | 0.387472684 / 0.398428596 | 0.370387972 / 0.382592464 |

## Resource evidence and decisions

expertium: Recorded E33 (accepted iteration-0033 Expertium configuration: E31, F42), F48 (deterministic algorithms without uninitialized-memory fill), F50 (skip zero hierarchy padding sources in exact row backward) GPU activity is 66.54% and worker-pool CPU 53.23%. Main process CPU averaged 105% of one core; system load averaged 6.4 busy logical threads on 8 physical cores. Launch the unchanged accepted four-row NVRTC WKV kernels through a C++ dispatcher and C++ autograd node, replacing Python/CuPy forward/backward wrapping while preserving tensor types, checkpoint layout, current stream and argument bytes.

F28 starts worker imports during setup: **6 distinct workers** acknowledged initializer imports before training. Setup-to-ready elapsed **15.293 s**, final ready wait **0.000414 s**. These costs are outside the training timer and inside the reported process wall; shutdown remains inside training.

E33 (accepted iteration-0033 Expertium configuration: E31, F42), F48 (deterministic algorithms without uninitialized-memory fill), F50 (skip zero hierarchy padding sources in exact row backward), F65b (native WKV dispatcher/autograd using unchanged NVRTC kernels): measured GPU input-stage time **2.958 s**, with **0.009 s** overlapping the preceding training batch. The input-stage event covers copying and GPU feature materialization; this is separate device timing, not a wall-time saving estimate.

E33 (accepted iteration-0033 Expertium configuration: E31, F42), F48 (deterministic algorithms without uninitialized-memory fill), F50 (skip zero hierarchy padding sources in exact row backward), F65b (native WKV dispatcher/autograd using unchanged NVRTC kernels): GPU activity **63.6%**, worker-pool CPU **62.8%**, peak reserved VRAM **11.90 GiB**, minimum available system RAM **9.48 GiB**. [Resource profile](profiles/expertium-E33-F48-F50-F65b/profile.md). Full-history sampling-window sufficiency: `False`; GPU activity is not occupancy.

Speed change: **faster** under the applicable rule. Gains within ±0.77% (median spread over pairs of repeats of an identical configuration) count as the same speed. Retuning requires a material resource reduction.

Decision: **retain main improvement; preliminary only**. Exact recorded parity versus previous accepted run: **True**; logged update mismatches: **0**. Consecutive iterations without a 5% gain: **0**. Three-miss threshold reached: **False**. Round stopping rule: complete the explicit 10-iteration count despite consecutive misses.

| Binary loss | Candidate | Δ vs previous / original | Familywise upper vs previous / original |
|---|---:|---|---|
| curve_log_loss_label_weighted | 0.387472684 | +0.00% / -3.25% | +0.00% / -0.49% |
| curve_log_loss_user_average | 0.370387972 | +0.00% / -4.48% | +0.00% / -0.62% |
| instant_log_loss_label_weighted | 0.398428596 | +0.00% / -0.76% | +0.00% / +0.59% |
| instant_log_loss_user_average | 0.382592464 | +0.00% / -0.99% | +0.00% / +0.86% |

Replicate blocks: **2** reference/candidate pairs; trimmed per-update gain mean **+1.81%**, lower bound **+0.38%** at z = 1.9927 (block SD used 1.01%); block decision **promote**. Replicates exact: reference True, candidate True.

## Evidence and provenance

Each process captures its own dirty source snapshot and full commit at start; the shared training-implementation fingerprint is checked at every user boundary. Historical references retain their original revisions. [Recorded ordered-prefetch contracts](../iteration-0004/prefetch-checks-v1/result.json); functional smoke results are separate from full-history timing.

- [expertium: E33 (accepted iteration-0033 Expertium configuration: E31, F42), F48 (deterministic algorithms without uninitialized-memory fill), F50 (skip zero hierarchy padding sources in exact row backward), F65b (native WKV dispatcher/autograd using unchanged NVRTC kernels)](expertium/E33-F48-F50-F65b/result.json): complete; training start UTC 2026-10-13T11:05:20.234675+00:00; commit `50424589f196b51555988c981a71856afd152d98`, dirty `True`; [source snapshot](../../../sources/experiment-sources/b5a7c6f5149b10a8370c939b8f615f5c757f2aa370424ee2852a37b10c1dd817/source.zip), [updates](expertium/E33-F48-F50-F65b/updates.jsonl), [checkpoint](expertium/E33-F48-F50-F65b/master.pth).
- [expertium: E33 (accepted iteration-0033 Expertium configuration: E31, F42), F48 (deterministic algorithms without uninitialized-memory fill), F50 (skip zero hierarchy padding sources in exact row backward), F65b (native WKV dispatcher/autograd using unchanged NVRTC kernels) smoke](functional-smoke-v1/expertium/E33-F48-F50-F65b/result.json): complete; training start UTC 2026-10-13T11:04:48.829382+00:00; commit `50424589f196b51555988c981a71856afd152d98`, dirty `True`; [source snapshot](../../../sources/experiment-sources/b5a7c6f5149b10a8370c939b8f615f5c757f2aa370424ee2852a37b10c1dd817/source.zip), [updates](functional-smoke-v1/expertium/E33-F48-F50-F65b/updates.jsonl), [checkpoint](functional-smoke-v1/expertium/E33-F48-F50-F65b/master.pth).

Implementation commit: `7dfa19dbe3627971543d9c8204d4bba805ef7421` (after experiments).

20 distinct held-out users; one training seed and one timing/configuration; no broader validation or timing-repeatability claim. This iteration is complete. Recorded scope: Explicit human count: iteration 2 of 10; complete the count despite consecutive misses; reuse accepted references; fixed cohort; no broader validation; no broader validation is authorized.
