# R01 iteration 0087

Status: **complete preliminary screen**. Authorization: Go for 9 iterations of R01 !.

Same fixed **80 training / 20 unseen held-out users**, **7,654,975 training / 868,370 evaluation reviews**, seed 2025, one epoch, BF16 compute/FP32 master. Chunks contain 128 reviews; prefetch is bounded at twice the worker count. Review exposure stays fixed; declared batch-size or cross-user packing changes may alter update membership, counts, batch seeds, dropout and the learning-rate trajectory. Exact worker/chunk settings and explanations are preserved in the protocol and raw results.

Effective rules: [loop definition](loop-definition.md).

[Frozen protocol](protocol.json), [selection from recorded profiles](selection.json), [loop definition](loop-definition.md), [raw comparison](comparison.json). References retain their original process provenance. Timing includes complete training loads, worker startup/shutdown, preparation, transfers, updates and logs. Setup/primer, save and evaluation are separate; process wall includes them.

Acceptance: measured relative regression ≤1% for all four binary losses against the previous accepted run; ≤2% versus the fixed original baseline is preferred. Confidence bounds are reported separately. Faster but less accurate variants remain separate opt-in fastmode. Accurate resource-saving candidates without a speed gain remain pending retuning. Target ≥5% reviews/s gain versus the previous accepted run; complete the explicit 9-iteration count despite consecutive misses. This remains a preliminary screen; broader validation is pending.

## Full-history measurements

| Model / IDs | Training s | Reviews/s | Gain vs previous / original baseline | Process wall s | Weighted Curve / Instant | User-average Curve / Instant |
|---|---:|---:|---|---:|---|---|
| expertium: E83 (accepted iteration-0083 Expertium configuration: E80, F33b, F57, F90 (alias for Windows path length)), F77a (WDDM queue flushed (stream query) after the forward and after the backward) | 37.736 | 202,858 | +8.17% / +576.08% | 73.644 | 0.387472684 / 0.398428596 | 0.370387972 / 0.382592464 |

## Resource evidence and decisions

expertium: Recorded E80 (accepted iteration-0080 Expertium configuration: E79, F65d (alias for Windows path length)), F33b (loss/norm reads deferred by two queued updates instead of one), F57 (per-user RSS sampling and progress JSON every eight users instead of every user), F90 (native exact WKV with checkpoint tile 16 instead of 32) GPU activity is 73.00% and worker-pool CPU 51.29%. Main process CPU averaged 95% of one core; system load averaged 5.0 busy logical threads on 8 physical cores. Flush the WDDM software command queue with a stream query after the forward and after the backward on the accepted 0083 stack.

F28 starts worker imports during setup: **5 distinct workers** acknowledged initializer imports before training. Setup-to-ready elapsed **3.835 s**, final ready wait **0.000296 s**. These costs are outside the training timer and inside the reported process wall; shutdown remains inside training.

E83 (accepted iteration-0083 Expertium configuration: E80, F33b, F57, F90 (alias for Windows path length)), F77a (WDDM queue flushed (stream query) after the forward and after the backward): measured GPU input-stage time **1.109 s**, with **0.008 s** overlapping the preceding training batch. The input-stage event covers copying and GPU feature materialization; this is separate device timing, not a wall-time saving estimate.

E83 (accepted iteration-0083 Expertium configuration: E80, F33b, F57, F90 (alias for Windows path length)), F77a (WDDM queue flushed (stream query) after the forward and after the backward): GPU activity **70.3%**, worker-pool CPU **54.9%**, peak reserved VRAM **12.20 GiB**, minimum available system RAM **8.37 GiB**. [Resource profile](profiles/expertium-E83-F77a/profile.md). Full-history sampling-window sufficiency: `False`; GPU activity is not occupancy.

Speed change: **same within recorded repeat spread** under the applicable rule. Gains within ±0.87% (median spread over pairs of repeats of an identical configuration) count as the same speed. Retuning requires a material resource reduction.

Decision: **retain previous accepted run; speed target missed**. Exact recorded parity versus previous accepted run: **True**; logged update mismatches: **0**. Consecutive iterations without a 5% gain: **4**. Three-miss threshold reached: **True**. Round stopping rule: complete the explicit 9-iteration count despite consecutive misses.

| Binary loss | Candidate | Δ vs previous / original | Familywise upper vs previous / original |
|---|---:|---|---|
| curve_log_loss_label_weighted | 0.387472684 | +0.00% / -3.25% | +0.00% / -0.49% |
| curve_log_loss_user_average | 0.370387972 | +0.00% / -4.48% | +0.00% / -0.62% |
| instant_log_loss_label_weighted | 0.398428596 | +0.00% / -0.76% | +0.00% / +0.59% |
| instant_log_loss_user_average | 0.382592464 | +0.00% / -0.99% | +0.00% / +0.86% |

Replicate blocks: **6** reference/candidate pairs; trimmed per-update gain mean **+0.52%**, lower bound **-0.30%** at z = 1.9927 (block SD used 1.01%); block decision **stack candidate**. Replicates exact: reference True, candidate True.

Stack candidate: **positive +8.17% below the promotion rule**; keep the option for later stacking with other small gains (2026-10-12 human request).

## Evidence and provenance

Each process captures its own dirty source snapshot and full commit at start; the shared training-implementation fingerprint is checked at every user boundary. Historical references retain their original revisions. [Recorded ordered-prefetch contracts](../iteration-0004/prefetch-checks-v1/result.json); functional smoke results are separate from full-history timing.

- [expertium: E83 (accepted iteration-0083 Expertium configuration: E80, F33b, F57, F90 (alias for Windows path length)), F77a (WDDM queue flushed (stream query) after the forward and after the backward)](expertium/E83-F77a/result.json): complete; training start UTC 2026-10-13T17:09:13.284249+00:00; commit `fa72498bf8585aad0743222c428cd0023ba473a7`, dirty `True`; [source snapshot](../../../sources/experiment-sources/8e2b85c3730ec8d456693369b685930bdef699bf38605af4500278628bd04870/source.zip), [updates](expertium/E83-F77a/updates.jsonl), [checkpoint](expertium/E83-F77a/master.pth).
- [expertium: E83 (accepted iteration-0083 Expertium configuration: E80, F33b, F57, F90 (alias for Windows path length)), F77a (WDDM queue flushed (stream query) after the forward and after the backward) smoke](functional-smoke-v1/expertium/E83-F77a/result.json): complete; training start UTC 2026-10-13T17:07:22.834846+00:00; commit `fa72498bf8585aad0743222c428cd0023ba473a7`, dirty `True`; [source snapshot](../../../sources/experiment-sources/8e2b85c3730ec8d456693369b685930bdef699bf38605af4500278628bd04870/source.zip), [updates](functional-smoke-v1/expertium/E83-F77a/updates.jsonl), [checkpoint](functional-smoke-v1/expertium/E83-F77a/master.pth).

Implementation commit: `7fced6223659f08303718bd8e14055f3f4c956a4` (after experiments).

20 distinct held-out users; one training seed and one timing/configuration; no broader validation or timing-repeatability claim. This iteration is complete. Recorded scope: Explicit human count: iteration 8 of 9; complete the count despite consecutive misses; reuse accepted references; fixed cohort; no broader validation; no broader validation is authorized.
