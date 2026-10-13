# R01 iteration 0083

Status: **complete preliminary screen**. Authorization: Go for 9 iterations of R01 !.

Same fixed **80 training / 20 unseen held-out users**, **7,654,975 training / 868,370 evaluation reviews**, seed 2025, one epoch, BF16 compute/FP32 master. Chunks contain 128 reviews; prefetch is bounded at twice the worker count. Review exposure stays fixed; declared batch-size or cross-user packing changes may alter update membership, counts, batch seeds, dropout and the learning-rate trajectory. Exact worker/chunk settings and explanations are preserved in the protocol and raw results.

Effective rules: [loop definition](loop-definition.md).

[Frozen protocol](protocol.json), [selection from recorded profiles](selection.json), [loop definition](loop-definition.md), [raw comparison](comparison.json). References retain their original process provenance. Timing includes complete training loads, worker startup/shutdown, preparation, transfers, updates and logs. Setup/primer, save and evaluation are separate; process wall includes them.

Acceptance: measured relative regression ≤1% for all four binary losses against the previous accepted run; ≤2% versus the fixed original baseline is preferred. Confidence bounds are reported separately. Faster but less accurate variants remain separate opt-in fastmode. Accurate resource-saving candidates without a speed gain remain pending retuning. Target ≥5% reviews/s gain versus the previous accepted run; complete the explicit 9-iteration count despite consecutive misses. This remains a preliminary screen; broader validation is pending.

## Full-history measurements

| Model / IDs | Training s | Reviews/s | Gain vs previous / original baseline | Process wall s | Weighted Curve / Instant | User-average Curve / Instant |
|---|---:|---:|---|---:|---|---|
| expertium: E80 (accepted iteration-0080 Expertium configuration: E79, F65d (alias for Windows path length)), F33b (loss/norm reads deferred by two queued updates instead of one), F57 (per-user RSS sampling and progress JSON every eight users instead of every user), F90 (native exact WKV with checkpoint tile 16 instead of 32) | 40.819 | 187,535 | -6.40% / +525.01% | 81.552 | 0.387472684 / 0.398428596 | 0.370387972 / 0.382592464 |

## Resource evidence and decisions

expertium: Recorded E79 (accepted iteration-0079 Expertium configuration: E76, F86, F65c (alias for Windows path length)), F65d (native exact shift-gather and padded row-select dispatcher/autograd using unchanged F26/F27 kernels) GPU activity is 74.53% and worker-pool CPU 55.35%. Main process CPU averaged 95% of one core; system load averaged 5.1 busy logical threads on 8 physical cores. Stacks three small exact changes on accepted 0080 (E80): F33b two outstanding loss/norm reads (0082 stack candidate, +0.55%), F57 RSS sampling and progress JSON every eight users instead of every user, and F90 native WKV checkpoint tile 16 (fewer registers per recompute tile). E80 aliases the accepted 0080 options.

F28 starts worker imports during setup: **5 distinct workers** acknowledged initializer imports before training. Setup-to-ready elapsed **8.351 s**, final ready wait **0.000300 s**. These costs are outside the training timer and inside the reported process wall; shutdown remains inside training.

E80 (accepted iteration-0080 Expertium configuration: E79, F65d (alias for Windows path length)), F33b (loss/norm reads deferred by two queued updates instead of one), F57 (per-user RSS sampling and progress JSON every eight users instead of every user), F90 (native exact WKV with checkpoint tile 16 instead of 32): measured GPU input-stage time **1.300 s**, with **0.011 s** overlapping the preceding training batch. The input-stage event covers copying and GPU feature materialization; this is separate device timing, not a wall-time saving estimate.

E80 (accepted iteration-0080 Expertium configuration: E79, F65d (alias for Windows path length)), F33b (loss/norm reads deferred by two queued updates instead of one), F57 (per-user RSS sampling and progress JSON every eight users instead of every user), F90 (native exact WKV with checkpoint tile 16 instead of 32): GPU activity **73.0%**, worker-pool CPU **51.3%**, peak reserved VRAM **12.20 GiB**, minimum available system RAM **8.54 GiB**. [Resource profile](profiles/expertium-E80-F33b-F57-F90/profile.md). Full-history sampling-window sufficiency: `False`; GPU activity is not occupancy.

Speed change: **faster** under the applicable rule. Gains within ±0.86% (median spread over pairs of repeats of an identical configuration) count as the same speed. Retuning requires a material resource reduction.

Decision: **retain main improvement; preliminary only**. Exact recorded parity versus previous accepted run: **True**; logged update mismatches: **0**. Consecutive iterations without a 5% gain: **0**. Three-miss threshold reached: **False**. Round stopping rule: complete the explicit 9-iteration count despite consecutive misses.

| Binary loss | Candidate | Δ vs previous / original | Familywise upper vs previous / original |
|---|---:|---|---|
| curve_log_loss_label_weighted | 0.387472684 | +0.00% / -3.25% | +0.00% / -0.49% |
| curve_log_loss_user_average | 0.370387972 | +0.00% / -4.48% | +0.00% / -0.62% |
| instant_log_loss_label_weighted | 0.398428596 | +0.00% / -0.76% | +0.00% / +0.59% |
| instant_log_loss_user_average | 0.382592464 | +0.00% / -0.99% | +0.00% / +0.86% |

Replicate blocks: **2** reference/candidate pairs; trimmed per-update gain mean **+1.95%**, lower bound **+0.51%** at z = 1.9927 (block SD used 1.01%); block decision **promote**. Replicates exact: reference True, candidate True.

## Evidence and provenance

Each process captures its own dirty source snapshot and full commit at start; the shared training-implementation fingerprint is checked at every user boundary. Historical references retain their original revisions. [Recorded ordered-prefetch contracts](../iteration-0004/prefetch-checks-v1/result.json); functional smoke results are separate from full-history timing.

- [expertium: E80 (accepted iteration-0080 Expertium configuration: E79, F65d (alias for Windows path length)), F33b (loss/norm reads deferred by two queued updates instead of one), F57 (per-user RSS sampling and progress JSON every eight users instead of every user), F90 (native exact WKV with checkpoint tile 16 instead of 32)](expertium/E80-F33b-F57-F90/result.json): complete; training start UTC 2026-10-13T16:29:41.665075+00:00; commit `125d0f864e1df4c0c526f25a6b080c6472512b80`, dirty `True`; [source snapshot](../../../sources/experiment-sources/53dbdf56a59a017896f9bbe09a8652585f6c3ab8e024cfa6dc874db6a7eff929/source.zip), [updates](expertium/E80-F33b-F57-F90/updates.jsonl), [checkpoint](expertium/E80-F33b-F57-F90/master.pth).
- [expertium: E80 (accepted iteration-0080 Expertium configuration: E79, F65d (alias for Windows path length)), F33b (loss/norm reads deferred by two queued updates instead of one), F57 (per-user RSS sampling and progress JSON every eight users instead of every user), F90 (native exact WKV with checkpoint tile 16 instead of 32) smoke](functional-smoke-v1/expertium/E80-F33b-F57-F90/result.json): complete; training start UTC 2026-10-13T16:27:57.931533+00:00; commit `125d0f864e1df4c0c526f25a6b080c6472512b80`, dirty `True`; [source snapshot](../../../sources/experiment-sources/53dbdf56a59a017896f9bbe09a8652585f6c3ab8e024cfa6dc874db6a7eff929/source.zip), [updates](functional-smoke-v1/expertium/E80-F33b-F57-F90/updates.jsonl), [checkpoint](functional-smoke-v1/expertium/E80-F33b-F57-F90/master.pth).

Implementation commit: `7fced6223659f08303718bd8e14055f3f4c956a4` (after experiments).

20 distinct held-out users; one training seed and one timing/configuration; no broader validation or timing-repeatability claim. This iteration is complete. Recorded scope: Explicit human count: iteration 4 of 9; complete the count despite consecutive misses; reuse accepted references; fixed cohort; no broader validation; no broader validation is authorized.
