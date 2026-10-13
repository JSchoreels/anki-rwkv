# R01 iteration 0080

Status: **complete preliminary screen**. Authorization: Go for 9 iterations of R01 !.

Same fixed **80 training / 20 unseen held-out users**, **7,654,975 training / 868,370 evaluation reviews**, seed 2025, one epoch, BF16 compute/FP32 master. Chunks contain 128 reviews; prefetch is bounded at twice the worker count. Review exposure stays fixed; declared batch-size or cross-user packing changes may alter update membership, counts, batch seeds, dropout and the learning-rate trajectory. Exact worker/chunk settings and explanations are preserved in the protocol and raw results.

Effective rules: [loop definition](loop-definition.md).

[Frozen protocol](protocol.json), [selection from recorded profiles](selection.json), [loop definition](loop-definition.md), [raw comparison](comparison.json). References retain their original process provenance. Timing includes complete training loads, worker startup/shutdown, preparation, transfers, updates and logs. Setup/primer, save and evaluation are separate; process wall includes them.

Acceptance: measured relative regression ≤1% for all four binary losses against the previous accepted run; ≤2% versus the fixed original baseline is preferred. Confidence bounds are reported separately. Faster but less accurate variants remain separate opt-in fastmode. Accurate resource-saving candidates without a speed gain remain pending retuning. Target ≥5% reviews/s gain versus the previous accepted run; complete the explicit 9-iteration count despite consecutive misses. This remains a preliminary screen; broader validation is pending.

## Full-history measurements

| Model / IDs | Training s | Reviews/s | Gain vs previous / original baseline | Process wall s | Weighted Curve / Instant | User-average Curve / Instant |
|---|---:|---:|---|---:|---|---|
| expertium: E79 (accepted iteration-0079 Expertium configuration: E76, F86, F65c (alias for Windows path length)), F65d (native exact shift-gather and padded row-select dispatcher/autograd using unchanged F26/F27 kernels) | 38.207 | 200,356 | +9.96% / +567.74% | 82.439 | 0.387472684 / 0.398428596 | 0.370387972 / 0.382592464 |

## Resource evidence and decisions

expertium: Recorded E76 (accepted iteration-0076 Expertium configuration: E72, F71, F73a (alias for Windows path length)), F86 (main process and preparation workers opted out of Windows power throttling at above-normal priority), F65c (native exact GroupNorm dispatcher/autograd using unchanged reduction kernels) GPU activity is 66.38% and worker-pool CPU 50.52%. Main process CPU averaged 97% of one core; system load averaged 5.0 busy logical threads on 8 physical cores. F65d moves the exact shift-gather (F26, two per layer instance) and padded row-select (F27/F50, one per split) autograd from Python Functions to the native dispatcher, removing Python backward nodes while launching the same kernels. E79 aliases the accepted 0079 options for Windows path length.

F28 starts worker imports during setup: **5 distinct workers** acknowledged initializer imports before training. Setup-to-ready elapsed **11.931 s**, final ready wait **0.000338 s**. These costs are outside the training timer and inside the reported process wall; shutdown remains inside training.

E79 (accepted iteration-0079 Expertium configuration: E76, F86, F65c (alias for Windows path length)), F65d (native exact shift-gather and padded row-select dispatcher/autograd using unchanged F26/F27 kernels): measured GPU input-stage time **1.738 s**, with **0.009 s** overlapping the preceding training batch. The input-stage event covers copying and GPU feature materialization; this is separate device timing, not a wall-time saving estimate.

E79 (accepted iteration-0079 Expertium configuration: E76, F86, F65c (alias for Windows path length)), F65d (native exact shift-gather and padded row-select dispatcher/autograd using unchanged F26/F27 kernels): GPU activity **74.5%**, worker-pool CPU **55.3%**, peak reserved VRAM **11.95 GiB**, minimum available system RAM **8.39 GiB**. [Resource profile](profiles/expertium-E79-F65d/profile.md). Full-history sampling-window sufficiency: `False`; GPU activity is not occupancy.

Speed change: **faster** under the applicable rule. Gains within ±0.85% (median spread over pairs of repeats of an identical configuration) count as the same speed. Retuning requires a material resource reduction.

Decision: **retain main improvement; preliminary only**. Exact recorded parity versus previous accepted run: **True**; logged update mismatches: **0**. Consecutive iterations without a 5% gain: **0**. Three-miss threshold reached: **False**. Round stopping rule: complete the explicit 9-iteration count despite consecutive misses.

| Binary loss | Candidate | Δ vs previous / original | Familywise upper vs previous / original |
|---|---:|---|---|
| curve_log_loss_label_weighted | 0.387472684 | +0.00% / -3.25% | +0.00% / -0.49% |
| curve_log_loss_user_average | 0.370387972 | +0.00% / -4.48% | +0.00% / -0.62% |
| instant_log_loss_label_weighted | 0.398428596 | +0.00% / -0.76% | +0.00% / +0.59% |
| instant_log_loss_user_average | 0.382592464 | +0.00% / -0.99% | +0.00% / +0.86% |

Replicate blocks: **2** reference/candidate pairs; trimmed per-update gain mean **+2.13%**, lower bound **+0.59%** at z = 1.9927 (block SD used 1.08%); block decision **promote**. Replicates exact: reference True, candidate True.

## Evidence and provenance

Each process captures its own dirty source snapshot and full commit at start; the shared training-implementation fingerprint is checked at every user boundary. Historical references retain their original revisions. [Recorded ordered-prefetch contracts](../iteration-0004/prefetch-checks-v1/result.json); functional smoke results are separate from full-history timing.

- [expertium: E79 (accepted iteration-0079 Expertium configuration: E76, F86, F65c (alias for Windows path length)), F65d (native exact shift-gather and padded row-select dispatcher/autograd using unchanged F26/F27 kernels)](expertium/E79-F65d/result.json): complete; training start UTC 2026-10-13T15:40:10.324975+00:00; commit `a58fd9223ba8646ac947223bed9cab70851adb97`, dirty `True`; [source snapshot](../../../sources/experiment-sources/49e3553559ff6a0ac1e82c8ccc71c6256d0b74aafb30edf806ad43da22c28ed2/source.zip), [updates](expertium/E79-F65d/updates.jsonl), [checkpoint](expertium/E79-F65d/master.pth).
- [expertium: E79 (accepted iteration-0079 Expertium configuration: E76, F86, F65c (alias for Windows path length)), F65d (native exact shift-gather and padded row-select dispatcher/autograd using unchanged F26/F27 kernels) smoke](functional-smoke-v1/expertium/E79-F65d/result.json): complete; training start UTC 2026-10-13T15:38:09.543949+00:00; commit `a58fd9223ba8646ac947223bed9cab70851adb97`, dirty `True`; [source snapshot](../../../sources/experiment-sources/49e3553559ff6a0ac1e82c8ccc71c6256d0b74aafb30edf806ad43da22c28ed2/source.zip), [updates](functional-smoke-v1/expertium/E79-F65d/updates.jsonl), [checkpoint](functional-smoke-v1/expertium/E79-F65d/master.pth).

Implementation commit: `7fced6223659f08303718bd8e14055f3f4c956a4` (after experiments).

20 distinct held-out users; one training seed and one timing/configuration; no broader validation or timing-repeatability claim. This iteration is complete. Recorded scope: Explicit human count: iteration 1 of 9; complete the count despite consecutive misses; reuse accepted references; fixed cohort; no broader validation; no broader validation is authorized.
