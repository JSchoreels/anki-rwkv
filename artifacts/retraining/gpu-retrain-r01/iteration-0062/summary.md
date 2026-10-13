# R01 iteration 0062

Status: **complete preliminary screen**. Authorization: I want you to follow R01 cycles, spawn each time a new GPT 6.1 Sol medium thread for 3 iterations, until you reach iteration number 70. Go up to 70 and then shutdown the computer..

Same fixed **80 training / 20 unseen held-out users**, **7,654,975 training / 868,370 evaluation reviews**, seed 2025, one epoch, BF16 compute/FP32 master. Chunks contain 128 reviews; prefetch is bounded at twice the worker count. Review exposure stays fixed; declared batch-size or cross-user packing changes may alter update membership, counts, batch seeds, dropout and the learning-rate trajectory. Exact worker/chunk settings and explanations are preserved in the protocol and raw results.

Effective rules: [loop definition](loop-definition.md).

[Frozen protocol](protocol.json), [selection from recorded profiles](selection.json), [loop definition](loop-definition.md), [raw comparison](comparison.json). References retain their original process provenance. Timing includes complete training loads, worker startup/shutdown, preparation, transfers, updates and logs. Setup/primer, save and evaluation are separate; process wall includes them.

Acceptance: measured relative regression ≤1% for all four binary losses against the previous accepted run; ≤2% versus the fixed original baseline is preferred. Confidence bounds are reported separately. Faster but less accurate variants remain separate opt-in fastmode. Accurate resource-saving candidates without a speed gain remain pending retuning. Target ≥5% reviews/s gain versus the previous accepted run; complete the explicit 23-iteration count despite consecutive misses. This remains a preliminary screen; broader validation is pending.

## Full-history measurements

| Model / IDs | Training s | Reviews/s | Gain vs previous / original baseline | Process wall s | Weighted Curve / Instant | User-average Curve / Instant |
|---|---:|---:|---|---:|---|---|
| expertium: E33 (accepted iteration-0033 Expertium configuration: E31, F42), F48 (deterministic algorithms without uninitialized-memory fill), F50 (skip zero hierarchy padding sources in exact row backward), F58f (recomputation row-value-only pipeline) | 45.328 | 168,881 | -14.49% / +462.84% | 99.778 | 0.387472684 / 0.398428596 | 0.370387972 / 0.382592464 |

## Resource evidence and decisions

expertium: Recorded E33 (accepted iteration-0033 Expertium configuration: E31, F42), F48 (deterministic algorithms without uninitialized-memory fill), F50 (skip zero hierarchy padding sources in exact row backward), F58e (forward recurrence scalar and row-value pipeline) GPU activity is 58.83% and worker-pool CPU 59.78%. Main process CPU averaged 109% of one core; system load averaged 6.3 busy logical threads on 8 physical cores. Pipeline recomputation per-row value loads without scalar prefetch before current shuffle work, preserving arithmetic and reduction order. Test independently on accepted 0042; omit unproven scalar F58 after inconclusive P3b attribution.

F28 starts worker imports during setup: **6 distinct workers** acknowledged initializer imports before training. Setup-to-ready elapsed **15.368 s**, final ready wait **0.000448 s**. These costs are outside the training timer and inside the reported process wall; shutdown remains inside training.

E33 (accepted iteration-0033 Expertium configuration: E31, F42), F48 (deterministic algorithms without uninitialized-memory fill), F50 (skip zero hierarchy padding sources in exact row backward), F58f (recomputation row-value-only pipeline): measured GPU input-stage time **2.828 s**, with **0.010 s** overlapping the preceding training batch. The input-stage event covers copying and GPU feature materialization; this is separate device timing, not a wall-time saving estimate.

E33 (accepted iteration-0033 Expertium configuration: E31, F42), F48 (deterministic algorithms without uninitialized-memory fill), F50 (skip zero hierarchy padding sources in exact row backward), F58f (recomputation row-value-only pipeline): GPU activity **69.9%**, worker-pool CPU **61.0%**, peak reserved VRAM **11.90 GiB**, minimum available system RAM **6.32 GiB**. [Resource profile](profiles/expertium-E33-F48-F50-F58f/profile.md). Full-history sampling-window sufficiency: `False`; GPU activity is not occupancy.

Speed change: **slower** under the applicable rule. Gains within ±0.73% (median spread over pairs of repeats of an identical configuration) count as the same speed. Retuning requires a material resource reduction.

Decision: **retain previous accepted run; speed target missed**. Exact recorded parity versus previous accepted run: **True**; logged update mismatches: **0**. Consecutive iterations without a 5% gain: **15**. Three-miss threshold reached: **True**. Round stopping rule: complete the explicit 23-iteration count despite consecutive misses.

| Binary loss | Candidate | Δ vs previous / original | Familywise upper vs previous / original |
|---|---:|---|---|
| curve_log_loss_label_weighted | 0.387472684 | +0.00% / -3.25% | +0.00% / -0.49% |
| curve_log_loss_user_average | 0.370387972 | +0.00% / -4.48% | +0.00% / -0.62% |
| instant_log_loss_label_weighted | 0.398428596 | +0.00% / -0.76% | +0.00% / +0.59% |
| instant_log_loss_user_average | 0.382592464 | +0.00% / -0.99% | +0.00% / +0.86% |

Replicate blocks: **2** reference/candidate pairs; trimmed per-update gain mean **-0.06%**, lower bound **-1.47%** at z = 1.9927 (block SD used 1.01%); block decision **reject**. Replicates exact: reference True, candidate True.

## Evidence and provenance

Each process captures its own dirty source snapshot and full commit at start; the shared training-implementation fingerprint is checked at every user boundary. Historical references retain their original revisions. [Recorded ordered-prefetch contracts](../iteration-0004/prefetch-checks-v1/result.json); functional smoke results are separate from full-history timing.

- [expertium: E33 (accepted iteration-0033 Expertium configuration: E31, F42), F48 (deterministic algorithms without uninitialized-memory fill), F50 (skip zero hierarchy padding sources in exact row backward), F58f (recomputation row-value-only pipeline)](expertium/E33-F48-F50-F58f/result.json): complete; training start UTC 2026-10-13T03:33:51.826937+00:00; commit `43b6c2a2e426d2d4c3db6df14a573b3797e808d6`, dirty `True`; [source snapshot](../../../sources/experiment-sources/dfc4e5aa3d87fd13a6ef6432404350f001f0f144fe302d744b8eed3c23ff5df6/source.zip), [updates](expertium/E33-F48-F50-F58f/updates.jsonl), [checkpoint](expertium/E33-F48-F50-F58f/master.pth).
- [expertium: E33 (accepted iteration-0033 Expertium configuration: E31, F42), F48 (deterministic algorithms without uninitialized-memory fill), F50 (skip zero hierarchy padding sources in exact row backward), F58f (recomputation row-value-only pipeline) smoke](functional-smoke-v1/expertium/E33-F48-F50-F58f/result.json): complete; training start UTC 2026-10-13T03:33:19.868293+00:00; commit `43b6c2a2e426d2d4c3db6df14a573b3797e808d6`, dirty `True`; [source snapshot](../../../sources/experiment-sources/dfc4e5aa3d87fd13a6ef6432404350f001f0f144fe302d744b8eed3c23ff5df6/source.zip), [updates](functional-smoke-v1/expertium/E33-F48-F50-F58f/updates.jsonl), [checkpoint](functional-smoke-v1/expertium/E33-F48-F50-F58f/master.pth).

Implementation commit: `f127c14f93da8d9ea7f82426fc8f891cb0e65b27` (after experiments).

20 distinct held-out users; one training seed and one timing/configuration; no broader validation or timing-repeatability claim. This iteration is complete. Recorded scope: Explicit human count: iteration 15 of 23; complete the count despite consecutive misses; reuse accepted references; fixed cohort; no broader validation; no broader validation is authorized.
