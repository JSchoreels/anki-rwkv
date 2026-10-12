# R01 iteration 0042

Status: **complete preliminary screen**. Authorization: Ok, here is its feedback. Pop now a new thread for 9 iterations..

Same fixed **80 training / 20 unseen held-out users**, **7,654,975 training / 868,370 evaluation reviews**, seed 2025, one epoch, BF16 compute/FP32 master. Chunks contain 128 reviews; prefetch is bounded at twice the worker count. Review exposure stays fixed; declared batch-size or cross-user packing changes may alter update membership, counts, batch seeds, dropout and the learning-rate trajectory. Exact worker/chunk settings and explanations are preserved in the protocol and raw results.

Effective rules: [loop definition](loop-definition.md). Exact recorded parity uses the human-authorized **2% measured throughput gate**, with timing spread reported separately; other changes retain the **5% target** and original timing rule.

[Frozen protocol](protocol.json), [selection from recorded profiles](selection.json), [loop definition](loop-definition.md), [raw comparison](comparison.json). References retain their original process provenance. Timing includes complete training loads, worker startup/shutdown, preparation, transfers, updates and logs. Setup/primer, save and evaluation are separate; process wall includes them.

Acceptance: measured relative regression ≤1% for all four binary losses against the previous accepted run; ≤2% versus the fixed original baseline is preferred. Confidence bounds are reported separately. Faster but less accurate variants remain separate opt-in fastmode. Accurate resource-saving candidates without a speed gain remain pending retuning. Target ≥5% reviews/s gain versus the previous accepted run for nonexact changes, or ≥2% measured gain for exact recorded parity; stop each model after three consecutive misses of the applicable target. This remains a preliminary screen; broader validation is pending.

## Full-history measurements

| Model / IDs | Training s | Reviews/s | Gain vs previous / original baseline | Process wall s | Weighted Curve / Instant | User-average Curve / Instant |
|---|---:|---:|---|---:|---|---|
| expertium: E33 (accepted iteration-0033 Expertium configuration: E31, F42), F48 (deterministic algorithms without uninitialized-memory fill), F50 (skip zero hierarchy padding sources in exact row backward) | 38.759 | 197,499 | +15.69% / +558.22% | 90.735 | 0.387472684 / 0.398428596 | 0.370387972 / 0.382592464 |

## Resource evidence and decisions

expertium: Recorded E30 (accepted iteration-0030 Expertium configuration: E27, F35, F38), F37 (loss without host synchronization), F42 (batch-wide column extraction for exact sample preparation) GPU activity is 69.09% and worker-pool CPU 42.43%. Main process CPU averaged 95% of one core; system load averaged 5.5 busy logical threads on 8 physical cores. Combine F48 with an exact row-selection backward that excludes zero hierarchy padding sources while preserving sequential real-source order and the unchanged clamped forward.

F28 starts worker imports during setup: **6 distinct workers** acknowledged initializer imports before training. Setup-to-ready elapsed **15.361 s**, final ready wait **0.000444 s**. These costs are outside the training timer and inside the reported process wall; shutdown remains inside training.

E33 (accepted iteration-0033 Expertium configuration: E31, F42), F48 (deterministic algorithms without uninitialized-memory fill), F50 (skip zero hierarchy padding sources in exact row backward): measured GPU input-stage time **2.279 s**, with **0.008 s** overlapping the preceding training batch. The input-stage event covers copying and GPU feature materialization; this is separate device timing, not a wall-time saving estimate.

E33 (accepted iteration-0033 Expertium configuration: E31, F42), F48 (deterministic algorithms without uninitialized-memory fill), F50 (skip zero hierarchy padding sources in exact row backward): GPU activity **66.5%**, worker-pool CPU **53.2%**, peak reserved VRAM **11.91 GiB**, minimum available system RAM **7.15 GiB**. [Resource profile](profiles/expertium-E33-F48-F50/profile.md). Full-history sampling-window sufficiency: `False`; GPU activity is not occupancy.

Speed change: **faster** under the applicable rule. The recorded ±7.05% repeat spread remains descriptive under the authorized exact-parity gate; the measured gain does not establish repeatability.

Decision: **retain main improvement; preliminary only**. Exact recorded parity versus previous accepted run: **True**; logged update mismatches: **0**. Consecutive iterations without a 2% gain: **0**. Three-iteration stop threshold reached: **False**.

| Binary loss | Candidate | Δ vs previous / original | Familywise upper vs previous / original |
|---|---:|---|---|
| curve_log_loss_label_weighted | 0.387472684 | +0.00% / -3.25% | +0.00% / -0.49% |
| curve_log_loss_user_average | 0.370387972 | +0.00% / -4.48% | +0.00% / -0.62% |
| instant_log_loss_label_weighted | 0.398428596 | +0.00% / -0.76% | +0.00% / +0.59% |
| instant_log_loss_user_average | 0.382592464 | +0.00% / -0.99% | +0.00% / +0.86% |

## Evidence and provenance

Each process captures its own dirty source snapshot and full commit at start; the shared training-implementation fingerprint is checked at every user boundary. Historical references retain their original revisions. [Recorded ordered-prefetch contracts](../iteration-0004/prefetch-checks-v1/result.json); functional smoke results are separate from full-history timing.

- [expertium: E33 (accepted iteration-0033 Expertium configuration: E31, F42), F48 (deterministic algorithms without uninitialized-memory fill), F50 (skip zero hierarchy padding sources in exact row backward)](expertium/E33-F48-F50/result.json): complete; training start UTC 2026-10-12T19:23:29.228461+00:00; commit `32ad65319fbabc00871ab6a797177f227d5e5fa6`, dirty `True`; [source snapshot](../../../sources/experiment-sources/b6f8db5fef06815f95bd7423d3a273db130ec3c6a62c18ad6d53f3d1d5dca804/source.zip), [updates](expertium/E33-F48-F50/updates.jsonl), [checkpoint](expertium/E33-F48-F50/master.pth).
- [expertium: E33 (accepted iteration-0033 Expertium configuration: E31, F42), F48 (deterministic algorithms without uninitialized-memory fill), F50 (skip zero hierarchy padding sources in exact row backward) smoke](functional-smoke-v1/expertium/E33-F48-F50/result.json): complete; training start UTC 2026-10-12T19:23:00.853623+00:00; commit `32ad65319fbabc00871ab6a797177f227d5e5fa6`, dirty `True`; [source snapshot](../../../sources/experiment-sources/b6f8db5fef06815f95bd7423d3a273db130ec3c6a62c18ad6d53f3d1d5dca804/source.zip), [updates](functional-smoke-v1/expertium/E33-F48-F50/updates.jsonl), [checkpoint](functional-smoke-v1/expertium/E33-F48-F50/master.pth).

Implementation commit: `66e0bf6e7c70b4580b9078fc8fe9747744ba400b` (after experiments).

20 distinct held-out users; one training seed and one timing/configuration; no broader validation or timing-repeatability claim. This iteration is complete. Recorded scope: reuse recorded original baselines and accepted runs; one selected candidate per active model; focused checks as needed; no larger cohort; no broader validation is authorized.
