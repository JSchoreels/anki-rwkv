# R01 iteration 0037

Status: **complete preliminary screen**. Authorization: 2026-10-12 human request: pop a new worker for 3 new.

Same fixed **80 training / 20 unseen held-out users**, **7,654,975 training / 868,370 evaluation reviews**, seed 2025, one epoch, BF16 compute/FP32 master. Chunks contain 128 reviews; prefetch is bounded at twice the worker count. Review exposure stays fixed; declared batch-size or cross-user packing changes may alter update membership, counts, batch seeds, dropout and the learning-rate trajectory. Exact worker/chunk settings and explanations are preserved in the protocol and raw results.

Effective rules: [loop definition](loop-definition.md). Exact recorded parity uses the human-authorized **2% measured throughput gate**, with timing spread reported separately; other changes retain the **5% target** and original timing rule.

[Frozen protocol](protocol.json), [selection from recorded profiles](selection.json), [loop definition](loop-definition.md), [raw comparison](comparison.json). References retain their original process provenance. Timing includes complete training loads, worker startup/shutdown, preparation, transfers, updates and logs. Setup/primer, save and evaluation are separate; process wall includes them.

Acceptance: measured relative regression ≤1% for all four binary losses against the previous accepted run; ≤2% versus the fixed original baseline is preferred. Confidence bounds are reported separately. Faster but less accurate variants remain separate opt-in fastmode. Accurate resource-saving candidates without a speed gain remain pending retuning. Target ≥5% reviews/s gain versus the previous accepted run for nonexact changes, or ≥2% measured gain for exact recorded parity; stop each model after three consecutive misses of the applicable target. This remains a preliminary screen; broader validation is pending.

## Full-history measurements

| Model / IDs | Training s | Reviews/s | Gain vs previous / original baseline | Process wall s | Weighted Curve / Instant | User-average Curve / Instant |
|---|---:|---:|---|---:|---|---|
| expertium: E33 (accepted iteration-0033 Expertium configuration: E31, F42), F44 (batch-wide chunk-local segment transforms) | 50.861 | 150,508 | -11.84% / +401.61% | 107.521 | 0.387472684 / 0.398428596 | 0.370387972 / 0.382592464 |

## Resource evidence and decisions

expertium: Recorded E33 (accepted iteration-0033 Expertium configuration: E31, F42), F43 (four preparation workers after columnar samples) GPU activity is 67.52% and worker-pool CPU 65.69%. Main process CPU averaged 93% of one core; system load averaged 5.5 busy logical threads on 8 physical cores. Hoist chunk-local day minima, card-first offsets and weekday transforms into one batch-wide pandas pass. Preserve six workers, four recurrence rows and all E33 sample arithmetic and boundaries.

F28 starts worker imports during setup: **6 distinct workers** acknowledged initializer imports before training. Setup-to-ready elapsed **15.029 s**, final ready wait **0.000403 s**. These costs are outside the training timer and inside the reported process wall; shutdown remains inside training.

E33 (accepted iteration-0033 Expertium configuration: E31, F42), F44 (batch-wide chunk-local segment transforms): measured GPU input-stage time **3.423 s**, with **0.011 s** overlapping the preceding training batch. The input-stage event covers copying and GPU feature materialization; this is separate device timing, not a wall-time saving estimate.

E33 (accepted iteration-0033 Expertium configuration: E31, F42), F44 (batch-wide chunk-local segment transforms): GPU activity **65.6%**, worker-pool CPU **44.4%**, peak reserved VRAM **11.96 GiB**, minimum available system RAM **2.89 GiB**. [Resource profile](profiles/expertium-E33-F44/profile.md). Full-history sampling-window sufficiency: `False`; GPU activity is not occupancy.

Speed change: **slower** under the applicable rule. The recorded ±5.68% repeat spread remains descriptive under the authorized exact-parity gate; the measured gain does not establish repeatability.

Decision: **retain previous accepted run; speed target missed**. Exact recorded parity versus previous accepted run: **True**; logged update mismatches: **0**. Consecutive iterations without a 2% gain: **1**. Three-iteration stop threshold reached: **False**.

| Binary loss | Candidate | Δ vs previous / original | Familywise upper vs previous / original |
|---|---:|---|---|
| curve_log_loss_label_weighted | 0.387472684 | +0.00% / -3.25% | +0.00% / -0.49% |
| curve_log_loss_user_average | 0.370387972 | +0.00% / -4.48% | +0.00% / -0.62% |
| instant_log_loss_label_weighted | 0.398428596 | +0.00% / -0.76% | +0.00% / +0.59% |
| instant_log_loss_user_average | 0.382592464 | +0.00% / -0.99% | +0.00% / +0.86% |

## Evidence and provenance

Each process captures its own dirty source snapshot and full commit at start; the shared training-implementation fingerprint is checked at every user boundary. Historical references retain their original revisions. [Recorded ordered-prefetch contracts](../iteration-0004/prefetch-checks-v1/result.json); functional smoke results are separate from full-history timing.

- [expertium: E33 (accepted iteration-0033 Expertium configuration: E31, F42), F44 (batch-wide chunk-local segment transforms)](expertium/E33-F44/result.json): complete; training start UTC 2026-10-12T18:29:05.585334+00:00; commit `f9d314fa6cda6179ae547f4659fc2062da94e8ec`, dirty `True`; [source snapshot](../../../sources/experiment-sources/f517e78cf67ed1b2ca3a59e96f80e7db19b3a96c34b0989e7651545733a18958/source.zip), [updates](expertium/E33-F44/updates.jsonl), [checkpoint](expertium/E33-F44/master.pth).
- [expertium: E33 (accepted iteration-0033 Expertium configuration: E31, F42), F44 (batch-wide chunk-local segment transforms) smoke](functional-smoke-v1/expertium/E33-F44/result.json): complete; training start UTC 2026-10-12T18:28:36.514154+00:00; commit `f9d314fa6cda6179ae547f4659fc2062da94e8ec`, dirty `True`; [source snapshot](../../../sources/experiment-sources/f517e78cf67ed1b2ca3a59e96f80e7db19b3a96c34b0989e7651545733a18958/source.zip), [updates](functional-smoke-v1/expertium/E33-F44/updates.jsonl), [checkpoint](functional-smoke-v1/expertium/E33-F44/master.pth).

Implementation commit: `a8da8767391a005d21c344cd3898905f94de61a1` (after experiments).

20 distinct held-out users; one training seed and one timing/configuration; no broader validation or timing-repeatability claim. This iteration is complete. Recorded scope: New user-authorized round; reuse recorded original baselines and accepted runs; one selected candidate per active model; fixed 100-user cohort, no broader validation; three-miss stop applies within this round; no broader validation is authorized.
