# R01 iteration 0039

Status: **complete preliminary screen**. Authorization: 2026-10-12 human request: pop a new worker for 3 new.

Same fixed **80 training / 20 unseen held-out users**, **7,654,975 training / 868,370 evaluation reviews**, seed 2025, one epoch, BF16 compute/FP32 master. Chunks contain 128 reviews; prefetch is bounded at twice the worker count. Review exposure stays fixed; declared batch-size or cross-user packing changes may alter update membership, counts, batch seeds, dropout and the learning-rate trajectory. Exact worker/chunk settings and explanations are preserved in the protocol and raw results.

Effective rules: [loop definition](loop-definition.md). Exact recorded parity uses the human-authorized **2% measured throughput gate**, with timing spread reported separately; other changes retain the **5% target** and original timing rule.

[Frozen protocol](protocol.json), [selection from recorded profiles](selection.json), [loop definition](loop-definition.md), [raw comparison](comparison.json). References retain their original process provenance. Timing includes complete training loads, worker startup/shutdown, preparation, transfers, updates and logs. Setup/primer, save and evaluation are separate; process wall includes them.

Acceptance: measured relative regression ≤1% for all four binary losses against the previous accepted run; ≤2% versus the fixed original baseline is preferred. Confidence bounds are reported separately. Faster but less accurate variants remain separate opt-in fastmode. Accurate resource-saving candidates without a speed gain remain pending retuning. Target ≥5% reviews/s gain versus the previous accepted run for nonexact changes, or ≥2% measured gain for exact recorded parity; stop each model after three consecutive misses of the applicable target. This remains a preliminary screen; broader validation is pending.

## Full-history measurements

| Model / IDs | Training s | Reviews/s | Gain vs previous / original baseline | Process wall s | Weighted Curve / Instant | User-average Curve / Instant |
|---|---:|---:|---|---:|---|---|
| expertium: E33 (accepted iteration-0033 Expertium configuration: E31, F42), F46 (load-pipelined exact shift backward) | 49.597 | 154,344 | -9.59% / +414.39% | 106.543 | 0.387472684 / 0.398428596 | 0.370387972 / 0.382592464 |

## Resource evidence and decisions

expertium: Recorded E30 (accepted iteration-0030 Expertium configuration: E27, F35, F38), F37 (loss without host synchronization), F42 (batch-wide column extraction for exact sample preparation) GPU activity is 69.09% and worker-pool CPU 42.43%. Main process CPU averaged 95% of one core; system load averaged 5.5 busy logical threads on 8 physical cores. Pipeline BF16 shift-backward source-index and gradient loads across 32-source passes. Preserve each lane accumulation order, warp-tree reduction, sequential tail and final rounding; all E33 host and recurrence options stay fixed.

F28 starts worker imports during setup: **6 distinct workers** acknowledged initializer imports before training. Setup-to-ready elapsed **15.230 s**, final ready wait **0.000489 s**. These costs are outside the training timer and inside the reported process wall; shutdown remains inside training.

E33 (accepted iteration-0033 Expertium configuration: E31, F42), F46 (load-pipelined exact shift backward): measured GPU input-stage time **4.027 s**, with **0.011 s** overlapping the preceding training batch. The input-stage event covers copying and GPU feature materialization; this is separate device timing, not a wall-time saving estimate.

E33 (accepted iteration-0033 Expertium configuration: E31, F42), F46 (load-pipelined exact shift backward): GPU activity **58.1%**, worker-pool CPU **56.5%**, peak reserved VRAM **11.96 GiB**, minimum available system RAM **2.80 GiB**. [Resource profile](profiles/expertium-E33-F46/profile.md). Full-history sampling-window sufficiency: `False`; GPU activity is not occupancy.

Speed change: **slower** under the applicable rule. The recorded ±5.68% repeat spread remains descriptive under the authorized exact-parity gate; the measured gain does not establish repeatability.

Decision: **retain previous accepted run; speed target missed**. Exact recorded parity versus previous accepted run: **True**; logged update mismatches: **0**. Consecutive iterations without a 2% gain: **3**. Three-iteration stop threshold reached: **True**.

| Binary loss | Candidate | Δ vs previous / original | Familywise upper vs previous / original |
|---|---:|---|---|
| curve_log_loss_label_weighted | 0.387472684 | +0.00% / -3.25% | +0.00% / -0.49% |
| curve_log_loss_user_average | 0.370387972 | +0.00% / -4.48% | +0.00% / -0.62% |
| instant_log_loss_label_weighted | 0.398428596 | +0.00% / -0.76% | +0.00% / +0.59% |
| instant_log_loss_user_average | 0.382592464 | +0.00% / -0.99% | +0.00% / +0.86% |

## Evidence and provenance

Each process captures its own dirty source snapshot and full commit at start; the shared training-implementation fingerprint is checked at every user boundary. Historical references retain their original revisions. [Recorded ordered-prefetch contracts](../iteration-0004/prefetch-checks-v1/result.json); functional smoke results are separate from full-history timing.

- [expertium: E33 (accepted iteration-0033 Expertium configuration: E31, F42), F46 (load-pipelined exact shift backward)](expertium/E33-F46/result.json): complete; training start UTC 2026-10-12T18:34:25.105404+00:00; commit `206c75fab46613f982f832cd9e87abb93169db50`, dirty `True`; [source snapshot](../../../sources/experiment-sources/d81407b4401a7a2978177afd90d5223f3edbdb5734e11ec6134d9ffbd89685bb/source.zip), [updates](expertium/E33-F46/updates.jsonl), [checkpoint](expertium/E33-F46/master.pth).
- [expertium: E33 (accepted iteration-0033 Expertium configuration: E31, F42), F46 (load-pipelined exact shift backward) smoke](functional-smoke-v1/expertium/E33-F46/result.json): complete; training start UTC 2026-10-12T18:33:55.189304+00:00; commit `206c75fab46613f982f832cd9e87abb93169db50`, dirty `True`; [source snapshot](../../../sources/experiment-sources/d81407b4401a7a2978177afd90d5223f3edbdb5734e11ec6134d9ffbd89685bb/source.zip), [updates](functional-smoke-v1/expertium/E33-F46/updates.jsonl), [checkpoint](functional-smoke-v1/expertium/E33-F46/master.pth).

Implementation commit: `a8da8767391a005d21c344cd3898905f94de61a1` (after experiments).

20 distinct held-out users; one training seed and one timing/configuration; no broader validation or timing-repeatability claim. This iteration is complete. Recorded scope: reuse recorded original baselines and accepted runs; one selected candidate per active model; focused checks as needed; no larger cohort; no broader validation is authorized.
