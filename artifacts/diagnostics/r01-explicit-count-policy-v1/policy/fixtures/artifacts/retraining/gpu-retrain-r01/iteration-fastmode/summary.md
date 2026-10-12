# R01 iteration fastmode

Status: **complete preliminary screen**. Authorization: 2026-10-11: user requested one new iteration.

Same fixed **80 training / 20 unseen held-out users**, **960 training / 40 evaluation reviews**, seed 2025, one epoch, BF16 compute/FP32 master. Chunks contain 128 reviews; prefetch is bounded at twice the worker count. Review exposure stays fixed; declared batch-size or cross-user packing changes may alter update membership, counts, batch seeds, dropout and the learning-rate trajectory. Exact worker/chunk settings and explanations are preserved in the protocol and raw results.

Effective rules: [loop definition](../../../../docs/GPU_RETRAIN_LOOP.md). Exact recorded parity uses the human-authorized **2% measured throughput gate**, with timing spread reported separately; other changes retain the **5% target** and original timing rule.

[Frozen protocol](protocol.json), [selection from recorded profiles](selection.json), [loop definition](loop-definition.md), [raw comparison](comparison.json). References retain their original process provenance. Timing includes complete training loads, worker startup/shutdown, preparation, transfers, updates and logs. Setup/primer, save and evaluation are separate; process wall includes them.

Acceptance: measured relative regression ≤1% for all four binary losses against the previous accepted run; ≤2% versus the fixed original baseline is preferred. Confidence bounds are reported separately. Faster but less accurate variants remain separate opt-in fastmode. Accurate resource-saving candidates without a speed gain remain pending retuning. Target ≥5% reviews/s gain versus the previous accepted run for nonexact changes, or ≥2% measured gain for exact recorded parity; stop each model after three consecutive misses of the applicable target. This remains a preliminary screen; broader validation is pending.

## Full-history measurements

| Model / IDs | Training s | Reviews/s | Gain vs previous / original baseline | Process wall s | Weighted Curve / Instant | User-average Curve / Instant |
|---|---:|---:|---|---:|---|---|
| baseline: F01 (ordered user prefetch), F08 (fused shift operations) | 6.400 | 150 | +25.00% / +50.00% | 20.000 | 0.502500000 / 0.502500000 | 0.502500000 / 0.502500000 |

## Resource evidence and decisions

baseline: Recorded F01 (ordered user prefetch) GPU activity is 70.00% and worker-pool CPU 90.00%. Alex is near the GPU-activity target; remove internal NaN/count waits, consolidate metric transfers and buffer log writes, preserving finite-loss/gradient checks and loss arithmetic.

F01 (ordered user prefetch), F08 (fused shift operations): GPU activity **40.0%**, worker-pool CPU **90.0%**, peak reserved VRAM **2.00 GiB**, minimum available system RAM **8.00 GiB**. [Resource profile](profiles/baseline-F01-F08/profile.md). Full-history sampling-window sufficiency: `False`; GPU activity is not occupancy.

Decision: **retain separate opt-in fastmode**. Exact recorded parity versus previous accepted run: **False**; logged update mismatches: **0**. Consecutive iterations without a 5% gain: **0**. Three-miss threshold reached: **False**. Round stopping rule: stop each model after three consecutive misses of the applicable target.

| Binary loss | Candidate | Δ vs previous / original | Familywise upper vs previous / original |
|---|---:|---|---|
| curve_log_loss_label_weighted | 0.502500000 | +0.50% / +0.50% | +0.50% / +0.50% |
| curve_log_loss_user_average | 0.502500000 | +0.50% / +0.50% | +0.50% / +0.50% |
| instant_log_loss_label_weighted | 0.502500000 | +0.50% / +0.50% | +0.50% / +0.50% |
| instant_log_loss_user_average | 0.502500000 | +0.50% / +0.50% | +0.50% / +0.50% |

## Evidence and provenance

Each process captures its own dirty source snapshot and full commit at start; the shared training-implementation fingerprint is checked at every user boundary. Historical references retain their original revisions. [Recorded ordered-prefetch contracts](../iteration-0004/prefetch-checks-v1/result.json); functional smoke results are separate from full-history timing.

- [baseline: F01 (ordered user prefetch), F08 (fused shift operations)](baseline/F01-F08/result.json): complete; training start UTC 2026-01-02T00:00:00+00:00; commit `synthetic-only`, dirty `False`; [source snapshot](../../../synthetic-only-source.zip), [updates](baseline/F01-F08/updates.jsonl), no checkpoint saved.

Implementation commit pending the completed speed/quality decision.

20 distinct held-out users; one training seed and one timing/configuration; no broader validation or timing-repeatability claim. This iteration is complete. Recorded scope: reuse recorded original baselines and accepted runs; one selected candidate per active model; focused checks as needed; no larger cohort; no broader validation is authorized.
