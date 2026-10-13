# R01 iteration 0077

Status: **complete preliminary screen**. Authorization: Ten full R01 training cycles.

Same fixed **80 training / 20 unseen held-out users**, **7,654,975 training / 868,370 evaluation reviews**, seed 2025, one epoch, BF16 compute/FP32 master. Chunks contain 128 reviews; prefetch is bounded at twice the worker count. Review exposure stays fixed; declared batch-size or cross-user packing changes may alter update membership, counts, batch seeds, dropout and the learning-rate trajectory. Exact worker/chunk settings and explanations are preserved in the protocol and raw results.

Effective rules: [loop definition](loop-definition.md).

[Frozen protocol](protocol.json), [selection from recorded profiles](selection.json), [loop definition](loop-definition.md), [raw comparison](comparison.json). References retain their original process provenance. Timing includes complete training loads, worker startup/shutdown, preparation, transfers, updates and logs. Setup/primer, save and evaluation are separate; process wall includes them.

Acceptance: measured relative regression ≤1% for all four binary losses against the previous accepted run; ≤2% versus the fixed original baseline is preferred. Confidence bounds are reported separately. Faster but less accurate variants remain separate opt-in fastmode. Accurate resource-saving candidates without a speed gain remain pending retuning. Target ≥5% reviews/s gain versus the previous accepted run; complete the explicit 10-iteration count despite consecutive misses. This remains a preliminary screen; broader validation is pending.

## Full-history measurements

| Model / IDs | Training s | Reviews/s | Gain vs previous / original baseline | Process wall s | Weighted Curve / Instant | User-average Curve / Instant |
|---|---:|---:|---|---:|---|---|
| expertium: E72 (accepted iteration-0072 Expertium configuration: E33, F48, F50, F65b (alias for Windows path length)), F71 (one preparation worker per physical core, main process on the remaining cores), F73b (four preparation workers with F71 core placement) | 39.559 | 193,507 | +4.57% / +544.91% | 83.021 | 0.387472684 / 0.398428596 | 0.370387972 / 0.382592464 |

## Resource evidence and decisions

expertium: Recorded E72 (accepted iteration-0072 Expertium configuration: E33, F48, F50, F65b (alias for Windows path length)), F71 (one preparation worker per physical core, main process on the remaining cores), F73a (five preparation workers with F71 core placement) GPU activity is 64.40% and worker-pool CPU 51.60%. Main process CPU averaged 101% of one core; system load averaged 5.1 busy logical threads on 8 physical cores. F73b drops to four preparation workers on F71 physical-core placement, giving the main process four dedicated cores instead of three. Batch bytes and RNG are identical across 6/5/4 workers.

F28 starts worker imports during setup: **4 distinct workers** acknowledged initializer imports before training. Setup-to-ready elapsed **9.485 s**, final ready wait **0.000275 s**. These costs are outside the training timer and inside the reported process wall; shutdown remains inside training.

E72 (accepted iteration-0072 Expertium configuration: E33, F48, F50, F65b (alias for Windows path length)), F71 (one preparation worker per physical core, main process on the remaining cores), F73b (four preparation workers with F71 core placement): measured GPU input-stage time **2.004 s**, with **0.009 s** overlapping the preceding training batch. The input-stage event covers copying and GPU feature materialization; this is separate device timing, not a wall-time saving estimate.

E72 (accepted iteration-0072 Expertium configuration: E33, F48, F50, F65b (alias for Windows path length)), F71 (one preparation worker per physical core, main process on the remaining cores), F73b (four preparation workers with F71 core placement): GPU activity **76.6%**, worker-pool CPU **64.6%**, peak reserved VRAM **11.90 GiB**, minimum available system RAM **9.74 GiB**. [Resource profile](profiles/expertium-E72-F71-F73b/profile.md). Full-history sampling-window sufficiency: `False`; GPU activity is not occupancy.

Speed change: **slower** under the applicable rule. Gains within ±0.83% (median spread over pairs of repeats of an identical configuration) count as the same speed. Retuning requires a material resource reduction.

Decision: **retain previous accepted run; speed target missed**. Exact recorded parity versus previous accepted run: **True**; logged update mismatches: **0**. Consecutive iterations without a 5% gain: **1**. Three-miss threshold reached: **False**. Round stopping rule: complete the explicit 10-iteration count despite consecutive misses.

| Binary loss | Candidate | Δ vs previous / original | Familywise upper vs previous / original |
|---|---:|---|---|
| curve_log_loss_label_weighted | 0.387472684 | +0.00% / -3.25% | +0.00% / -0.49% |
| curve_log_loss_user_average | 0.370387972 | +0.00% / -4.48% | +0.00% / -0.62% |
| instant_log_loss_label_weighted | 0.398428596 | +0.00% / -0.76% | +0.00% / +0.59% |
| instant_log_loss_user_average | 0.382592464 | +0.00% / -0.99% | +0.00% / +0.86% |

Replicate blocks: **2** reference/candidate pairs; trimmed per-update gain mean **-0.12%**, lower bound **-1.53%** at z = 1.9927 (block SD used 1.01%); block decision **reject**. Replicates exact: reference True, candidate True.

## Evidence and provenance

Each process captures its own dirty source snapshot and full commit at start; the shared training-implementation fingerprint is checked at every user boundary. Historical references retain their original revisions. [Recorded ordered-prefetch contracts](../iteration-0004/prefetch-checks-v1/result.json); functional smoke results are separate from full-history timing.

- [expertium: E72 (accepted iteration-0072 Expertium configuration: E33, F48, F50, F65b (alias for Windows path length)), F71 (one preparation worker per physical core, main process on the remaining cores), F73b (four preparation workers with F71 core placement)](expertium/E72-F71-F73b/result.json): complete; training start UTC 2026-10-13T14:36:55.714669+00:00; commit `ba1253fe612ca2bb98982f39c54f4fe70dc1ed82`, dirty `True`; [source snapshot](../../../sources/experiment-sources/fe85a3a9db950de47be1660442f29dd699570599bcb17b67d32a390472e0d9b4/source.zip), [updates](expertium/E72-F71-F73b/updates.jsonl), [checkpoint](expertium/E72-F71-F73b/master.pth).
- [expertium: E72 (accepted iteration-0072 Expertium configuration: E33, F48, F50, F65b (alias for Windows path length)), F71 (one preparation worker per physical core, main process on the remaining cores), F73b (four preparation workers with F71 core placement) smoke](functional-smoke-v1/expertium/E72-F71-F73b/result.json): complete; training start UTC 2026-10-13T14:35:08.333570+00:00; commit `ba1253fe612ca2bb98982f39c54f4fe70dc1ed82`, dirty `True`; [source snapshot](../../../sources/experiment-sources/fe85a3a9db950de47be1660442f29dd699570599bcb17b67d32a390472e0d9b4/source.zip), [updates](functional-smoke-v1/expertium/E72-F71-F73b/updates.jsonl), [checkpoint](functional-smoke-v1/expertium/E72-F71-F73b/master.pth).

Implementation commit: `187c81ad028aa78da1b18c348a1ae23b8ac61228` (after experiments).

20 distinct held-out users; one training seed and one timing/configuration; no broader validation or timing-repeatability claim. This iteration is complete. Recorded scope: Explicit human count: iteration 6 of 10; complete the count despite consecutive misses; reuse accepted references; fixed cohort; no broader validation; no broader validation is authorized.
