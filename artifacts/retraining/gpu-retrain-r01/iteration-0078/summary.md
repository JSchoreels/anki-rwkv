# R01 iteration 0078

Status: **complete preliminary screen**. Authorization: Ten full R01 training cycles.

Same fixed **80 training / 20 unseen held-out users**, **7,654,975 training / 868,370 evaluation reviews**, seed 2025, one epoch, BF16 compute/FP32 master. Chunks contain 128 reviews; prefetch is bounded at twice the worker count. Review exposure stays fixed; declared batch-size or cross-user packing changes may alter update membership, counts, batch seeds, dropout and the learning-rate trajectory. Exact worker/chunk settings and explanations are preserved in the protocol and raw results.

Effective rules: [loop definition](loop-definition.md).

[Frozen protocol](protocol.json), [selection from recorded profiles](selection.json), [loop definition](loop-definition.md), [raw comparison](comparison.json). References retain their original process provenance. Timing includes complete training loads, worker startup/shutdown, preparation, transfers, updates and logs. Setup/primer, save and evaluation are separate; process wall includes them.

Acceptance: measured relative regression ≤1% for all four binary losses against the previous accepted run; ≤2% versus the fixed original baseline is preferred. Confidence bounds are reported separately. Faster but less accurate variants remain separate opt-in fastmode. Accurate resource-saving candidates without a speed gain remain pending retuning. Target ≥5% reviews/s gain versus the previous accepted run; complete the explicit 10-iteration count despite consecutive misses. This remains a preliminary screen; broader validation is pending.

## Full-history measurements

| Model / IDs | Training s | Reviews/s | Gain vs previous / original baseline | Process wall s | Weighted Curve / Instant | User-average Curve / Instant |
|---|---:|---:|---|---:|---|---|
| expertium: E72 (accepted iteration-0072 Expertium configuration: E33, F48, F50, F65b (alias for Windows path length)), F71 (one preparation worker per physical core, main process on the remaining cores), F73a (five preparation workers with F71 core placement), F86 (main process and preparation workers opted out of Windows power throttling at above-normal priority) | 39.104 | 195,758 | +5.78% / +552.41% | 81.841 | 0.387472684 / 0.398428596 | 0.370387972 / 0.382592464 |

## Resource evidence and decisions

expertium: Recorded E72 (accepted iteration-0072 Expertium configuration: E33, F48, F50, F65b (alias for Windows path length)), F71 (one preparation worker per physical core, main process on the remaining cores), F73a (five preparation workers with F71 core placement) GPU activity is 64.40% and worker-pool CPU 51.60%. Main process CPU averaged 101% of one core; system load averaged 5.1 busy logical threads on 8 physical cores. F86 opts the main process and the five pinned preparation workers out of Windows power throttling (EcoQoS) and raises them to above-normal priority, recording the prior state and timer resolution. Scheduling only; arithmetic and order are unchanged.

F28 starts worker imports during setup: **5 distinct workers** acknowledged initializer imports before training. Setup-to-ready elapsed **9.547 s**, final ready wait **0.000282 s**. These costs are outside the training timer and inside the reported process wall; shutdown remains inside training.

E72 (accepted iteration-0072 Expertium configuration: E33, F48, F50, F65b (alias for Windows path length)), F71 (one preparation worker per physical core, main process on the remaining cores), F73a (five preparation workers with F71 core placement), F86 (main process and preparation workers opted out of Windows power throttling at above-normal priority): measured GPU input-stage time **1.960 s**, with **0.008 s** overlapping the preceding training batch. The input-stage event covers copying and GPU feature materialization; this is separate device timing, not a wall-time saving estimate.

E72 (accepted iteration-0072 Expertium configuration: E33, F48, F50, F65b (alias for Windows path length)), F71 (one preparation worker per physical core, main process on the remaining cores), F73a (five preparation workers with F71 core placement), F86 (main process and preparation workers opted out of Windows power throttling at above-normal priority): GPU activity **70.7%**, worker-pool CPU **53.5%**, peak reserved VRAM **11.90 GiB**, minimum available system RAM **8.92 GiB**. [Resource profile](profiles/expertium-E72-F71-F73a-F86/profile.md). Full-history sampling-window sufficiency: `False`; GPU activity is not occupancy.

Speed change: **same within recorded repeat spread** under the applicable rule. Gains within ±0.84% (median spread over pairs of repeats of an identical configuration) count as the same speed. Retuning requires a material resource reduction.

Decision: **retain previous accepted run; speed target missed**. Exact recorded parity versus previous accepted run: **True**; logged update mismatches: **0**. Consecutive iterations without a 5% gain: **2**. Three-miss threshold reached: **False**. Round stopping rule: complete the explicit 10-iteration count despite consecutive misses.

| Binary loss | Candidate | Δ vs previous / original | Familywise upper vs previous / original |
|---|---:|---|---|
| curve_log_loss_label_weighted | 0.387472684 | +0.00% / -3.25% | +0.00% / -0.49% |
| curve_log_loss_user_average | 0.370387972 | +0.00% / -4.48% | +0.00% / -0.62% |
| instant_log_loss_label_weighted | 0.398428596 | +0.00% / -0.76% | +0.00% / +0.59% |
| instant_log_loss_user_average | 0.382592464 | +0.00% / -0.99% | +0.00% / +0.86% |

Replicate blocks: **6** reference/candidate pairs; trimmed per-update gain mean **+1.74%**, lower bound **-0.22%** at z = 1.9927 (block SD used 2.41%); block decision **stack candidate**. Replicates exact: reference True, candidate True.

Stack candidate: **positive +5.78% below the promotion rule**; keep the option for later stacking with other small gains (2026-10-12 human request).

## Evidence and provenance

Each process captures its own dirty source snapshot and full commit at start; the shared training-implementation fingerprint is checked at every user boundary. Historical references retain their original revisions. [Recorded ordered-prefetch contracts](../iteration-0004/prefetch-checks-v1/result.json); functional smoke results are separate from full-history timing.

- [expertium: E72 (accepted iteration-0072 Expertium configuration: E33, F48, F50, F65b (alias for Windows path length)), F71 (one preparation worker per physical core, main process on the remaining cores), F73a (five preparation workers with F71 core placement), F86 (main process and preparation workers opted out of Windows power throttling at above-normal priority)](expertium/E72-F71-F73a-F86/result.json): complete; training start UTC 2026-10-13T14:44:36.554767+00:00; commit `0c937dd3328529336c0967b95633706522eaaca7`, dirty `True`; [source snapshot](../../../sources/experiment-sources/19b087d1d97d277c3ef8b2a89c93757823bc58919dff554664f9ac1e5a243409/source.zip), [updates](expertium/E72-F71-F73a-F86/updates.jsonl), [checkpoint](expertium/E72-F71-F73a-F86/master.pth).
- [expertium: E72 (accepted iteration-0072 Expertium configuration: E33, F48, F50, F65b (alias for Windows path length)), F71 (one preparation worker per physical core, main process on the remaining cores), F73a (five preparation workers with F71 core placement), F86 (main process and preparation workers opted out of Windows power throttling at above-normal priority) smoke](functional-smoke-v1/expertium/E72-F71-F73a-F86/result.json): complete; training start UTC 2026-10-13T14:42:38.116351+00:00; commit `0c937dd3328529336c0967b95633706522eaaca7`, dirty `True`; [source snapshot](../../../sources/experiment-sources/19b087d1d97d277c3ef8b2a89c93757823bc58919dff554664f9ac1e5a243409/source.zip), [updates](functional-smoke-v1/expertium/E72-F71-F73a-F86/updates.jsonl), [checkpoint](functional-smoke-v1/expertium/E72-F71-F73a-F86/master.pth).

Implementation commit: `187c81ad028aa78da1b18c348a1ae23b8ac61228` (after experiments).

20 distinct held-out users; one training seed and one timing/configuration; no broader validation or timing-repeatability claim. This iteration is complete. Recorded scope: Explicit human count: iteration 7 of 10; complete the count despite consecutive misses; reuse accepted references; fixed cohort; no broader validation; no broader validation is authorized.
