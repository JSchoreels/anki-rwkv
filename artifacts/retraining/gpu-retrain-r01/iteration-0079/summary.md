# R01 iteration 0079

Status: **complete preliminary screen**. Authorization: Ten full R01 training cycles.

Same fixed **80 training / 20 unseen held-out users**, **7,654,975 training / 868,370 evaluation reviews**, seed 2025, one epoch, BF16 compute/FP32 master. Chunks contain 128 reviews; prefetch is bounded at twice the worker count. Review exposure stays fixed; declared batch-size or cross-user packing changes may alter update membership, counts, batch seeds, dropout and the learning-rate trajectory. Exact worker/chunk settings and explanations are preserved in the protocol and raw results.

Effective rules: [loop definition](loop-definition.md).

[Frozen protocol](protocol.json), [selection from recorded profiles](selection.json), [loop definition](loop-definition.md), [raw comparison](comparison.json). References retain their original process provenance. Timing includes complete training loads, worker startup/shutdown, preparation, transfers, updates and logs. Setup/primer, save and evaluation are separate; process wall includes them.

Acceptance: measured relative regression ≤1% for all four binary losses against the previous accepted run; ≤2% versus the fixed original baseline is preferred. Confidence bounds are reported separately. Faster but less accurate variants remain separate opt-in fastmode. Accurate resource-saving candidates without a speed gain remain pending retuning. Target ≥5% reviews/s gain versus the previous accepted run; complete the explicit 10-iteration count despite consecutive misses. This remains a preliminary screen; broader validation is pending.

## Full-history measurements

| Model / IDs | Training s | Reviews/s | Gain vs previous / original baseline | Process wall s | Weighted Curve / Instant | User-average Curve / Instant |
|---|---:|---:|---|---:|---|---|
| expertium: E76 (accepted iteration-0076 Expertium configuration: E72, F71, F73a (alias for Windows path length)), F86 (main process and preparation workers opted out of Windows power throttling at above-normal priority), F65c (native exact GroupNorm dispatcher/autograd using unchanged reduction kernels) | 42.013 | 182,206 | -1.54% / +507.25% | 86.216 | 0.387472684 / 0.398428596 | 0.370387972 / 0.382592464 |

## Resource evidence and decisions

expertium: Recorded E72 (accepted iteration-0072 Expertium configuration: E33, F48, F50, F65b (alias for Windows path length)), F71 (one preparation worker per physical core, main process on the remaining cores), F73a (five preparation workers with F71 core placement) GPU activity is 64.40% and worker-pool CPU 51.60%. Main process CPU averaged 101% of one core; system load averaged 5.1 busy logical threads on 8 physical cores. Stacks the two exact stack candidates on accepted 0076 (E76): F86 power-throttling opt-out at above-normal priority for the main process and pinned workers, and F65c native GroupNorm dispatcher/autograd with unchanged reduction kernels. E76 aliases the accepted 0076 options for Windows path length.

F28 starts worker imports during setup: **5 distinct workers** acknowledged initializer imports before training. Setup-to-ready elapsed **12.073 s**, final ready wait **0.000381 s**. These costs are outside the training timer and inside the reported process wall; shutdown remains inside training.

E76 (accepted iteration-0076 Expertium configuration: E72, F71, F73a (alias for Windows path length)), F86 (main process and preparation workers opted out of Windows power throttling at above-normal priority), F65c (native exact GroupNorm dispatcher/autograd using unchanged reduction kernels): measured GPU input-stage time **1.897 s**, with **0.010 s** overlapping the preceding training batch. The input-stage event covers copying and GPU feature materialization; this is separate device timing, not a wall-time saving estimate.

E76 (accepted iteration-0076 Expertium configuration: E72, F71, F73a (alias for Windows path length)), F86 (main process and preparation workers opted out of Windows power throttling at above-normal priority), F65c (native exact GroupNorm dispatcher/autograd using unchanged reduction kernels): GPU activity **66.4%**, worker-pool CPU **50.5%**, peak reserved VRAM **11.90 GiB**, minimum available system RAM **8.20 GiB**. [Resource profile](profiles/expertium-E76-F86-F65c/profile.md). Full-history sampling-window sufficiency: `False`; GPU activity is not occupancy.

Speed change: **faster** under the applicable rule. Gains within ±0.85% (median spread over pairs of repeats of an identical configuration) count as the same speed. Retuning requires a material resource reduction.

Decision: **retain main improvement; preliminary only**. Exact recorded parity versus previous accepted run: **True**; logged update mismatches: **0**. Consecutive iterations without a 5% gain: **0**. Three-miss threshold reached: **False**. Round stopping rule: complete the explicit 10-iteration count despite consecutive misses.

| Binary loss | Candidate | Δ vs previous / original | Familywise upper vs previous / original |
|---|---:|---|---|
| curve_log_loss_label_weighted | 0.387472684 | +0.00% / -3.25% | +0.00% / -0.49% |
| curve_log_loss_user_average | 0.370387972 | +0.00% / -4.48% | +0.00% / -0.62% |
| instant_log_loss_label_weighted | 0.398428596 | +0.00% / -0.76% | +0.00% / +0.59% |
| instant_log_loss_user_average | 0.382592464 | +0.00% / -0.99% | +0.00% / +0.86% |

Replicate blocks: **2** reference/candidate pairs; trimmed per-update gain mean **+6.16%**, lower bound **+4.67%** at z = 1.9927 (block SD used 1.01%); block decision **promote**. Replicates exact: reference True, candidate True.

## Evidence and provenance

Each process captures its own dirty source snapshot and full commit at start; the shared training-implementation fingerprint is checked at every user boundary. Historical references retain their original revisions. [Recorded ordered-prefetch contracts](../iteration-0004/prefetch-checks-v1/result.json); functional smoke results are separate from full-history timing.

- [expertium: E76 (accepted iteration-0076 Expertium configuration: E72, F71, F73a (alias for Windows path length)), F86 (main process and preparation workers opted out of Windows power throttling at above-normal priority), F65c (native exact GroupNorm dispatcher/autograd using unchanged reduction kernels)](expertium/E76-F86-F65c/result.json): complete; training start UTC 2026-10-13T15:06:25.559811+00:00; commit `72b00c62f6997e2e546a756dc6e3572a37b9e8b3`, dirty `True`; [source snapshot](../../../sources/experiment-sources/273fdb2f8251ec037d3ea2d1f349b9c5ac0552ed52bc4919cb0f6213f8b4adbb/source.zip), [updates](expertium/E76-F86-F65c/updates.jsonl), [checkpoint](expertium/E76-F86-F65c/master.pth).
- [expertium: E76 (accepted iteration-0076 Expertium configuration: E72, F71, F73a (alias for Windows path length)), F86 (main process and preparation workers opted out of Windows power throttling at above-normal priority), F65c (native exact GroupNorm dispatcher/autograd using unchanged reduction kernels) smoke](functional-smoke-v1/expertium/E76-F86-F65c/result.json): complete; training start UTC 2026-10-13T15:04:01.210762+00:00; commit `72b00c62f6997e2e546a756dc6e3572a37b9e8b3`, dirty `True`; [source snapshot](../../../sources/experiment-sources/273fdb2f8251ec037d3ea2d1f349b9c5ac0552ed52bc4919cb0f6213f8b4adbb/source.zip), [updates](functional-smoke-v1/expertium/E76-F86-F65c/updates.jsonl), [checkpoint](functional-smoke-v1/expertium/E76-F86-F65c/master.pth).

Implementation commit: `187c81ad028aa78da1b18c348a1ae23b8ac61228` (after experiments).

20 distinct held-out users; one training seed and one timing/configuration; no broader validation or timing-repeatability claim. This iteration is complete. Recorded scope: Explicit human count: iteration 8 of 10; complete the count despite consecutive misses; reuse accepted references; fixed cohort; no broader validation; no broader validation is authorized.
