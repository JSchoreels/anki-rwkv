# R01 iteration 0076

Status: **complete preliminary screen**. Authorization: Ten full R01 training cycles.

Same fixed **80 training / 20 unseen held-out users**, **7,654,975 training / 868,370 evaluation reviews**, seed 2025, one epoch, BF16 compute/FP32 master. Chunks contain 128 reviews; prefetch is bounded at twice the worker count. Review exposure stays fixed; declared batch-size or cross-user packing changes may alter update membership, counts, batch seeds, dropout and the learning-rate trajectory. Exact worker/chunk settings and explanations are preserved in the protocol and raw results.

Effective rules: [loop definition](loop-definition.md).

[Frozen protocol](protocol.json), [selection from recorded profiles](selection.json), [loop definition](loop-definition.md), [raw comparison](comparison.json). References retain their original process provenance. Timing includes complete training loads, worker startup/shutdown, preparation, transfers, updates and logs. Setup/primer, save and evaluation are separate; process wall includes them.

Acceptance: measured relative regression ≤1% for all four binary losses against the previous accepted run; ≤2% versus the fixed original baseline is preferred. Confidence bounds are reported separately. Faster but less accurate variants remain separate opt-in fastmode. Accurate resource-saving candidates without a speed gain remain pending retuning. Target ≥5% reviews/s gain versus the previous accepted run; complete the explicit 10-iteration count despite consecutive misses. This remains a preliminary screen; broader validation is pending.

## Full-history measurements

| Model / IDs | Training s | Reviews/s | Gain vs previous / original baseline | Process wall s | Weighted Curve / Instant | User-average Curve / Instant |
|---|---:|---:|---|---:|---|---|
| expertium: E72 (accepted iteration-0072 Expertium configuration: E33, F48, F50, F65b (alias for Windows path length)), F71 (one preparation worker per physical core, main process on the remaining cores), F73a (five preparation workers with F71 core placement) | 41.366 | 185,056 | +8.90% / +516.75% | 84.273 | 0.387472684 / 0.398428596 | 0.370387972 / 0.382592464 |

## Resource evidence and decisions

expertium: Recorded E33 (accepted iteration-0033 Expertium configuration: E31, F42), F48 (deterministic algorithms without uninitialized-memory fill), F50 (skip zero hierarchy padding sources in exact row backward), F65b (native WKV dispatcher/autograd using unchanged NVRTC kernels), F71 (one preparation worker per physical core, main process on the remaining cores) GPU activity is 68.96% and worker-pool CPU 44.26%. Main process CPU averaged 100% of one core; system load averaged 6.4 busy logical threads on 8 physical cores. F73a stacks five preparation workers on F71 core placement: each worker keeps its own physical core and the main process gets three dedicated cores instead of two. E72 aliases the accepted 0072 options for Windows path length. Batch bytes and RNG are identical across 6/5/4 workers.

F28 starts worker imports during setup: **5 distinct workers** acknowledged initializer imports before training. Setup-to-ready elapsed **9.694 s**, final ready wait **0.000295 s**. These costs are outside the training timer and inside the reported process wall; shutdown remains inside training.

E72 (accepted iteration-0072 Expertium configuration: E33, F48, F50, F65b (alias for Windows path length)), F71 (one preparation worker per physical core, main process on the remaining cores), F73a (five preparation workers with F71 core placement): measured GPU input-stage time **1.901 s**, with **0.007 s** overlapping the preceding training batch. The input-stage event covers copying and GPU feature materialization; this is separate device timing, not a wall-time saving estimate.

E72 (accepted iteration-0072 Expertium configuration: E33, F48, F50, F65b (alias for Windows path length)), F71 (one preparation worker per physical core, main process on the remaining cores), F73a (five preparation workers with F71 core placement): GPU activity **64.4%**, worker-pool CPU **51.6%**, peak reserved VRAM **11.90 GiB**, minimum available system RAM **8.87 GiB**. [Resource profile](profiles/expertium-E72-F71-F73a/profile.md). Full-history sampling-window sufficiency: `False`; GPU activity is not occupancy.

Speed change: **faster** under the applicable rule. Gains within ±0.83% (median spread over pairs of repeats of an identical configuration) count as the same speed. Retuning requires a material resource reduction.

Decision: **retain main improvement; preliminary only**. Exact recorded parity versus previous accepted run: **True**; logged update mismatches: **0**. Consecutive iterations without a 5% gain: **0**. Three-miss threshold reached: **False**. Round stopping rule: complete the explicit 10-iteration count despite consecutive misses.

| Binary loss | Candidate | Δ vs previous / original | Familywise upper vs previous / original |
|---|---:|---|---|
| curve_log_loss_label_weighted | 0.387472684 | +0.00% / -3.25% | +0.00% / -0.49% |
| curve_log_loss_user_average | 0.370387972 | +0.00% / -4.48% | +0.00% / -0.62% |
| instant_log_loss_label_weighted | 0.398428596 | +0.00% / -0.76% | +0.00% / +0.59% |
| instant_log_loss_user_average | 0.382592464 | +0.00% / -0.99% | +0.00% / +0.86% |

Replicate blocks: **2** reference/candidate pairs; trimmed per-update gain mean **+2.56%**, lower bound **+1.12%** at z = 1.9927 (block SD used 1.01%); block decision **promote**. Replicates exact: reference True, candidate True.

## Evidence and provenance

Each process captures its own dirty source snapshot and full commit at start; the shared training-implementation fingerprint is checked at every user boundary. Historical references retain their original revisions. [Recorded ordered-prefetch contracts](../iteration-0004/prefetch-checks-v1/result.json); functional smoke results are separate from full-history timing.

- [expertium: E72 (accepted iteration-0072 Expertium configuration: E33, F48, F50, F65b (alias for Windows path length)), F71 (one preparation worker per physical core, main process on the remaining cores), F73a (five preparation workers with F71 core placement)](expertium/E72-F71-F73a/result.json): complete; training start UTC 2026-10-13T14:08:58.636024+00:00; commit `f3d9dd1920b2109123b97c7315b994ef02162302`, dirty `True`; [source snapshot](../../../sources/experiment-sources/f2711276989489ba2e049285acb3973825f4f8c7afcea0e3b5ec950aea7bd5f4/source.zip), [updates](expertium/E72-F71-F73a/updates.jsonl), [checkpoint](expertium/E72-F71-F73a/master.pth).
- [expertium: E72 (accepted iteration-0072 Expertium configuration: E33, F48, F50, F65b (alias for Windows path length)), F71 (one preparation worker per physical core, main process on the remaining cores), F73a (five preparation workers with F71 core placement) smoke](functional-smoke-v1/expertium/E72-F71-F73a/result.json): complete; training start UTC 2026-10-13T14:02:41.255818+00:00; commit `f3d9dd1920b2109123b97c7315b994ef02162302`, dirty `True`; [source snapshot](../../../sources/experiment-sources/f2711276989489ba2e049285acb3973825f4f8c7afcea0e3b5ec950aea7bd5f4/source.zip), [updates](functional-smoke-v1/expertium/E72-F71-F73a/updates.jsonl), [checkpoint](functional-smoke-v1/expertium/E72-F71-F73a/master.pth).

Implementation commit: `530f1655c6369cff6e25810000f380a95554265b` (after experiments).

20 distinct held-out users; one training seed and one timing/configuration; no broader validation or timing-repeatability claim. This iteration is complete. Recorded scope: Explicit human count: iteration 5 of 10; complete the count despite consecutive misses; reuse accepted references; fixed cohort; no broader validation; no broader validation is authorized.

## Interpretation and caveats (baseline-drift-f71, added after the generated report)

- Decision evidence is the paired replicate gate, not the single full run: two counted blocks against accepted 0072 (E33-F48-F50-F65b), trimmed per-update gains +2.49% and +2.64% (mean +2.56%, lower bound +1.12% with the calibrated 1.01% SD) → `promote` at the first look; exact in every run. Total-throughput block ratios were +6.0% (180,947 → 191,834) and −1.7% (187,972 → 184,868); the frozen rule decides on the trimmed per-update estimator.
- The single-run "+8.90% versus previous" compares 0076 at stock GPU clocks (SM about 2,450–2,480 MHz; the MSI Afterburner overclock was not applied) with 0072 recorded at about 3,140 MHz, so it is not a like-for-like speed figure. All 0076 replicates ran at stock clocks.
- Unlike F71 alone (0074, about +0.5–1.5% per update when controls were spread), F73a gained with spread-state controls (summed preparation 113–117 s): five pinned workers (108–111 s summed preparation, queue wait ≤0.6 s) leave the main process three dedicated cores.
- Candidate track `E72-F71-F73a` uses the `E72` alias (accepted 0072 configuration: E33, F48, F50, F65b) because the same options as 0075 (`E33-F48-F50-F65b-F71-F73a`) failed at smoke setup on the Windows 260-character path limit; the smoke repeat carried that reason. Worker-count byte/RNG check: `artifacts/diagnostics/r01-f73-worker-counts-v1` (6/5/4 equal).
