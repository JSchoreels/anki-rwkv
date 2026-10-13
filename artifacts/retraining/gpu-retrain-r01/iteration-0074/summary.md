# R01 iteration 0074

Status: **complete preliminary screen**. Authorization: Ten full R01 training cycles.

Same fixed **80 training / 20 unseen held-out users**, **7,654,975 training / 868,370 evaluation reviews**, seed 2025, one epoch, BF16 compute/FP32 master. Chunks contain 128 reviews; prefetch is bounded at twice the worker count. Review exposure stays fixed; declared batch-size or cross-user packing changes may alter update membership, counts, batch seeds, dropout and the learning-rate trajectory. Exact worker/chunk settings and explanations are preserved in the protocol and raw results.

Effective rules: [loop definition](loop-definition-amendment-001.md). [Protocol amendment](reference-reuse-amendment.json) records the transition; the original protocol remains preserved.

[Frozen protocol](protocol.json), [selection from recorded profiles](selection.json), [loop definition](loop-definition.md), [raw comparison](comparison.json). References retain their original process provenance. Timing includes complete training loads, worker startup/shutdown, preparation, transfers, updates and logs. Setup/primer, save and evaluation are separate; process wall includes them.

Acceptance: measured relative regression ≤1% for all four binary losses against the previous accepted run; ≤2% versus the fixed original baseline is preferred. Confidence bounds are reported separately. Faster but less accurate variants remain separate opt-in fastmode. Accurate resource-saving candidates without a speed gain remain pending retuning. Target ≥5% reviews/s gain versus the previous accepted run; complete the explicit 10-iteration count despite consecutive misses. This remains a preliminary screen; broader validation is pending.

## Full-history measurements

| Model / IDs | Training s | Reviews/s | Gain vs previous / original baseline | Process wall s | Weighted Curve / Instant | User-average Curve / Instant |
|---|---:|---:|---|---:|---|---|
| expertium: E33 (accepted iteration-0033 Expertium configuration: E31, F42), F48 (deterministic algorithms without uninitialized-memory fill), F50 (skip zero hierarchy padding sources in exact row backward), F65b (native WKV dispatcher/autograd using unchanged NVRTC kernels), F71 (one preparation worker per physical core, main process on the remaining cores) | 46.243 | 165,538 | -2.59% / +451.70% | 92.100 | 0.387472684 / 0.398428596 | 0.370387972 / 0.382592464 |

## Resource evidence and decisions

expertium: Recorded E33 (accepted iteration-0033 Expertium configuration: E31, F42), F48 (deterministic algorithms without uninitialized-memory fill), F50 (skip zero hierarchy padding sources in exact row backward), F65b (native WKV dispatcher/autograd using unchanged NVRTC kernels) GPU activity is 63.55% and worker-pool CPU 62.81%. Main process CPU averaged 110% of one core; system load averaged 6.2 busy logical threads on 8 physical cores. F71 pins each of the six prestarted preparation workers to its own physical core and gives the main process the two best-ranked remaining cores, restoring its affinity before evaluation. Placement only; arithmetic and batch order are unchanged.

F28 starts worker imports during setup: **6 distinct workers** acknowledged initializer imports before training. Setup-to-ready elapsed **12.402 s**, final ready wait **0.000329 s**. These costs are outside the training timer and inside the reported process wall; shutdown remains inside training.

E33 (accepted iteration-0033 Expertium configuration: E31, F42), F48 (deterministic algorithms without uninitialized-memory fill), F50 (skip zero hierarchy padding sources in exact row backward), F65b (native WKV dispatcher/autograd using unchanged NVRTC kernels), F71 (one preparation worker per physical core, main process on the remaining cores): measured GPU input-stage time **2.102 s**, with **0.009 s** overlapping the preceding training batch. The input-stage event covers copying and GPU feature materialization; this is separate device timing, not a wall-time saving estimate.

E33 (accepted iteration-0033 Expertium configuration: E31, F42), F48 (deterministic algorithms without uninitialized-memory fill), F50 (skip zero hierarchy padding sources in exact row backward), F65b (native WKV dispatcher/autograd using unchanged NVRTC kernels), F71 (one preparation worker per physical core, main process on the remaining cores): GPU activity **69.0%**, worker-pool CPU **44.3%**, peak reserved VRAM **11.90 GiB**, minimum available system RAM **7.40 GiB**. [Resource profile](profiles/expertium-E33-F48-F50-F65b-F71/profile.md). Full-history sampling-window sufficiency: `False`; GPU activity is not occupancy.

Speed change: **same within recorded repeat spread** under the applicable rule. Gains within ±0.80% (median spread over pairs of repeats of an identical configuration) count as the same speed. Retuning requires a material resource reduction.

Decision: **investigate retuning freed resources**. Exact recorded parity versus previous accepted run: **True**; logged update mismatches: **0**. Consecutive iterations without a 5% gain: **2**. Three-miss threshold reached: **False**. Round stopping rule: complete the explicit 10-iteration count despite consecutive misses.

| Binary loss | Candidate | Δ vs previous / original | Familywise upper vs previous / original |
|---|---:|---|---|
| curve_log_loss_label_weighted | 0.387472684 | +0.00% / -3.25% | +0.00% / -0.49% |
| curve_log_loss_user_average | 0.370387972 | +0.00% / -4.48% | +0.00% / -0.62% |
| instant_log_loss_label_weighted | 0.398428596 | +0.00% / -0.76% | +0.00% / +0.59% |
| instant_log_loss_user_average | 0.382592464 | +0.00% / -0.99% | +0.00% / +0.86% |

Replicate blocks: **6** reference/candidate pairs; trimmed per-update gain mean **+2.42%**, lower bound **-0.72%** at z = 1.9927 (block SD used 3.91%); block decision **stack candidate**. Replicates exact: reference True, candidate True.

Stack candidate: **positive -2.59% below the promotion rule**; keep the option for later stacking with other small gains (2026-10-12 human request).

Use the recorded resource reduction to investigate higher preparation parallelism or batching; preserve cohort and disclose any changed update schedule. Conditional capacity estimates are not measured gains.

## Evidence and provenance

Each process captures its own dirty source snapshot and full commit at start; the shared training-implementation fingerprint is checked at every user boundary. Historical references retain their original revisions. [Recorded ordered-prefetch contracts](../iteration-0004/prefetch-checks-v1/result.json); functional smoke results are separate from full-history timing.

- [expertium: E33 (accepted iteration-0033 Expertium configuration: E31, F42), F48 (deterministic algorithms without uninitialized-memory fill), F50 (skip zero hierarchy padding sources in exact row backward), F65b (native WKV dispatcher/autograd using unchanged NVRTC kernels), F71 (one preparation worker per physical core, main process on the remaining cores)](expertium/E33-F48-F50-F65b-F71/result.json): complete; training start UTC 2026-10-13T13:16:03.638498+00:00; commit `1e95e3521920c8c5ee8be4041cef73de1740cbdc`, dirty `True`; [source snapshot](../../../sources/experiment-sources/a744227140653bd770ad0256e5d2ebbb538d1f6d3570099490bc8c292d1971d6/source.zip), [updates](expertium/E33-F48-F50-F65b-F71/updates.jsonl), [checkpoint](expertium/E33-F48-F50-F65b-F71/master.pth).
- [expertium: E33 (accepted iteration-0033 Expertium configuration: E31, F42), F48 (deterministic algorithms without uninitialized-memory fill), F50 (skip zero hierarchy padding sources in exact row backward), F65b (native WKV dispatcher/autograd using unchanged NVRTC kernels), F71 (one preparation worker per physical core, main process on the remaining cores) smoke](functional-smoke-v1/expertium/E33-F48-F50-F65b-F71/result.json): complete; training start UTC 2026-10-13T13:15:05.589986+00:00; commit `1e95e3521920c8c5ee8be4041cef73de1740cbdc`, dirty `True`; [source snapshot](../../../sources/experiment-sources/a744227140653bd770ad0256e5d2ebbb538d1f6d3570099490bc8c292d1971d6/source.zip), [updates](functional-smoke-v1/expertium/E33-F48-F50-F65b-F71/updates.jsonl), [checkpoint](functional-smoke-v1/expertium/E33-F48-F50-F65b-F71/master.pth).

Implementation commit pending the completed speed/quality decision.

20 distinct held-out users; one training seed and one timing/configuration; no broader validation or timing-repeatability claim. This iteration is complete. Recorded scope: Explicit human count: iteration 4 of 10; complete the count despite consecutive misses; reuse accepted references; fixed cohort; no broader validation; no broader validation is authorized.

## Interpretation and caveats (baseline-drift-f71, added after the generated report)

- Decision: final look at six counted blocks, trimmed per-update gains +10.70, +0.49, +0.48, +1.52, +0.57, +1.17% (mean +2.42%, lower bound −0.72%) → `stack candidate`; the report's `investigate retuning freed resources` reflects lower worker CPU (~250% versus ~290%). F71 is exact (all runs share final parameters `8141e9b1…`) and is committed opt-in on `main` (`530f165`, default off), not accepted.
- Mechanism: the gain appears only when the control falls into the packed state. Block 1's control was packed (summed preparation 161.8 s) and F71 gained +10.7%; the other controls ran spread (114–129 s) and F71 was within about ±1.5% per update. Uncounted diagnostic 9171, where every control was packed, measured +9.22% (lower bound +8.32%). F71 is therefore insurance against Windows packing workers onto SMT siblings, not a constant speedup ([F71 summary](../../../diagnostics/r01-native-toolchain/baseline-drift/f71/summary.md)).
- GPU clocks: every 0074 run (smoke, full and replicates) ran at stock clocks (SM about 2,430–2,540 MHz) because the MSI Afterburner overclock (+300 MHz, 111% power) was stopped and, after Afterburner restarted at 13:27 UTC, its profile was not applied. Accepted 0072 ran at about 3,150 MHz, so the single-run "−2.59% versus previous" compares different GPU states and is not a speed result; the paired blocks share one state and remain valid.
- Order and pairing: replicates r3/r4 ran candidate-then-reference (alternating order) and are unpaired under the frozen strict reference-then-candidate rule; they are preserved. r5 is an empty attempt refused by the launch guard after `docs/GPU_RETRAIN_LOOP.md` changed (commit c3b34b8, inactive P-A/P-B tooling); [amendment 001](reference-reuse-amendment.json) snapshots the new loop text with unchanged acceptance before r6–r15.
- The preflight was re-recorded as `artifacts/diagnostics/r01-round-0071-0075/preflight-0074.json` after the machine reboot changed the runtime record.
