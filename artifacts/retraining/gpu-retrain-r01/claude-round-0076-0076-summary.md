# Claude R01 round: iterations 0076–0076

The accepted preliminary expertium reference is **0076: 185,056 reviews/s**, +8.90% versus the round’s starting accepted reference and +516.75% versus the fixed original baseline. Implementation commit `530f1655c6369cff6e25810000f380a95554265b` was made after the experiments.

Authorization: Ten full R01 training cycles. Explicit count: 5 of 10 iterations completed; 5 remain authorized. Consecutive misses do not end this counted round.

## Full-history results

Training seconds and raw review presentations/s include user transitions; complete process seconds also include setup, discarded primer, checkpoint saving and evaluation. Gains below use each candidate’s own recorded previous accepted reference, which can differ between rows. One full timing per candidate does not establish repeatability; broader validation remains pending.

| Iteration | Bundle | Training seconds | Reviews/s | Gain vs prior accepted | Complete process seconds | Queue wait s | Summed worker preparation s | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [0072](iteration-0072/summary.md) | E33-F48-F50-F65b | 45.047 | 169,934 | — | 99.676 | 0.84 | 161.1 | Starting accepted reference |
| [0076](iteration-0076/summary.md) | E72-F71-F73a | 41.366 | 185,056 | +8.90% | 84.273 | 0.50 | 108.1 | retain main improvement; preliminary only |

**0076, E72-F71-F73a.** F73a stacks five preparation workers on F71 core placement: each worker keeps its own physical core and the main process gets three dedicated cores instead of two. E72 aliases the accepted 0072 options for Windows path length. Batch bytes and RNG are identical across 6/5/4 workers. Recorded decision: retain main improvement; preliminary only; exact full recorded parity: True. Training throughput changed +8.90% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.50 versus 0.84; summed preparation s: 108.12 versus 161.11; validation s: 28.22 versus 32.44; primer s: 9.65 versus 15.24; worker CPU %: 51.60 versus 62.81; GPU activity %: 64.40 versus 63.55. These observations include machine-state variation and do not independently attribute a component’s effect.

## Separate held-out binary losses

| Iteration | Metric | Loss | Delta vs prior accepted | Delta vs original baseline |
|---|---|---:|---:|---:|
| 0076 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0076 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0076 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0076 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |

## Checks and provenance

Full recorded parity is exact in 1/1 cases. The recorded acceptance gates and decisions are reused; this report does not change them or substitute parity for broader quality validation.

- 0076: experiment `f3d9dd1920b2109123b97c7315b994ef02162302`, origin `9abe549f645752a5a830fa797982b96b08e10487`, dirty at process start: True; [source archive](../../sources/experiment-sources/f2711276989489ba2e049285acb3973825f4f8c7afcea0e3b5ec950aea7bd5f4/source.zip), [protocol](iteration-0076/protocol.json), [comparison](iteration-0076/comparison.json).
- 0076 pre-run background: 5.8% total CPU; 0 other Python GPU processes detected.

## Pending next steps

- F73b: four pinned workers on E72-F71
- F72 main-thread budget audit (per-phase thread_time) before zero-copy slots
- Enable Windows long paths (human) or keep E72-style aliases for long track names

## Interpretation and caveats for 0076 (baseline-drift-f71)

- Decision evidence is the paired replicate gate, not the single full run: two counted blocks against accepted 0072 (E33-F48-F50-F65b), trimmed per-update gains +2.49% and +2.64% (mean +2.56%, lower bound +1.12% with the calibrated 1.01% SD) → `promote` at the first look; exact in every run. Total-throughput block ratios were +6.0% (180,947 → 191,834) and −1.7% (187,972 → 184,868); the frozen rule decides on the trimmed per-update estimator.
- The single-run "+8.90% versus previous" compares 0076 at stock GPU clocks (SM about 2,450–2,480 MHz; the MSI Afterburner overclock was not applied) with 0072 recorded at about 3,140 MHz, so it is not a like-for-like speed figure. All 0076 replicates ran at stock clocks.
- Unlike F71 alone (0074, about +0.5–1.5% per update when controls were spread), F73a gained with spread-state controls (summed preparation 113–117 s): five pinned workers (108–111 s summed preparation, queue wait ≤0.6 s) leave the main process three dedicated cores.
- Candidate track `E72-F71-F73a` uses the `E72` alias (accepted 0072 configuration: E33, F48, F50, F65b) because the same options as 0075 (`E33-F48-F50-F65b-F71-F73a`) failed at smoke setup on the Windows 260-character path limit; the smoke repeat carried that reason. Worker-count byte/RNG check: `artifacts/diagnostics/r01-f73-worker-counts-v1` (6/5/4 equal).
