# Review of R01 round 0041–0045: telemetry and turnaround notes

Analysis only, made from recorded results. No training, smoke, protocol or check was run, and no accepted reference or decision changes. Source: Git `2b1b802e6b10b569f3e2bcbe494a9f2da6de4ac9`, clean tree. [telemetry.py](telemetry.py) reads each run's `result.json` and the one-second `gpu_power.jsonl` monitor; output is in [telemetry.json](telemetry.json). Busy samples are those with utilization ≥20%. nvidia-smi utilization is the share of time any kernel runs. It does not measure SM throughput, so power per utilization point is only a proxy.

## Findings

| Question | Observation | Limits |
|---|---|---|
| Was F49 (0043) slow because streams overlap badly? | Probably not mainly. Peak reserved VRAM was 15,460 MiB, against 12,196 MiB in 0042, which is close to the 95% cap of a 16 GB card. nvidia-smi `memory.used` reached 15,934 MiB. Utilization sat at a 100% median while power was 95.5 W, against 194.2 W in 0042, and the GPU ran 9 °C cooler. The GPU was occupied but stalled, which fits VRAM exhaustion (Windows WDDM paging or allocator free/retry) better than compute contention. F17's 640-chunk 0019 also collapsed (−50%) at 14.13 GiB reserved. | Allocator retry counts, shared-memory use and clocks were not recorded. This is a hypothesis to test, not an attribution. |
| Why was F48 alone (0041) so slow (−23.50%)? | Other processes held more VRAM than in any other normal run: `memory.used` median minus own reserved was 3,016 MiB, against 1,907–2,497 MiB elsewhere. Its max was 15,487 MiB. Preparation (173.7 s) and validation (34.7 s) were also slow. The F48 diagnostic measured −4.68% synchronous update time on the large batch, so F48 alone should have been faster. | One run. CPU and VRAM states are confounded. |
| Is there a GPU-state signal behind run-to-run drift? | Busy-sample power per utilization point splits into two groups: about 1.7–1.96 W/% (0027, 0030, 0033–0038, 0041) and 2.36–2.56 W/% (0031, 0039, 0040, 0042, 0044, 0045). The same-configuration pair 0033 → 0040 moves 1.69 → 2.45 W/% at +7.05% reviews/s. | Power also depends on workload mix. Without SM/memory clocks, P-state and throttle reasons, this cannot separate clock state from workload. |
| How large is the 0042 gain against the same-configuration repeat? | 197,499 / 182,745 (0040 E33 repeat) = +8.07%, against +15.69% vs 0033. Both references ran exact E33. | Both are single runs. |
| Where does process time go now? | 0042 case: 88.48 s. Training 38.76 s (44%), held-out evaluation 29.63 s, primer 15.31 s, model setup 4.53 s. `pool_prestart_seconds` is read after the primer, so it matches primer time and hides the real worker spawn time. Evaluation spawns a fresh six-worker spawn-context pool after the training pool closes (`preparation_pool(workers)` in `scripts/gpu_r01_training.py`), so that spawn and import time falls inside `validation_seconds`. | The spawn's share of the 29.6 s is unmeasured. |

## Would a best-of (max reviews/s) estimator reduce noise?

[segments.py](segments.py) aligns exact runs by work done, using console progress lines (cumulative reviews ÷ average rate). It splits each run into eight equal-work segments ([segments.json](segments.json)); segment sums are within 0.4 s of `training_seconds`. Exact runs process identical updates in identical order, so segment times are paired.

| Pair | Total time ratio | Per-segment ratios | Reading |
|---|---|---|---|
| 0033 / 0040 (same E33 config) | 1.071 | 1.02 1.08 1.08 1.08 1.16 1.06 1.08 1.07 | Uniform whole-run slow state, not spikes. Segment-wise minimum equals the better run (41.52 s). |
| 0041 / 0033 (F48 alone vs E33) | 1.308 | 1.26 0.97 1.03 1.12 0.98 1.20 2.15 1.78 | A late transient: the last two segments are 2.15× and 1.78×. The median segment ratio is 1.16. |
| 0045 / 0042 (F54 vs accepted) | 1.004 | 1.01 0.96 0.96 1.04 0.97 1.05 1.03 0.99 | About ±4% segment noise, cancelling in the total. |
| 0044 / 0042 (F53 vs accepted) | 1.051 | 1.09 1.02 1.07 1.09 1.01 1.04 1.09 0.97 | A consistent ~5% in 6 of 8 segments. |

Conclusion: the data contain two kinds of noise. Transient spikes (0041's tail) are removed by taking the per-segment minimum across repeats, or a median of paired segment ratios. Whole-run slow states (0033 vs 0040) are untouched by any estimator inside a pair. Best-of-N runs helps only if every arm gets at least one fast-state run. Interleave the arms (A B A B) and compare best-of-N with best-of-N. If 6 of 15 runs being in the higher-power group is representative and runs are independent, best-of-2 reaches a fast run about 64% of the time and best-of-3 about 78%. A best-of statistic also hides fragile candidates (F49-style memory cliffs), so report it alongside the total and resource flags. Segment 1 holds all queue wait (0.47–0.56 s) and runs 257 ms per update at a 31k mean update size, against 181–292 ms per update later at 40–66k (F64).

The resulting proposals are F58–F64 and the F49/F55 next-step updates in [GPU_OPTIMIZATIONS.md](../../../docs/GPU_OPTIMIZATIONS.md). Each needs fresh human authorization before any training run.
