# Resource profile: expertium/E33-F48-F50-F65b-F65c

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 20 | 19.6 | 8.3 / 10.0 | 2.5 | 12.5 | 13.11 | 4.84 | 0.14 |
| training including user transitions | 43 | 43.6 | 64.8 / 92.0 | 19.0 | 63.5 | 9.46 | 8.33 | 11.90 |
| checkpoint save/evaluation | 32 | 32.1 | 27.8 / 39.0 | 1.9 | 51.5 | 11.60 | 6.35 | 11.90 |

Logged coverage: **80/80 training users**, **167 updates**, **7,654,975 reviews**. Held-out cohort: **20 users**.
Complete training observation: **True**. Resource telemetry does not establish predictive quality.
Full-history profile available: **False**; observed training window: **43.6 s**.

| History decile | Logged users | Completed training histories | Reviews | Updates |
|---|---:|---:|---:|---:|
| 1 | 4 | 4 | 21,917 | 4 |
| 2 | 11 | 11 | 103,948 | 11 |
| 3 | 6 | 6 | 80,507 | 6 |
| 4 | 14 | 14 | 262,889 | 14 |
| 5 | 8 | 8 | 224,226 | 8 |
| 6 | 7 | 7 | 279,717 | 7 |
| 7 | 10 | 10 | 536,500 | 10 |
| 8 | 4 | 4 | 299,415 | 8 |
| 9 | 7 | 7 | 939,563 | 18 |
| 10 | 9 | 9 | 4,906,293 | 81 |

Packed updates count once in each participating history decile; decile update counts need not sum to the optimizer update total.

Summed preparation across workers: 162.2 s; summed future waits: 0.9 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- swap/paging: PdhAddEnglishCounterW failed. Performance counters may be disabled..
- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.389 s; sampler-thread CPU: 0.344 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0073/expertium/E33-F48-F50-F65b-F65c/result.json](../../expertium/E33-F48-F50-F65b-F65c/result.json): 03712c10043782ba7d6fa3b2cc1979d52633be7720bedf2178106579bec50e76.
- [artifacts/retraining/gpu-retrain-r01/iteration-0073/protocol.json](../../protocol.json): 65056bc77a9ac285769254bd535c8c9410c1ea2a739f4fd8fbc67e02f7c9ef3a.
- [artifacts/retraining/gpu-retrain-r01/iteration-0073/expertium/E33-F48-F50-F65b-F65c/resources.jsonl](../../expertium/E33-F48-F50-F65b-F65c/resources.jsonl): 97fc72abfc895de4a276394d8c9e7aca8f56cc8448f1697fb551eeb8497a3d8b.
- [artifacts/retraining/gpu-retrain-r01/iteration-0073/expertium/E33-F48-F50-F65b-F65c/updates.jsonl](../../expertium/E33-F48-F50-F65b-F65c/updates.jsonl): a685c6a13cd5eb6c2d5445e85b710644aaded953831762eb4a26caed489da674.
- [artifacts/retraining/gpu-retrain-r01/iteration-0073/expertium/E33-F48-F50-F65b-F65c/resources-summary.json](../../expertium/E33-F48-F50-F65b-F65c/resources-summary.json): 18268b2885aa5f28de94d7d9bdd91cb9070554f78896091a9dc7bd3f3f08f54b.
