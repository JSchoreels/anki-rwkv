# Resource profile: expertium/E33-F48-F50-F34f

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 20 | 19.7 | 8.4 / 12.0 | 1.7 | 14.1 | 9.64 | 4.71 | 0.14 |
| training including user transitions | 44 | 44.7 | 65.5 / 93.0 | 18.1 | 61.3 | 5.92 | 8.38 | 11.90 |
| checkpoint save/evaluation | 30 | 30.1 | 29.1 / 41.0 | 2.2 | 53.7 | 8.11 | 6.34 | 11.90 |

Logged coverage: **80/80 training users**, **167 updates**, **7,654,975 reviews**. Held-out cohort: **20 users**.
Complete training observation: **True**. Resource telemetry does not establish predictive quality.
Full-history profile available: **False**; observed training window: **44.7 s**.

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

Summed preparation across workers: 160.6 s; summed future waits: 0.8 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- swap/paging: PdhAddEnglishCounterW failed. Performance counters may be disabled..
- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.544 s; sampler-thread CPU: 0.359 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0069/expertium/E33-F48-F50-F34f/result.json](../../expertium/E33-F48-F50-F34f/result.json): 3f18e75984076dbec50123eab2c81bcb091542b927cd8feb39b4ba957fc3b78d.
- [artifacts/retraining/gpu-retrain-r01/iteration-0069/protocol.json](../../protocol.json): b1fca819abd6d6f39b1ed4ba016eff57cc9310477ce89d0000181c561e750d73.
- [artifacts/retraining/gpu-retrain-r01/iteration-0069/expertium/E33-F48-F50-F34f/resources.jsonl](../../expertium/E33-F48-F50-F34f/resources.jsonl): ecbe822a16e0d74f671d9d67d946877dc130d7aaccf04eb840410147ee3230a7.
- [artifacts/retraining/gpu-retrain-r01/iteration-0069/expertium/E33-F48-F50-F34f/updates.jsonl](../../expertium/E33-F48-F50-F34f/updates.jsonl): cc73c5ef85bbbf2078da7a1217f153ff1cd2e397a39a6becd17d97877053354e.
- [artifacts/retraining/gpu-retrain-r01/iteration-0069/expertium/E33-F48-F50-F34f/resources-summary.json](../../expertium/E33-F48-F50-F34f/resources-summary.json): db32b76b944048a80513259ab00b82610a54df2aa0eb55d6dbddda163c2de1ba.
