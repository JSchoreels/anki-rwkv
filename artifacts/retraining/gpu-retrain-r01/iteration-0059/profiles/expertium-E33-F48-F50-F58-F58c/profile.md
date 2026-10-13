# Resource profile: expertium/E33-F48-F50-F58-F58c

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 20 | 19.7 | 7.4 / 8.0 | 1.0 | 13.8 | 10.53 | 4.80 | 0.14 |
| training including user transitions | 43 | 43.9 | 65.0 / 92.0 | 17.5 | 59.8 | 6.72 | 8.34 | 11.90 |
| checkpoint save/evaluation | 31 | 31.1 | 27.9 / 40.0 | 1.8 | 52.2 | 8.94 | 6.37 | 11.90 |

Logged coverage: **80/80 training users**, **167 updates**, **7,654,975 reviews**. Held-out cohort: **20 users**.
Complete training observation: **True**. Resource telemetry does not establish predictive quality.
Full-history profile available: **False**; observed training window: **43.9 s**.

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

Summed preparation across workers: 154.2 s; summed future waits: 0.7 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- swap/paging: PdhAddEnglishCounterW failed. Performance counters may be disabled..
- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.813 s; sampler-thread CPU: 0.391 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0059/expertium/E33-F48-F50-F58-F58c/result.json](../../expertium/E33-F48-F50-F58-F58c/result.json): c92a7c2f802c9f757b83e07a4963ee59bd8f67807624ea6e4485914e078d9aad.
- [artifacts/retraining/gpu-retrain-r01/iteration-0059/protocol.json](../../protocol.json): 6dae3735c3839b6f9f0249cc7b26d42be5ce9635d797ae72f75193728039c3f8.
- [artifacts/retraining/gpu-retrain-r01/iteration-0059/expertium/E33-F48-F50-F58-F58c/resources.jsonl](../../expertium/E33-F48-F50-F58-F58c/resources.jsonl): 256bafc7d1473127a60472ccd33ea41890faa4bd34bd8b685d07cc1704cff707.
- [artifacts/retraining/gpu-retrain-r01/iteration-0059/expertium/E33-F48-F50-F58-F58c/updates.jsonl](../../expertium/E33-F48-F50-F58-F58c/updates.jsonl): fb63674da17d8b9bc3dfda2ade077cce98f816846bf1b586a4b66104f09594a1.
- [artifacts/retraining/gpu-retrain-r01/iteration-0059/expertium/E33-F48-F50-F58-F58c/resources-summary.json](../../expertium/E33-F48-F50-F58-F58c/resources-summary.json): 03567e38ee584ce90d91b3bc497328582a46acdc6c0f02443406d245afa1baf1.
