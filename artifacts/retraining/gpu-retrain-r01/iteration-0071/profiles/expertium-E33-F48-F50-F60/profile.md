# Resource profile: expertium/E33-F48-F50-F60

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 20 | 19.6 | 8.6 / 11.0 | 2.5 | 12.8 | 12.84 | 4.83 | 0.14 |
| training including user transitions | 45 | 45.8 | 65.8 / 93.0 | 18.9 | 60.5 | 9.34 | 8.35 | 11.90 |
| checkpoint save/evaluation | 32 | 32.1 | 29.2 / 41.0 | 3.0 | 51.3 | 11.55 | 6.39 | 11.90 |

Logged coverage: **80/80 training users**, **167 updates**, **7,654,975 reviews**. Held-out cohort: **20 users**.
Complete training observation: **True**. Resource telemetry does not establish predictive quality.
Full-history profile available: **False**; observed training window: **45.8 s**.

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

Summed preparation across workers: 162.4 s; summed future waits: 0.7 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- swap/paging: PdhAddEnglishCounterW failed. Performance counters may be disabled..
- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.550 s; sampler-thread CPU: 0.281 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0071/expertium/E33-F48-F50-F60/result.json](../../expertium/E33-F48-F50-F60/result.json): 07d5646cdcf2e9e6aa17b356ed8910b15fe9da9fd2a03448bb8e3b5d162c87ce.
- [artifacts/retraining/gpu-retrain-r01/iteration-0071/protocol.json](../../protocol.json): f2945281da09b0cbab46dea79f20bb6a5bfe30d52876672a9b96db1b274a3c37.
- [artifacts/retraining/gpu-retrain-r01/iteration-0071/expertium/E33-F48-F50-F60/resources.jsonl](../../expertium/E33-F48-F50-F60/resources.jsonl): 5e391e8130a7c5b4d4ebfe87f27508bd90d46cc19a0e81692d1dc3e562d8cb66.
- [artifacts/retraining/gpu-retrain-r01/iteration-0071/expertium/E33-F48-F50-F60/updates.jsonl](../../expertium/E33-F48-F50-F60/updates.jsonl): bcea62f53ad7937456f1cfdd1be8a3b83b16ff7f8ff9f31991d0b9091d4b601d.
- [artifacts/retraining/gpu-retrain-r01/iteration-0071/expertium/E33-F48-F50-F60/resources-summary.json](../../expertium/E33-F48-F50-F60/resources-summary.json): d15b607abedec27768e5bc9291c8ea2a7ee3de00e2426affb35d2fdb48825f1d.
