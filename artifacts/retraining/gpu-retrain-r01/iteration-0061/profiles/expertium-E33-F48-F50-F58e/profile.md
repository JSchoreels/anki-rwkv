# Resource profile: expertium/E33-F48-F50-F58e

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 20 | 19.6 | 8.1 / 10.0 | 1.7 | 14.0 | 10.22 | 4.68 | 0.14 |
| training including user transitions | 43 | 44.1 | 58.8 / 92.0 | 17.0 | 59.8 | 6.35 | 8.42 | 11.90 |
| checkpoint save/evaluation | 30 | 30.2 | 28.8 / 40.0 | 1.8 | 52.9 | 8.55 | 6.37 | 11.90 |

Logged coverage: **80/80 training users**, **167 updates**, **7,654,975 reviews**. Held-out cohort: **20 users**.
Complete training observation: **True**. Resource telemetry does not establish predictive quality.
Full-history profile available: **False**; observed training window: **44.1 s**.

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

Summed preparation across workers: 154.9 s; summed future waits: 0.8 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- swap/paging: PdhAddEnglishCounterW failed. Performance counters may be disabled..
- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.913 s; sampler-thread CPU: 0.297 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0061/expertium/E33-F48-F50-F58e/result.json](../../expertium/E33-F48-F50-F58e/result.json): 314d06c36a491e573f4188caca55cb992b0e676568b486816b5d91f8feb9cf52.
- [artifacts/retraining/gpu-retrain-r01/iteration-0061/protocol.json](../../protocol.json): dd006748527b0ad896fb403c07e110fc0cb84414acaa8af7b3026a09f6bd097d.
- [artifacts/retraining/gpu-retrain-r01/iteration-0061/expertium/E33-F48-F50-F58e/resources.jsonl](../../expertium/E33-F48-F50-F58e/resources.jsonl): c2a19b24d6f8ebbfd00f36a0ebe6b5a279c417f7643b18cb0b2c16e7e348ae65.
- [artifacts/retraining/gpu-retrain-r01/iteration-0061/expertium/E33-F48-F50-F58e/updates.jsonl](../../expertium/E33-F48-F50-F58e/updates.jsonl): 6ed8ba3212d6944d3170fef7aee1f58c9ad17eef6d76972d0c9f605852181035.
- [artifacts/retraining/gpu-retrain-r01/iteration-0061/expertium/E33-F48-F50-F58e/resources-summary.json](../../expertium/E33-F48-F50-F58e/resources-summary.json): fbe50df1b5721cca02a28c6989a93dc3efc9d02ec6e7b53eefda228cfea2c33e.
