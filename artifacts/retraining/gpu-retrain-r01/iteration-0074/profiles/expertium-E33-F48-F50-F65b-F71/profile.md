# Resource profile: expertium/E33-F48-F50-F65b-F71

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 15 | 14.6 | 1.8 / 6.0 | 1.0 | 13.0 | 11.02 | 4.99 | 0.14 |
| training including user transitions | 44 | 45.0 | 69.0 / 93.0 | 16.9 | 44.3 | 7.40 | 8.30 | 11.90 |
| checkpoint save/evaluation | 28 | 28.2 | 21.8 / 39.0 | 1.2 | 46.1 | 11.49 | 6.38 | 11.90 |

Logged coverage: **80/80 training users**, **167 updates**, **7,654,975 reviews**. Held-out cohort: **20 users**.
Complete training observation: **True**. Resource telemetry does not establish predictive quality.
Full-history profile available: **False**; observed training window: **45.0 s**.

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

Summed preparation across workers: 131.6 s; summed future waits: 0.4 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 4.055 s; sampler-thread CPU: 0.969 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0074/expertium/E33-F48-F50-F65b-F71/result.json](../../expertium/E33-F48-F50-F65b-F71/result.json): 8076d5dbe592a2ea0b615838fecb2c7996f497f04f92266acf1d298d100ffa89.
- [artifacts/retraining/gpu-retrain-r01/iteration-0074/protocol.json](../../protocol.json): f7fb7cd92cef6bce7bdbd4b0df7c9e6112a81cc9957dc35df0a6ceafd1bb4443.
- [artifacts/retraining/gpu-retrain-r01/iteration-0074/expertium/E33-F48-F50-F65b-F71/resources.jsonl](../../expertium/E33-F48-F50-F65b-F71/resources.jsonl): 0d3a62cda5f4b6ba2476edff613db8a7294ac34a44dab741e4e2b8a47f0d1a10.
- [artifacts/retraining/gpu-retrain-r01/iteration-0074/expertium/E33-F48-F50-F65b-F71/updates.jsonl](../../expertium/E33-F48-F50-F65b-F71/updates.jsonl): 61c333c901bd9a92c36816c80085ffff81e67765455664b850f9694e68a27030.
- [artifacts/retraining/gpu-retrain-r01/iteration-0074/expertium/E33-F48-F50-F65b-F71/resources-summary.json](../../expertium/E33-F48-F50-F65b-F71/resources-summary.json): 2c4a4afcb80060dac067942d893624eae67b69d54665a310a60c84ad5069d2ce.
