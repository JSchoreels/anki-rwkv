# Resource profile: expertium/E33-F48-F50-F58-F51a

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 20 | 19.6 | 7.6 / 10.0 | 1.5 | 14.1 | 10.69 | 4.77 | 0.14 |
| training including user transitions | 43 | 44.0 | 65.5 / 93.0 | 17.9 | 60.1 | 6.92 | 8.39 | 11.90 |
| checkpoint save/evaluation | 31 | 31.2 | 29.5 / 41.0 | 1.9 | 52.5 | 9.24 | 6.36 | 11.90 |

Logged coverage: **80/80 training users**, **167 updates**, **7,654,975 reviews**. Held-out cohort: **20 users**.
Complete training observation: **True**. Resource telemetry does not establish predictive quality.
Full-history profile available: **False**; observed training window: **44.0 s**.

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

Summed preparation across workers: 155.3 s; summed future waits: 0.9 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- swap/paging: PdhAddEnglishCounterW failed. Performance counters may be disabled..
- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.854 s; sampler-thread CPU: 0.406 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0055/expertium/E33-F48-F50-F58-F51a/result.json](../../expertium/E33-F48-F50-F58-F51a/result.json): f6e62be3c0c6209fa3333c97e773916ecddee405e4c786b3b86d70553ba60acc.
- [artifacts/retraining/gpu-retrain-r01/iteration-0055/protocol.json](../../protocol.json): f32f6b3bd85301418016d5af571cd0c8b5d66a813dc3a2873c5df608f86661ea.
- [artifacts/retraining/gpu-retrain-r01/iteration-0055/expertium/E33-F48-F50-F58-F51a/resources.jsonl](../../expertium/E33-F48-F50-F58-F51a/resources.jsonl): 256ccc5bd38caad700d223335de9a0df828dadd0d363d4a28fa253cdf90c5472.
- [artifacts/retraining/gpu-retrain-r01/iteration-0055/expertium/E33-F48-F50-F58-F51a/updates.jsonl](../../expertium/E33-F48-F50-F58-F51a/updates.jsonl): 4fd144b1582c541e573be8814488e43fa3d9891b32dfe01077b4dc2be7986136.
- [artifacts/retraining/gpu-retrain-r01/iteration-0055/expertium/E33-F48-F50-F58-F51a/resources-summary.json](../../expertium/E33-F48-F50-F58-F51a/resources-summary.json): 3c419b252314dbdffaca3a24a461d1f6d71ee0f3d236504fa7c11f7fc9b0b32b.
