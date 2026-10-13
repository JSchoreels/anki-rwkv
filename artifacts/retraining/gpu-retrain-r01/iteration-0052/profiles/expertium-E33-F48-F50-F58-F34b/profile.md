# Resource profile: expertium/E33-F48-F50-F58-F34b

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 20 | 19.6 | 7.6 / 12.0 | 1.1 | 13.8 | 11.23 | 4.71 | 0.14 |
| training including user transitions | 44 | 44.8 | 62.5 / 93.0 | 16.6 | 58.8 | 7.42 | 8.39 | 11.90 |
| checkpoint save/evaluation | 30 | 30.1 | 29.5 / 40.0 | 1.9 | 53.4 | 9.62 | 6.40 | 11.90 |

Logged coverage: **80/80 training users**, **167 updates**, **7,654,975 reviews**. Held-out cohort: **20 users**.
Complete training observation: **True**. Resource telemetry does not establish predictive quality.
Full-history profile available: **False**; observed training window: **44.8 s**.

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

Summed preparation across workers: 154.8 s; summed future waits: 0.8 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- swap/paging: PdhAddEnglishCounterW failed. Performance counters may be disabled..
- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.669 s; sampler-thread CPU: 0.281 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0052/expertium/E33-F48-F50-F58-F34b/result.json](../../expertium/E33-F48-F50-F58-F34b/result.json): ee12dc2a54034e55d042454e12bc4736f36f21cb46c84c80334d876672a0c93b.
- [artifacts/retraining/gpu-retrain-r01/iteration-0052/protocol.json](../../protocol.json): 911c9e2791e2252af26b7f289bfbc85b55a4a0392f01b6aff181901004a53cfc.
- [artifacts/retraining/gpu-retrain-r01/iteration-0052/expertium/E33-F48-F50-F58-F34b/resources.jsonl](../../expertium/E33-F48-F50-F58-F34b/resources.jsonl): e054e6124f9ba916749195a4d686e46299e9fb24b428181fe03e25ac83c6e239.
- [artifacts/retraining/gpu-retrain-r01/iteration-0052/expertium/E33-F48-F50-F58-F34b/updates.jsonl](../../expertium/E33-F48-F50-F58-F34b/updates.jsonl): 539bc7231665746c36ba1ca8b3d58b13d24d47ae9c068bdb3eb4de435d29b164.
- [artifacts/retraining/gpu-retrain-r01/iteration-0052/expertium/E33-F48-F50-F58-F34b/resources-summary.json](../../expertium/E33-F48-F50-F58-F34b/resources-summary.json): 6d816dc3347912cd9f6402db61cdf184cc7c0a5790d16fd5e5836083e3b48aba.
