# Resource profile: expertium/E33-F46

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 20 | 19.8 | 6.5 / 15.0 | 1.3 | 14.5 | 8.85 | 4.27 | 0.06 |
| training including user transitions | 48 | 49.4 | 58.1 / 92.0 | 16.3 | 56.5 | 2.80 | 8.28 | 11.96 |
| checkpoint save/evaluation | 32 | 32.2 | 22.7 / 47.0 | 1.3 | 50.8 | 5.44 | 6.33 | 11.96 |

Logged coverage: **80/80 training users**, **167 updates**, **7,654,975 reviews**. Held-out cohort: **20 users**.
Complete training observation: **True**. Resource telemetry does not establish predictive quality.
Full-history profile available: **False**; observed training window: **49.4 s**.

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

Summed preparation across workers: 164.1 s; summed future waits: 0.9 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- swap/paging: PdhAddEnglishCounterW failed. Performance counters may be disabled..
- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 4.344 s; sampler-thread CPU: 0.328 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0039/expertium/E33-F46/result.json](../../expertium/E33-F46/result.json): 030c057a3d8676dd2f4a69f4daa1eaa15e285a5f35b6eb059b28cd3bd19405c3.
- [artifacts/retraining/gpu-retrain-r01/iteration-0039/protocol.json](../../protocol.json): 34ce5cc0f3c4d6368feaa7014ee0dfcaad4bac19004158bd923b2c896ab08f8f.
- [artifacts/retraining/gpu-retrain-r01/iteration-0039/expertium/E33-F46/resources.jsonl](../../expertium/E33-F46/resources.jsonl): ad5f3e92bb800234d7e5d98f813e95ffbed542da3b808136fd1686f21e77a9df.
- [artifacts/retraining/gpu-retrain-r01/iteration-0039/expertium/E33-F46/updates.jsonl](../../expertium/E33-F46/updates.jsonl): 30c79fa3c9f227ff5fde3d569333e857d88deb09a0dddee60348a10a5d0ebf0c.
- [artifacts/retraining/gpu-retrain-r01/iteration-0039/expertium/E33-F46/resources-summary.json](../../expertium/E33-F46/resources-summary.json): a391a706d5cf24139209736403b0b159636cf87cafa2dd1e8d98fcd17d512952.
