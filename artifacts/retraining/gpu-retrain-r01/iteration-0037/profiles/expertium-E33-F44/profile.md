# Resource profile: expertium/E33-F44

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 20 | 19.7 | 1.5 / 3.0 | 1.0 | 14.0 | 6.06 | 4.53 | 0.14 |
| training including user transitions | 49 | 50.1 | 65.6 / 93.0 | 16.7 | 44.4 | 2.89 | 8.28 | 11.96 |
| checkpoint save/evaluation | 33 | 33.3 | 25.9 / 44.0 | 1.5 | 49.8 | 5.56 | 6.36 | 11.96 |

Logged coverage: **80/80 training users**, **167 updates**, **7,654,975 reviews**. Held-out cohort: **20 users**.
Complete training observation: **True**. Resource telemetry does not establish predictive quality.
Full-history profile available: **False**; observed training window: **50.1 s**.

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

Summed preparation across workers: 130.8 s; summed future waits: 0.6 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- swap/paging: PdhAddEnglishCounterW failed. Performance counters may be disabled..
- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 4.077 s; sampler-thread CPU: 0.547 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0037/expertium/E33-F44/result.json](../../expertium/E33-F44/result.json): efea9c629fdff8c53a5f3fcb8cf93b638dda8d57963f3747958f6c47863c197b.
- [artifacts/retraining/gpu-retrain-r01/iteration-0037/protocol.json](../../protocol.json): 12a29a61bc667836f5c272e2c3adfa27d6b536c8d774276ce521effc3beaf887.
- [artifacts/retraining/gpu-retrain-r01/iteration-0037/expertium/E33-F44/resources.jsonl](../../expertium/E33-F44/resources.jsonl): 0fea05fc1de32895afdc95b5fdd1a1fcbf0acecbeff1704184fa801536e257f6.
- [artifacts/retraining/gpu-retrain-r01/iteration-0037/expertium/E33-F44/updates.jsonl](../../expertium/E33-F44/updates.jsonl): db809ec2795bdf9713d50e8c4f744cc2e7bf7f30b5d35fddd361e9377762ef7a.
- [artifacts/retraining/gpu-retrain-r01/iteration-0037/expertium/E33-F44/resources-summary.json](../../expertium/E33-F44/resources-summary.json): ec98072d6ee329113ef4dc2776ec5d64b8ff9259ae521d6110cffb49a55e31b8.
