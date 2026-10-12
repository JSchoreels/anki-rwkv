# Resource profile: expertium/E33-F48-F50-F53

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 20 | 19.6 | 3.2 / 12.0 | 1.1 | 13.1 | 8.65 | 5.18 | 0.18 |
| training including user transitions | 39 | 39.8 | 58.7 / 92.0 | 17.8 | 57.5 | 4.92 | 8.82 | 11.93 |
| checkpoint save/evaluation | 27 | 27.0 | 23.0 / 33.0 | 1.0 | 50.0 | 7.23 | 6.79 | 11.93 |

Logged coverage: **80/80 training users**, **167 updates**, **7,654,975 reviews**. Held-out cohort: **20 users**.
Complete training observation: **True**. Resource telemetry does not establish predictive quality.
Full-history profile available: **False**; observed training window: **39.8 s**.

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

Summed preparation across workers: 135.1 s; summed future waits: 0.7 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- swap/paging: PdhAddEnglishCounterW failed. Performance counters may be disabled..
- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.488 s; sampler-thread CPU: 0.328 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0044/expertium/E33-F48-F50-F53/result.json](../../expertium/E33-F48-F50-F53/result.json): e75136d1d45de9b2dd04fddb198ecdee4f2e48dfc8bae0bbf8d258dc78265d88.
- [artifacts/retraining/gpu-retrain-r01/iteration-0044/protocol.json](../../protocol.json): 69479030192442d9610f81070db6734e9b097931e154b9a968558c712e7ff794.
- [artifacts/retraining/gpu-retrain-r01/iteration-0044/expertium/E33-F48-F50-F53/resources.jsonl](../../expertium/E33-F48-F50-F53/resources.jsonl): 108c57ebff0a01b0685d3bd2d96b1fbc7532a67986c78584a03bb077fb5ccd3e.
- [artifacts/retraining/gpu-retrain-r01/iteration-0044/expertium/E33-F48-F50-F53/updates.jsonl](../../expertium/E33-F48-F50-F53/updates.jsonl): 646171a63723e76a61e2cdc982dcf1a0808da49b099f1e094e593540a92fa006.
- [artifacts/retraining/gpu-retrain-r01/iteration-0044/expertium/E33-F48-F50-F53/resources-summary.json](../../expertium/E33-F48-F50-F53/resources-summary.json): 2db95b90eada9691d5fc10ad3de12de027cd1f3e3f2fc0903e3f2c222f36c455.
