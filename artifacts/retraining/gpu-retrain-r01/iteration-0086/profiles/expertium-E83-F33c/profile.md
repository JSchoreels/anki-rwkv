# Resource profile: expertium/E83-F33c

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 11 | 10.4 | 8.2 / 11.0 | 1.8 | 9.0 | 11.60 | 4.67 | 0.18 |
| training including user transitions | 39 | 39.7 | 71.2 / 93.0 | 19.0 | 50.3 | 8.25 | 7.85 | 12.20 |
| checkpoint save/evaluation | 27 | 27.1 | 31.0 / 39.0 | 2.1 | 63.2 | 10.54 | 5.83 | 12.20 |

Logged coverage: **80/80 training users**, **167 updates**, **7,654,975 reviews**. Held-out cohort: **20 users**.
Complete training observation: **True**. Resource telemetry does not establish predictive quality.
Full-history profile available: **False**; observed training window: **39.7 s**.

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

Summed preparation across workers: 105.7 s; summed future waits: 0.3 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.455 s; sampler-thread CPU: 0.797 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0086/expertium/E83-F33c/result.json](../../expertium/E83-F33c/result.json): e219ac6b43b846708d5501526a8dc30b8da939cba5373ca4f6e1ec0d90299f73.
- [artifacts/retraining/gpu-retrain-r01/iteration-0086/protocol.json](../../protocol.json): c4fb310a25b1d5437d16ea8785cb62be2076f4352ac6e45394f5d098dd514032.
- [artifacts/retraining/gpu-retrain-r01/iteration-0086/expertium/E83-F33c/resources.jsonl](../../expertium/E83-F33c/resources.jsonl): 4290316ce7b9e3af252bc940455e1372a27e728a7336c23f3533656663f73a18.
- [artifacts/retraining/gpu-retrain-r01/iteration-0086/expertium/E83-F33c/updates.jsonl](../../expertium/E83-F33c/updates.jsonl): 1b16ed9ec6fd064ba446a3a092303b79100b8a95aef5f2713c3a629629398a5b.
- [artifacts/retraining/gpu-retrain-r01/iteration-0086/expertium/E83-F33c/resources-summary.json](../../expertium/E83-F33c/resources-summary.json): 288210cef130c9b856b41ead17c5188856697e0a960e8bdd8cd172d2dfe057a6.
