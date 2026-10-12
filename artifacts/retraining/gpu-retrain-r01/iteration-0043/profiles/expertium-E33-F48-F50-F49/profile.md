# Resource profile: expertium/E33-F48-F50-F49

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 20 | 19.7 | 2.2 / 4.0 | 1.0 | 14.0 | 10.52 | 5.18 | 0.39 |
| training including user transitions | 153 | 201.8 | 92.4 / 100.0 | 3.6 | 11.0 | 3.93 | 10.06 | 15.10 |
| checkpoint save/evaluation | 29 | 29.1 | 22.8 / 34.0 | 1.1 | 49.4 | 7.22 | 6.67 | 15.10 |

Logged coverage: **80/80 training users**, **167 updates**, **7,654,975 reviews**. Held-out cohort: **20 users**.
Complete training observation: **True**. Resource telemetry does not establish predictive quality.
Full-history profile available: **True**; observed training window: **201.8 s**.

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

Summed preparation across workers: 115.9 s; summed future waits: 0.6 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- swap/paging: PdhAddEnglishCounterW failed. Performance counters may be disabled..
- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 50.130 s; sampler-thread CPU: 0.812 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0043/expertium/E33-F48-F50-F49/result.json](../../expertium/E33-F48-F50-F49/result.json): 38ce2003523ac84942c5fb7df19befebf47dc8d67bb6d7ef70bb72c4bbb160a9.
- [artifacts/retraining/gpu-retrain-r01/iteration-0043/protocol.json](../../protocol.json): 5b71d42c4ec7a90d3e960d1f9c0a8c7f9c538a05007c936101439862833c827e.
- [artifacts/retraining/gpu-retrain-r01/iteration-0043/expertium/E33-F48-F50-F49/resources.jsonl](../../expertium/E33-F48-F50-F49/resources.jsonl): 8971a65b10d2409b4a0bf2a81f8bae6353e3777478d14a829a78c73ca1fcb0b6.
- [artifacts/retraining/gpu-retrain-r01/iteration-0043/expertium/E33-F48-F50-F49/updates.jsonl](../../expertium/E33-F48-F50-F49/updates.jsonl): 6ae3cd70bae34643abfddb932bd6dac0dc210c9ca421255f9f6145261557ee5f.
- [artifacts/retraining/gpu-retrain-r01/iteration-0043/expertium/E33-F48-F50-F49/resources-summary.json](../../expertium/E33-F48-F50-F49/resources-summary.json): 3a38d03bc0192fec195576845e7b9f73129a978e84e7eab3e34f83faa8fc20e6.
