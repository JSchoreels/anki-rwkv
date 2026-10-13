# Resource profile: expertium/E33-F48-F50-F58-F62

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 20 | 19.6 | 7.1 / 8.0 | 1.0 | 14.0 | 11.55 | 5.00 | 0.14 |
| training including user transitions | 43 | 44.0 | 66.4 / 92.0 | 17.4 | 59.9 | 7.96 | 8.37 | 11.91 |
| checkpoint save/evaluation | 30 | 30.2 | 28.2 / 40.0 | 1.9 | 53.0 | 10.19 | 6.38 | 11.91 |

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

Summed preparation across workers: 154.0 s; summed future waits: 0.8 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- swap/paging: PdhAddEnglishCounterW failed. Performance counters may be disabled..
- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.848 s; sampler-thread CPU: 0.375 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0050/expertium/E33-F48-F50-F58-F62/result.json](../../expertium/E33-F48-F50-F58-F62/result.json): 0d35a5bcdc9d76087c13988db4c0152ba428742f13b3c0489e0b8a03470710fe.
- [artifacts/retraining/gpu-retrain-r01/iteration-0050/protocol.json](../../protocol.json): be8e911eef6cef6e64fce014efa4f3fac164190e35d0fe7602a44a0f454f543b.
- [artifacts/retraining/gpu-retrain-r01/iteration-0050/expertium/E33-F48-F50-F58-F62/resources.jsonl](../../expertium/E33-F48-F50-F58-F62/resources.jsonl): 290e066085dbc7c421114278bddb1e5f57b04acc64e7bb766e0f5f25ca3e0a01.
- [artifacts/retraining/gpu-retrain-r01/iteration-0050/expertium/E33-F48-F50-F58-F62/updates.jsonl](../../expertium/E33-F48-F50-F58-F62/updates.jsonl): 9d0efea26bbc1f4fa22a2c24d3790b9c05a7386c5693f0dd3be86b8447da2b26.
- [artifacts/retraining/gpu-retrain-r01/iteration-0050/expertium/E33-F48-F50-F58-F62/resources-summary.json](../../expertium/E33-F48-F50-F58-F62/resources-summary.json): bfe4d3bbee036ba7a85521b7026bcb99e1371d25fddb75e1f53386e045c5cba4.
