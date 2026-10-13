# Resource profile: expertium/E83-F77a-F77b

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 12 | 11.5 | 1.5 / 3.0 | 1.0 | 18.6 | 12.17 | 4.52 | 0.15 |
| training including user transitions | 39 | 39.6 | 68.8 / 94.0 | 15.8 | 51.3 | 8.84 | 7.74 | 12.20 |
| checkpoint save/evaluation | 27 | 27.1 | 21.5 / 31.0 | 1.1 | 52.2 | 11.09 | 5.86 | 12.20 |

Logged coverage: **80/80 training users**, **167 updates**, **7,654,975 reviews**. Held-out cohort: **20 users**.
Complete training observation: **True**. Resource telemetry does not establish predictive quality.
Full-history profile available: **False**; observed training window: **39.6 s**.

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

Summed preparation across workers: 107.0 s; summed future waits: 0.6 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.373 s; sampler-thread CPU: 0.812 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0088/expertium/E83-F77a-F77b/result.json](../../expertium/E83-F77a-F77b/result.json): e4120a34837a466744c9c2ce968c9d14b58795f918fc075b28954354a8da0e6a.
- [artifacts/retraining/gpu-retrain-r01/iteration-0088/protocol.json](../../protocol.json): ad031e5caff36676d4f9947dcfb164243ee669fe8edd76d4133e0081f6d91ba2.
- [artifacts/retraining/gpu-retrain-r01/iteration-0088/expertium/E83-F77a-F77b/resources.jsonl](../../expertium/E83-F77a-F77b/resources.jsonl): eeeff915ea8a232e0d56b1936e8095464e7f5ca928700b4eb113640149d33162.
- [artifacts/retraining/gpu-retrain-r01/iteration-0088/expertium/E83-F77a-F77b/updates.jsonl](../../expertium/E83-F77a-F77b/updates.jsonl): 4146cf817caf5be750efe987c56e9b23c747f88062bc9318ae26cffc9cfaa89c.
- [artifacts/retraining/gpu-retrain-r01/iteration-0088/expertium/E83-F77a-F77b/resources-summary.json](../../expertium/E83-F77a-F77b/resources-summary.json): 666496da973e883f354da227f7425ffd721db0321b22256a3391b9a791612b9d.
