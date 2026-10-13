# Resource profile: expertium/E83-F72a

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 7 | 6.3 | 1.0 / 2.0 | 1.0 | 66.4 | 11.91 | 4.62 | 0.15 |
| training including user transitions | 36 | 36.5 | 77.9 / 94.0 | 19.3 | 54.7 | 8.56 | 7.77 | 12.20 |
| checkpoint save/evaluation | 27 | 27.1 | 21.2 / 33.0 | 1.0 | 51.7 | 10.75 | 5.82 | 12.20 |

Logged coverage: **80/80 training users**, **167 updates**, **7,654,975 reviews**. Held-out cohort: **20 users**.
Complete training observation: **True**. Resource telemetry does not establish predictive quality.
Full-history profile available: **False**; observed training window: **36.5 s**.

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

Summed preparation across workers: 107.4 s; summed future waits: 0.9 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.047 s; sampler-thread CPU: 0.750 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0089/expertium/E83-F72a/result.json](../../expertium/E83-F72a/result.json): 0c0415d473af1aadfb1b114890ed464ff040d6dc77f9026c93b879790872b34f.
- [artifacts/retraining/gpu-retrain-r01/iteration-0089/protocol.json](../../protocol.json): a559cb637eef3be8da2de26193cb83eb27dd39ceda3662196e96ce228ee810b7.
- [artifacts/retraining/gpu-retrain-r01/iteration-0089/expertium/E83-F72a/resources.jsonl](../../expertium/E83-F72a/resources.jsonl): c13949e30b37037f91c0da96c5c8c3d62c6d3790700934a859b4174dbf14156e.
- [artifacts/retraining/gpu-retrain-r01/iteration-0089/expertium/E83-F72a/updates.jsonl](../../expertium/E83-F72a/updates.jsonl): dd3bec8e109eeee8ce83ea4224498bb3d352a6c9afbfbde777f9f01b0f37f8dd.
- [artifacts/retraining/gpu-retrain-r01/iteration-0089/expertium/E83-F72a/resources-summary.json](../../expertium/E83-F72a/resources-summary.json): 3ae48b85318066891df524aadbdf8a5ee33d28f8a7f3624cc712a4cb8b7ecc32.
