# Resource profile: expertium/E83-F90b

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 15 | 14.6 | 7.4 / 10.0 | 1.9 | 10.5 | 11.89 | 4.71 | 0.18 |
| training including user transitions | 62 | 63.7 | 88.2 / 99.0 | 13.1 | 31.3 | 8.45 | 7.86 | 13.27 |
| checkpoint save/evaluation | 26 | 26.0 | 30.1 / 41.0 | 3.2 | 64.4 | 10.39 | 5.99 | 13.27 |

Logged coverage: **80/80 training users**, **167 updates**, **7,654,975 reviews**. Held-out cohort: **20 users**.
Complete training observation: **True**. Resource telemetry does not establish predictive quality.
Full-history profile available: **False**; observed training window: **63.7 s**.

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

Summed preparation across workers: 105.3 s; summed future waits: 0.8 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 4.457 s; sampler-thread CPU: 1.094 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0084/expertium/E83-F90b/result.json](../../expertium/E83-F90b/result.json): fb91075ca2755c11d4b2766242a93b4206c4f03b8feeb4439b3f6a0bbc4704be.
- [artifacts/retraining/gpu-retrain-r01/iteration-0084/protocol.json](../../protocol.json): e58aa3b38d37b9524da99ce9fd19c55baef5a4942d6a5c5d6843ca0c5a5a1980.
- [artifacts/retraining/gpu-retrain-r01/iteration-0084/expertium/E83-F90b/resources.jsonl](../../expertium/E83-F90b/resources.jsonl): 88a0434bf10a8dd1284f5c4011cc13e2320e9013fffce44e975e25fae4f347d8.
- [artifacts/retraining/gpu-retrain-r01/iteration-0084/expertium/E83-F90b/updates.jsonl](../../expertium/E83-F90b/updates.jsonl): 4eb2df2f8947bb9f1c5c9bfe89cb55b3d6967e740eeef5e22af6fc0999ba179b.
- [artifacts/retraining/gpu-retrain-r01/iteration-0084/expertium/E83-F90b/resources-summary.json](../../expertium/E83-F90b/resources-summary.json): 2dac235f5d71dcf1240dcfab336ec19a29796f4a74e492b79ffef3cd8d69c877.
