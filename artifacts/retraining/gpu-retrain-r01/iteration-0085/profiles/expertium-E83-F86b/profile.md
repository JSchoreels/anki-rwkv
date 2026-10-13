# Resource profile: expertium/E83-F86b

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 13 | 12.5 | 7.7 / 11.0 | 2.0 | 6.5 | 11.56 | 4.70 | 0.18 |
| training including user transitions | 37 | 37.7 | 71.1 / 93.0 | 18.7 | 52.7 | 8.34 | 7.77 | 12.20 |
| checkpoint save/evaluation | 28 | 28.1 | 28.7 / 39.0 | 1.9 | 61.4 | 10.55 | 5.82 | 12.20 |

Logged coverage: **80/80 training users**, **167 updates**, **7,654,975 reviews**. Held-out cohort: **20 users**.
Complete training observation: **True**. Resource telemetry does not establish predictive quality.
Full-history profile available: **False**; observed training window: **37.7 s**.

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

Summed preparation across workers: 105.2 s; summed future waits: 0.6 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.464 s; sampler-thread CPU: 0.906 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0085/expertium/E83-F86b/result.json](../../expertium/E83-F86b/result.json): f1018a4bb877ee12227dc09c8b2c104a5f921d3824405f333b0da5b5e76df7c5.
- [artifacts/retraining/gpu-retrain-r01/iteration-0085/protocol.json](../../protocol.json): c5207f65499a992082b81fd3499d21ed5c91230ec22d78f998c39e31af413cc9.
- [artifacts/retraining/gpu-retrain-r01/iteration-0085/expertium/E83-F86b/resources.jsonl](../../expertium/E83-F86b/resources.jsonl): c71ec55fd8f9e6a000d1079898f77f28207df5bf68614c8a1000d6a88b8e5b78.
- [artifacts/retraining/gpu-retrain-r01/iteration-0085/expertium/E83-F86b/updates.jsonl](../../expertium/E83-F86b/updates.jsonl): 8c3e95539a3b45663c5f0c4f6075d61e4c4917435c66f76468fb0ce88ee2119b.
- [artifacts/retraining/gpu-retrain-r01/iteration-0085/expertium/E83-F86b/resources-summary.json](../../expertium/E83-F86b/resources-summary.json): 76dc4b35c47c40f237915b7b10384230851be794c9867a3cd021dfe0a845bc74.
