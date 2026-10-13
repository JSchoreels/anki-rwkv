# Resource profile: expertium/E83-F77a

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 7 | 6.3 | 7.7 / 9.0 | 1.5 | 50.3 | 11.66 | 4.62 | 0.15 |
| training including user transitions | 36 | 36.5 | 70.3 / 92.0 | 17.8 | 54.9 | 8.37 | 7.76 | 12.20 |
| checkpoint save/evaluation | 27 | 27.2 | 31.5 / 38.0 | 2.0 | 64.5 | 10.43 | 5.83 | 12.20 |

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

Summed preparation across workers: 107.4 s; summed future waits: 1.1 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.111 s; sampler-thread CPU: 0.781 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0087/expertium/E83-F77a/result.json](../../expertium/E83-F77a/result.json): 9cfbb26119df00c91f57c57a04e060a5c14ed729640afbdf43580e19f031fcec.
- [artifacts/retraining/gpu-retrain-r01/iteration-0087/protocol.json](../../protocol.json): 35c85be7e36619dbb3dfa8903602b5e043a4e85fad7529c866808788fa03feb8.
- [artifacts/retraining/gpu-retrain-r01/iteration-0087/expertium/E83-F77a/resources.jsonl](../../expertium/E83-F77a/resources.jsonl): caa59c1e94a5da798f3506e8b042159071aa42d304400f79ac33569456fd35da.
- [artifacts/retraining/gpu-retrain-r01/iteration-0087/expertium/E83-F77a/updates.jsonl](../../expertium/E83-F77a/updates.jsonl): 1f7fbcd631135601a41bdffbfa473fbb31cbd78e450e74c59175cf5b7a7c5c44.
- [artifacts/retraining/gpu-retrain-r01/iteration-0087/expertium/E83-F77a/resources-summary.json](../../expertium/E83-F77a/resources-summary.json): 262e774e4a3128cb61771faaf443f10ef2ba853b04b9b4d4a8c54fb8e86555e3.
