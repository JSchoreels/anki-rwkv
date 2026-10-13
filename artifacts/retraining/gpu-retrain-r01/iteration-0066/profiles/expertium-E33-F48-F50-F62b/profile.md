# Resource profile: expertium/E33-F48-F50-F62b

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 20 | 19.6 | 8.7 / 10.0 | 2.2 | 14.2 | 9.89 | 4.69 | 0.14 |
| training including user transitions | 44 | 44.8 | 66.2 / 92.0 | 19.0 | 61.2 | 6.16 | 8.30 | 11.90 |
| checkpoint save/evaluation | 30 | 30.0 | 31.9 / 42.0 | 2.1 | 55.3 | 8.32 | 6.39 | 11.90 |

Logged coverage: **80/80 training users**, **167 updates**, **7,654,975 reviews**. Held-out cohort: **20 users**.
Complete training observation: **True**. Resource telemetry does not establish predictive quality.
Full-history profile available: **False**; observed training window: **44.8 s**.

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

Summed preparation across workers: 160.3 s; summed future waits: 0.9 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- swap/paging: PdhAddEnglishCounterW failed. Performance counters may be disabled..
- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.445 s; sampler-thread CPU: 0.359 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0066/expertium/E33-F48-F50-F62b/result.json](../../expertium/E33-F48-F50-F62b/result.json): cfd0d8054a0c856b00b4b1041f035ffb551a1ec4f5e5db651893ec3d04526aa2.
- [artifacts/retraining/gpu-retrain-r01/iteration-0066/protocol.json](../../protocol.json): 74430fb52fa83f461866cc995e989c4b8054e5325a6b51a583ae0f5254e3a898.
- [artifacts/retraining/gpu-retrain-r01/iteration-0066/expertium/E33-F48-F50-F62b/resources.jsonl](../../expertium/E33-F48-F50-F62b/resources.jsonl): 5a4b92a72e3cc6b1836add03494c6927a7c1b37aba42ab25d670c874dfe45b22.
- [artifacts/retraining/gpu-retrain-r01/iteration-0066/expertium/E33-F48-F50-F62b/updates.jsonl](../../expertium/E33-F48-F50-F62b/updates.jsonl): b5c48243a090d29ac1054ecb67fd935613b13ffca0290c4440745486e64a5f8f.
- [artifacts/retraining/gpu-retrain-r01/iteration-0066/expertium/E33-F48-F50-F62b/resources-summary.json](../../expertium/E33-F48-F50-F62b/resources-summary.json): 26c5e4baaf22a57735c9e81cf0baf3efc41689e889dd04c3f4affa41bee4f720.
