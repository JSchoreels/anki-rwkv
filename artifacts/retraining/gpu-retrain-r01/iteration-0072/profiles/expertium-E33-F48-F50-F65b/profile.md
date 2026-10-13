# Resource profile: expertium/E33-F48-F50-F65b

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 20 | 19.7 | 7.2 / 8.0 | 1.9 | 11.8 | 12.84 | 5.22 | 0.18 |
| training including user transitions | 43 | 43.8 | 63.6 / 93.0 | 17.4 | 62.8 | 9.48 | 8.40 | 11.90 |
| checkpoint save/evaluation | 31 | 31.0 | 30.0 / 38.0 | 2.1 | 52.7 | 11.74 | 6.41 | 11.90 |

Logged coverage: **80/80 training users**, **167 updates**, **7,654,975 reviews**. Held-out cohort: **20 users**.
Complete training observation: **True**. Resource telemetry does not establish predictive quality.
Full-history profile available: **False**; observed training window: **43.8 s**.

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

Summed preparation across workers: 161.1 s; summed future waits: 0.8 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- swap/paging: PdhAddEnglishCounterW failed. Performance counters may be disabled..
- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.632 s; sampler-thread CPU: 0.266 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0072/expertium/E33-F48-F50-F65b/result.json](../../expertium/E33-F48-F50-F65b/result.json): 44480bc4659aef91c89db451b8efb8dffed308f4f94202b4cbc50bc4efb3f77e.
- [artifacts/retraining/gpu-retrain-r01/iteration-0072/protocol.json](../../protocol.json): c79dd26fc0e401569f933a476b06c888c22383b85272614dba6920e554bda16f.
- [artifacts/retraining/gpu-retrain-r01/iteration-0072/expertium/E33-F48-F50-F65b/resources.jsonl](../../expertium/E33-F48-F50-F65b/resources.jsonl): af8522df16c24f7db0142a8259a703eb0c932b3649d3e0fbc38eb1ebfae10fe8.
- [artifacts/retraining/gpu-retrain-r01/iteration-0072/expertium/E33-F48-F50-F65b/updates.jsonl](../../expertium/E33-F48-F50-F65b/updates.jsonl): 1491ce4f3a02582bfa82d2bcb36ae73d9caf7e326ba78cf71aa2ca67e55c7ec9.
- [artifacts/retraining/gpu-retrain-r01/iteration-0072/expertium/E33-F48-F50-F65b/resources-summary.json](../../expertium/E33-F48-F50-F65b/resources-summary.json): dc04e3eb2b4bb2bf23eade82dddc51f0a07d313079e68e22d4e7c7e576aecd4e.
