# Resource profile: expertium/E33-F48-F50-F58d

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 20 | 19.6 | 7.8 / 10.0 | 1.1 | 13.9 | 10.00 | 4.92 | 0.14 |
| training including user transitions | 43 | 44.0 | 68.0 / 92.0 | 19.6 | 59.8 | 6.42 | 8.34 | 11.90 |
| checkpoint save/evaluation | 31 | 31.2 | 30.0 / 41.0 | 2.3 | 52.2 | 8.64 | 6.34 | 11.90 |

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

Summed preparation across workers: 154.4 s; summed future waits: 0.8 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- swap/paging: PdhAddEnglishCounterW failed. Performance counters may be disabled..
- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.820 s; sampler-thread CPU: 0.359 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0060/expertium/E33-F48-F50-F58d/result.json](../../expertium/E33-F48-F50-F58d/result.json): ccdb90643b2dd645a5eb87344bc67e895881cdb38081567aaeda41514140706e.
- [artifacts/retraining/gpu-retrain-r01/iteration-0060/protocol.json](../../protocol.json): b706913df053e09b8b85e1d7f60a3d1d28a5f78a40d8f041efe0e4738b033b90.
- [artifacts/retraining/gpu-retrain-r01/iteration-0060/expertium/E33-F48-F50-F58d/resources.jsonl](../../expertium/E33-F48-F50-F58d/resources.jsonl): 4b9c249c4763e66ba99e59c6ce1a02a138806e65900e070a80b9820b6c4e8f0c.
- [artifacts/retraining/gpu-retrain-r01/iteration-0060/expertium/E33-F48-F50-F58d/updates.jsonl](../../expertium/E33-F48-F50-F58d/updates.jsonl): cf6147174bb76dfdc7cdf4d7cc61fe07afb7e5093a2951aeb09a158da7113327.
- [artifacts/retraining/gpu-retrain-r01/iteration-0060/expertium/E33-F48-F50-F58d/resources-summary.json](../../expertium/E33-F48-F50-F58d/resources-summary.json): f3a69dd797bd2b014c1ac22b53f199421ce5460bc85e92442a2a1c9fb539f50a.
