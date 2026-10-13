# Resource profile: expertium/E33-F48-F50-F58-F58b

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 20 | 19.6 | 7.5 / 8.0 | 1.0 | 13.8 | 10.36 | 4.90 | 0.14 |
| training including user transitions | 43 | 43.9 | 64.2 / 94.0 | 16.9 | 59.8 | 6.72 | 8.36 | 11.90 |
| checkpoint save/evaluation | 31 | 31.2 | 28.7 / 40.0 | 1.9 | 52.6 | 8.89 | 6.40 | 11.90 |

Logged coverage: **80/80 training users**, **167 updates**, **7,654,975 reviews**. Held-out cohort: **20 users**.
Complete training observation: **True**. Resource telemetry does not establish predictive quality.
Full-history profile available: **False**; observed training window: **43.9 s**.

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

Summed preparation across workers: 153.4 s; summed future waits: 0.8 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- swap/paging: PdhAddEnglishCounterW failed. Performance counters may be disabled..
- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.779 s; sampler-thread CPU: 0.406 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0058/expertium/E33-F48-F50-F58-F58b/result.json](../../expertium/E33-F48-F50-F58-F58b/result.json): 56e890d0c0ca35f9309415eb33f8cf9dcfd158496b9e35091d63aa5de2b8c1c2.
- [artifacts/retraining/gpu-retrain-r01/iteration-0058/protocol.json](../../protocol.json): 5ef19acd527ee3c558172f93362a8b1e253ddc7e96565d3db75a7776a5d6015e.
- [artifacts/retraining/gpu-retrain-r01/iteration-0058/expertium/E33-F48-F50-F58-F58b/resources.jsonl](../../expertium/E33-F48-F50-F58-F58b/resources.jsonl): 61d5d192ad130a049b3e706577c331c38a009512e62ccb1cc836f0323dd260bd.
- [artifacts/retraining/gpu-retrain-r01/iteration-0058/expertium/E33-F48-F50-F58-F58b/updates.jsonl](../../expertium/E33-F48-F50-F58-F58b/updates.jsonl): 73d36dc1c144e34d95c8add7c01c97db1a3c77ac8067c3ceb063ff7e4315f376.
- [artifacts/retraining/gpu-retrain-r01/iteration-0058/expertium/E33-F48-F50-F58-F58b/resources-summary.json](../../expertium/E33-F48-F50-F58-F58b/resources-summary.json): e1a31ae58a3c1d93b498684646959fd9fb22a90ec3d1c3a8cd1c99011066b1b4.
