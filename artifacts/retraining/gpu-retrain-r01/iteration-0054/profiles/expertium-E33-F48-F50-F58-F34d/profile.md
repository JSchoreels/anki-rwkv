# Resource profile: expertium/E33-F48-F50-F58-F34d

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 20 | 19.6 | 9.3 / 10.0 | 2.9 | 14.6 | 10.77 | 4.62 | 0.14 |
| training including user transitions | 43 | 43.9 | 62.9 / 92.0 | 16.7 | 59.8 | 7.09 | 8.35 | 11.90 |
| checkpoint save/evaluation | 31 | 31.3 | 32.1 / 42.0 | 2.8 | 52.5 | 9.28 | 6.35 | 11.90 |

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

Summed preparation across workers: 153.8 s; summed future waits: 0.8 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- swap/paging: PdhAddEnglishCounterW failed. Performance counters may be disabled..
- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.933 s; sampler-thread CPU: 0.359 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0054/expertium/E33-F48-F50-F58-F34d/result.json](../../expertium/E33-F48-F50-F58-F34d/result.json): 9db5aa5a23b44a850d24530c90a42277d99aa150775f8ba0cd1d36636d30db0a.
- [artifacts/retraining/gpu-retrain-r01/iteration-0054/protocol.json](../../protocol.json): 1aa27f4de0233134c2a3653e7d39ac952d7b1d5ece03104a8ec8b969d8c560c0.
- [artifacts/retraining/gpu-retrain-r01/iteration-0054/expertium/E33-F48-F50-F58-F34d/resources.jsonl](../../expertium/E33-F48-F50-F58-F34d/resources.jsonl): b9692a36c13eccdcf6cac68229302c8f51c37f0e918c4eb82afed1268216920b.
- [artifacts/retraining/gpu-retrain-r01/iteration-0054/expertium/E33-F48-F50-F58-F34d/updates.jsonl](../../expertium/E33-F48-F50-F58-F34d/updates.jsonl): d268949b80700d554524ad1ae07403c53105da2b3995c25edd2af43a92070fe9.
- [artifacts/retraining/gpu-retrain-r01/iteration-0054/expertium/E33-F48-F50-F58-F34d/resources-summary.json](../../expertium/E33-F48-F50-F58-F34d/resources-summary.json): 5fac214671fa76197e580bc0c8483a03bdf2a894d2968d17b03fa536e6f05ea6.
