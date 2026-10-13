# Resource profile: expertium/E33-F48-F50-F58g

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 20 | 19.7 | 7.6 / 10.0 | 1.1 | 13.9 | 10.19 | 4.64 | 0.14 |
| training including user transitions | 44 | 44.7 | 68.3 / 93.0 | 20.6 | 61.3 | 6.40 | 8.29 | 11.90 |
| checkpoint save/evaluation | 30 | 30.0 | 28.4 / 40.0 | 1.8 | 53.6 | 8.41 | 6.34 | 11.90 |

Logged coverage: **80/80 training users**, **167 updates**, **7,654,975 reviews**. Held-out cohort: **20 users**.
Complete training observation: **True**. Resource telemetry does not establish predictive quality.
Full-history profile available: **False**; observed training window: **44.7 s**.

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

Summed preparation across workers: 161.2 s; summed future waits: 0.8 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- swap/paging: PdhAddEnglishCounterW failed. Performance counters may be disabled..
- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.469 s; sampler-thread CPU: 0.297 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0064/expertium/E33-F48-F50-F58g/result.json](../../expertium/E33-F48-F50-F58g/result.json): 3e35b5fc7cd88c66b1d718dfa487d2da02d2cd96f83023e73782c6828f0ce183.
- [artifacts/retraining/gpu-retrain-r01/iteration-0064/protocol.json](../../protocol.json): 2c5875fecb12915e5b2a047eaa79d01fa71df8697dd7c82dd750967768b82ad7.
- [artifacts/retraining/gpu-retrain-r01/iteration-0064/expertium/E33-F48-F50-F58g/resources.jsonl](../../expertium/E33-F48-F50-F58g/resources.jsonl): ce2987e23260f3e63a65404085001f4d7f71d6b551ae4cf28522bc77e254545a.
- [artifacts/retraining/gpu-retrain-r01/iteration-0064/expertium/E33-F48-F50-F58g/updates.jsonl](../../expertium/E33-F48-F50-F58g/updates.jsonl): aef16a1607b5c19c2fbeb7d9563d772456a64496a292f4d14e4d47459fd36ee9.
- [artifacts/retraining/gpu-retrain-r01/iteration-0064/expertium/E33-F48-F50-F58g/resources-summary.json](../../expertium/E33-F48-F50-F58g/resources-summary.json): 16016ad47fce612b51a75df5114cd2d44e3334e97eefeba0ccea74fc091173d8.
