# Resource profile: expertium/E30-F37-F42

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 19 | 18.7 | 2.2 / 5.0 | 1.0 | 14.4 | 7.73 | 4.23 | 0.04 |
| training including user transitions | 40 | 40.7 | 59.7 / 91.0 | 17.2 | 47.2 | 3.70 | 8.31 | 11.96 |
| checkpoint save/evaluation | 29 | 29.1 | 19.1 / 35.0 | 1.2 | 48.3 | 5.54 | 6.36 | 11.96 |

Logged coverage: **80/80 training users**, **167 updates**, **7,654,975 reviews**. Held-out cohort: **20 users**.
Complete training observation: **True**. Resource telemetry does not establish predictive quality.
Full-history profile available: **False**; observed training window: **40.7 s**.

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

Summed preparation across workers: 114.0 s; summed future waits: 0.6 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- swap/paging: PdhAddEnglishCounterW failed. Performance counters may be disabled..
- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.516 s; sampler-thread CPU: 0.250 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0040/expertium/E30-F37-F42/result.json](../../expertium/E30-F37-F42/result.json): c81926c814d881b24f599cda18c8610e4ee7e9c7752d6b36156509ba63676510.
- [artifacts/retraining/gpu-retrain-r01/iteration-0040/protocol.json](../../protocol.json): a30f3e65143be9f4617da6f3399aecce228fd17eb4c422220c7052f2d22e5549.
- [artifacts/retraining/gpu-retrain-r01/iteration-0040/expertium/E30-F37-F42/resources.jsonl](../../expertium/E30-F37-F42/resources.jsonl): 6ca4b5839d63c698e419124245950ad18165b172ee97c7e9008de499bfb798fd.
- [artifacts/retraining/gpu-retrain-r01/iteration-0040/expertium/E30-F37-F42/updates.jsonl](../../expertium/E30-F37-F42/updates.jsonl): 705072c79225a9e160f869146aeb99520b78ba8fcf60fc008c8e696c9a014f9f.
- [artifacts/retraining/gpu-retrain-r01/iteration-0040/expertium/E30-F37-F42/resources-summary.json](../../expertium/E30-F37-F42/resources-summary.json): ee0f5c9edf91b3df3e65f248907798acb19de0d11921de6a60c0a4c729a17570.
