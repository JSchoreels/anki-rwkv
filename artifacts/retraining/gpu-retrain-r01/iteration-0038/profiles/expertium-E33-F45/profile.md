# Resource profile: expertium/E33-F45

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 20 | 19.7 | 2.1 / 11.0 | 1.1 | 15.5 | 7.66 | 4.52 | 0.06 |
| training including user transitions | 51 | 52.0 | 61.5 / 92.0 | 14.6 | 52.7 | 5.64 | 8.26 | 11.96 |
| checkpoint save/evaluation | 34 | 34.3 | 24.7 / 40.0 | 1.0 | 51.1 | 9.63 | 6.36 | 11.96 |

Logged coverage: **80/80 training users**, **167 updates**, **7,654,975 reviews**. Held-out cohort: **20 users**.
Complete training observation: **True**. Resource telemetry does not establish predictive quality.
Full-history profile available: **False**; observed training window: **52.0 s**.

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

Summed preparation across workers: 164.0 s; summed future waits: 0.8 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- swap/paging: PdhAddEnglishCounterW failed. Performance counters may be disabled..
- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 4.125 s; sampler-thread CPU: 0.516 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0038/expertium/E33-F45/result.json](../../expertium/E33-F45/result.json): 3d9eececf365a9249d2f4038b50b7b3b4721ccb1289e2d425453aaff7de7e2fe.
- [artifacts/retraining/gpu-retrain-r01/iteration-0038/protocol.json](../../protocol.json): 83ecbe25383dcb048c205cc340285b80cc13080a249ccc320e3c75b6f711b484.
- [artifacts/retraining/gpu-retrain-r01/iteration-0038/expertium/E33-F45/resources.jsonl](../../expertium/E33-F45/resources.jsonl): 00e9a6cd4364306d53dde0a087d9ff276ee3f95d3b769334c219b5237ff5849c.
- [artifacts/retraining/gpu-retrain-r01/iteration-0038/expertium/E33-F45/updates.jsonl](../../expertium/E33-F45/updates.jsonl): 3a29422dcb7a65f4dbf9e300d896cb0229af62e47042f5727210bfc5847f2c74.
- [artifacts/retraining/gpu-retrain-r01/iteration-0038/expertium/E33-F45/resources-summary.json](../../expertium/E33-F45/resources-summary.json): ba516d5d4fbe662c5bfbe04323c765171627f3ca6e82eacbeb95aaa2558a5086.
