# Resource profile: expertium/E33-F48-F50

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 20 | 19.7 | 2.1 / 3.0 | 1.0 | 13.7 | 10.24 | 5.18 | 0.18 |
| training including user transitions | 37 | 37.5 | 66.5 / 92.0 | 19.4 | 53.2 | 7.15 | 8.35 | 11.91 |
| checkpoint save/evaluation | 29 | 29.1 | 23.4 / 43.0 | 1.2 | 48.9 | 9.18 | 6.41 | 11.91 |

Logged coverage: **80/80 training users**, **167 updates**, **7,654,975 reviews**. Held-out cohort: **20 users**.
Complete training observation: **True**. Resource telemetry does not establish predictive quality.
Full-history profile available: **False**; observed training window: **37.5 s**.

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

Summed preparation across workers: 118.3 s; summed future waits: 0.5 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- swap/paging: PdhAddEnglishCounterW failed. Performance counters may be disabled..
- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.368 s; sampler-thread CPU: 0.312 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0042/expertium/E33-F48-F50/result.json](../../expertium/E33-F48-F50/result.json): 1b9c548c19811f1e5648639810cb34db1eed09ff8d871eaab0444941b62141ad.
- [artifacts/retraining/gpu-retrain-r01/iteration-0042/protocol.json](../../protocol.json): 39f6da9f14338dc6074ac00cf93b919ebc34ce9dacda37933fc7c0dc39a9b521.
- [artifacts/retraining/gpu-retrain-r01/iteration-0042/expertium/E33-F48-F50/resources.jsonl](../../expertium/E33-F48-F50/resources.jsonl): b0c50082f4addefb14d82a24fcea8a4ee5ca30c9db25b0d2c04bd28087d0e08b.
- [artifacts/retraining/gpu-retrain-r01/iteration-0042/expertium/E33-F48-F50/updates.jsonl](../../expertium/E33-F48-F50/updates.jsonl): 06aa282008d51caecc7deb6708a769e2569dd2afe3576507ad698de0593d300e.
- [artifacts/retraining/gpu-retrain-r01/iteration-0042/expertium/E33-F48-F50/resources-summary.json](../../expertium/E33-F48-F50/resources-summary.json): 8a3c59f68529afea0c87d8cafe9caf5011400df681f6d18dcc9e4babf9cdc57c.
