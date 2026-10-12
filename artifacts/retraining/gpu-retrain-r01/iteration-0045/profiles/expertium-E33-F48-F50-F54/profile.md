# Resource profile: expertium/E33-F48-F50-F54

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 20 | 19.6 | 2.8 / 6.0 | 1.0 | 12.7 | 8.90 | 5.18 | 0.18 |
| training including user transitions | 37 | 37.5 | 64.1 / 92.0 | 15.3 | 53.7 | 5.66 | 8.28 | 11.91 |
| checkpoint save/evaluation | 28 | 28.0 | 21.2 / 31.0 | 1.0 | 49.1 | 7.80 | 6.37 | 11.91 |

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

Summed preparation across workers: 120.1 s; summed future waits: 0.5 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- swap/paging: PdhAddEnglishCounterW failed. Performance counters may be disabled..
- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.294 s; sampler-thread CPU: 0.266 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0045/expertium/E33-F48-F50-F54/result.json](../../expertium/E33-F48-F50-F54/result.json): 9ca0d7271facee177e909771d5a6b7ea3d7517704af008a88ba18b0e50cb2ab3.
- [artifacts/retraining/gpu-retrain-r01/iteration-0045/protocol.json](../../protocol.json): 372422106e0ab3aaec8ed1eb041678870142cf548c06ace0ea2c26fd70a43b8e.
- [artifacts/retraining/gpu-retrain-r01/iteration-0045/expertium/E33-F48-F50-F54/resources.jsonl](../../expertium/E33-F48-F50-F54/resources.jsonl): 4f76226ad817ec63bc5f47fdf8b1cc065fc96e72fb32914d689ff88900afddab.
- [artifacts/retraining/gpu-retrain-r01/iteration-0045/expertium/E33-F48-F50-F54/updates.jsonl](../../expertium/E33-F48-F50-F54/updates.jsonl): ca80c60dddca5d3412fede5048195017b87578e19d2c98d296f9572de78265f1.
- [artifacts/retraining/gpu-retrain-r01/iteration-0045/expertium/E33-F48-F50-F54/resources-summary.json](../../expertium/E33-F48-F50-F54/resources-summary.json): 96b92e27bce38d270c1ec128909e4b2eaf1015513e5e29225dd837df98c5f8bb.
