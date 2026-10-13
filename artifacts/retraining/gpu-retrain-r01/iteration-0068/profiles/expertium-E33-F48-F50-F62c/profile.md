# Resource profile: expertium/E33-F48-F50-F62c

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 20 | 19.6 | 9.6 / 10.0 | 2.9 | 14.3 | 9.85 | 4.72 | 0.14 |
| training including user transitions | 44 | 44.8 | 69.2 / 93.0 | 20.0 | 60.7 | 5.91 | 8.43 | 11.91 |
| checkpoint save/evaluation | 31 | 31.1 | 27.9 / 40.0 | 1.8 | 52.7 | 8.25 | 6.41 | 11.91 |

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

Summed preparation across workers: 159.4 s; summed future waits: 0.8 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- swap/paging: PdhAddEnglishCounterW failed. Performance counters may be disabled..
- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.566 s; sampler-thread CPU: 0.375 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0068/expertium/E33-F48-F50-F62c/result.json](../../expertium/E33-F48-F50-F62c/result.json): 3a9ffaa4001590b47793406b0fc16239cce348f4766691f574c8bb930093d327.
- [artifacts/retraining/gpu-retrain-r01/iteration-0068/protocol.json](../../protocol.json): abe7b1235d544287ef2f3d37c0581d7dddd9e6804717d891abc4c94681db1412.
- [artifacts/retraining/gpu-retrain-r01/iteration-0068/expertium/E33-F48-F50-F62c/resources.jsonl](../../expertium/E33-F48-F50-F62c/resources.jsonl): 04792052f91aa053872d9579536e64a26737c6b520d78addd1bde96aef70ca7b.
- [artifacts/retraining/gpu-retrain-r01/iteration-0068/expertium/E33-F48-F50-F62c/updates.jsonl](../../expertium/E33-F48-F50-F62c/updates.jsonl): d8ab5a78576724d934c7784b2a6d7d21191c8d007738c16a41337fc4d5233358.
- [artifacts/retraining/gpu-retrain-r01/iteration-0068/expertium/E33-F48-F50-F62c/resources-summary.json](../../expertium/E33-F48-F50-F62c/resources-summary.json): 98982a3de7fc49a52e91a3a9c7d26e8532419ebbf0d8a24aea3943a55fbc20a6.
