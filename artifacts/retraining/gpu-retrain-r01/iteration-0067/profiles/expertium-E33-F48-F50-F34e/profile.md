# Resource profile: expertium/E33-F48-F50-F34e

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 20 | 19.7 | 9.6 / 11.0 | 2.7 | 14.6 | 9.87 | 4.71 | 0.14 |
| training including user transitions | 44 | 44.7 | 62.7 / 93.0 | 17.7 | 61.1 | 5.82 | 8.67 | 11.90 |
| checkpoint save/evaluation | 30 | 30.1 | 26.4 / 40.0 | 1.8 | 54.6 | 8.31 | 6.35 | 11.90 |

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

Summed preparation across workers: 159.9 s; summed future waits: 0.8 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- swap/paging: PdhAddEnglishCounterW failed. Performance counters may be disabled..
- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.487 s; sampler-thread CPU: 0.328 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0067/expertium/E33-F48-F50-F34e/result.json](../../expertium/E33-F48-F50-F34e/result.json): 59b5a3beb7556a39ec11e01df01a5ecffe4faecaacb77331642a3e2c54e0b783.
- [artifacts/retraining/gpu-retrain-r01/iteration-0067/protocol.json](../../protocol.json): 36c2c72fc3d2fe13f2a980d112b6f7c41a3f786d80dd291405d272c1a6c8ac28.
- [artifacts/retraining/gpu-retrain-r01/iteration-0067/expertium/E33-F48-F50-F34e/resources.jsonl](../../expertium/E33-F48-F50-F34e/resources.jsonl): 692d624b590c0b2b010f51eabbfb60b98fe20fef20dba0542de36831884a7733.
- [artifacts/retraining/gpu-retrain-r01/iteration-0067/expertium/E33-F48-F50-F34e/updates.jsonl](../../expertium/E33-F48-F50-F34e/updates.jsonl): 9d48a6065c902afc956162fa3fd20b7ef661c5ce9f20ab14da48b6b7d0354c5c.
- [artifacts/retraining/gpu-retrain-r01/iteration-0067/expertium/E33-F48-F50-F34e/resources-summary.json](../../expertium/E33-F48-F50-F34e/resources-summary.json): 63e31c3cebb46ce8c0fa8a29e50e779b39903bd6331d732952cbb46ff0904b84.
