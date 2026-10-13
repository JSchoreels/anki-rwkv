# Resource profile: expertium/E33-F48-F50-F58-F52a

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 20 | 19.6 | 7.9 / 10.0 | 1.6 | 14.1 | 10.79 | 4.71 | 0.14 |
| training including user transitions | 43 | 43.9 | 58.3 / 93.0 | 15.4 | 59.4 | 6.95 | 8.41 | 11.90 |
| checkpoint save/evaluation | 31 | 31.2 | 27.8 / 41.0 | 1.8 | 52.4 | 9.20 | 6.40 | 11.90 |

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

Summed preparation across workers: 153.5 s; summed future waits: 0.8 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- swap/paging: PdhAddEnglishCounterW failed. Performance counters may be disabled..
- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.783 s; sampler-thread CPU: 0.391 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0056/expertium/E33-F48-F50-F58-F52a/result.json](../../expertium/E33-F48-F50-F58-F52a/result.json): 7d654d87f4588573dbd4998e5e75c9bdf0dda119fa236153eae933330e56b487.
- [artifacts/retraining/gpu-retrain-r01/iteration-0056/protocol.json](../../protocol.json): ba288e283f10264e83f35aa6c95c887ccf7eca37681245cd13e3849bcbcc7017.
- [artifacts/retraining/gpu-retrain-r01/iteration-0056/expertium/E33-F48-F50-F58-F52a/resources.jsonl](../../expertium/E33-F48-F50-F58-F52a/resources.jsonl): 621ee7983c2ce241b44cf92d97948e1d7fffcfa5923c12d86d1678c2e28cc85a.
- [artifacts/retraining/gpu-retrain-r01/iteration-0056/expertium/E33-F48-F50-F58-F52a/updates.jsonl](../../expertium/E33-F48-F50-F58-F52a/updates.jsonl): 8c495f5f4fad33fabfcc1f72f60a54372c59e60270c2c50a886b05afd80a0870.
- [artifacts/retraining/gpu-retrain-r01/iteration-0056/expertium/E33-F48-F50-F58-F52a/resources-summary.json](../../expertium/E33-F48-F50-F58-F52a/resources-summary.json): fda32732fbc39da794ae189ebc566a539fed780464819b732d2a8103cb089184.
