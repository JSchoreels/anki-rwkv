# Resource profile: expertium/E33-F48-F50-F58-F34c

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 20 | 19.6 | 7.8 / 10.0 | 1.4 | 13.8 | 11.21 | 4.66 | 0.14 |
| training including user transitions | 44 | 45.0 | 66.3 / 94.0 | 18.1 | 58.4 | 7.36 | 8.34 | 11.90 |
| checkpoint save/evaluation | 30 | 30.2 | 29.8 / 40.0 | 1.9 | 53.0 | 9.65 | 6.35 | 11.90 |

Logged coverage: **80/80 training users**, **167 updates**, **7,654,975 reviews**. Held-out cohort: **20 users**.
Complete training observation: **True**. Resource telemetry does not establish predictive quality.
Full-history profile available: **False**; observed training window: **45.0 s**.

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

Summed preparation across workers: 154.5 s; summed future waits: 0.8 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- swap/paging: PdhAddEnglishCounterW failed. Performance counters may be disabled..
- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.838 s; sampler-thread CPU: 0.422 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0053/expertium/E33-F48-F50-F58-F34c/result.json](../../expertium/E33-F48-F50-F58-F34c/result.json): 7b70036f926486a80885ca76d08f9d76c95a3ae40e7824be09d14de9d9776022.
- [artifacts/retraining/gpu-retrain-r01/iteration-0053/protocol.json](../../protocol.json): 2ae51dfde5555820882364998c73e1977b85c7a467676d0062ab01034aea2d55.
- [artifacts/retraining/gpu-retrain-r01/iteration-0053/expertium/E33-F48-F50-F58-F34c/resources.jsonl](../../expertium/E33-F48-F50-F58-F34c/resources.jsonl): 2ed62c2ddeb5860b57440726341236142ac930245cfdb787179639180cd6d9d8.
- [artifacts/retraining/gpu-retrain-r01/iteration-0053/expertium/E33-F48-F50-F58-F34c/updates.jsonl](../../expertium/E33-F48-F50-F58-F34c/updates.jsonl): fdeca3b8b35083d0cd799864764acadf6c6ee84a2d4b7a7f2e94eb80ffed54c2.
- [artifacts/retraining/gpu-retrain-r01/iteration-0053/expertium/E33-F48-F50-F58-F34c/resources-summary.json](../../expertium/E33-F48-F50-F58-F34c/resources-summary.json): 88fbd4aa38e53a18755692fd59861609e331efd0a924f7d6736d590174967ac8.
