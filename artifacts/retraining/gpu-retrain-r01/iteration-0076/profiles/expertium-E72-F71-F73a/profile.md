# Resource profile: expertium/E72-F71-F73a

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 13 | 12.5 | 0.3 / 1.0 | 1.0 | 17.2 | 11.99 | 4.63 | 0.18 |
| training including user transitions | 39 | 39.7 | 64.4 / 94.0 | 17.4 | 51.6 | 8.87 | 7.69 | 11.90 |
| checkpoint save/evaluation | 28 | 28.1 | 22.0 / 32.0 | 1.1 | 51.4 | 11.02 | 5.84 | 11.90 |

Logged coverage: **80/80 training users**, **167 updates**, **7,654,975 reviews**. Held-out cohort: **20 users**.
Complete training observation: **True**. Resource telemetry does not establish predictive quality.
Full-history profile available: **False**; observed training window: **39.7 s**.

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

Summed preparation across workers: 108.1 s; summed future waits: 0.5 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.467 s; sampler-thread CPU: 0.828 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0076/expertium/E72-F71-F73a/result.json](../../expertium/E72-F71-F73a/result.json): 0f103491b1f4c6f7b483dc079f48fbd2e482b1bf0000537304388cc52744e3d9.
- [artifacts/retraining/gpu-retrain-r01/iteration-0076/protocol.json](../../protocol.json): c59a70f62999245bebaf26ab7813bd3102862acf9aae0f9c4d113f789413397c.
- [artifacts/retraining/gpu-retrain-r01/iteration-0076/expertium/E72-F71-F73a/resources.jsonl](../../expertium/E72-F71-F73a/resources.jsonl): f97d6e3f55166f1f2e5cf4115bd9d3b3c54729ff1b6b14e78b365728384a995b.
- [artifacts/retraining/gpu-retrain-r01/iteration-0076/expertium/E72-F71-F73a/updates.jsonl](../../expertium/E72-F71-F73a/updates.jsonl): 8515dfa210ae92660f19732067f9fdbc22acd5a0c78c0bd5564b4a838d9f4ab4.
- [artifacts/retraining/gpu-retrain-r01/iteration-0076/expertium/E72-F71-F73a/resources-summary.json](../../expertium/E72-F71-F73a/resources-summary.json): 91abc883318f8f583733676c7745081c9251a3c60c2ed0c113cee0e381c52ac5.
