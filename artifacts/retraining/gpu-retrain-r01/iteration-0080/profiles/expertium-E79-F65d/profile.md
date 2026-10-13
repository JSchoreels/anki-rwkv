# Resource profile: expertium/E79-F65d

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 16 | 15.6 | 0.9 / 3.0 | 1.1 | 10.3 | 11.47 | 4.71 | 0.18 |
| training including user transitions | 36 | 36.5 | 74.5 / 93.0 | 19.6 | 55.3 | 8.39 | 7.67 | 11.95 |
| checkpoint save/evaluation | 26 | 26.1 | 23.4 / 33.0 | 1.0 | 67.9 | 10.55 | 5.83 | 11.95 |

Logged coverage: **80/80 training users**, **167 updates**, **7,654,975 reviews**. Held-out cohort: **20 users**.
Complete training observation: **True**. Resource telemetry does not establish predictive quality.
Full-history profile available: **False**; observed training window: **36.5 s**.

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

Sampler wall: 3.400 s; sampler-thread CPU: 0.938 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0080/expertium/E79-F65d/result.json](../../expertium/E79-F65d/result.json): 7d58aaf4655fcf5ae29cc556996c306ecf4c9387f5d346690033d2ac1ef3ecc2.
- [artifacts/retraining/gpu-retrain-r01/iteration-0080/protocol.json](../../protocol.json): 02588fe7389a4ee7ac90cba0083d5a80db65962ec4f5658741afc8c38605763a.
- [artifacts/retraining/gpu-retrain-r01/iteration-0080/expertium/E79-F65d/resources.jsonl](../../expertium/E79-F65d/resources.jsonl): 7719296dc3ecf2ca0a8ae9d76519f28e711c0f5e626d7ae72d8de2b0a04ab71b.
- [artifacts/retraining/gpu-retrain-r01/iteration-0080/expertium/E79-F65d/updates.jsonl](../../expertium/E79-F65d/updates.jsonl): de1218e2a63e6ec8b67ad1118e63679e19cfcf90766a89813737232efb2ad75a.
- [artifacts/retraining/gpu-retrain-r01/iteration-0080/expertium/E79-F65d/resources-summary.json](../../expertium/E79-F65d/resources-summary.json): b1bee11d006de0fb479d200ea92c71be63e7ce903f64a09531763ec0f56ac0b4.
