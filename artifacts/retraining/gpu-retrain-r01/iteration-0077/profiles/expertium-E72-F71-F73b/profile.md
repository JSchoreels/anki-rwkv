# Resource profile: expertium/E72-F71-F73b

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 13 | 12.5 | 1.6 / 3.0 | 1.0 | 10.4 | 12.70 | 4.03 | 0.14 |
| training including user transitions | 38 | 38.7 | 76.6 / 94.0 | 20.5 | 64.6 | 9.74 | 7.05 | 11.90 |
| checkpoint save/evaluation | 28 | 28.1 | 23.8 / 32.0 | 1.0 | 60.4 | 11.75 | 5.34 | 11.90 |

Logged coverage: **80/80 training users**, **167 updates**, **7,654,975 reviews**. Held-out cohort: **20 users**.
Complete training observation: **True**. Resource telemetry does not establish predictive quality.
Full-history profile available: **False**; observed training window: **38.7 s**.

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

Summed preparation across workers: 96.8 s; summed future waits: 0.6 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.443 s; sampler-thread CPU: 0.922 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0077/expertium/E72-F71-F73b/result.json](../../expertium/E72-F71-F73b/result.json): 733eed683cf145b49b39b553b4e3219fad0bf02fd7c3dbe85271f0a210959b4c.
- [artifacts/retraining/gpu-retrain-r01/iteration-0077/protocol.json](../../protocol.json): 0080c8d67ed3013750dd85b626c7c6d9b05e2a93c948aa3d13a0d216c0e842f4.
- [artifacts/retraining/gpu-retrain-r01/iteration-0077/expertium/E72-F71-F73b/resources.jsonl](../../expertium/E72-F71-F73b/resources.jsonl): a39e4c3d4f5674699ab02ebf290b10713c292da53a44dd863f364705273e71fe.
- [artifacts/retraining/gpu-retrain-r01/iteration-0077/expertium/E72-F71-F73b/updates.jsonl](../../expertium/E72-F71-F73b/updates.jsonl): 8e20e9c6d166d73578e94c121f0da77deb8090184caa627a990daa9bdf06ea97.
- [artifacts/retraining/gpu-retrain-r01/iteration-0077/expertium/E72-F71-F73b/resources-summary.json](../../expertium/E72-F71-F73b/resources-summary.json): 6deb4f31f1095b6bfebec5aca1aa7a0307d5926b5cf0c9a4bbafb544e75c415a.
