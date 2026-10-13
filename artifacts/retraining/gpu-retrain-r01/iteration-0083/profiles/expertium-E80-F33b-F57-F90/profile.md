# Resource profile: expertium/E80-F33b-F57-F90

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 12 | 11.5 | 7.3 / 8.0 | 1.6 | 8.0 | 11.79 | 4.70 | 0.18 |
| training including user transitions | 39 | 39.8 | 73.0 / 93.0 | 19.6 | 51.3 | 8.54 | 7.78 | 12.20 |
| checkpoint save/evaluation | 26 | 26.1 | 32.2 / 39.0 | 2.0 | 64.2 | 10.62 | 5.85 | 12.20 |

Logged coverage: **80/80 training users**, **167 updates**, **7,654,975 reviews**. Held-out cohort: **20 users**.
Complete training observation: **True**. Resource telemetry does not establish predictive quality.
Full-history profile available: **False**; observed training window: **39.8 s**.

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

Summed preparation across workers: 107.5 s; summed future waits: 0.3 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.416 s; sampler-thread CPU: 0.875 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0083/expertium/E80-F33b-F57-F90/result.json](../../expertium/E80-F33b-F57-F90/result.json): a913f328bde85653e11efcce7de40d8ae16f1c02bdf9b3dab912656d7076f582.
- [artifacts/retraining/gpu-retrain-r01/iteration-0083/protocol.json](../../protocol.json): 1fe2b9b5ce42ff427c68e186e16c6411c4f52d6492615e89e887e566bccaec76.
- [artifacts/retraining/gpu-retrain-r01/iteration-0083/expertium/E80-F33b-F57-F90/resources.jsonl](../../expertium/E80-F33b-F57-F90/resources.jsonl): 20deeba8529a70baa06030cb7a46fe510c5357505fe45d2409fbb6909bd53eaa.
- [artifacts/retraining/gpu-retrain-r01/iteration-0083/expertium/E80-F33b-F57-F90/updates.jsonl](../../expertium/E80-F33b-F57-F90/updates.jsonl): 61464ba4ab8880a754a4e550b5a2f0fe4bd7721afc8518717cd1d46e4d2ff059.
- [artifacts/retraining/gpu-retrain-r01/iteration-0083/expertium/E80-F33b-F57-F90/resources-summary.json](../../expertium/E80-F33b-F57-F90/resources-summary.json): c6f2f8bbdc46f5a6cc39882b17f07b5205776dc83407d4fc2ac1257872453584.
