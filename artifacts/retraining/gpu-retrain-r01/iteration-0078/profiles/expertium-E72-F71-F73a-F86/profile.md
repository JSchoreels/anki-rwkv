# Resource profile: expertium/E72-F71-F73a-F86

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 13 | 12.5 | 1.2 / 3.0 | 1.0 | 11.0 | 12.19 | 4.53 | 0.14 |
| training including user transitions | 37 | 37.8 | 70.7 / 94.0 | 18.7 | 53.5 | 8.92 | 7.70 | 11.90 |
| checkpoint save/evaluation | 27 | 27.1 | 25.1 / 32.0 | 1.0 | 51.4 | 11.03 | 5.84 | 11.90 |

Logged coverage: **80/80 training users**, **167 updates**, **7,654,975 reviews**. Held-out cohort: **20 users**.
Complete training observation: **True**. Resource telemetry does not establish predictive quality.
Full-history profile available: **False**; observed training window: **37.8 s**.

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

Summed preparation across workers: 107.3 s; summed future waits: 0.5 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.496 s; sampler-thread CPU: 0.906 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0078/expertium/E72-F71-F73a-F86/result.json](../../expertium/E72-F71-F73a-F86/result.json): eb0b4c6106031d2e5c5a2b79f3e93118952be7aca37e65c73e776ad74ddc37cf.
- [artifacts/retraining/gpu-retrain-r01/iteration-0078/protocol.json](../../protocol.json): cbee85ef8f0c9c76ac6944f3fb27cc4fc86b20b0b3bbb55645215d93b896539d.
- [artifacts/retraining/gpu-retrain-r01/iteration-0078/expertium/E72-F71-F73a-F86/resources.jsonl](../../expertium/E72-F71-F73a-F86/resources.jsonl): 856996a0f0cdd2b4833c31ea606493feb33f65c2e5eacc18750ef2c0ea36e777.
- [artifacts/retraining/gpu-retrain-r01/iteration-0078/expertium/E72-F71-F73a-F86/updates.jsonl](../../expertium/E72-F71-F73a-F86/updates.jsonl): 4ef43ace2cf922c0b949a55af19e426c186c0e79c73c5c7cf0dec4baedfc5490.
- [artifacts/retraining/gpu-retrain-r01/iteration-0078/expertium/E72-F71-F73a-F86/resources-summary.json](../../expertium/E72-F71-F73a-F86/resources-summary.json): c4ca060db64a843f6208d45b5e0469da7dffb2b9752c2901feaff23677796b05.
