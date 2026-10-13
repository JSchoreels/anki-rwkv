# Resource profile: expertium/E80-F65e

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 15 | 14.6 | 1.3 / 5.0 | 1.0 | 11.2 | 12.21 | 4.14 | 0.14 |
| training including user transitions | 37 | 37.6 | 74.7 / 93.0 | 18.6 | 53.6 | 8.52 | 7.72 | 11.95 |
| checkpoint save/evaluation | 27 | 27.1 | 30.2 / 39.0 | 2.6 | 62.4 | 10.33 | 5.85 | 11.95 |

Logged coverage: **80/80 training users**, **167 updates**, **7,654,975 reviews**. Held-out cohort: **20 users**.
Complete training observation: **True**. Resource telemetry does not establish predictive quality.
Full-history profile available: **False**; observed training window: **37.6 s**.

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

Summed preparation across workers: 107.4 s; summed future waits: 0.7 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.470 s; sampler-thread CPU: 0.859 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0081/expertium/E80-F65e/result.json](../../expertium/E80-F65e/result.json): 0ee91fde4cdd8a89d7ce7cfc5ecf6095c06617ae9ac5f5d55024bcd6e36399f8.
- [artifacts/retraining/gpu-retrain-r01/iteration-0081/protocol.json](../../protocol.json): 8c94914332bc5d996729e19e7e1728eb653a2039871735e0c07d7b21a882b0af.
- [artifacts/retraining/gpu-retrain-r01/iteration-0081/expertium/E80-F65e/resources.jsonl](../../expertium/E80-F65e/resources.jsonl): 81256f81e95b94621e2d9b1dea70f6ab28f547e865822e1bf8af2700eb5ad577.
- [artifacts/retraining/gpu-retrain-r01/iteration-0081/expertium/E80-F65e/updates.jsonl](../../expertium/E80-F65e/updates.jsonl): b243c69b0aceb62414c6ad9962479f9e2cd91ec54d69c4fa50379153e444bf39.
- [artifacts/retraining/gpu-retrain-r01/iteration-0081/expertium/E80-F65e/resources-summary.json](../../expertium/E80-F65e/resources-summary.json): ee1e2091bc7d550c430f97e283c0b0c336dc2858dd0ee4fb6596e63a25ef3de1.
