# Resource profile: expertium/E33-F48-F50-F58-F58e

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 20 | 19.7 | 7.8 / 9.0 | 1.2 | 14.4 | 10.10 | 4.65 | 0.14 |
| training including user transitions | 44 | 44.7 | 66.6 / 93.0 | 17.7 | 60.9 | 6.30 | 8.36 | 11.90 |
| checkpoint save/evaluation | 30 | 30.1 | 29.5 / 40.0 | 1.9 | 53.6 | 8.48 | 6.36 | 11.90 |

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

Summed preparation across workers: 159.2 s; summed future waits: 0.9 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- swap/paging: PdhAddEnglishCounterW failed. Performance counters may be disabled..
- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.499 s; sampler-thread CPU: 0.312 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0063/expertium/E33-F48-F50-F58-F58e/result.json](../../expertium/E33-F48-F50-F58-F58e/result.json): 67d9a71d3eb6b7da6ee72a5bd3219dcb117b850022b0a9f9322d36ecb6432d97.
- [artifacts/retraining/gpu-retrain-r01/iteration-0063/protocol.json](../../protocol.json): 8e23254d5260abbcec459edc06d6ee8472ca9df314b50863766f28c2e457c82e.
- [artifacts/retraining/gpu-retrain-r01/iteration-0063/expertium/E33-F48-F50-F58-F58e/resources.jsonl](../../expertium/E33-F48-F50-F58-F58e/resources.jsonl): 2af49a66bd72a0f23c8cd3f8e47157e6904302f4eb8092faf6daa6da06210457.
- [artifacts/retraining/gpu-retrain-r01/iteration-0063/expertium/E33-F48-F50-F58-F58e/updates.jsonl](../../expertium/E33-F48-F50-F58-F58e/updates.jsonl): ca4118f83246a92e976ad3daeeb30c3ce8649f8e95974ef36828cd7eff525d15.
- [artifacts/retraining/gpu-retrain-r01/iteration-0063/expertium/E33-F48-F50-F58-F58e/resources-summary.json](../../expertium/E33-F48-F50-F58-F58e/resources-summary.json): 39bc393ebf170985f81cb9bc068a3e3de4343e2d6c813f17f85c072746a00f76.
