# Resource profile: expertium/E33-F48

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 20 | 19.7 | 3.1 / 6.0 | 1.0 | 13.2 | 7.33 | 4.51 | 0.06 |
| training including user transitions | 56 | 57.6 | 64.9 / 99.0 | 12.8 | 50.0 | 3.48 | 8.37 | 11.96 |
| checkpoint save/evaluation | 34 | 34.4 | 21.3 / 32.0 | 1.2 | 50.2 | 5.96 | 6.49 | 11.96 |

Logged coverage: **80/80 training users**, **167 updates**, **7,654,975 reviews**. Held-out cohort: **20 users**.
Complete training observation: **True**. Resource telemetry does not establish predictive quality.
Full-history profile available: **False**; observed training window: **57.6 s**.

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

Summed preparation across workers: 173.7 s; summed future waits: 0.8 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- swap/paging: PdhAddEnglishCounterW failed. Performance counters may be disabled..
- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 4.696 s; sampler-thread CPU: 0.297 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0041/expertium/E33-F48/result.json](../../expertium/E33-F48/result.json): d4fd13552028f0df52ccb1261242c0f3de5651d861e8bb9a162eb301f257cac2.
- [artifacts/retraining/gpu-retrain-r01/iteration-0041/protocol.json](../../protocol.json): adefc828cd64d5adc8892f81c2debecb5e9215f67edf6fc3c5f2b64ef7cd3aeb.
- [artifacts/retraining/gpu-retrain-r01/iteration-0041/expertium/E33-F48/resources.jsonl](../../expertium/E33-F48/resources.jsonl): d8c17579d4167c9cfacdbfa541370c44ebcebdd053d53296d4d3951d3a55e09f.
- [artifacts/retraining/gpu-retrain-r01/iteration-0041/expertium/E33-F48/updates.jsonl](../../expertium/E33-F48/updates.jsonl): 929359152448ffbfba6092c05ef27a144e8bb7c44bb54d7180d215ae8db05f8b.
- [artifacts/retraining/gpu-retrain-r01/iteration-0041/expertium/E33-F48/resources-summary.json](../../expertium/E33-F48/resources-summary.json): e4b8b61cc233e89fcb35e47f89ef4cc71cb25a34deb403e46f2b4123eea651f5.
