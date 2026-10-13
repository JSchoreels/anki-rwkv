# Resource profile: expertium/E33-F48-F50-F58-F34a

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 20 | 19.6 | 8.5 / 11.0 | 1.9 | 13.6 | 11.19 | 4.77 | 0.14 |
| training including user transitions | 44 | 45.1 | 64.1 / 93.0 | 16.7 | 58.7 | 7.47 | 8.36 | 11.90 |
| checkpoint save/evaluation | 30 | 30.2 | 28.3 / 40.0 | 1.9 | 53.2 | 9.66 | 6.38 | 11.90 |

Logged coverage: **80/80 training users**, **167 updates**, **7,654,975 reviews**. Held-out cohort: **20 users**.
Complete training observation: **True**. Resource telemetry does not establish predictive quality.
Full-history profile available: **False**; observed training window: **45.1 s**.

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

Summed preparation across workers: 154.8 s; summed future waits: 0.8 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- swap/paging: PdhAddEnglishCounterW failed. Performance counters may be disabled..
- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.959 s; sampler-thread CPU: 0.438 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0051/expertium/E33-F48-F50-F58-F34a/result.json](../../expertium/E33-F48-F50-F58-F34a/result.json): 8e5acc8365dabd3e5d9eb2bf371faaae82a239ee04b10084763f613aa0b7d015.
- [artifacts/retraining/gpu-retrain-r01/iteration-0051/protocol.json](../../protocol.json): 83042b548fd9bed5bb4031de9c48f8988b4ba690f4a75795882425cce238bbea.
- [artifacts/retraining/gpu-retrain-r01/iteration-0051/expertium/E33-F48-F50-F58-F34a/resources.jsonl](../../expertium/E33-F48-F50-F58-F34a/resources.jsonl): 34a48c6855b0364526544da37014c34a1b93d63a3a03472d9a5db5593b96dbca.
- [artifacts/retraining/gpu-retrain-r01/iteration-0051/expertium/E33-F48-F50-F58-F34a/updates.jsonl](../../expertium/E33-F48-F50-F58-F34a/updates.jsonl): b3d0569b31fabc98b096ab08fcc613b0867c7dd3c3a586348209813972eaa4d9.
- [artifacts/retraining/gpu-retrain-r01/iteration-0051/expertium/E33-F48-F50-F58-F34a/resources-summary.json](../../expertium/E33-F48-F50-F58-F34a/resources-summary.json): f97fbef4dd2875afa829023802daf8fe0654dfd8f489a38196ff67c40524a7f5.
