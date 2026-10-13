# Resource profile: expertium/E33-F48-F50-F58

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 20 | 19.7 | 3.9 / 11.0 | 1.1 | 13.8 | 10.54 | 5.19 | 0.18 |
| training including user transitions | 37 | 37.5 | 62.5 / 93.0 | 17.1 | 52.1 | 7.44 | 8.33 | 11.91 |
| checkpoint save/evaluation | 28 | 28.1 | 22.1 / 33.0 | 1.0 | 48.3 | 9.65 | 6.37 | 11.91 |

Logged coverage: **80/80 training users**, **167 updates**, **7,654,975 reviews**. Held-out cohort: **20 users**.
Complete training observation: **True**. Resource telemetry does not establish predictive quality.
Full-history profile available: **False**; observed training window: **37.5 s**.

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

Summed preparation across workers: 116.3 s; summed future waits: 0.5 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- swap/paging: PdhAddEnglishCounterW failed. Performance counters may be disabled..
- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.342 s; sampler-thread CPU: 0.266 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0048/expertium/E33-F48-F50-F58/result.json](../../expertium/E33-F48-F50-F58/result.json): 44a92f39466901a7a1223594817b57252bdab2b53bfd60630c99c45bbca0b066.
- [artifacts/retraining/gpu-retrain-r01/iteration-0048/protocol.json](../../protocol.json): 518be81d63acb864d06436d3486756d0451bc4ad1c76e2c8b885f1bd98b5aab5.
- [artifacts/retraining/gpu-retrain-r01/iteration-0048/expertium/E33-F48-F50-F58/resources.jsonl](../../expertium/E33-F48-F50-F58/resources.jsonl): 1c0e60b034e8e9762f0b97a37ff7cd497da1455512473076ee4ef30b7ec97b20.
- [artifacts/retraining/gpu-retrain-r01/iteration-0048/expertium/E33-F48-F50-F58/updates.jsonl](../../expertium/E33-F48-F50-F58/updates.jsonl): 634a53a8e754bab7b1d4b92219d24b5180fdf533fff56013e16eb98089e51579.
- [artifacts/retraining/gpu-retrain-r01/iteration-0048/expertium/E33-F48-F50-F58/resources-summary.json](../../expertium/E33-F48-F50-F58/resources-summary.json): 42c459853d27029ea8edeef7e4062a1654380433c6a01466440ac9519c398dd6.
