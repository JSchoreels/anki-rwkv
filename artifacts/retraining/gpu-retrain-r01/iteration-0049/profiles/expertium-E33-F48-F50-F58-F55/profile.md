# Resource profile: expertium/E33-F48-F50-F58-F55

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 20 | 19.6 | 8.4 / 10.0 | 1.8 | 14.1 | 11.91 | 4.89 | 0.14 |
| training including user transitions | 43 | 44.2 | 62.2 / 93.0 | 18.3 | 59.9 | 8.19 | 8.45 | 11.91 |
| checkpoint save/evaluation | 31 | 31.2 | 28.9 / 41.0 | 2.2 | 52.3 | 10.39 | 6.36 | 11.91 |

Logged coverage: **80/80 training users**, **167 updates**, **7,654,975 reviews**. Held-out cohort: **20 users**.
Complete training observation: **True**. Resource telemetry does not establish predictive quality.
Full-history profile available: **False**; observed training window: **44.2 s**.

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

Summed preparation across workers: 155.6 s; summed future waits: 0.8 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- swap/paging: PdhAddEnglishCounterW failed. Performance counters may be disabled..
- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.959 s; sampler-thread CPU: 0.281 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0049/expertium/E33-F48-F50-F58-F55/result.json](../../expertium/E33-F48-F50-F58-F55/result.json): a965f857dc673ad38b868aeac48b226551c0dd19024fa43b8b35510c8ce4aa4b.
- [artifacts/retraining/gpu-retrain-r01/iteration-0049/protocol.json](../../protocol.json): 25efb6639ef4281355c53d74890a3392e765073fac1874f80cd4e113b0894dce.
- [artifacts/retraining/gpu-retrain-r01/iteration-0049/expertium/E33-F48-F50-F58-F55/resources.jsonl](../../expertium/E33-F48-F50-F58-F55/resources.jsonl): 6cb98d8bc03a6f121671681a366cc76d0254ebca794acb2ce611673ff4fad486.
- [artifacts/retraining/gpu-retrain-r01/iteration-0049/expertium/E33-F48-F50-F58-F55/updates.jsonl](../../expertium/E33-F48-F50-F58-F55/updates.jsonl): 783767f20eb1959dd224509915b0ca2eca8c6b218995a62e1c55c479945e4233.
- [artifacts/retraining/gpu-retrain-r01/iteration-0049/expertium/E33-F48-F50-F58-F55/resources-summary.json](../../expertium/E33-F48-F50-F58-F55/resources-summary.json): b870a6450f4611b971abafe5b229cfb1abb9dc653e1f98ee0d5931b9a1d43733.
