# Resource profile: expertium/E33-F48-F50-F34g

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 20 | 19.7 | 7.7 / 10.0 | 1.1 | 15.2 | 9.86 | 4.52 | 0.14 |
| training including user transitions | 44 | 44.7 | 68.0 / 93.0 | 19.5 | 61.5 | 5.86 | 8.33 | 11.90 |
| checkpoint save/evaluation | 31 | 31.1 | 30.1 / 42.0 | 2.3 | 53.4 | 8.00 | 6.38 | 11.90 |

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

Summed preparation across workers: 161.8 s; summed future waits: 0.8 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- swap/paging: PdhAddEnglishCounterW failed. Performance counters may be disabled..
- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.560 s; sampler-thread CPU: 0.250 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0070/expertium/E33-F48-F50-F34g/result.json](../../expertium/E33-F48-F50-F34g/result.json): 277de1d70b4e70ab51790a38e061fec79876b32203acd8cc46e688e24325f429.
- [artifacts/retraining/gpu-retrain-r01/iteration-0070/protocol.json](../../protocol.json): f5002b6280d6079d9c77f09a21478a267cbcc8b390a9dd6ab80259c55d2737dd.
- [artifacts/retraining/gpu-retrain-r01/iteration-0070/expertium/E33-F48-F50-F34g/resources.jsonl](../../expertium/E33-F48-F50-F34g/resources.jsonl): 75b6c190acfbd23cce84337f7f0b885f0efa03dd11015a92399d7b2f7efaa187.
- [artifacts/retraining/gpu-retrain-r01/iteration-0070/expertium/E33-F48-F50-F34g/updates.jsonl](../../expertium/E33-F48-F50-F34g/updates.jsonl): 8d0f334d612c56c4b00da854caf6ec19c6eabd1d440f48a8df657c5707aead55.
- [artifacts/retraining/gpu-retrain-r01/iteration-0070/expertium/E33-F48-F50-F34g/resources-summary.json](../../expertium/E33-F48-F50-F34g/resources-summary.json): 06101aa572cb9b2e6d67e5e2251aa68910e1a8f015a4a04fc635b788160587a8.
