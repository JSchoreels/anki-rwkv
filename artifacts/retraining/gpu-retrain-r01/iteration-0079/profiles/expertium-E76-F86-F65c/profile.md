# Resource profile: expertium/E76-F86-F65c

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 15 | 14.6 | 1.1 / 2.0 | 1.0 | 5.8 | 11.43 | 4.69 | 0.18 |
| training including user transitions | 40 | 40.7 | 66.4 / 94.0 | 17.1 | 50.5 | 8.20 | 7.83 | 11.90 |
| checkpoint save/evaluation | 27 | 27.1 | 24.0 / 31.0 | 1.0 | 66.1 | 10.60 | 5.83 | 11.90 |

Logged coverage: **80/80 training users**, **167 updates**, **7,654,975 reviews**. Held-out cohort: **20 users**.
Complete training observation: **True**. Resource telemetry does not establish predictive quality.
Full-history profile available: **False**; observed training window: **40.7 s**.

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

Summed preparation across workers: 108.8 s; summed future waits: 0.3 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.563 s; sampler-thread CPU: 0.906 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0079/expertium/E76-F86-F65c/result.json](../../expertium/E76-F86-F65c/result.json): 0853722909311d3d90fbb454bfc0c65545bf447ed246f516e8e5331d8d1172d4.
- [artifacts/retraining/gpu-retrain-r01/iteration-0079/protocol.json](../../protocol.json): f4d5e434deb92d24e49069444236d5bb2ca74aca6d816d8b7a1090b38866e784.
- [artifacts/retraining/gpu-retrain-r01/iteration-0079/expertium/E76-F86-F65c/resources.jsonl](../../expertium/E76-F86-F65c/resources.jsonl): 8155fcd9bdd65e119cc7741f67690f96e8f728272eec2280fc0ed01cb6381b6c.
- [artifacts/retraining/gpu-retrain-r01/iteration-0079/expertium/E76-F86-F65c/updates.jsonl](../../expertium/E76-F86-F65c/updates.jsonl): fdb3dad45aaaa6840d95e458ecd18de80a75e4c78dcc680e5c5cbbaed1b23be1.
- [artifacts/retraining/gpu-retrain-r01/iteration-0079/expertium/E76-F86-F65c/resources-summary.json](../../expertium/E76-F86-F65c/resources-summary.json): 353925972116453b4076ce4995de2070a36da6e42c4ada6d849f32b964062ba6.
