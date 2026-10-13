# Resource profile: expertium/E33-F48-F50-F58-F58a

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 20 | 19.7 | 7.3 / 8.0 | 1.0 | 13.9 | 10.68 | 4.61 | 0.14 |
| training including user transitions | 43 | 43.9 | 69.7 / 93.0 | 20.8 | 59.7 | 6.71 | 8.44 | 11.90 |
| checkpoint save/evaluation | 31 | 31.2 | 28.5 / 41.0 | 2.1 | 52.1 | 8.99 | 6.36 | 11.90 |

Logged coverage: **80/80 training users**, **167 updates**, **7,654,975 reviews**. Held-out cohort: **20 users**.
Complete training observation: **True**. Resource telemetry does not establish predictive quality.
Full-history profile available: **False**; observed training window: **43.9 s**.

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

Summed preparation across workers: 153.4 s; summed future waits: 0.8 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- swap/paging: PdhAddEnglishCounterW failed. Performance counters may be disabled..
- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.830 s; sampler-thread CPU: 0.406 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0057/expertium/E33-F48-F50-F58-F58a/result.json](../../expertium/E33-F48-F50-F58-F58a/result.json): d4323ce5db674d3295df394cd15d8cd76a1b8f21da9a6c9beec05d18cf9f2a01.
- [artifacts/retraining/gpu-retrain-r01/iteration-0057/protocol.json](../../protocol.json): 2594a878b15b1db699f3c9021bdf0b97ab9482d96f353fdc1b841ab1d333614a.
- [artifacts/retraining/gpu-retrain-r01/iteration-0057/expertium/E33-F48-F50-F58-F58a/resources.jsonl](../../expertium/E33-F48-F50-F58-F58a/resources.jsonl): 457da3a8912fdd73fb7bc852cfc982aab6cd81ea6b7ab1eeec96e2d016eeb9ed.
- [artifacts/retraining/gpu-retrain-r01/iteration-0057/expertium/E33-F48-F50-F58-F58a/updates.jsonl](../../expertium/E33-F48-F50-F58-F58a/updates.jsonl): 784ed7068b254650e11bd49bf9a6f3d7bd741f2acbe0d8896dd0090a34ffbfbc.
- [artifacts/retraining/gpu-retrain-r01/iteration-0057/expertium/E33-F48-F50-F58-F58a/resources-summary.json](../../expertium/E33-F48-F50-F58-F58a/resources-summary.json): 93e8121c9c3a7aceacd33a666f5123bfefd6b9f5d38eb228296c794e6a34ee76.
