# Resource profile: expertium/E33-F48-F50-F58f

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 20 | 19.7 | 8.2 / 10.0 | 1.4 | 14.5 | 10.18 | 4.51 | 0.08 |
| training including user transitions | 44 | 44.7 | 69.9 / 93.0 | 20.7 | 61.0 | 6.32 | 8.36 | 11.90 |
| checkpoint save/evaluation | 31 | 31.1 | 28.9 / 41.0 | 2.4 | 53.0 | 8.53 | 6.36 | 11.90 |

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

Summed preparation across workers: 159.9 s; summed future waits: 0.9 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- swap/paging: PdhAddEnglishCounterW failed. Performance counters may be disabled..
- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.536 s; sampler-thread CPU: 0.344 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0062/expertium/E33-F48-F50-F58f/result.json](../../expertium/E33-F48-F50-F58f/result.json): 7b0b4b2f28a538cd7917e0cf40c04f3f6bf8abd1f55a73d924543a54bd25d015.
- [artifacts/retraining/gpu-retrain-r01/iteration-0062/protocol.json](../../protocol.json): 88c74260d115e7ffe245863c881d9ebf1b8039a2a2087f237d427fcd53522348.
- [artifacts/retraining/gpu-retrain-r01/iteration-0062/expertium/E33-F48-F50-F58f/resources.jsonl](../../expertium/E33-F48-F50-F58f/resources.jsonl): 09d6ce28ff3cafc0d468f9e22d0c5a8ee4516ee0c5f266711f9dc0fe6aaa47cc.
- [artifacts/retraining/gpu-retrain-r01/iteration-0062/expertium/E33-F48-F50-F58f/updates.jsonl](../../expertium/E33-F48-F50-F58f/updates.jsonl): a24b6564b11b67c41a789c6ce376d61bda273c83419d8a561c598c3a792dfead.
- [artifacts/retraining/gpu-retrain-r01/iteration-0062/expertium/E33-F48-F50-F58f/resources-summary.json](../../expertium/E33-F48-F50-F58f/resources-summary.json): 22165690a79a4ca1d176a5e989fa0b7417bd89681a5f4876537529619cec829e.
