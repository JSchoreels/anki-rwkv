# Resource profile: expertium/E80-F33b

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 19 | 18.7 | 7.3 / 10.0 | 2.0 | 5.0 | 10.77 | 4.69 | 0.18 |
| training including user transitions | 37 | 37.6 | 74.3 / 93.0 | 21.7 | 54.4 | 7.68 | 7.75 | 11.95 |
| checkpoint save/evaluation | 27 | 27.1 | 26.5 / 38.0 | 2.7 | 62.5 | 10.87 | 5.85 | 11.95 |

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

Summed preparation across workers: 108.4 s; summed future waits: 0.6 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.683 s; sampler-thread CPU: 0.969 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0082/expertium/E80-F33b/result.json](../../expertium/E80-F33b/result.json): c0695cd01fbfdf4a9950687f90cd42a038eb08b1c902dadcc04078b2adba3611.
- [artifacts/retraining/gpu-retrain-r01/iteration-0082/protocol.json](../../protocol.json): fc0cbf235db278fe526c31004c2dc91d267e958d6c9c6d6bf35669cdeb5b28f5.
- [artifacts/retraining/gpu-retrain-r01/iteration-0082/expertium/E80-F33b/resources.jsonl](../../expertium/E80-F33b/resources.jsonl): c557e5ddbab124d27c6d935622caa0d04077c93b9f3e94e4359d1f55f883078a.
- [artifacts/retraining/gpu-retrain-r01/iteration-0082/expertium/E80-F33b/updates.jsonl](../../expertium/E80-F33b/updates.jsonl): b31a16926368307ed9465d7b6880749e13a09f63ef7ac2a7199686e33fac09bc.
- [artifacts/retraining/gpu-retrain-r01/iteration-0082/expertium/E80-F33b/resources-summary.json](../../expertium/E80-F33b/resources-summary.json): a494ec2d70c58388efa1fc528a87b0be0be07b26adaf07efd589a21ed6bbbd1c.
