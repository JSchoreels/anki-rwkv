# Resource profile: expertium/E89-F65e-F77a

Case status: **complete**. Scope: **complete-case telemetry**.

Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.

| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startup/primer | 11 | 10.4 | 1.5 / 6.0 | 1.0 | 11.5 | 12.38 | 4.68 | 0.18 |
| training including user transitions | 38 | 38.6 | 60.8 / 92.0 | 15.9 | 55.1 | 8.68 | 7.90 | 12.21 |
| checkpoint save/evaluation | 28 | 28.2 | 18.6 / 27.0 | 1.0 | 53.0 | 11.28 | 5.86 | 12.21 |

Logged coverage: **80/80 training users**, **167 updates**, **7,654,975 reviews**. Held-out cohort: **20 users**.
Complete training observation: **True**. Resource telemetry does not establish predictive quality.
Full-history profile available: **False**; observed training window: **38.6 s**.

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

Summed preparation across workers: 113.7 s; summed future waits: 1.6 s.
worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time

per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate

Unavailable/limited counters:

- actual_GPU_bytes_per_second: not sampled.
- physical_disk_IO: process I/O includes IPC/device traffic; physical disk counters not sampled.
- Windows_swap_in_out_bytes: installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero.
- precise_steady_vs_user_transition: current telemetry combines training and user transitions; exact phase timestamps not recorded.
- kernel_occupancy: not sampled.

Sampler wall: 3.411 s; sampler-thread CPU: 0.859 s. This does not estimate critical-path slowdown.

Source evidence:

- [artifacts/retraining/gpu-retrain-r01/iteration-0090/expertium/E89-F65e-F77a/result.json](../../expertium/E89-F65e-F77a/result.json): d95b3fbfd9785adb9b216625d04e91e9fa9b0d9dac1b0663f21024c45ae550a7.
- [artifacts/retraining/gpu-retrain-r01/iteration-0090/protocol.json](../../protocol.json): 4b91120ab822c8fc503c79d348a54d9d275afee0f0715289450de5add2c99041.
- [artifacts/retraining/gpu-retrain-r01/iteration-0090/expertium/E89-F65e-F77a/resources.jsonl](../../expertium/E89-F65e-F77a/resources.jsonl): 07deb918279554c20e4e6f29be3dd935344355678bc47d6a64cb58e4f6fff313.
- [artifacts/retraining/gpu-retrain-r01/iteration-0090/expertium/E89-F65e-F77a/updates.jsonl](../../expertium/E89-F65e-F77a/updates.jsonl): 93d57ea067b54e121f6555d1cba51e627dbd799d0dcc1967527cddd72c9e40bd.
- [artifacts/retraining/gpu-retrain-r01/iteration-0090/expertium/E89-F65e-F77a/resources-summary.json](../../expertium/E89-F65e-F77a/resources-summary.json): 1c6f0a5e53fda71fbe21b1fba03517a00e3d5a40f68fdd24583b34a90b7c8c32.
