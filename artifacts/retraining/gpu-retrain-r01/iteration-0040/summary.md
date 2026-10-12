# E33 reproducibility during desktop use: one authorized repeat

The historical E33 implementation completed at **182,745 reviews/s in 41.889 training seconds**, versus **170,718 reviews/s in 44.840 seconds** in iteration 0033: throughput **+7.05%**, training time **−6.58%**. Exact recorded numerical parity held. This diagnostic leaves accepted iteration 0033 unchanged and does not promote a new reference, restart the optimization loop, or validate quality more broadly.

Authorization: the human requested “pop a worker to re-run iteration 33 to see if we still achieve the same results with me using the dekstop”. Exactly one full matched repeat ran while desktop use remained unrestricted. Alex stayed paused; no new smoke, arithmetic checks, optimization candidates, or multi-user validation were launched. The previous 0037–0039 worker completed and its clean closure was observed before the repeat; no competing Python GPU process was present at launch.

Open [raw comparison](repeat-comparison.json), [repeat result](expertium/E30-F37-F42/result.json), [original result](../iteration-0033/expertium/E30-F37-F42/result.json), [protocol](protocol.json), [compatibility record](repeat-compatibility.json) and [resource profile](profiles/expertium-E30-F37-F42/result.json). [Console log](console.log) and [controller timing](console.log.run.json) preserve launch output and elapsed time; [checkpoint](expertium/E30-F37-F42/master.pth), [updates](expertium/E30-F37-F42/updates.jsonl), [resource samples](expertium/E30-F37-F42/resources.jsonl), [power samples](expertium/E30-F37-F42/gpu_power.jsonl) and their summaries remain local raw evidence.

## Matched exposure and implementation

Both cases use E30-F37-F42: fixed 80 training users and 20 held-out users, cohort SHA256 `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`, dataset revision `75299740cff05894ef42d7ad990666691efdd2da`, seed 2025, one epoch, 7,654,975 training review presentations, 868,370 evaluation reviews, 167 updates and 6,764,613 Curve/Instant training labels each. Held-out Curve/Instant label counts are 741,950 each. BF16 compute, FP32 master weights, 512 batch chunks, 128 reviews/chunk, six preparation workers, prefetch depth 12, one main thread, four configured CPU threads, four recurrence rows, 95% GPU memory fraction, peak LR 0.0007, five warmup updates and original clipping/update order are preserved. Initial weights and optimizer were freshly seeded; the original checkpoint was not reused.

The current workspace `scripts/run_isolated_gpu_r01.py` applied live LOOP guards and issued the launch ticket to the existing recorded 0033 runtime. All 189 archived source files were checked against their preserved manifest; training implementation SHA256 matches `2a10c1602934b4587d4b084d4e66f9665cc497283b98eff4b70f661385956452`. Both training processes record full Git revision `999baf9f085949e05e4c00628a5f82e197240922`, dirty only for untracked `isolation.json`, and the same preserved process source archive `artifacts/sources/experiment-sources/07043a6f8ee874cc18abebe24eeee89d251c9ee1201f9720a7969d9fb13456f6/source.zip`. The historical archive's origin revision is `dc56f0d38818b9a04d6fcc68d6d5f91973f6ff39` with preserved local edits; it is distinct from the runtime revision and later implementation commits. This repeat did not use current training source. Controller/analysis provenance is separately recorded in the launch and profile records.

Configuration, options, execution mode, dropout, master precision, TF32/matmul settings, Torch `2.9.1+cu128`, CUDA `12.8` and RTX 5080 identity match the original. The compatible [preflight](../../../diagnostics/r01-e33-desktop-repeat-v1/preflight.json) records driver 617.14, CuPy 14.2.0 and NumPy 2.5.4; it is a CPU configuration/temp check, with no new CUDA probe. Existing historical arithmetic/smoke/full-run evidence was reused because the implementation/options were unchanged. Source and runtime metadata are preserved in [source isolation](source-isolation.json); no guards were changed.

## Timing and resources

| Measurement | Original 0033 | Desktop repeat | Change |
|---|---:|---:|---:|
| Training seconds | 44.840 | 41.889 | −6.58% |
| Raw review presentations/s | 170,718 | 182,745 | +7.05% |
| Complete training-process seconds | 96.242 | 93.626 | −2.72% |
| Case elapsed seconds | 94.053 | 91.364 | −2.86% |
| Summed worker preparation seconds | 110.322 | 114.022 | +3.35% |
| Queue wait seconds | 0.590 | 0.556 | −5.81% |
| Validation seconds | 29.190 | 29.625 | +1.49% |
| Discarded primer seconds | 15.032 | 14.964 | −0.45% |
| Worker-pool mean CPU, one core = 100% | 42.43% | 47.17% | +4.74 points / +11.17% |
| Whole-device GPU activity | 69.09% | 59.69% | −9.40 points / −13.61% |
| Total host CPU during sampled training | 34.56% | 38.51% | +3.94 points |
| Experiment-tree CPU, normalized to all 16 CPUs | 21.94% | 24.17% | +2.23 points |
| Background CPU residual proxy | 12.62% | 14.34% | +1.71 points |

Training timing includes user transitions, preparation/transfer/update work and logs; the worker preparation sum overlaps across workers and is not a critical-path duration. Complete process timing additionally includes setup, discarded primer, checkpoint saving and evaluation. The repeat's outer controller/export elapsed time was **94.562 s**; no comparable original outer-controller sidecar is available, so no matched percentage is claimed for that scope. Model setup was 4.738→4.630 s and pool prestart 15.084→15.013 s; these overlap and must not be added to infer wall time.

[Original pre-launch load](../iteration-0033/background-load.json) was 8.2% total CPU with zero other Python GPU processes; [repeat pre-launch load](background-load.json) was 5.3% CPU, 2% GPU activity, 45.32 W, 780 MHz and 44°C, also with zero other Python GPU processes. During training, the residual CPU proxy rose despite the lower pre-launch reading. It subtracts sampled experiment-tree CPU divided by 16 from mean per-core host CPU; intervals are not perfectly synchronized, so it is approximate and provides no per-app attribution. GPU activity is whole-device telemetry and cannot separate desktop activity from training.

The original has 43 training samples spanning 43.705 s; the repeat has 40 spanning 40.735 s. Both cover the full history but fail the profiler's 120-second resource observation threshold, so GPU/CPU counters are descriptive and cannot establish saturation or an independent component effect. The faster repeat under intended desktop use is an observation, not causal proof that desktop use improves training. One repeat cannot establish a timing distribution, reliability bound or isolated desktop-load effect.

## Numerical reproducibility

All 167 update sequence entries match for step, user, offset, reviews, seed, labels, LR, logged loss and gradient norm; serialized floating-point bits of all logged losses/norms also match. Initial/final parameter SHA256, held-out input SHA256, review/label counts, per-user coverage and the full evaluation object match. Full gradient tensor hashes were not recorded by either full run, so this is exact recorded update/norm/parameter/input evidence, not a fresh full-tensor gradient comparison. Historical focused arithmetic evidence remains applicable to the unchanged source.

| Held-out binary log loss | Original | Repeat | Absolute / relative change |
|---|---:|---:|---:|
| Curve, label-weighted | 0.3874726840790628 | 0.3874726840790628 | 0 / 0% |
| Curve, user-average | 0.37038797217215763 | 0.37038797217215763 | 0 / 0% |
| Instant, label-weighted | 0.3984285960019747 | 0.3984285960019747 | 0 / 0% |
| Instant, user-average | 0.3825924635443113 | 0.3825924635443113 | 0 / 0% |

The composite training objective and rating cross-entropy are distinct from these four metrics. This is the original 20-user held-out screen and does not satisfy the explicitly authorized broader-validation requirement. Pending: timing distribution and causal workload attribution remain unmeasured; any further repeat or optimization round requires a new human request. The accepted reference remains 0033 and the existing three-miss optimization stop remains closed.
