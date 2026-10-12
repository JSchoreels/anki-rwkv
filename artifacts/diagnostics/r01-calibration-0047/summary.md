# R01 noise calibration from iteration 0047 (analysis only)

Six identical, exact full-history runs of the 0042 configuration ([iteration summary](../../retraining/gpu-retrain-r01/iteration-0047/summary.md)) give 15 identical-run pairs. Each pair's "gain" should be zero; its spread is the noise of each estimator. [estimators.py](estimators.py) → [estimators.json](estimators.json); [replicate-stats.json](replicate-stats.json) from `scripts/r01_replicate_stats.py`.

| Estimator over a pair of runs | SD over 15 identical pairs | Mean absolute "gain" |
|---|---:|---:|
| Total training time (the current gate) | 3.74% | 2.86% |
| Total device span | 3.19% | 2.74% |
| Trimmed time-weighted per-update wall, 5 MAD | 1.01% | 0.81% |
| Trimmed time-weighted per-update wall, 3 MAD | 0.57% | 0.50% |
| Median per-update wall ratio | 0.46% | 0.35% |
| Median per-update device span ratio | 0.27% | 0.23% |

- **Why totals are noisy.** Run-level noise comes mostly from a few transient updates. Replicate r4's whole deficit sits in its first work segment (−28%) while the other seven segments are within ±4%. Large updates' median wall time is 268–272 ms in every run.
- **Bias check.** Speeding up only the 87 largest updates of one run by 3% (true total effect +1.79%) is recovered as +1.78% (3 MAD) and +1.81% (5 MAD). The median ratio describes the typical update instead, so it is a secondary statistic.
- **Blocks needed** (one-sided 5%, power 80%, block SD 1.01% with 5 MAD): 2 blocks for a 2% or 3% gain, 7 for 1%, 26 for 0.5%. With total time (SD 2.57% per run): 4 blocks for 5%, 10 for 3%, 21 for 2%, 83 for 1%. One replicate takes about 90 s end to end, so a block (reference plus candidate) costs about 3–4 minutes.
- **Proposed gate.** Per block, the 5-MAD trimmed per-update gain of the candidate against its adjacent reference run. Mean over blocks; standard error from max(calibrated 1.01%, observed block SD). Promote if the one-sided 95% lower bound exceeds zero; stack candidate if the mean is positive; otherwise reject. A neutral change is promoted about 5% of the time, against about 25% for the median single-run band.
- **Limits.** One machine state (all six at about 1.7 W per utilization point) and one configuration. A whole-run state switch between the two runs of a block is not removed by per-update pairing; the observed-SD term and the quiet-machine gate guard against it. Recalibrate if the state changes; the R–R pairs of each iteration's blocks provide continuing A/A data.
