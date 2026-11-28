# Small + Tiny ensemble (SM184): result

CPU only, SM160 dumps, T24, 718 users (users with all six dumps; Big's 5 out-of-memory users drop out), weights fitted on the 100 `wfit` users only. Numbers from `sm184-result.json` (made by `sm184_ensemble.py`). % = instant-head LogLoss change, paired user bootstrap 95% CI.

Fitted weights: (a) 0.92 Small Instant-Only (SM113) / 0.08 Tiny Instant-Only (SM130); (b) 4-way 0.756 Small Instant-Only, 0.233 Small two-head (SM118), 0.011 Tiny Instant-Only, 0.000 Tiny two-head; (c) 1.00 on the Small ensemble (Tiny gets 0); (d) 0.68 Big Instant-Only (SM99) / 0.32 Small Instant-Only.

| Instant head | vs Small Instant-Only alone | vs Big Instant-Only alone | AUC change vs Small |
|---|---|---|---|
| Big Instant-Only alone (SM99) | -0.012% [-0.036, +0.014] | 0 | +0.0003 |
| Tiny shipped ensemble | +0.529% [+0.486, +0.575] | +0.540% [+0.494, +0.587] | -0.0020 |
| Small two-model ensemble (0.78) | -0.021% [-0.024, -0.019] | -0.010% [-0.036, +0.015] | +0.0001 |
| (a) Small IO + Tiny IO | -0.003% [-0.005, -0.002] | +0.008% [-0.017, +0.033] | +0.0000 |
| (b) 4-way | -0.021% [-0.024, -0.018] | -0.010% [-0.036, +0.015] | +0.0001 |
| (c) Small ens + Tiny ens | -0.021% [-0.024, -0.019] | -0.010% [-0.036, +0.015] | +0.0001 |
| (d) Big IO + Small IO (reference) | -0.192% [-0.213, -0.172] | -0.180% [-0.193, -0.168] | +0.0010 |

Curve head (report only, two-head models): Small two-head + Tiny two-head (w 0.74) is -0.026% [-0.051, -0.004] vs Small two-head alone and +0.182% [+0.027, +0.337] vs Big two-head alone.

## Verdict
No Small + Tiny ensemble is worth considering: none reaches -0.05% (go rule), and the fits give Tiny almost no weight (0 to 8%). The Small + Tiny gain over Small is the Small two-model ensemble's own gain (-0.021%), which SM160 already measured. Tiny is 0.53% worse than Small, with errors too correlated to add information. Big Instant-Only is statistically level with Small Instant-Only on this set (-0.012%, CI contains 0). The cross-size pair that does help is (d) Big + Small (-0.19% vs Small and -0.18% vs Big), which is not a Small + Tiny option.

## Deploy cost
No parameter counts or state bytes for Small and Tiny were found in the SM162/SM164 RESULT files; running both costs about 1.5x Small per review and per card state (Tiny added on top of Small).
