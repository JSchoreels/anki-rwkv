# AUC ceiling (SM182): result tables

Rule users: 723 of the 732 set (show-time inputs, int8 weights and states, REF). Integrity vs SM171's rows: {'users': 723, 'max_abs_logloss_diff': 0.0, 'ok': True}. Dump vs rows: {'LogLoss': 4.99967097056242e-07, 'AUC': 4.999272493755313e-07, 'RMSE(bins)': 4.999140323230788e-07}.

**Verdict for M (mixed): NOT USEFUL** (useful = within-user AUC up with the CI above 0 on a head, and LogLoss CI upper <= +0.05% on both heads; R is a ceiling measurement, no verdict).

## Absolute values

| model | head | LogLoss | within-user AUC | overall AUC | RMSE(bins) | ECE (20 bins) | LogLoss after Platt | within-user AUC after Platt | RMSE(bins) after Platt |
|---|---|---|---|---|---|---|---|---|---|
| control cool-down (SM171 control) | curve | 0.30337 | 0.7951 | 0.8450 | 0.0397 | 0.0016 | 0.30284 | 0.7920 | 0.0368 |
| control cool-down (SM171 control) | instant | 0.28009 | 0.8418 | 0.8719 | 0.0228 | 0.0015 | 0.28010 | 0.8413 | 0.0225 |
| pure ranking (R) | curve | 0.30412 | 0.7961 | 0.8449 | 0.0422 | 0.0089 | 0.30263 | 0.7923 | 0.0368 |
| pure ranking (R) | instant | 0.28057 | 0.8420 | 0.8719 | 0.0249 | 0.0089 | 0.28008 | 0.8415 | 0.0225 |
| mixed (M) | curve | 0.30346 | 0.7959 | 0.8451 | 0.0403 | 0.0030 | 0.30265 | 0.7925 | 0.0367 |
| mixed (M) | instant | 0.28006 | 0.8420 | 0.8720 | 0.0227 | 0.0015 | 0.28005 | 0.8416 | 0.0224 |

## Against the control (paired user bootstrap, 95% CI)

| arm | head | LogLoss % | within-user AUC change | within-user AUC % | overall AUC change | RMSE(bins) % | after Platt: LogLoss % vs control raw | vs control after Platt | gap recovered by Platt |
|---|---|---|---|---|---|---|---|---|---|
| pure ranking (R) | curve | +0.248% [+0.158, +0.331] | +0.0010 [+0.0007, +0.0012] | +0.12% [+0.09, +0.15] | -0.0001 [-0.0004, +0.0002] | +6.3% [+5.2, +7.6] | -0.241% [-0.560, +0.004] | -0.069% [-0.114, -0.035] | +1.97 [+0.98, +3.92] |
| pure ranking (R) | instant | +0.171% [+0.152, +0.192] | +0.0002 [+0.0001, +0.0003] | +0.03% [+0.02, +0.04] | -0.0000 [-0.0000, +0.0000] | +9.3% [+8.2, +10.6] | -0.004% [-0.013, +0.005] | -0.008% [-0.013, -0.003] | +1.03 [+0.97, +1.08] |
| mixed (M) | curve | +0.029% [-0.022, +0.071] | +0.0008 [+0.0006, +0.0009] | +0.09% [+0.07, +0.12] | +0.0001 [-0.0001, +0.0003] | +1.6% [+1.2, +2.0] | -0.235% [-0.551, +0.006] | -0.062% [-0.090, -0.040] | +8.99 [-76.65, +87.80] |
| mixed (M) | instant | -0.014% [-0.020, -0.008] | +0.0002 [+0.0001, +0.0003] | +0.03% [+0.02, +0.04] | +0.0000 [+0.0000, +0.0001] | -0.7% [-1.0, -0.3] | -0.015% [-0.022, -0.007] | -0.018% [-0.022, -0.014] | -0.05 [-0.57, +0.40] |

## Training

| model | reviews/s | minutes | peak allocated / reserved MiB |
|---|---|---|---|
| control cool-down (SM171 control) | 55,723 | 20.1 | 4,216 / 8,354 |
| pure ranking (R) | 63,989 | 17.5 | 4,212 / 8,330 |
| mixed (M) | 64,187 | 17.5 | 4,212 / 8,330 |
