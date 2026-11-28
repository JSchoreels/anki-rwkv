# AUC ceiling (SM182): result

Finished Sat 2026-11-28 03:56 (`SM182 DONE`). Tables: `sm182-result.md`, `sm182-result.json`. Rule users: 723 of the 732 set, none left out for a non-finite row; integrity against the state-energy penalty screen's (SM171) rows: max difference 0.0; the per-review dump reproduces the scorer within 5e-7. Arm RA was not needed (R trained to the end with finite losses).

## Question and rules

Our models are trained on LogLoss only. How much discrimination (AUC) is left on the table? Arm R (pure ranking, pair weight 1.0, no LogLoss term) measures the ceiling of a short cool-down and has no verdict. Arm M (mixed, pair weight 0.91 plus the stock loss) is "useful" if, on at least one head, the within-user AUC change has its CI lower bound above 0, and on both heads the LogLoss change has its CI upper bound at most +0.05%. The control is the cool-down of the state-energy penalty screen (SM171), not retrained. Paired user bootstrap, 2,000 draws, 95% CI.

## Results against the control (95% CI)

| arm | head | within-user AUC change | LogLoss | overall AUC change | RMSE(bins) | ECE |
|---|---|---|---|---|---|---|
| pure ranking (R) | curve | +0.0010 [+0.0007, +0.0012] (+0.12%) | +0.248% [+0.158, +0.331] | -0.0001 [-0.0004, +0.0002] | +6.3% | 0.0089 (control 0.0016) |
| pure ranking (R) | instant | +0.0002 [+0.0001, +0.0003] (+0.03%) | +0.171% [+0.152, +0.192] | -0.0000 | +9.3% | 0.0089 (control 0.0015) |
| mixed (M) | curve | +0.0008 [+0.0006, +0.0009] (+0.09%) | +0.029% [-0.022, +0.071] | +0.0001 [-0.0001, +0.0003] | +1.6% | 0.0030 |
| mixed (M) | instant | +0.0002 [+0.0001, +0.0003] (+0.03%) | -0.014% [-0.020, -0.008] | +0.0000 [+0.0000, +0.0001] | -0.7% | 0.0015 |

Control values: within-user AUC 0.7951 (curve) and 0.8418 (instant); LogLoss 0.30337 and 0.28009.

## Platt recalibration (causal per-user fit)

For R, a causal per-user Platt fit removes the whole LogLoss loss: curve LogLoss after Platt is -0.241% [-0.560, +0.004] against the control raw (the share of the gap recovered is above 100%, +1.97 with a very wide CI) and -0.069% [-0.114, -0.035] against the control after its own Platt; instant is -0.004% [-0.013, +0.005] against the control raw and the share recovered is +1.03 [+0.97, +1.08]. So R's LogLoss loss is calibration drift, not worse ranking. After Platt, RMSE(bins) is equal to the control's after Platt (0.0368 and 0.0225). M after Platt: curve -0.062% [-0.090, -0.040] and instant -0.018% [-0.022, -0.014] against the control after Platt.

## Verdict for M: NOT USEFUL

Within-user AUC condition: passes on both heads (CI lower bounds above 0). LogLoss condition: passes on instant (upper bound -0.008%), fails on curve (upper bound +0.071% is above +0.05%). Per-head flags in `sm182-result.json` (`rule_M`).

## Training

| model | reviews/s | minutes | peak allocated / reserved MiB |
|---|---|---|---|
| control cool-down (SM171 control) | 55,723 | 20.1 | 4,216 / 8,354 |
| pure ranking (R) | 63,989 | 17.5 | 4,212 / 8,330 |
| mixed (M) | 64,187 | 17.5 | 4,212 / 8,330 |

(The arms ran faster than the control, probably because less else shared the GPU; the pair term costs no visible speed.) Scoring took about 11 to 24 minutes per model.

## Conclusion

Training for ranking directly, even with no LogLoss term, raises within-user AUC by only +0.12% on the curve head and +0.03% on the instant head, with no gain in overall AUC. LogLoss training therefore leaves almost no discrimination on the table for this model in a short cool-down. The mix is a small, real within-user AUC gain at no LogLoss cost on instant but the curve head's LogLoss CI crosses the bar, and the gain (+0.09% / +0.03%) is far below the 0.05%-style effect sizes that matter. Ranking losses are not worth pursuing for this model family; the follow-up weight sweep is optional and low value.
