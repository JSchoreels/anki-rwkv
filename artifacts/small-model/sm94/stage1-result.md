# Fast-recall prediction (SM94): stage 1 result

Rules applied exactly as pre-registered in [DESIGN.md](DESIGN.md). Model: the released small model (the release from epoch 25 (SM63)) with its shipped answer-time head (the answer-time head release (SM83)), normal inputs, 723 users, 21,783,607 later-day reviews with a recorded time.

Sign convention of the tables: "vs C" = LogLoss of the predictor minus LogLoss of C, in % of C's LogLoss; **positive = worse than C**. A win needs the whole 95% interval below zero and at least 0.05 points.

## Speed and sanity

- Dump (GPU ticket 000947-sm94): 723 users, 21,783,607 scored reviews, 17,320 reviews/s, 1,258 s (queue elapsed 21 min). Analysis (CPU): 110.8 s.
- A0 (identity vs the answer-time head on richer frozen features (SM82c)): same review sets for 723 of 723 users, max relative gap 0.0, 723 >= 722 users. PASS.

## (a) Fast-pass prediction (grade Hard / Good / Easy and answer time under N)

All 723 users and 21,783,607 rows at every threshold. C = composed from the shipped heads, B0 = running per-user rate, B1 = P(pass) x running P(fast | pass), S0 = recalibrated C, S = stack with the 26 time features (cross-fitted).

| Threshold | Event rate | LogLoss C | B0 vs C [95% CI] | B1 vs C [95% CI] | S0 vs C [95% CI] | S vs C [95% CI] | S vs S0 [95% CI] | C mean pred / obs | ECE |
|---|---|---|---|---|---|---|---|---|---|
| under 3.9 s | 0.268 | 0.35947 | +35.21 [+32.79, +37.88] | +29.53 [+27.20, +32.13] | -0.022 [-0.046, +0.004] | +0.072 [+0.001, +0.148] | +0.094 [+0.032, +0.162] | 0.2625 / 0.2680 | 0.0057 |
| under 5.5 s | 0.392 | 0.43207 | +32.46 [+30.65, +34.45] | +25.79 [+23.93, +27.79] | +0.011 [-0.001, +0.025] | +0.091 [+0.051, +0.136] | +0.080 [+0.038, +0.126] | 0.3893 / 0.3915 | 0.0028 |
| under 10.9 s | 0.599 | 0.46412 | +26.95 [+25.42, +28.55] | +17.52 [+15.95, +19.22] | -0.007 [-0.034, +0.022] | +0.038 [-0.001, +0.082] | +0.045 [+0.013, +0.077] | 0.6046 / 0.5994 | 0.0053 |
| under 21.5 s | 0.732 | 0.42433 | +22.86 [+21.36, +24.49] | +9.81 [+8.54, +11.25] | -0.008 [-0.037, +0.020] | +0.021 [-0.017, +0.062] | +0.029 [+0.007, +0.052] | 0.7366 / 0.7318 | 0.0051 |
| under the user's typical time | 0.428 | 0.50786 | +32.40 [+30.73, +34.25] | +26.33 [+24.58, +28.19] | -0.007 [-0.036, +0.023] | +0.008 [-0.097, +0.152] | +0.016 [-0.073, +0.144] | 0.4252 / 0.4276 | 0.0036 |

Calibration slope (Platt, fold 0): 0.993 / 0.981 / 0.967 / 0.960 / 0.963 (C is slightly over-confident at long thresholds).

- **A1 (the composed probability is usable): FAIL by the letter.** C beats B0 and B1 by far at every threshold (9.8 to 35 points, CIs far from zero). ECE <= 0.01 everywhere. But the mean-gap bound |mean predicted - observed| <= 0.005 is missed at under 3.9 s (0.0054) and under 10.9 s (0.0053); it holds at 5.5 s (0.0022), 21.5 s (0.0048) and the typical time (0.0023). The misses are 0.0003 to 0.0004 over the bound; stated here so the user can judge, the rule is not changed.
- **A2 (a direct fast-pass target is worth training): NO-GO.** S is worse than C at every threshold (positive; at 5.5 s the CI is entirely above zero, +0.09), so it never beats C. S0 (post-hoc recalibration) does not beat C either (all CIs include zero, best -0.022). "Recalibrate C post hoc" is therefore not supported either; C is already close to calibrated.
- Limitation: a no-go on the stacking proxies does not prove a trained head or trunk could not do better; it ranks stage 2 down, it does not close it.

## (b) Does the speed signal help plain recall (R = instant head P(pass))

Later-day rows only (60.1% of the instant labels); points = change of the all-label instant LogLoss if only these rows changed, in % of the big model (1 leak-free epoch) instant LogLoss (0.27613, 722 users). LogLoss on the 21.78M rows: R 0.27067, Rb0 (recalibrated) 0.27080, Rb (with speed features) 0.27085.

| Comparison | Points [95% CI] |
|---|---|
| Rb vs Rb0 (the rule: speed beyond recalibration) | +0.0095 [+0.0071, +0.0120] |
| Rb0 vs R (calibration note) | +0.0288 [+0.0248, +0.0329] |
| Rb vs R | +0.0384 [+0.0334, +0.0432] |

- **B1-rule (speed helps recall, go for stage 2b): NO-GO.** Rb is slightly worse than Rb0, not better (positive = worse; the bar is at least 0.05 points better with the CI below zero). Both recalibrations are slightly worse than R itself, so R is already well calibrated, as the real-data check of the time-aware target (SM41r) found. This matches the expected no-go.

## Stage 2 cost if it had been a go

- 2a (direct fast-pass head on the frozen release): about 1.2 GPU h. Not warranted by A2.
- 2b (speed as an auxiliary target in the trunk): about 4.8 GPU h (+ 6.5 h for the release-matched run). Not warranted by the B1-rule.
- Stage 1c (FSRS7-Instant x speed baseline, reported only) and stage 2 are not started without the user's go.
