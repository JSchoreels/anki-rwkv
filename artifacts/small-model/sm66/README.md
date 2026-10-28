# Per-user adaptation (SM66): does a tiny per-user calibration lower LogLoss?

CPU-only, no training. Question: can a correction fitted only on a user's own earlier reviews lower LogLoss on that user's later reviews? Script: `scripts/sm66_per_user_calibration.py` (tables: `scripts/sm66_report.py`); numbers: `summary.json`, `tables.md`.

## Answer in one paragraph

On the instant head, no: every per-user, global and online correction moves LogLoss by 0.00001 to 0.00008 (0.01 to 0.02 percent), under 7 percent of the small-vs-big gap of 0.00122, and per-user is indistinguishable from the global calibration control. On the curve head, an online per-user bias plus temperature (refit every 500 reviews) lowers the released small model's (SM56) LogLoss by 0.00101 [-0.00226, -0.00020] (0.33 percent); the offline split (fit on the first half, score the second) gives only -0.00016 [-0.00069, +0.00023], not significant. The online curve gain is real but is almost entirely adaptation to drift inside a user's history, and it is of the same size for the big model (-0.00071 [-0.00114, -0.00033]), so it does not close the small-vs-big gap (curve gap change -0.00030 [-0.00145, +0.00044]; instant gap widens slightly, +0.00003 [+0.00001, +0.00006]). The global calibration control is about zero for both heads (curve -0.00005, instant -0.00001), so the models are well calibrated on average and the online gain is not plain miscalibration. Verdict at the end.

## Method

- Data: the 200 held-out benchmark users of the feature-discovery dump (`wt/sm70/artifacts/small-model/sm70/dump`, SM56 w8+s8, 7,058,890 benchmark test reviews) and the big model's per-review dump of the same users and reviews (`wt/small-model/artifacts/small-model/sm24-dump`, `fork_w8+s8`; review ids and labels checked identical). Each user's dumped test reviews are taken in time order (`review_th`); "history" below means these reviews. Users: 2,000 to 175,290 reviews; terciles T1 2,000-10,540 (67 users), T2 10,720-32,875 (66), T3 33,050-175,290 (67).
- Correction: logit' = a * logit(p) + b, fitted separately for the curve head (`p_ahead`) and the instant head (`p_instant`) by minimising LogLoss with an L2 prior toward the identity (a = 1, b = 0). Forms: B = bias only (a = 1), BT = bias + temperature. Prior strength (in pseudo-reviews, grid 3, 30, 300, 3,000, 30,000; separate grids for b and a) is chosen by 5-fold cross-validation across users, one value per model, head and scheme (never per user): for each held-out fold the value minimising the pooled loss of the other four folds' users is used.
- Offline: fit on the first half of each user's reviews, score the second half. Online: expanding window, refit after every N reviews on all reviews seen so far, predict only the next block; the first block is uncorrected. Headline N = 500, sensitivity N = 2,000. Online is scored on the same second half as offline (comparable), and also on all blocks after the first (own baseline, labelled "all blocks").
- Controls: (1) global calibration: one (a, b) per head fitted on the first halves of all users outside the user's CV fold, applied unchanged; (2) the per-user correction stacked on top of the global one. The "per-user - global" table is the direct test of whether per-user adds anything beyond plain miscalibration.
- Aggregation: review-weighted pooled LogLoss. Delta = with correction - without (negative = better), 95 percent CI from a user bootstrap (2,000 resamples, seed 0). The gap rows are the paired change of (small - big) LogLoss on identical reviews, each model corrected with its own CV-chosen prior.
- Caveat: only the benchmark test reviews of a user are available, not the full history, so "earlier reviews" for the first corrected block are the early part of the test set, and the model's own training-time view of the user is not part of the correction. A fitted correction in the app would see the user's full history.

## Results

Delta = LogLoss with the correction - without, review-weighted over users, 95% user bootstrap (2,000 resamples, seed 0). Negative = better.

### Curve head

| Scheme | Small model (SM56) | Big model | Gap small-big change (before -> after) |
|---|---|---|---|
| global calibration (control) | -0.00005 [-0.00019, +0.00008] (-0.02%) | -0.00009 [-0.00025, +0.00007] (-0.03%) | +0.00003 [-0.00005, +0.00011] (0.00054 -> 0.00057) |
| per-user bias, offline | -0.00022 [-0.00065, +0.00008] (-0.07%) | +0.00010 [-0.00014, +0.00043] (+0.03%) | -0.00032 [-0.00103, +0.00007] (0.00054 -> 0.00022) |
| per-user bias+temperature, offline | -0.00016 [-0.00069, +0.00023] (-0.05%) | +0.00005 [-0.00028, +0.00047] (+0.02%) | -0.00022 [-0.00108, +0.00026] (0.00054 -> 0.00032) |
| per-user bias on top of global, offline | -0.00025 [-0.00073, +0.00013] (-0.08%) | +0.00004 [-0.00023, +0.00036] (+0.01%) | -0.00028 [-0.00099, +0.00012] (0.00054 -> 0.00025) |
| per-user bias+temperature on top of global, offline | -0.00021 [-0.00078, +0.00024] (-0.07%) | +0.00001 [-0.00035, +0.00043] (+0.00%) | -0.00022 [-0.00109, +0.00026] (0.00054 -> 0.00032) |
| per-user bias, online (block 500) | -0.00073 [-0.00164, -0.00018] (-0.24%) | -0.00059 [-0.00095, -0.00029] (-0.20%) | -0.00014 [-0.00091, +0.00039] (0.00054 -> 0.00040) |
| per-user bias+temperature, online (block 500) | -0.00101 [-0.00226, -0.00020] (-0.33%) | -0.00071 [-0.00114, -0.00033] (-0.24%) | -0.00030 [-0.00145, +0.00044] (0.00054 -> 0.00024) |
| per-user bias, online (block 500), all blocks after the first | -0.00060 [-0.00111, -0.00025] (-0.20%) | -0.00064 [-0.00101, -0.00033] (-0.21%) | +0.00004 [-0.00040, +0.00040] (0.00036 -> 0.00040) |
| per-user bias+temperature, online (block 500), all blocks after the first | -0.00079 [-0.00146, -0.00033] (-0.27%) | -0.00077 [-0.00116, -0.00043] (-0.26%) | -0.00002 [-0.00059, +0.00040] (0.00036 -> 0.00034) |
| per-user bias, online (block 2000) | -0.00044 [-0.00114, -0.00000] (-0.15%) | -0.00041 [-0.00071, -0.00015] (-0.14%) | -0.00003 [-0.00065, +0.00040] (0.00054 -> 0.00050) |
| per-user bias+temperature, online (block 2000) | -0.00067 [-0.00192, +0.00020] (-0.22%) | -0.00030 [-0.00069, +0.00007] (-0.10%) | -0.00036 [-0.00150, +0.00033] (0.00054 -> 0.00018) |
| per-user bias, online (block 2000), all blocks after the first | -0.00029 [-0.00069, -0.00002] (-0.10%) | -0.00030 [-0.00056, -0.00008] (-0.10%) | +0.00001 [-0.00033, +0.00028] (0.00038 -> 0.00040) |
| per-user bias+temperature, online (block 2000), all blocks after the first | -0.00054 [-0.00119, -0.00011] (-0.18%) | -0.00044 [-0.00077, -0.00016] (-0.15%) | -0.00010 [-0.00069, +0.00030] (0.00038 -> 0.00028) |

### Instant head

| Scheme | Small model (SM56) | Big model | Gap small-big change (before -> after) |
|---|---|---|---|
| global calibration (control) | -0.00001 [-0.00002, -0.00000] (-0.00%) | -0.00002 [-0.00004, -0.00001] (-0.01%) | +0.00001 [-0.00000, +0.00002] (0.00122 -> 0.00124) |
| per-user bias, offline | -0.00001 [-0.00002, -0.00000] (-0.00%) | -0.00002 [-0.00004, -0.00001] (-0.01%) | +0.00002 [+0.00000, +0.00003] (0.00122 -> 0.00124) |
| per-user bias+temperature, offline | -0.00002 [-0.00003, -0.00000] (-0.01%) | -0.00005 [-0.00008, -0.00002] (-0.02%) | +0.00003 [+0.00000, +0.00006] (0.00122 -> 0.00125) |
| per-user bias on top of global, offline | -0.00001 [-0.00003, -0.00000] (-0.01%) | -0.00003 [-0.00005, -0.00002] (-0.01%) | +0.00002 [+0.00000, +0.00004] (0.00122 -> 0.00124) |
| per-user bias+temperature on top of global, offline | -0.00002 [-0.00003, -0.00000] (-0.01%) | -0.00005 [-0.00009, -0.00002] (-0.02%) | +0.00003 [+0.00001, +0.00006] (0.00122 -> 0.00126) |
| per-user bias, online (block 500) | -0.00002 [-0.00003, -0.00001] (-0.01%) | -0.00003 [-0.00005, -0.00002] (-0.01%) | +0.00002 [+0.00000, +0.00003] (0.00122 -> 0.00124) |
| per-user bias+temperature, online (block 500) | -0.00003 [-0.00005, -0.00002] (-0.01%) | -0.00007 [-0.00011, -0.00003] (-0.02%) | +0.00003 [+0.00001, +0.00006] (0.00122 -> 0.00126) |
| per-user bias, online (block 500), all blocks after the first | -0.00002 [-0.00003, -0.00001] (-0.01%) | -0.00004 [-0.00006, -0.00002] (-0.01%) | +0.00002 [+0.00001, +0.00004] (0.00118 -> 0.00121) |
| per-user bias+temperature, online (block 500), all blocks after the first | -0.00003 [-0.00004, -0.00002] (-0.01%) | -0.00006 [-0.00009, -0.00004] (-0.02%) | +0.00003 [+0.00001, +0.00006] (0.00118 -> 0.00122) |
| per-user bias, online (block 2000) | -0.00001 [-0.00003, -0.00000] (-0.01%) | -0.00003 [-0.00005, -0.00002] (-0.01%) | +0.00002 [+0.00000, +0.00003] (0.00122 -> 0.00124) |
| per-user bias+temperature, online (block 2000) | -0.00003 [-0.00005, -0.00002] (-0.01%) | -0.00006 [-0.00011, -0.00003] (-0.02%) | +0.00003 [+0.00001, +0.00006] (0.00122 -> 0.00126) |
| per-user bias, online (block 2000), all blocks after the first | -0.00002 [-0.00002, -0.00001] (-0.01%) | -0.00004 [-0.00006, -0.00002] (-0.01%) | +0.00002 [+0.00000, +0.00003] (0.00119 -> 0.00121) |
| per-user bias+temperature, online (block 2000), all blocks after the first | -0.00003 [-0.00004, -0.00002] (-0.01%) | -0.00006 [-0.00009, -0.00003] (-0.02%) | +0.00003 [+0.00001, +0.00006] (0.00119 -> 0.00122) |

### Per-user correction versus the global control (small model; delta of LogLoss, same evaluation reviews)

| Head | Scheme | Small: per-user - global | Big: per-user - global |
|---|---|---|---|
| curve | per-user bias, offline | -0.00017 [-0.00051, +0.00006] | +0.00019 [-0.00010, +0.00060] |
| curve | per-user bias+temperature, offline | -0.00011 [-0.00055, +0.00022] | +0.00014 [-0.00021, +0.00063] |
| curve | per-user bias on top of global, offline | -0.00019 [-0.00057, +0.00008] | +0.00012 [-0.00012, +0.00048] |
| curve | per-user bias+temperature on top of global, offline | -0.00015 [-0.00063, +0.00021] | +0.00010 [-0.00024, +0.00055] |
| curve | per-user bias, online (block 500) | -0.00068 [-0.00150, -0.00018] | -0.00051 [-0.00081, -0.00023] |
| curve | per-user bias+temperature, online (block 500) | -0.00095 [-0.00214, -0.00023] | -0.00062 [-0.00100, -0.00028] |
| curve | per-user bias, online (block 2000) | -0.00039 [-0.00098, -0.00002] | -0.00032 [-0.00058, -0.00009] |
| curve | per-user bias+temperature, online (block 2000) | -0.00062 [-0.00177, +0.00015] | -0.00022 [-0.00054, +0.00010] |
| instant | per-user bias, offline | +0.00000 [-0.00001, +0.00001] | -0.00000 [-0.00003, +0.00002] |
| instant | per-user bias+temperature, offline | -0.00001 [-0.00002, +0.00001] | -0.00002 [-0.00006, +0.00001] |
| instant | per-user bias on top of global, offline | -0.00000 [-0.00001, +0.00000] | -0.00001 [-0.00003, +0.00001] |
| instant | per-user bias+temperature on top of global, offline | -0.00000 [-0.00002, +0.00001] | -0.00003 [-0.00007, +0.00000] |
| instant | per-user bias, online (block 500) | -0.00000 [-0.00002, +0.00001] | -0.00001 [-0.00003, +0.00001] |
| instant | per-user bias+temperature, online (block 500) | -0.00002 [-0.00004, -0.00001] | -0.00004 [-0.00009, -0.00001] |
| instant | per-user bias, online (block 2000) | -0.00000 [-0.00002, +0.00001] | -0.00001 [-0.00003, +0.00001] |
| instant | per-user bias+temperature, online (block 2000) | -0.00002 [-0.00004, -0.00001] | -0.00004 [-0.00009, -0.00001] |

### By user size (terciles of reviews per user; small model, then big model delta)

Terciles (min-max dumped test reviews, users): T1 2000-10540 (67); T2 10720-32875 (66); T3 33050-175290 (67)

| Head | Scheme | T1 small | T2 small | T3 small | T1 big | T2 big | T3 big |
|---|---|---|---|---|---|---|---|
| curve | global calibration (control) | -0.00007 [-0.00029, +0.00019] | +0.00009 [-0.00011, +0.00030] | -0.00009 [-0.00026, +0.00008] | -0.00022 [-0.00049, +0.00007] | -0.00000 [-0.00019, +0.00020] | -0.00010 [-0.00030, +0.00010] |
| curve | per-user bias, offline | -0.00009 [-0.00027, +0.00006] | +0.00004 [-0.00012, +0.00022] | -0.00030 [-0.00085, +0.00010] | -0.00001 [-0.00006, +0.00005] | -0.00002 [-0.00007, +0.00002] | +0.00014 [-0.00017, +0.00060] |
| curve | per-user bias+temperature, offline | -0.00022 [-0.00044, -0.00003] | -0.00005 [-0.00030, +0.00020] | -0.00019 [-0.00089, +0.00031] | -0.00010 [-0.00022, +0.00001] | -0.00007 [-0.00016, +0.00002] | +0.00009 [-0.00034, +0.00067] |
| curve | per-user bias, online (block 500) | -0.00045 [-0.00119, +0.00025] | -0.00027 [-0.00063, +0.00008] | -0.00086 [-0.00204, -0.00015] | -0.00081 [-0.00171, -0.00011] | -0.00023 [-0.00058, +0.00010] | -0.00066 [-0.00112, -0.00027] |
| curve | per-user bias+temperature, online (block 500) | -0.00095 [-0.00216, +0.00010] | -0.00057 [-0.00144, +0.00021] | -0.00111 [-0.00277, -0.00013] | -0.00186 [-0.00364, -0.00047] | -0.00097 [-0.00200, -0.00015] | -0.00057 [-0.00107, -0.00015] |
| instant | global calibration (control) | -0.00002 [-0.00005, +0.00000] | -0.00001 [-0.00003, +0.00000] | -0.00001 [-0.00002, -0.00000] | -0.00003 [-0.00006, +0.00001] | -0.00001 [-0.00003, +0.00002] | -0.00003 [-0.00004, -0.00001] |
| instant | per-user bias, offline | -0.00001 [-0.00002, -0.00000] | -0.00001 [-0.00003, +0.00000] | -0.00001 [-0.00002, +0.00000] | -0.00001 [-0.00002, +0.00000] | -0.00001 [-0.00002, +0.00000] | -0.00003 [-0.00005, -0.00001] |
| instant | per-user bias+temperature, offline | -0.00002 [-0.00005, -0.00000] | -0.00000 [-0.00003, +0.00003] | -0.00002 [-0.00004, -0.00000] | -0.00003 [-0.00005, -0.00000] | -0.00000 [-0.00004, +0.00003] | -0.00006 [-0.00010, -0.00002] |
| instant | per-user bias, online (block 500) | -0.00002 [-0.00006, +0.00001] | -0.00002 [-0.00005, +0.00000] | -0.00001 [-0.00003, -0.00000] | -0.00001 [-0.00005, +0.00003] | -0.00001 [-0.00003, +0.00001] | -0.00004 [-0.00007, -0.00002] |
| instant | per-user bias+temperature, online (block 500) | -0.00002 [-0.00006, +0.00002] | -0.00002 [-0.00006, +0.00001] | -0.00004 [-0.00006, -0.00002] | -0.00003 [-0.00006, -0.00001] | -0.00002 [-0.00005, +0.00001] | -0.00008 [-0.00013, -0.00004] |

### Share of users improved and CV-chosen prior strengths (small model)

| Head | Scheme | Users improved | Prior strength per CV fold |
|---|---|---|---|
| curve | per-user bias, offline | 62% | f0: lam_b=3000; f1: lam_b=3000; f2: lam_b=3000; f3: lam_b=3000; f4: lam_b=3000 |
| curve | per-user bias+temperature, offline | 56% | f0: lam_b=3000,lam_a=30000; f1: lam_b=3000,lam_a=30000; f2: lam_b=30000,lam_a=3000; f3: lam_b=3000,lam_a=30000; f4: lam_b=30000,lam_a=3000 |
| curve | per-user bias, online (block 500) | 54% | f0: lam_b=300; f1: lam_b=300; f2: lam_b=300; f3: lam_b=300; f4: lam_b=300 |
| curve | per-user bias+temperature, online (block 500) | 52% | f0: lam_b=30,lam_a=300; f1: lam_b=30,lam_a=300; f2: lam_b=300,lam_a=300; f3: lam_b=300,lam_a=300; f4: lam_b=300,lam_a=300 |
| instant | per-user bias, offline | 60% | f0: lam_b=3000; f1: lam_b=3000; f2: lam_b=3000; f3: lam_b=3000; f4: lam_b=3000 |
| instant | per-user bias+temperature, offline | 60% | f0: lam_b=3000,lam_a=3000; f1: lam_b=3000,lam_a=3000; f2: lam_b=3000,lam_a=3000; f3: lam_b=3000,lam_a=3000; f4: lam_b=3000,lam_a=3000 |
| instant | per-user bias, online (block 500) | 49% | f0: lam_b=300; f1: lam_b=3000; f2: lam_b=300; f3: lam_b=3000; f4: lam_b=300 |
| instant | per-user bias+temperature, online (block 500) | 54% | f0: lam_b=300,lam_a=3000; f1: lam_b=3000,lam_a=3000; f2: lam_b=300,lam_a=3000; f3: lam_b=300,lam_a=3000; f4: lam_b=300,lam_a=3000 |

## Reading the tables

- Instant head: nothing to gain. The best scheme (online bias+temperature) is -0.00003 small / -0.00007 big, and the gap to the big model widens by 0.00003. The big model benefits slightly more because it has slightly more residual miscalibration.
- Curve head, offline: -0.0002 (bias) and -0.0002 (bias+temperature), CIs include zero, same size as the global control's -0.00005. Fixed per-user offsets from the first half do not carry to the second half.
- Curve head, online: -0.0007 (bias) and -0.0010 (bias+temperature) at N = 500, CI excludes zero, and it beats the global control (-0.0010 [-0.0021, -0.0002]). With N = 2,000 the gain shrinks (-0.0004 / -0.0007) and the bias+temperature CI includes zero. The gain is tracking short-term drift of each user's recall rate against the prediction, which a refit every 500 reviews follows. The big model gains almost as much (-0.0006 / -0.0007), so the gap does not narrow significantly. Only 52 to 54 percent of users improve under online bias+temperature (CV prior is strong, 30 to 300 pseudo-reviews on b and 300 on a): the pooled gain comes from a minority of users with large drift.
- User size: the online curve gain is not larger for big users (small model, bias+temperature: T1 -0.00095, T2 -0.00057, T3 -0.00111; all CIs wide). The offline gain is T1 -0.00022 [-0.00044, -0.00003] only for the smallest tercile. The instant head is flat across terciles.

## Verdict

Not worth building as an app-side or training-time feature for the instant head: the whole possible gain (0.00003) is under 3 percent of the small-vs-big gap. For the curve head, an online per-user bias+temperature recalibration gives about 0.001 LogLoss (0.3 percent), larger than the curve gap (0.00054 on these users), but it helps the big model equally, so it is an accuracy bonus for both and not a way for the small model to catch up. If cheap to ship (two floats per user per head, refit every few hundred reviews on the device), it is a small curve-head improvement; it would need a check on the full per-user histories and on the app's real curve usage before committing to it, and a fixed offline per-user calibration is not worth it.
