# Robustness to other apps (SM29): pass / fail ratings and no answer times

Leak-free scoring (deleted cards: neighbour), w8+s8, the 732 set; review-weighted benchmark LogLoss over all test reviews, curve and instant heads (the answer-time head is not evaluated); deltas in points = % of the big model (1 leak-free epoch) with normal inputs on the same users, paired user bootstrap 2,000 resamples, seed 0, 95% CI. Every variant scores the same test reviews with the same labels; only inputs change. No retraining: the released models as they are. `passfail`: Hard and Easy reviews get the Good rating one-hot in the inputs (an app with two buttons); `notime`: every review's answer time set to missing (duration 0, as missing times are encoded); `passfail-notime`: both. Labels (pass / fail) are unchanged.

## 732 (rule) pool, 722 users

| model | inputs | curve LogLoss | curve vs normal, points [95% CI] | instant LogLoss | instant vs normal, points [95% CI] |
| --- | --- | --- | --- | --- | --- |
| small model (released: the leak-free small model (SM72s), cool-down from epoch 25) | normal | 0.30003 | (reference) | 0.27654 | (reference) |
|  | pass / fail ratings only (`passfail`) | 0.30390 | +1.30 [+1.07, +1.57] | 0.27919 | +0.96 [+0.85, +1.09] |
|  | no answer times (`notime`) | 0.32560 | +8.55 [+8.01, +9.10] | 0.30700 | +11.04 [+10.52, +11.62] |
|  | pass / fail, no times (`passfail-notime`) | 0.33182 | +10.63 [+9.95, +11.39] | 0.31109 | +12.53 [+12.08, +13.01] |
| big model (1 leak-free epoch) | normal | 0.29897 | (reference) | 0.27576 | (reference) |
|  | pass / fail ratings only (`passfail`) | 0.30321 | +1.42 [+1.18, +1.69] | 0.27838 | +0.95 [+0.84, +1.07] |
|  | no answer times (`notime`) | 0.34210 | +14.43 [+13.21, +15.84] | 0.32635 | +18.35 [+17.55, +19.23] |
|  | pass / fail, no times (`passfail-notime`) | 0.34947 | +16.89 [+15.73, +18.26] | 0.35295 | +27.99 [+26.82, +29.15] |

Small model minus big model (1 leak-free epoch) under the same inputs, points [95% CI] (positive = the small model is worse):

| inputs | curve gap | instant gap | curve gap change vs normal | instant gap change vs normal |
| --- | --- | --- | --- | --- |
| normal | +0.36 [+0.20, +0.53] | +0.28 [+0.25, +0.31] | | |
| pass / fail ratings only (`passfail`) | +0.23 [+0.02, +0.49] | +0.29 [+0.27, +0.32] | -0.12 [-0.33, +0.06] | +0.01 [-0.01, +0.03] |
| no answer times (`notime`) | -5.52 [-6.49, -4.66] | -7.02 [-7.76, -6.34] | -5.87 [-6.93, -4.95] | -7.30 [-8.05, -6.62] |
| pass / fail, no times (`passfail-notime`) | -5.90 [-6.81, -5.13] | -15.18 [-16.30, -14.13] | -6.26 [-7.20, -5.45] | -15.46 [-16.58, -14.41] |

## 732+long pool, 725 users

| model | inputs | curve LogLoss | curve vs normal, points [95% CI] | instant LogLoss | instant vs normal, points [95% CI] |
| --- | --- | --- | --- | --- | --- |
| small model (released: the leak-free small model (SM72s), cool-down from epoch 25) | normal | 0.30199 | (reference) | 0.27742 | (reference) |
|  | pass / fail ratings only (`passfail`) | 0.30550 | +1.17 [+0.74, +1.56] | 0.27996 | +0.92 [+0.78, +1.07] |
|  | no answer times (`notime`) | 0.32814 | +8.68 [+8.14, +9.25] | 0.30841 | +11.21 [+10.68, +11.79] |
|  | pass / fail, no times (`passfail-notime`) | 0.33430 | +10.73 [+10.00, +11.52] | 0.31264 | +12.73 [+12.24, +13.22] |
| big model (1 leak-free epoch) | normal | 0.30123 | (reference) | 0.27658 | (reference) |
|  | pass / fail ratings only (`passfail`) | 0.30468 | +1.15 [+0.52, +1.61] | 0.27914 | +0.93 [+0.81, +1.06] |
|  | no answer times (`notime`) | 0.34564 | +14.74 [+13.48, +16.19] | 0.32801 | +18.60 [+17.69, +19.64] |
|  | pass / fail, no times (`passfail-notime`) | 0.35182 | +16.80 [+15.62, +18.15] | 0.35325 | +27.72 [+26.52, +28.86] |

Small model minus big model (1 leak-free epoch) under the same inputs, points [95% CI] (positive = the small model is worse):

| inputs | curve gap | instant gap | curve gap change vs normal | instant gap change vs normal |
| --- | --- | --- | --- | --- |
| normal | +0.25 [+0.02, +0.47] | +0.31 [+0.26, +0.38] | | |
| pass / fail ratings only (`passfail`) | +0.27 [+0.04, +0.53] | +0.30 [+0.27, +0.33] | +0.02 [-0.25, +0.30] | -0.01 [-0.06, +0.03] |
| no answer times (`notime`) | -5.81 [-6.84, -4.87] | -7.09 [-7.97, -6.31] | -6.06 [-7.10, -5.12] | -7.39 [-8.33, -6.59] |
| pass / fail, no times (`passfail-notime`) | -5.82 [-6.75, -5.00] | -14.68 [-15.84, -13.56] | -6.07 [-7.03, -5.23] | -14.99 [-16.15, -13.86] |

Inputs checked on CPU: artifacts/small-model/sm31/check-cpu-sm31-inputs.json.
