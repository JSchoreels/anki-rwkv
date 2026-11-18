# Duration-leak scoring (SM31): normal vs deployment-realistic intervals

Leak-free scoring (deleted cards: neighbour), w8+s8, the 732 set; review-weighted benchmark LogLoss over all test reviews, curve and instant heads (the answer-time head is not evaluated); deltas in points = % of the big model (1 leak-free epoch) with normal inputs on the same users, paired user bootstrap 2,000 resamples, seed 0, 95% CI. Every variant scores the same test reviews with the same labels; only inputs change. No retraining: the released models as they are. Normal inputs: `elapsed_seconds` runs from the card's previous answer to this answer, so it holds the review's own answer time. Show-time intervals (`e2s-query`, deployment-realistic): everything asked before the answer uses the interval from the card's previous answer to the moment the card is shown, floor(max(elapsed_seconds - duration, 0)): the instant query row's interval, its sin / cos, the cumulative interval (earlier answer-to-answer intervals + the show-time one) and its sin / cos, and the curve query time; update rows (after the answer) keep the full interval. `e2s-all` = the outside-research definition, the same subtraction on every row (update rows and the cumulative interval too).

## 732 (rule) pool, 722 users

| model | inputs | curve LogLoss | curve vs normal, points [95% CI] | instant LogLoss | instant vs normal, points [95% CI] |
| --- | --- | --- | --- | --- | --- |
| small model (released: the leak-free small model (SM72s), cool-down from epoch 25) | normal | 0.30003 | (reference) | 0.27654 | (reference) |
|  | show-time intervals (deployment-realistic) (`e2s-query`) | 0.30041 | +0.13 [+0.11, +0.16] | 0.27772 | +0.43 [+0.34, +0.55] |
|  | end-to-start on every row (the outside-research definition) (`e2s-all`) | 0.30036 | +0.11 [+0.09, +0.14] | 0.27721 | +0.24 [+0.20, +0.29] |
| big model (1 leak-free epoch) | normal | 0.29897 | (reference) | 0.27576 | (reference) |
|  | show-time intervals (deployment-realistic) (`e2s-query`) | 0.29958 | +0.20 [+0.17, +0.26] | 0.27716 | +0.51 [+0.41, +0.63] |
|  | end-to-start on every row (the outside-research definition) (`e2s-all`) | 0.29947 | +0.17 [+0.14, +0.21] | 0.27657 | +0.30 [+0.25, +0.36] |

Small model minus big model (1 leak-free epoch) under the same inputs, points [95% CI] (positive = the small model is worse):

| inputs | curve gap | instant gap | curve gap change vs normal | instant gap change vs normal |
| --- | --- | --- | --- | --- |
| normal | +0.36 [+0.20, +0.53] | +0.28 [+0.25, +0.31] | | |
| show-time intervals (deployment-realistic) (`e2s-query`) | +0.28 [+0.12, +0.46] | +0.20 [+0.17, +0.24] | -0.08 [-0.10, -0.06] | -0.08 [-0.10, -0.06] |
| end-to-start on every row (the outside-research definition) (`e2s-all`) | +0.30 [+0.14, +0.48] | +0.23 [+0.20, +0.26] | -0.06 [-0.08, -0.04] | -0.05 [-0.07, -0.04] |

## 732+long pool, 725 users

| model | inputs | curve LogLoss | curve vs normal, points [95% CI] | instant LogLoss | instant vs normal, points [95% CI] |
| --- | --- | --- | --- | --- | --- |
| small model (released: the leak-free small model (SM72s), cool-down from epoch 25) | normal | 0.30199 | (reference) | 0.27742 | (reference) |
|  | show-time intervals (deployment-realistic) (`e2s-query`) | 0.30236 | +0.12 [+0.10, +0.15] | 0.27858 | +0.42 [+0.33, +0.54] |
|  | end-to-start on every row (the outside-research definition) (`e2s-all`) | 0.30231 | +0.11 [+0.09, +0.13] | 0.27808 | +0.24 [+0.20, +0.28] |
| big model (1 leak-free epoch) | normal | 0.30123 | (reference) | 0.27658 | (reference) |
|  | show-time intervals (deployment-realistic) (`e2s-query`) | 0.30184 | +0.20 [+0.16, +0.25] | 0.27796 | +0.50 [+0.40, +0.62] |
|  | end-to-start on every row (the outside-research definition) (`e2s-all`) | 0.30173 | +0.17 [+0.14, +0.20] | 0.27738 | +0.29 [+0.24, +0.34] |

Small model minus big model (1 leak-free epoch) under the same inputs, points [95% CI] (positive = the small model is worse):

| inputs | curve gap | instant gap | curve gap change vs normal | instant gap change vs normal |
| --- | --- | --- | --- | --- |
| normal | +0.25 [+0.02, +0.47] | +0.31 [+0.26, +0.38] | | |
| show-time intervals (deployment-realistic) (`e2s-query`) | +0.17 [-0.05, +0.40] | +0.23 [+0.17, +0.30] | -0.08 [-0.10, -0.06] | -0.08 [-0.10, -0.06] |
| end-to-start on every row (the outside-research definition) (`e2s-all`) | +0.19 [-0.04, +0.41] | +0.25 [+0.20, +0.33] | -0.06 [-0.08, -0.04] | -0.05 [-0.07, -0.04] |

Inputs checked on CPU: artifacts/small-model/sm31/check-cpu-sm31-inputs.json (which columns move on which rows; normal bit-identical to the scorer) and cpu-proof/ (default scorer rows identical to the previous scorer).
