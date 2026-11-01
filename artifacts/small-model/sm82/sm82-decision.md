# Response-time prediction (SM82) stage 1: answer-time head vs the same recipe without it

Grade: leak-free scoring, w8+s8, 722 users of the 732 set, review-weighted LogLoss, points of % of the big model (1 leak-free epoch), paired user bootstrap 2,000 seed 0, 95% CI. Arm = the leak-free small model (SM72s) epoch 21 + the quarter-epoch 32k Muon cool-down with the answer-time head; reference = the same cool-down without it (the tiny architecture additions (SM77) reference).

| head | arm | reference | delta | points [95% CI] | users better | bar |
|---|---|---|---|---|---|---|
| curve | 0.30108 | 0.30073 | +0.000350 | +0.12 [+0.04, +0.18] | 247 | <= +0.03 |
| instant | 0.27795 | 0.27722 | +0.000733 | +0.27 [+0.24, +0.29] | 58 | <= +0.03 |

Time (722 users, 21,585,938 test reviews with a recorded time on a later day than the card's previous review, 20,440,806 under the cap): censored 16-bin time LogLoss given the true grade, and the mean |ln error| of the predicted median (uncensored reviews). Baselines get a residual distribution per grade fit on the users' non-test reviews.

| predictor | time LogLoss | mean abs ln error |
|---|---|---|
| answer-time head (given the true grade) | 1.9300 | 0.4625 |
| answer-time head (grade mixed by the grade head) | 1.9547 | 0.4744 |
| baseline: the card's last time | 2.4466 | 0.7222 |
| baseline: the user's running median per grade | 2.3664 | 0.6428 |

| head vs | LogLoss delta [95% CI] | relative | abs ln error delta [95% CI] | relative |
|---|---|---|---|---|
| the card's last time | -0.5167 [-0.5411, -0.4952] | -21.1% | -0.2596 [-0.2753, -0.2455] | -36.0% |
| the user's running median per grade | -0.4365 [-0.4651, -0.4099] | -18.4% | -0.1802 [-0.1950, -0.1663] | -28.0% |

| grade | reviews | head LogLoss | card's last time | user's median per grade | P(under 3.9 s) predicted / actual |
|---|---|---|---|---|---|
| 1 | 2,833,721 | 2.0394 | 2.5233 | 2.3602 | 0.148 / 0.140 |
| 2 | 4,414,931 | 1.9167 | 2.4233 | 2.3682 | 0.222 / 0.214 |
| 3 | 12,582,574 | 1.9181 | 2.4488 | 2.3830 | 0.330 / 0.330 |
| 4 | 1,754,712 | 1.8714 | 2.3664 | 2.2528 | 0.400 / 0.399 |

Threshold: bins 0-6 (under 3.891 s); earlier versions of this file said 5.5 s.

Size: 331,511 -> 339,767 parameters; scorer w8 weight bytes 410,459 -> 419,163 (+8,704 B); state unchanged.

Verdict: **fail-grade** (grade within the bar: False; time head beats both baselines: True).
