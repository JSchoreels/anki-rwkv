# The small model in the review simulator (SM41): diagnostic tables

20 users x 1 split (diag/sample.json), 60 days, target 0.90, instant-head desired retention (ihr-0.90) and FSRS-7 desired retention (fsrs7-0.90), 4 seeds each, daily shared-stream re-anchoring. Per-review logs from `scripts/sm41_engine.py run --log-dir`. Days 30-59 unless stated. 'True recall' = the instant head's P(pass) at the actual show moment: the simulated learner draws the grade from it. Seeds are averaged per user; mean ± 95% CI over users (t); 'paired' = FSRS-7 minus instant head per user.

## 1. Lanes in the sample (days 30-59)

| metric | instant head | FSRS-7 | FSRS-7 - instant head (paired) |
|---|---|---|---|
| realised retention | 0.869 ± 0.024 | 0.873 ± 0.030 | +0.004 ± 0.027 |
| mean true recall | 0.869 ± 0.025 | 0.873 ± 0.030 | +0.005 ± 0.028 |
| the policy's aim (its own prediction at the chosen interval) | 0.870 ± 0.018 | 0.868 ± 0.019 | -0.002 ± 0.006 |
| later-day reviews | 1819 ± 564 | 1946 ± 604 | +128 ± 315 |
| minutes (every review) | 765 ± 395 | 702 ± 360 | -64 ± 99 |
| memorised at day 60 (instant head) | 1102 ± 351 | 1090 ± 348 | -12 ± 13 |
| memorised at day 60 (curve head) | 1095 ± 335 | 1090 ± 335 | -5 ± 11 |

Cost gap at target 0.90 in the sample (instant head / FSRS-7 - 1, minutes 30-59 per memorised card): +8.3 ± 9.6 %.

## 2. True recall at review time (H1, H5): later-day reviews, days 30-59

| statistic | instant head | FSRS-7 | paired difference |
|---|---|---|---|
| mean | 0.869 ± 0.025 | 0.873 ± 0.030 | +0.005 ± 0.028 |
| SD over reviews | 0.094 ± 0.028 | 0.104 ± 0.023 | +0.010 ± 0.015 |
| 10th percentile | 0.761 ± 0.068 | 0.744 ± 0.062 | -0.016 ± 0.060 |
| 25th percentile | 0.841 ± 0.038 | 0.839 ± 0.041 | -0.003 ± 0.050 |
| median | 0.896 ± 0.015 | 0.903 ± 0.027 | +0.007 ± 0.027 |
| 75th percentile | 0.928 ± 0.011 | 0.943 ± 0.018 | +0.015 ± 0.016 |
| 90th percentile | 0.946 ± 0.011 | 0.965 ± 0.013 | +0.019 ± 0.012 |
| share below 0.80 | 0.139 ± 0.067 | 0.185 ± 0.081 | +0.046 ± 0.077 |
| share below 0.70 | 0.074 ± 0.044 | 0.089 ± 0.047 | +0.015 ± 0.046 |
| SD over cards of the card's mean true recall | 0.063 ± 0.020 | 0.082 ± 0.017 | +0.019 ± 0.009 |
| 10th percentile over cards | 0.807 ± 0.046 | 0.771 ± 0.044 | -0.036 ± 0.036 |
| 90th percentile over cards | 0.936 ± 0.013 | 0.956 ± 0.017 | +0.020 ± 0.015 |
| SD over cards of the realised pass share (cards with >= 2 reviews) | 0.190 ± 0.015 | 0.170 ± 0.022 | -0.020 ± 0.023 |

## 3. Intervals

| statistic, days | instant head | FSRS-7 | paired difference |
|---|---|---|---|
| median interval chosen (every card leaving the day, days 30-59) | 63.3 ± 103.6 | 230.4 ± 458.1 | +167.1 ± 355.0 |
| 90th percentile of the same | 217.8 ± 164.0 | 334.6 ± 553.6 | +116.8 ± 485.5 |
| mean of the same | 107.4 ± 100.3 | 178.6 ± 318.8 | +71.2 ± 226.9 |
| median interval chosen after a later-day review | 63.7 ± 103.6 | 17.3 ± 9.9 | -46.4 ± 95.9 |
| 90th percentile of the same | 125.2 ± 104.3 | 135.1 ± 156.8 | +9.9 ± 63.1 |
| median elapsed days at later-day reviews | 10.8 ± 4.9 | 9.4 ± 4.4 | -1.5 ± 3.7 |
| 90th percentile of the same | 34.2 ± 11.8 | 31.8 ± 9.3 | -2.4 ± 7.8 |

The 1-day floor (reviews at 1 day whose aim was < 0.88, i.e. P(pass) at 1 day was already below the target): share of later-day reviews instant head 15.7 ± 7.4 %, FSRS-7 19.4 ± 8.5 %; their mean true recall instant head 0.803, FSRS-7 0.827; share of later-day minutes instant head 19.4 ± 9.2 %, FSRS-7 23.6 ± 10.1 %.

## 4. Where FSRS-7 saves (days 30-59)

Card difficulty = tercile of the card's real later-day Again share over its whole real history (per user).

| card group | later-day reviews: instant head / FSRS-7 | paired difference | minutes: paired difference | memorised: paired difference | mean true recall: instant head / FSRS-7 |
|---|---|---|---|---|---|
| easiest third | 623 / 692 | +69 ± 134 | -12.6 ± 34.9 | -3.6 ± 4.9 | 0.869 / 0.873 |
| middle third | 406 / 472 | +66 ± 97 | -12.1 ± 27.9 | -0.6 ± 3.2 | 0.863 / 0.865 |
| hardest third | 717 / 712 | -5 ± 107 | -34.9 ± 34.2 | -6.9 ± 7.6 | 0.869 / 0.871 |
| thin history (< 2 real later-day reviews) | 73 / 71 | -2 ± 14 | -4.2 ± 7.9 | -0.4 ± 0.7 | 0.878 / 0.889 |

| elapsed days at the review | later-day reviews: instant head / FSRS-7 | paired difference | minutes: paired difference |
|---|---|---|---|
| 1 | 613 / 714 | +102 ± 177 | -20.4 ± 53.6 |
| 2-3 | 269 / 258 | -12 ± 67 | -16.1 ± 23.8 |
| 4-7 | 325 / 343 | +18 ± 173 | -13.3 ± 39.3 |
| 8-15 | 276 / 307 | +31 ± 57 | -2.6 ± 13.8 |
| 16-31 | 197 / 195 | -2 ± 56 | -8.4 ± 26.2 |
| 32-63 | 106 / 101 | -6 ± 24 | -2.8 ± 5.0 |
| 64+ | 32 / 29 | -3 ± 15 | -0.2 ± 3.5 |

## 5. Calibration at the chosen interval (H3), later-day reviews days 30-59

aim = the policy's own predicted recall at the interval it chose (instant head: the counterfactual candidate row; FSRS-7: its own curve); true = the instant head at the actual show moment; realised = the drawn pass share. Means over users (users with reviews in the band).

| elapsed days | instant head: aim / true / realised | instant head aim - true | FSRS-7: aim / true / realised | FSRS-7 aim - true |
|---|---|---|---|---|
| 1 | 0.801 / 0.841 / 0.836 | -4.0 ± 2.2 pts | 0.821 / 0.865 / 0.866 | -4.4 ± 3.1 pts |
| 2-3 | 0.901 / 0.905 / 0.907 | -0.5 ± 0.9 pts | 0.900 / 0.923 / 0.929 | -2.3 ± 2.1 pts |
| 4-7 | 0.900 / 0.905 / 0.911 | -0.5 ± 1.0 pts | 0.900 / 0.904 / 0.908 | -0.4 ± 2.6 pts |
| 8-15 | 0.900 / 0.899 / 0.893 | +0.1 ± 1.0 pts | 0.900 / 0.882 / 0.883 | +1.8 ± 3.4 pts |
| 16-31 | 0.900 / 0.886 / 0.889 | +1.4 ± 1.5 pts | 0.900 / 0.883 / 0.879 | +1.7 ± 2.6 pts |
| 32-63 | 0.900 / 0.867 / 0.874 | +3.3 ± 3.8 pts | 0.900 / 0.855 / 0.848 | +4.5 ± 3.8 pts |
| 64+ | 0.900 / 0.888 / 0.903 | +1.2 ± 4.6 pts | 0.900 / 0.855 / 0.851 | +4.5 ± 6.1 pts |

Overall: instant head aim - true +0.2 ± 1.6 points (mean |aim - true| 5.0 ± 1.9); FSRS-7 aim - true -0.5 ± 2.7 points (mean |aim - true| 7.7 ± 1.6). Curve head at the show moment - true: instant-head lanes +0.3 ± 1.7, FSRS-7 lanes -0.2 ± 1.4 points.

## 6. Memory bought per minute vs true recall at the review (H5), every later-day review, days 0-59

Episode = a later-day review + its same-day follow-ups (relearning steps). Gain = ln(t90 after the day) - ln(t90 before the review), t90 = the curve head's time to 90% recall (the card's 'stability' in the learner). Pooled over both policies' lanes; per user ratio of sums, then mean over users (users with >= 20 episodes in the bin).

| true recall | share of reviews: instant head / FSRS-7 | pass share | gain per review | minutes per review | gain per minute |
|---|---|---|---|---|---|
| 0.00-0.50 | 4.5% / 4.7% | 0.354 ± 0.039 | -0.650 ± 1.480 | 0.552 ± 0.183 | -2.66 ± 5.02 |
| 0.50-0.60 | 2.4% / 2.8% | 0.573 ± 0.023 | +0.187 ± 1.090 | 0.445 ± 0.103 | -0.52 ± 4.06 |
| 0.60-0.70 | 3.9% / 4.8% | 0.663 ± 0.021 | +0.413 ± 0.716 | 0.398 ± 0.083 | +0.76 ± 2.87 |
| 0.70-0.80 | 8.0% / 9.7% | 0.754 ± 0.006 | +0.484 ± 0.485 | 0.353 ± 0.077 | +1.78 ± 2.56 |
| 0.80-0.85 | 8.9% / 9.0% | 0.826 ± 0.005 | +0.515 ± 0.359 | 0.321 ± 0.070 | +2.67 ± 3.11 |
| 0.85-0.90 | 25.5% / 15.6% | 0.881 ± 0.003 | +0.394 ± 0.194 | 0.295 ± 0.063 | +1.70 ± 1.09 |
| 0.90-0.95 | 36.7% / 25.0% | 0.925 ± 0.002 | +0.501 ± 0.137 | 0.265 ± 0.059 | +2.27 ± 0.83 |
| 0.95-1.00 | 10.1% / 28.4% | 0.970 ± 0.003 | +0.577 ± 0.188 | 0.210 ± 0.052 | +3.51 ± 1.42 |

Stratified by the card's stability before the review (the curve head's t90), gain per minute (mean over users with >= 20 episodes in the cell):

| true recall | t90 0-3 d | t90 3-20 d | t90 20+ d |
|---|---|---|---|
| 0.00-0.50 | +2.11 ± 2.56 (16) | -20.15 ± 18.18 (12) | -15.95 ± 26.25 (5) |
| 0.50-0.60 | +5.59 ± 2.90 (18) | -12.50 ± 11.53 (13) | -20.03 ± 21.99 (8) |
| 0.60-0.70 | +7.13 ± 3.31 (18) | -7.79 ± 6.36 (15) | -10.93 ± 8.57 (11) |
| 0.70-0.80 | +8.56 ± 5.04 (19) | -3.97 ± 3.29 (20) | -6.78 ± 4.88 (16) |
| 0.80-0.85 | +9.18 ± 5.91 (19) | -1.60 ± 1.91 (20) | -4.12 ± 3.13 (17) |
| 0.85-0.90 | +7.64 ± 4.30 (19) | -0.05 ± 0.98 (20) | -1.97 ± 1.88 (20) |
| 0.90-0.95 | +7.46 ± 4.28 (19) | +1.34 ± 0.94 (20) | -0.23 ± 1.74 (20) |
| 0.95-1.00 | +11.14 ± 6.42 (19) | +3.13 ± 1.29 (20) | +2.35 ± 2.15 (20) |

Jensen check (each user's quadratic fit of gain and minutes on true recall, evaluated on the user's FSRS-7 true-recall distribution vs a constant at the same mean): gain per minute of the spread / of the constant - 1 = -0.4 ± 7.4 % over 20 users; curvature of gain(R) (quadratic coefficient) -1.08 ± 5.18 (positive = convex).

## 7. Where FSRS-7 lands low (H5)

Per user, over cards: Spearman correlation between FSRS-7's mean overestimate on the card (its aim - true recall, days 30-59) and the card's gain per minute in the instant-head lanes: -0.115 ± 0.100 over 20 users (positive = FSRS-7 overestimates, i.e. reviews late, the cards where a review buys more per minute). Correlation of the overestimate with the card's difficulty tercile: +0.040 ± 0.071.

| card group | FSRS-7 aim - true recall, points |
|---|---|
| easiest third | -0.4 ± 2.6 |
| middle third | +0.4 ± 2.9 |
| hardest third | +0.3 ± 2.5 |
| thin history (< 2 real later-day reviews) | -1.3 ± 3.6 |
## 8. Answer time: where the minutes go (later-day reviews, days 30-59)

Previous answer = the card's previous later-day answer time in the lane, in the user's real later-day answer-time terciles (teacher-forcing lane).

| quantity | instant head | FSRS-7 | FSRS-7 - instant head (paired) |
|---|---|---|---|
| seconds per later-day review | 16.138 | 14.682 | -1.456 ± 1.963 |
| later-day reviews | 1819 | 1946 | +128 ± 315 |
| seconds per review at true recall 0.70-0.80 | 18.046 | 17.452 | -0.594 ± 2.598 |
| seconds per review at true recall 0.80-0.85 | 17.593 | 17.030 | -0.564 ± 1.702 |
| seconds per review at true recall 0.85-0.90 | 16.391 | 15.456 | -0.935 ± 1.538 |
| seconds per review at true recall 0.90-0.95 | 15.266 | 14.073 | -1.192 ± 1.719 |
| seconds per review at true recall 0.95-1.00 | 14.320 | 12.000 | -2.320 ± 2.161 |
| the card's previous answer time, s | 17.068 | 15.017 | -2.051 ± 1.904 |
| previous answer fast (user's fastest third): share of reviews | 0.253 | 0.335 | +0.082 ± 0.050 |
| previous answer fast (user's fastest third): seconds per review | 12.709 | 11.351 | -1.357 ± 1.663 |
| previous answer fast (user's fastest third): true recall | 0.888 | 0.908 | +0.019 ± 0.018 |
| previous answer fast (user's fastest third): median elapsed days | 11.908 | 9.344 | -2.565 ± 2.531 |
| previous answer middle: share of reviews | 0.303 | 0.274 | -0.029 ± 0.025 |
| previous answer middle: seconds per review | 15.225 | 14.000 | -1.225 ± 1.598 |
| previous answer middle: true recall | 0.885 | 0.881 | -0.004 ± 0.018 |
| previous answer middle: median elapsed days | 9.844 | 8.812 | -1.031 ± 2.658 |
| previous answer slow (slowest third): share of reviews | 0.444 | 0.391 | -0.053 ± 0.034 |
| previous answer slow (slowest third): seconds per review | 19.120 | 17.775 | -1.345 ± 1.788 |
| previous answer slow (slowest third): true recall | 0.874 | 0.859 | -0.014 ± 0.022 |
| previous answer slow (slowest third): median elapsed days | 7.963 | 7.094 | -0.869 ± 2.161 |


## 9. From the full-run rows (all users, every target)

### H2: is the instant head's memorised count the issue? (matched realised retention, common range, paired per user, the primary metric's method)

| quantity read off the two curves | instant head / FSRS-7 - 1, % (95% CI) |
|---|---|
| cost, memorised by the instant head (the primary metric) | +14.5 ± 7.4 (median +7.6; 116 users) |
| cost, memorised by the curve head | +15.1 ± 7.5 (median +8.3; 116 users) |
| minutes on days 30-59 | +16.9 ± 7.7 (median +9.3; 116 users) |
| memorised at day 60, instant head | +2.2 ± 1.3 (median +1.3; 116 users) |
| memorised at day 60, curve head | +1.3 ± 0.3 (median +1.0; 116 users) |
| later-day reviews on days 30-59 | +71.0 ± 37.0 (median +8.9; 116 users) |
| seconds per later-day review | +7.9 ± 1.6 (median +7.6; 116 users) |
| minutes of later-day reviews | +78.6 ± 37.7 (median +20.0; 116 users) |
| minutes of same-day steps | +8.9 ± 8.1 (median +3.0; 116 users) |

Memorised by the curve head / by the instant head (mean over splits and targets): instant-head lanes 1.006, FSRS-7 lanes 1.017.

### H4: the minutes gap by 15-day bucket

Minutes per bucket, instant head / FSRS-7 - 1: at matched realised retention (days 30-59 retention as the common axis, the primary metric's method) and at the same target 0.90.

| days | matched realised retention, % (95% CI) | target 0.90, % (95% CI) |
|---|---|---|
| 0-14 | +120.1 ± 200.7 (median +8.9) | +23.4 ± 13.6 (median +11.8) |
| 15-29 | +58.3 ± 82.9 (median +10.1) | +23.1 ± 8.5 (median +10.8) |
| 30-44 | +18.1 ± 6.7 (median +10.3) | +18.9 ± 7.0 (median +12.0) |
| 45-59 | +24.6 ± 15.8 (median +10.8) | +27.9 ± 18.9 (median +13.3) |

Change from days 30-44 to days 45-59 (matched): +6.5 ± 12.9 points.

