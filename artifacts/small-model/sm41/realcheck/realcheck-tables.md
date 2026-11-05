# The real-data check of the time-aware target (SM41r): tables

117 users, 304 splits, 890,493 real later-day reviews in the 60-day windows (739,939 with a previous later-day answer); 1,392,736 window reviews in all. Strata: the rule's ln(last later-day answer ms / user's pre-split median), within-user terciles (fast / mid / slow). Recall bands: the instant head's P(pass) at the real show moment. Every number: per user first, then mean ± 95% t CI over users.

## 1. Calibration of the instant head by previous answer time

Realised pass rate - predicted P(pass), points (per user, mean ± CI over users). FSRS-7 (pre-split refit) on the same reviews for reference.

| stratum | reviews | mean ln ratio | predicted | realised | instant head: realised - predicted | FSRS-7: realised - predicted |
|---|---|---|---|---|---|---|
| fast | 245,557 | -0.83 | 0.907 | 0.908 | +0.1 ± 0.1 | +1.6 ± 0.6 |
| mid | 243,577 | -0.01 | 0.868 | 0.869 | +0.1 ± 0.2 | -0.4 ± 0.6 |
| slow | 250,805 | +0.95 | 0.841 | 0.841 | +0.0 ± 0.2 | -1.7 ± 0.8 |

Slow minus fast, instant head: -0.0 ± 0.2 points (117 users; > 0 = the model is too pessimistic about slow cards relative to fast ones).

Slow minus fast, FSRS-7: -3.3 ± 0.6 points (117 users; > 0 = the model is too pessimistic about slow cards relative to fast ones).

By recall band (instant head, realised - predicted, points):

| band | fast | mid | slow | slow - fast |
|---|---|---|---|---|
| < 0.70 | +1.0 ± 1.3 (17,588) | +0.8 ± 1.0 (28,834) | +0.4 ± 1.0 (40,302) | -0.2 ± 1.6 (93 users) |
| 0.70-0.80 | +0.9 ± 1.1 (13,923) | +0.5 ± 0.8 (22,523) | +0.5 ± 0.8 (27,898) | -0.7 ± 1.3 (105 users) |
| 0.80-0.85 | +0.1 ± 1.0 (13,001) | -0.1 ± 0.9 (19,313) | +0.2 ± 0.7 (22,493) | +0.1 ± 1.3 (105 users) |
| 0.85-0.90 | +0.2 ± 0.6 (22,593) | +0.2 ± 0.5 (29,444) | -0.3 ± 0.5 (31,908) | -0.6 ± 0.8 (113 users) |
| 0.90-0.95 | +0.2 ± 0.3 (41,486) | +0.1 ± 0.4 (45,638) | +0.1 ± 0.3 (46,425) | -0.1 ± 0.5 (114 users) |
| >= 0.95 | -0.0 ± 0.1 (136,966) | -0.1 ± 0.2 (97,825) | -0.3 ± 0.5 (81,779) | -0.2 ± 0.5 (115 users) |

Recalibration slope delta (per user logistic: pass ~ logit(pred) + delta x ln ratio, logit per unit ln ratio): instant head -0.009 ± 0.015 (117 users), FSRS-7 -0.198 ± 0.031 (117 users). A positive delta for the instant head = it over-penalises slow previous answers; the rule's beta then partly corrects a miscalibration (its share = delta / beta).

## 2. Cost of a review by previous answer time

Per review, standardised to the pooled recall-band mix (so slow and fast are compared at the same predicted recall):

| quantity | fast | mid | slow | slow vs fast, % |
|---|---|---|---|---|
| seconds of the review | 12.18 ± 1.14 | 16.37 ± 1.62 | 20.98 ± 1.91 | +77.7 ± 8.4 |
| seconds incl. same-day steps | 13.63 ± 1.25 | 18.26 ± 1.76 | 23.66 ± 2.12 | +79.5 ± 8.7 |
| answer-time head E[s | real grade] | 12.21 ± 1.14 | 16.29 ± 1.60 | 20.94 ± 1.86 | +74.3 ± 6.7 |
| answer-time head E[s] (grade-mixed) | 12.23 ± 1.15 | 16.30 ± 1.60 | 20.93 ± 1.86 | +74.3 ± 6.7 |

Lapse cost (same-day steps after a later-day Again), minutes per lapse: fast 0.18 ± 0.02, mid 0.23 ± 0.03, slow 0.32 ± 0.04; slow vs fast +92.2 ± 14.4 % (115 users).

Answer-time head bias, real ln s - E[ln s | real grade] (0 = calibrated): fast -0.016 ± 0.013, mid +0.000 ± 0.011, slow +0.003 ± 0.011.

Elasticity of answer time on the previous ln ratio (per-user least-squares slope of ln(time / median) on the ln ratio; 1 = fully persistent, 0 = no memory of the last answer):

- this review, real: +0.395 ± 0.024 (within recall band x card-difficulty tercile: +0.337 ± 0.025);
- this review, the answer-time head's E[s]: +0.368 ± 0.020;
- the next later-day review, real: +0.355 ± 0.024.

## 3. Cost per unit of memory gain vs recall at review, by stratum

Pooled over users (rows), per stratum and recall band: reviews, pass rate, seconds incl. same-day steps, curve-head ln t90 growth, the real next interval survived (days, next later-day review in the window; 0 if it failed), and seconds per unit ln t90 growth.

| stratum | band | reviews | pass | s/review | d ln t90 | next interval survived, d | s per d ln t90 |
|---|---|---|---|---|---|---|---|
| fast | < 0.70 | 17,588 | 0.484 | 24.3 | -0.917 | 4.0 | nan |
| fast | 0.70-0.80 | 13,923 | 0.759 | 17.5 | -0.101 | 6.4 | nan |
| fast | 0.80-0.85 | 13,001 | 0.830 | 15.5 | +0.155 | 7.4 | 99.8 |
| fast | 0.85-0.90 | 22,593 | 0.879 | 14.3 | +0.188 | 8.5 | 76.2 |
| fast | 0.90-0.95 | 41,486 | 0.929 | 12.9 | +0.284 | 9.9 | 45.3 |
| fast | >= 0.95 | 136,966 | 0.983 | 9.8 | +0.440 | 12.1 | 22.3 |
| mid | < 0.70 | 28,834 | 0.506 | 28.3 | -0.801 | 3.9 | nan |
| mid | 0.70-0.80 | 22,523 | 0.754 | 20.7 | +0.117 | 6.1 | 177.8 |
| mid | 0.80-0.85 | 19,313 | 0.824 | 18.8 | +0.292 | 7.2 | 64.6 |
| mid | 0.85-0.90 | 29,444 | 0.877 | 17.8 | +0.342 | 8.2 | 51.9 |
| mid | 0.90-0.95 | 45,638 | 0.929 | 16.6 | +0.399 | 9.1 | 41.5 |
| mid | >= 0.95 | 97,825 | 0.980 | 14.1 | +0.436 | 10.9 | 32.3 |
| slow | < 0.70 | 40,302 | 0.504 | 35.9 | -0.599 | 3.6 | nan |
| slow | 0.70-0.80 | 27,898 | 0.756 | 26.1 | +0.392 | 5.6 | 66.6 |
| slow | 0.80-0.85 | 22,493 | 0.822 | 23.9 | +0.513 | 6.6 | 46.6 |
| slow | 0.85-0.90 | 31,908 | 0.874 | 22.9 | +0.494 | 7.3 | 46.4 |
| slow | 0.90-0.95 | 46,425 | 0.926 | 21.9 | +0.483 | 8.3 | 45.4 |
| slow | >= 0.95 | 81,779 | 0.980 | 20.3 | +0.457 | 10.1 | 44.5 |

### Late (P < 0.85) minus on time (P >= 0.90), within user, matched on the card's state

Cells: user x stratum x curve-head t90 tercile x card-difficulty tercile (both within user); difference of means per cell, weighted by the smaller side; then mean ± CI over users. Slow - fast = the same contrast, slow minus fast, paired per user.

| outcome | fast | mid | slow | slow - fast |
|---|---|---|---|---|
| pass rate | -0.228 ± 0.023 (113) | -0.227 ± 0.021 (117) | -0.231 ± 0.016 (116) | -0.005 ± 0.016 (112) |
| seconds incl. steps | +5.702 ± 0.968 (113) | +6.558 ± 1.060 (117) | +8.126 ± 1.156 (116) | +2.306 ± 0.927 (112) |
| answer ms | +2968.812 ± 675.037 (113) | +3199.799 ± 706.945 (117) | +3571.941 ± 776.587 (116) | +515.906 ± 744.382 (112) |
| d ln t90 (curve head) | -1.210 ± 0.222 (113) | -1.167 ± 0.225 (117) | -1.075 ± 0.194 (116) | +0.127 ± 0.143 (112) |
| next interval survived, d | -1.639 ± 0.799 (113) | -1.138 ± 0.620 (115) | -0.852 ± 0.555 (115) | +0.798 ± 0.875 (111) |
| next review passed | -0.097 ± 0.020 (113) | -0.095 ± 0.016 (115) | -0.090 ± 0.016 (115) | +0.005 ± 0.020 (111) |
| lapse minutes (lapses only) | +0.031 ± 0.021 (100) | +0.018 ± 0.019 (109) | +0.044 ± 0.022 (109) | +0.011 ± 0.026 (95) |

### Cost per unit of memory, late vs on time (primary of point 3)

E = seconds per unit of ln t90 growth, on the late (P < 0.85) and on-time (P >= 0.90) sides of the same state cells; rho = E_late / E_on time (> 1: reviewing late costs more per unit of memory).

- ln rho, fast cards: +0.860 ± 0.627 (30 users); slow cards: +0.914 ± 0.518 (38 users).
- **Interaction ln rho_slow - ln rho_fast: +0.142 ± 0.370** (24 users, median +0.013; < 0 = late is relatively cheaper for slow cards). Users with a non-positive mean gain on a side (left out): 93.
- Positive currency (seconds per day of t90 after the review, no user left out): ln rho fast +2.386 ± 0.290, slow +1.818 ± 0.235; interaction -0.563 ± 0.250 (112 users, median -0.335).
- Holding view: X = seconds per card-day saved by reviewing late, per unit of recall lost: X = (rate_on - rate_late) / (mem_on - mem_late), rate = review seconds incl. steps / the real interval it closed, mem = (1 + P) / 2. Fast: +7.99 ± 4.78 (median 6.63), slow: +21.84 ± 9.36 (median 18.82); ln X_slow - ln X_fast +0.880 ± 0.181 (75 users; > 0 = deferring a slow card saves more seconds per unit of recall lost).

Pooled seconds per unit ln t90 growth by stratum and recall band (sums over users; bands with >= 200 reviews and a positive mean gain):

| stratum | mean x | < 0.70 | 0.70-0.80 | 0.80-0.85 | 0.85-0.90 | 0.90-0.95 | >= 0.95 | optimal P |
|---|---|---|---|---|---|---|---|---|
| fast | -0.83 | — | — | 100 | 76 | 45 | 22 | 0.984 |
| mid | -0.01 | — | 178 | 65 | 52 | 41 | 32 | 0.981 |
| slow | +0.95 | — | 67 | 47 | 46 | 45 | 45 | 0.979 |

Slope of the optimal recall's logit on x (slow vs fast): -0.14 (bootstrap over users 95%: -1.48 .. -0.11); the rule's beta = minus this slope, so this one-step metric implies beta = +0.14.

## 4. Implied beta

117 users. Elasticity of this review's answer time on x: real gamma +0.395 ± 0.024, the answer-time head +0.368 ± 0.020 (head - real -0.027 ± 0.007). Next later-day review: real +0.355 ± 0.024, head +0.330 ± 0.021. Calibration delta -0.009 ± 0.015.

beta_real = beta_sim x gamma_real / gamma_head + delta (bootstrap over users, 95%):

- beta_sim = 2: beta_real = 2.14 (2.10 .. 2.18)
- beta_sim = 3: beta_real = 3.21 (3.15 .. 3.27)

