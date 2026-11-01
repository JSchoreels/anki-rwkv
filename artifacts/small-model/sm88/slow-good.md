# Slow-Good signal (SM88)

150-user stage-2 sample of the 732 set (150 users), 1,788,213 (Good review j, next review k) pairs: j is Good on a later day than the card's previous review, k is the card's next review on a later day than j. Outcome: k is Again (overall 10.15%; the model's mean predicted lapse 10.26% at scheduling time, 10.32% at review time). Model: the leak-free small model (SM72s) cool-down from epoch 25; scheduling-time prediction = its curve head made at j, evaluated at k's real interval (scheduling-time vs review-time accuracy (SM81) dump, read only); review-time = its instant head at k. Relative time of j = ln(j's time / the user's running median Good time before j) (at least 20 earlier Goods); 'card' = ln(j's time / geometric mean of the card's earlier Good times). Predicted-recall control: 10 buckets of logit(predicted recall), standardised with pooled bucket weights. Intervals: user bootstrap, 2,000 resamples, seed 0.

## Quintiles of relative time (vs the user's running median), scheduling-time prediction

| Group | pairs | actual lapse % | predicted lapse % | actual / predicted [95% CI] |
| --- | --- | --- | --- | --- |
| fastest | 357,643 | 10.50 | 10.20 | 1.030 [0.974, 1.085] |
| middle 2 | 357,642 | 9.96 | 10.16 | 0.981 [0.948, 1.013] |
| middle 3 | 357,643 | 10.03 | 10.22 | 0.981 [0.945, 1.015] |
| middle 4 | 357,642 | 10.18 | 10.32 | 0.986 [0.952, 1.019] |
| slowest | 357,643 | 10.10 | 10.35 | 0.976 [0.950, 1.001] |

Slowest minus fastest at equal predicted recall: actual lapse -0.41 points [-1.10, +0.25]; residual (actual - predicted) -0.56 points [-1.12, -0.01].

## Same, review-time (instant) prediction at k

| Group | pairs | actual lapse % | predicted lapse % | actual / predicted [95% CI] |
| --- | --- | --- | --- | --- |
| fastest | 357,643 | 10.51 | 10.56 | 0.995 [0.977, 1.011] |
| middle 2 | 357,642 | 10.04 | 10.25 | 0.979 [0.968, 0.993] |
| middle 3 | 357,643 | 10.07 | 10.25 | 0.982 [0.969, 0.996] |
| middle 4 | 357,642 | 10.15 | 10.29 | 0.986 [0.977, 0.996] |
| slowest | 357,643 | 10.07 | 10.29 | 0.979 [0.972, 0.987] |

Slowest minus fastest at equal predicted recall: actual lapse -0.44 points [-0.80, -0.11]; residual (actual - predicted) -0.16 points [-0.35, +0.05].

## Within-user quintiles of relative time, scheduling-time prediction

| Group | pairs | actual lapse % | predicted lapse % | actual / predicted [95% CI] |
| --- | --- | --- | --- | --- |
| fastest | 357,703 | 10.37 | 10.20 | 1.017 [0.966, 1.073] |
| middle 2 | 357,645 | 10.17 | 10.27 | 0.991 [0.961, 1.022] |
| middle 3 | 357,641 | 10.14 | 10.24 | 0.991 [0.955, 1.024] |
| middle 4 | 357,645 | 9.98 | 10.24 | 0.975 [0.939, 1.008] |
| slowest | 357,579 | 10.13 | 10.34 | 0.979 [0.952, 1.006] |

Slowest minus fastest at equal predicted recall: actual lapse -0.24 points [-0.83, +0.35]; residual (actual - predicted) -0.39 points [-0.92, +0.08].

## Quintiles of relative time vs the card's own earlier Good times, scheduling-time prediction

| Group | pairs | actual lapse % | predicted lapse % | actual / predicted [95% CI] |
| --- | --- | --- | --- | --- |
| fastest | 341,594 | 10.41 | 10.27 | 1.013 [0.968, 1.062] |
| middle 2 | 341,593 | 10.13 | 10.30 | 0.984 [0.948, 1.017] |
| middle 3 | 341,594 | 10.16 | 10.34 | 0.983 [0.946, 1.017] |
| middle 4 | 341,593 | 10.26 | 10.39 | 0.987 [0.954, 1.017] |
| slowest | 341,594 | 10.28 | 10.45 | 0.983 [0.956, 1.008] |

Slowest minus fastest at equal predicted recall: actual lapse -0.13 points [-0.60, +0.24]; residual (actual - predicted) -0.31 points [-0.72, +0.05].

## Terciles (calibration by time tercile), scheduling-time prediction

| Group | pairs | actual lapse % | predicted lapse % | actual / predicted [95% CI] |
| --- | --- | --- | --- | --- |
| fastest | 596,071 | 10.29 | 10.18 | 1.010 [0.967, 1.054] |
| middle 2 | 596,071 | 10.04 | 10.23 | 0.981 [0.946, 1.014] |
| slowest | 596,071 | 10.14 | 10.34 | 0.981 [0.953, 1.007] |

Slowest minus fastest at equal predicted recall: actual lapse -0.15 points [-0.64, +0.30]; residual (actual - predicted) -0.30 points [-0.68, +0.06].

## Terciles, review-time prediction

| Group | pairs | actual lapse % | predicted lapse % | actual / predicted [95% CI] |
| --- | --- | --- | --- | --- |
| fastest | 596,071 | 10.34 | 10.46 | 0.988 [0.975, 1.002] |
| middle 2 | 596,071 | 10.07 | 10.25 | 0.983 [0.974, 0.993] |
| slowest | 596,071 | 10.10 | 10.29 | 0.981 [0.975, 0.988] |

Slowest minus fastest at equal predicted recall: actual lapse -0.24 points [-0.50, -0.02]; residual (actual - predicted) -0.07 points [-0.21, +0.07].

## Logistic regression: k Again on logit(predicted recall) + relative time of j

Coefficient (user-clustered robust SE); relative time in ln units.

| Reference time | Control | pairs | SD | slope on logit(pred), alone | slope with time | time coef per ln unit | time coef, user fixed effects |
| --- | --- | --- | --- | --- | --- | --- | --- |
| user running median | scheduling-time | 1,788,213 | 0.92 | -1.009 (0.015) | -1.011 (0.015) | -0.027 (0.014) | -0.022 (0.014) |
| user running median | review-time | 1,788,213 | 0.92 | -1.009 (0.007) | -1.009 (0.007) | -0.007 (0.004) | -0.007 (0.004) |
| card own Goods | scheduling-time | 1,707,968 | 0.90 | -1.009 (0.015) | -1.010 (0.015) | -0.017 (0.012) | -0.015 (0.010) |
| card own Goods | review-time | 1,707,968 | 0.90 | -1.009 (0.007) | -1.009 (0.007) | -0.004 (0.004) | -0.005 (0.004) |
