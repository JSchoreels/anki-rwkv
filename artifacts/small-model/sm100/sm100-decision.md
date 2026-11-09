# The show-time error sensitivity test (SM100): decision

Verdict: **not negligible**. the show-time instant-only small model (SM96), instant head, 722 users of the 732 set, points = % of the big model (1 leak-free epoch) normal-input instant LogLoss 0.275757; negligible needs the 95% CI of both +10 s and +60 s inside (-0.05, +0.05).

| show-time interval shift | instant LogLoss shifted | unshifted | points [95% CI] | users better / worse | mean abs dR | share abs dR > 0.01 |
|---|---|---|---|---|---|---|
| +10s (primary) | 0.276534 | 0.276287 | +0.090 [+0.069, +0.114] | 139 / 564 | 0.001213 | 0.02711 |
| +60s (primary) | 0.277381 | 0.276287 | +0.397 [+0.331, +0.476] | 62 / 652 | 0.004053 | 0.06703 |
| +300s | 0.27934 | 0.276287 | +1.107 [+0.951, +1.309] | 35 / 683 | 0.008344 | 0.10741 |

For scale, the model's own show-time cost (show-time - normal inputs): +0.055 [+0.042, +0.072] points.

## By show-time interval bucket (dumps)

| shift | bucket | share of reviews | mean abs dR | LogLoss change, % of bucket | points contribution |
|---|---|---|---|---|---|
| +10s | < 10 min | 0.1402 | 0.008504 | +0.559 | +0.089 |
| +10s | 10 min - 1 day | 0.2196 | 9e-05 | +0.001 | +0.000 |
| +10s | >= 1 day | 0.6402 | 1e-06 | +0.000 | +0.000 |
| +60s | < 10 min | 0.1402 | 0.028081 | +2.473 | +0.396 |
| +60s | 10 min - 1 day | 0.2196 | 0.000507 | +0.004 | +0.001 |
| +60s | >= 1 day | 0.6402 | 6e-06 | +0.000 | +0.000 |
| +300s | < 10 min | 0.1402 | 0.056115 | +6.864 | +1.098 |
| +300s | 10 min - 1 day | 0.2196 | 0.002081 | +0.039 | +0.009 |
| +300s | >= 1 day | 0.6402 | 2.9e-05 | +0.000 | +0.000 |

## Timeout subset (dumps + revlogs)

553 capped users (cap values: {'60000': 499, '120000': 12, '300000': 5, '180000': 12, '200000': 2, '1200000': 1, '90000': 9, '40000': 1, '100000': 1, '77000': 1, '70000': 3, '160000': 1, '600000': 3, '80000': 1, '500000': 1, '999000': 1}); timeout reviews 1,010,999 = 3.36% of 30,126,145 benchmark reviews; failure rate 0.2007 on timeout reviews vs 0.1435 on the capped users' other reviews.

| reviews | inputs a - b | a | b | LogLoss delta [95% CI] | % of b [95% CI] |
|---|---|---|---|---|---|
| timeout | show-time - normal | 0.3369 | 0.336281 | +0.000619 [+0.000431, +0.000850] | +0.184 [+0.131, +0.243] |
| timeout | +60 s - show-time | 0.336298 | 0.3369 | -0.000603 [-0.000834, -0.000417] | -0.179 [-0.237, -0.126] |
| capped users' other | show-time - normal | 0.276217 | 0.276057 | +0.000160 [+0.000111, +0.000220] | +0.058 [+0.040, +0.079] |

Reproduction vs the rows of the show-time instant-only small model (SM96): {"e2s-query": {"users": 722, "max_abs_instant_logloss_diff": 0.0}, "normal": {"users": 722, "max_abs_instant_logloss_diff": 0.0}}; dumps missing 0, review_th mismatches 0.
