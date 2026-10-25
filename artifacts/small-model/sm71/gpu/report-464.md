# SM71 GPU, 464: 458 users; benchmark test reviews all 19,833,530, existing 16,007,007, deleted 3,826,523 (review-weighted, user bootstrap 2,000, seed 0, 95% CI)

## LogLoss

| series | all curve | all instant | existing curve | existing instant | deleted curve | deleted instant |
|---|---|---|---|---|---|---|
| big/keep | 0.295839 | 0.272228 | 0.292059 | 0.268918 | 0.311651 | 0.286077 |
| big/neighbour | 0.295915 | 0.272604 | 0.292016 | 0.269007 | 0.312228 | 0.287651 |
| small/keep | 0.296797 | 0.273506 | 0.29318 | 0.270239 | 0.311927 | 0.287174 |
| small/neighbour | 0.297039 | 0.273837 | 0.293283 | 0.270337 | 0.312754 | 0.288476 |

## Change vs the same model with today's inputs (delta [95% CI], % of today)

| series | subset | curve | instant |
|---|---|---|---|
| big/neighbour | all | +0.000076 [-0.000342, +0.000405] (+0.03%) | +0.000376 [+0.000288, +0.000475] (+0.14%) |
| big/neighbour | deleted | +0.000577 [-0.000669, +0.001542] (+0.18%) | +0.001574 [+0.001329, +0.001828] (+0.55%) |
| big/neighbour | existing | -0.000043 [-0.000430, +0.000249] (-0.01%) | +0.000090 [+0.000059, +0.000124] (+0.03%) |
| small/neighbour | all | +0.000242 [+0.000064, +0.000430] (+0.08%) | +0.000331 [+0.000239, +0.000432] (+0.12%) |
| small/neighbour | deleted | +0.000828 [+0.000417, +0.001230] (+0.27%) | +0.001301 [+0.001009, +0.001648] (+0.45%) |
| small/neighbour | existing | +0.000103 [-0.000066, +0.000298] (+0.04%) | +0.000098 [+0.000064, +0.000136] (+0.04%) |

## Small vs big model (% vs big model [95% CI]; change vs today's all-review gap in points)

| pair / inputs | subset | curve gap | instant gap | curve change | instant change |
|---|---|---|---|---|---|
| small-big/keep | all | +0.32% [+0.16, +0.50] | +0.47% [+0.41, +0.54] | - | - |
| small-big/keep | deleted | +0.09% [-0.24, +0.38] | +0.38% [+0.32, +0.46] | -0.23 [-0.53, +0.01] | -0.09 [-0.16, -0.01] |
| small-big/keep | existing | +0.38% [+0.20, +0.58] | +0.49% [+0.42, +0.57] | +0.06 [-0.00, +0.13] | +0.02 [+0.00, +0.04] |
| small-big/neighbour | all | +0.38% [+0.17, +0.64] | +0.45% [+0.39, +0.52] | +0.06 [-0.04, +0.18] | -0.02 [-0.03, -0.00] |
| small-big/neighbour | deleted | +0.17% [-0.04, +0.41] | +0.29% [+0.22, +0.37] | -0.15 [-0.36, +0.06] | -0.18 [-0.27, -0.10] |
| small-big/neighbour | existing | +0.43% [+0.20, +0.73] | +0.49% [+0.42, +0.58] | +0.11 [-0.01, +0.26] | +0.03 [+0.00, +0.05] |
