# SM71 GPU, 732: 722 users; benchmark test reviews all 30,126,145, existing 24,031,663, deleted 6,094,482 (review-weighted, user bootstrap 2,000, seed 0, 95% CI)

## LogLoss

| series | all curve | all instant | existing curve | existing instant | deleted curve | deleted instant |
|---|---|---|---|---|---|---|
| big/keep | 0.299433 | 0.275844 | 0.294922 | 0.271831 | 0.317219 | 0.291668 |
| big/neighbour | 0.299661 | 0.276203 | 0.294933 | 0.271907 | 0.318301 | 0.293142 |
| small/keep | 0.300449 | 0.277142 | 0.296011 | 0.27317 | 0.317948 | 0.292805 |
| small/neighbour | 0.30066 | 0.277447 | 0.296075 | 0.273257 | 0.318741 | 0.293968 |

## Change vs the same model with today's inputs (delta [95% CI], % of today)

| series | subset | curve | instant |
|---|---|---|---|
| big/neighbour | all | +0.000228 [-0.000062, +0.000485] (+0.08%) | +0.000359 [+0.000292, +0.000437] (+0.13%) |
| big/neighbour | deleted | +0.001082 [+0.000287, +0.001811] (+0.34%) | +0.001474 [+0.001311, +0.001655] (+0.51%) |
| big/neighbour | existing | +0.000011 [-0.000270, +0.000250] (+0.00%) | +0.000076 [+0.000052, +0.000105] (+0.03%) |
| small/neighbour | all | +0.000211 [+0.000082, +0.000354] (+0.07%) | +0.000305 [+0.000242, +0.000379] (+0.11%) |
| small/neighbour | deleted | +0.000793 [+0.000428, +0.001178] (+0.25%) | +0.001163 [+0.000965, +0.001402] (+0.40%) |
| small/neighbour | existing | +0.000064 [-0.000044, +0.000204] (+0.02%) | +0.000087 [+0.000062, +0.000118] (+0.03%) |

## Small vs big model (% vs big model [95% CI]; change vs today's all-review gap in points)

| pair / inputs | subset | curve gap | instant gap | curve change | instant change |
|---|---|---|---|---|---|
| small-big/keep | all | +0.34% [+0.21, +0.47] | +0.47% [+0.43, +0.52] | - | - |
| small-big/keep | deleted | +0.23% [+0.01, +0.46] | +0.39% [+0.34, +0.44] | -0.11 [-0.31, +0.08] | -0.08 [-0.13, -0.03] |
| small-big/keep | existing | +0.37% [+0.23, +0.51] | +0.49% [+0.44, +0.55] | +0.03 [-0.02, +0.08] | +0.02 [+0.01, +0.04] |
| small-big/neighbour | all | +0.33% [+0.18, +0.50] | +0.45% [+0.41, +0.49] | -0.01 [-0.08, +0.07] | -0.02 [-0.03, -0.01] |
| small-big/neighbour | deleted | +0.14% [-0.02, +0.33] | +0.28% [+0.23, +0.34] | -0.20 [-0.35, -0.05] | -0.19 [-0.25, -0.13] |
| small-big/neighbour | existing | +0.39% [+0.21, +0.57] | +0.50% [+0.45, +0.55] | +0.05 [-0.04, +0.15] | +0.03 [+0.01, +0.04] |
