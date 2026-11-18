# The show-time two-head cool-down for Tiny (SM124): decision

Verdict: **go** (gain on: curve, instant; regression on: no head; 722 users of the 732 set; % of the big model (1 leak-free epoch) normal-input LogLoss of each head, curve 0.298966, instant 0.275757; paired user bootstrap 2,000, seed 0, 95% CI).

Rule (pre-registered): per head, gain = CI below 0 and at most -0.05%; regression = +0.05% or more, or CI above 0; go = a gain on at least one head and a regression on neither.

| the show-time two-head cool-down for Tiny (SM124) minus the Tiny two-head model (SM106), show-time | new | ref | % [95% CI] | better / worse | teacher gap closed |
|---|---|---|---|---|---|
| curve | 0.300369 | 0.300676 | -0.103% [-0.142, -0.067] | 498 / 224 | 0.244 (the show-time big model (SM91s2) minus SM106: -0.422% [-0.597, -0.257]) |
| instant | 0.277105 | 0.277369 | -0.096% [-0.106, -0.087] | 609 / 111 | 0.223 (the show-time big model (SM91s2) minus SM106: -0.430% [-0.467, -0.394]) |

## Read-out (732 set, %)

| minus | inputs | curve % [95% CI] | instant % [95% CI] | instant better / worse |
|---|---|---|---|---|
| the show-time big model (SM91s2) (the teacher) | show-time | +0.319% [+0.163, +0.483] | +0.334% [+0.304, +0.365] | 100 / 622 |
| the release (SM63) | show-time | -0.015% [-0.105, +0.097] | -0.222% [-0.320, -0.147] | 445 / 277 |
| the show-time retrain of the release (SM92) (Small) | show-time | +0.004% [-0.055, +0.053] | +0.083% [+0.066, +0.101] | 250 / 469 |
| the show-time two-head cool-down for Small (SM118) (Small, the same recipe) | show-time | +0.118% [+0.057, +0.175] | +0.183% [+0.164, +0.204] | 134 / 588 |
| the big model (1 leak-free epoch) (smaller-model bar, show-time: curve <= +0.5%, instant <= +0.4%) | show-time | +0.264% [+0.087, +0.469] | -0.019% [-0.130, +0.065] | 286 / 435 |
| the Tiny Instant-Only model (SM101) (instant only) | show-time | (not trained) | +0.140% [+0.125, +0.156] | 145 / 577 |
| the longer cool-down for Tiny (SM115) (instant only) | show-time | (not trained) | +0.239% [+0.219, +0.261] | 86 / 636 |
| the big model (1 leak-free epoch) (the smaller-model bar: curve <= +0.5%, instant <= +0.4%) | normal | +0.437% [+0.268, +0.639] | +0.458% [+0.421, +0.496] | 75 / 647 |
| the show-time two-head cool-down for Small (SM118) (Small, the same recipe) | normal | +0.119% [+0.058, +0.176] | +0.182% [+0.163, +0.203] | 130 / 592 |
| the new model on normal inputs (its show-time cost) | show-time vs normal | +0.032% [+0.026, +0.041] | +0.031% [+0.021, +0.043] | 192 / 504 |
| early stop: SM124 after pass 1 minus SM106's snapshot after 4,000 users | show-time | -0.052% [-0.148, +0.041] | -0.127% [-0.146, -0.111] | 532 / 189 |
| early stop read-out: SM124 after pass 1 minus SM106's final model | show-time | +0.145% [+0.009, +0.251] | +0.104% [+0.092, +0.117] | 162 / 560 |

## 1,500 validation users, show-time inputs (1487 users), % of the release's own LogLoss

| comparison | users | curve % [95% CI] | instant % [95% CI] |
|---|---|---|---|
| new minus the Tiny two-head model (SM106) | 1487 | -0.078% [-0.100, -0.055] | -0.098% [-0.109, -0.090] |
| new minus the show-time big model (SM91s2) | 1487 | +1.017% [+0.841, +1.217] | +0.489% [+0.448, +0.537] |
| new minus the show-time two-head cool-down for Small (SM118) | 1487 | +0.127% [+0.096, +0.156] | +0.184% [+0.166, +0.205] |
| new minus the release (SM63) | 1487 | -0.016% [-0.068, +0.037] | -0.181% [-0.230, -0.141] |
| new minus the Tiny two-head model (SM106), ids < 5000 (the 732 set) | 726 | -0.098% [-0.139, -0.062] | -0.106% [-0.127, -0.091] |
| new minus the Tiny two-head model (SM106), ids >= 5000 (training users of the big models) | 761 | -0.062% [-0.089, -0.033] | -0.092% [-0.101, -0.083] |

Users with rows of the show-time big model (SM91s2): 1487. Its ids >= 5000 are its own training users, so new minus SM91s2 there is not a held-out comparison. Row checks (732 rows vs 1,500-pool rows of the same users): {"new": {"users": 6, "max_abs_logloss_diff": 0.0}, "sm106": {"users": 6, "max_abs_logloss_diff": 0.0}}
