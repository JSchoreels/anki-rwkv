# The Tiny two-head model (SM106): decision

Verdict: **go** (722 users of the 732 set, show-time inputs; % = % of the big model (1 leak-free epoch)'s show-time LogLoss, curve 0.29958 / instant 0.277157; paired user bootstrap 2,000, seed 0, 95% CI).

## A, the pruning bar: the Tiny two-head model (SM106) minus the big model (1 leak-free epoch)

| head | new | big | % [95% CI] | bar (point estimate) | pass |
|---|---|---|---|---|---|
| curve | 0.300676 | 0.29958 | +0.366% [+0.183, +0.574] | <= +0.5% | pass |
| instant | 0.277369 | 0.277157 | +0.076% [-0.033, +0.163] | <= +0.4% | pass |

## B, report only (show-time inputs, % of the big model (1 leak-free epoch))

| new minus | head | new | ref | % [95% CI] | better / worse |
|---|---|---|---|---|---|
| the show-time retrain (SM92) | curve | 0.300676 | 0.300356 | +0.107% [+0.074, +0.137] | 213 / 509 |
| the show-time retrain (SM92) | instant | 0.277369 | 0.276876 | +0.178% [+0.158, +0.199] | 122 / 600 |
| the Tiny Instant-Only model (SM101) | instant | 0.277369 | 0.27672 | +0.234% [+0.215, +0.254] | 90 / 631 |
| the release (SM63) | curve | 0.300676 | 0.300414 | +0.088% [+0.001, +0.210] | 311 / 411 |
| the release (SM63) | instant | 0.277369 | 0.277718 | -0.126% [-0.223, -0.052] | 353 / 368 |

## 1,500 validation users, show-time inputs (1487 users), % of the release's own LogLoss

| comparison | head | new | ref | % [95% CI] | better / worse |
|---|---|---|---|---|---|
| new minus the release (SM63) | curve | 0.303517 | 0.303329 | +0.062% [+0.015, +0.118] | 651 / 836 |
| new minus the release (SM63) | instant | 0.280366 | 0.2806 | -0.083% [-0.132, -0.041] | 730 / 755 |
| new minus the Tiny Instant-Only model (SM101) | instant | 0.280366 | 0.279693 | +0.241% [+0.223, +0.261] | 151 / 1333 |

Row check: {"users": 6, "max_abs_logloss_diff": 0.0}
