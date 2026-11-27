# Show-time scoring of the headline models

Leak-free scoring (deleted cards: neighbour), w8+s8, review-weighted LogLoss. Show-time inputs = `e2s-query` (what the Anki fork gives at a query). 732 set = 722 users. Points = % of the big model (1 leak-free epoch) normal-input LogLoss on the same users (curve 0.298966, instant 0.275757), for both input kinds. Cost = show-time minus normal on the same model (points, paired user bootstrap 2,000, seed 0, 95% CI). The instant-only models never trained the curve head: their curve numbers are not meaningful.

## Curve head, 732 set

| model | users | normal LogLoss | show-time LogLoss | normal points | show-time points | show-time cost (points) [95% CI] |
|---|---|---|---|---|---|---|
| the release (SM63) | 722 | 0.30003 | 0.300414 | +0.356 | +0.484 | +0.128 [+0.106, +0.160] |
| the show-time retrain of the release (SM92) | 722 | 0.30024 | 0.300356 | +0.426 | +0.465 | +0.039 [+0.031, +0.048] |
| the instant-only small model (SM57) stage 1 | 722 | n/a (curve head untrained) | | | | |
| the instant-only small model (SM57) stage 4 | 722 | n/a (curve head untrained) | | | | |
| the previous release, cool-down from epoch 20 (SM72e20) | 722 | 0.300188 | 0.30057 | +0.409 | +0.537 | +0.128 [+0.105, +0.159] |
| the earlier release, cool-down from epoch 15 (SM72) | 722 | 0.300384 | 0.300773 | +0.474 | +0.604 | +0.130 [+0.106, +0.163] |
| the small model, cool-down from epoch 15 (SM56) | 722 | 0.300476 | 0.300867 | +0.505 | +0.636 | +0.131 [+0.106, +0.165] |
| the previous release, cool-down from epoch 10 (SM56 dec10) | 722 | 0.30066 | 0.301034 | +0.567 | +0.692 | +0.125 [+0.102, +0.157] |
| the shipped big model (fork RWKV) | 722 | 0.299661 | 0.300225 | +0.232 | +0.421 | +0.189 [+0.153, +0.236] |
| the big model (1 leak-free epoch) | 722 | 0.298966 | 0.29958 | +0.000 | +0.205 | +0.205 [+0.165, +0.258] |
| the show-time big model (SM91 step 2) | 722 | - | 0.299415 | - | +0.150 | - |

## Instant head, 732 set

| model | users | normal LogLoss | show-time LogLoss | normal points | show-time points | show-time cost (points) [95% CI] |
|---|---|---|---|---|---|---|
| the release (SM63) | 722 | 0.276539 | 0.277718 | +0.283 | +0.711 | +0.428 [+0.337, +0.546] |
| the show-time retrain of the release (SM92) | 722 | 0.276757 | 0.276876 | +0.362 | +0.406 | +0.043 [+0.033, +0.056] |
| the instant-only small model (SM57) stage 1 | 722 | 0.276093 | 0.277429 | +0.122 | +0.606 | +0.485 [+0.389, +0.602] |
| the instant-only small model (SM57) stage 4 | 722 | 0.275897 | 0.277236 | +0.051 | +0.536 | +0.485 [+0.396, +0.589] |
| the previous release, cool-down from epoch 20 (SM72e20) | 722 | 0.276642 | 0.277826 | +0.321 | +0.750 | +0.429 [+0.340, +0.541] |
| the earlier release, cool-down from epoch 15 (SM72) | 722 | 0.276872 | 0.27799 | +0.404 | +0.810 | +0.405 [+0.324, +0.507] |
| the small model, cool-down from epoch 15 (SM56) | 722 | 0.277145 | 0.278177 | +0.503 | +0.878 | +0.374 [+0.305, +0.462] |
| the previous release, cool-down from epoch 10 (SM56 dec10) | 722 | 0.277447 | 0.278403 | +0.613 | +0.960 | +0.347 [+0.278, +0.436] |
| the shipped big model (fork RWKV) | 722 | 0.276203 | 0.277559 | +0.162 | +0.653 | +0.492 [+0.394, +0.617] |
| the big model (1 leak-free epoch) | 722 | 0.275757 | 0.277157 | +0.000 | +0.508 | +0.508 [+0.411, +0.633] |
| the show-time big model (SM91 step 2) | 722 | - | 0.276183 | - | +0.155 | - |

## Pairs on the 732 set (722 users), unit points (% of the big model normal-input LogLoss)

| comparison | inputs | head | model | ref | points [95% CI] | users better / worse |
|---|---|---|---|---|---|---|
| stage 4 - stage 1 of the instant-only small model (SM57) | show-time | instant | 0.277236 | 0.277429 | -0.070 [-0.090, -0.056] | 512 / 208 |
| stage 4 - stage 1 of the instant-only small model (SM57) | show-time | curve (curve head untrained) | 0.452565 | 0.350514 | +34.135 [+30.876, +37.877] | 4 / 718 |
| stage 4 of the instant-only small model (SM57) - the release (SM63) | show-time | instant | 0.277236 | 0.277718 | -0.175 [-0.206, -0.149] | 575 / 147 |
| stage 4 of the instant-only small model (SM57) - the release (SM63) | show-time | curve (curve head untrained) | 0.452565 | 0.300414 | +50.893 [+46.336, +56.240] | 1 / 721 |
| stage 4 - stage 1 of the instant-only small model (SM57) | normal | instant | 0.275897 | 0.276093 | -0.071 [-0.079, -0.064] | 553 / 167 |
| stage 4 - stage 1 of the instant-only small model (SM57) | normal | curve (curve head untrained) | 0.450124 | 0.349112 | +33.787 [+30.565, +37.494] | 4 / 718 |
| stage 4 of the instant-only small model (SM57) - the release (SM63) | normal | instant | 0.275897 | 0.276539 | -0.233 [-0.252, -0.214] | 663 / 58 |
| stage 4 of the instant-only small model (SM57) - the release (SM63) | normal | curve (curve head untrained) | 0.450124 | 0.30003 | +50.205 [+45.730, +55.454] | 1 / 721 |

## 1,500 validation users, show-time inputs (1487 users), unit % of the release's own LogLoss

| comparison | head | model | ref | % [95% CI] | users better / worse |
|---|---|---|---|---|---|
| stage 4 - the release | instant | 0.280122 | 0.2806 | -0.170 [-0.190, -0.152] | 1178 / 308 |
| stage 4 - stage 1 | instant | 0.280122 | 0.280299 | -0.063 [-0.074, -0.054] | 1071 / 413 |
| stage 1 - the release | instant | 0.280299 | 0.2806 | -0.107 [-0.123, -0.093] | 1132 / 353 |
| stage 4 - the release, ids < 5000 (the 732 set) (726 users) | instant | 0.280006 | 0.280514 | -0.181 [-0.217, -0.152] | 579 / 147 |
| stage 4 - the release, ids >= 5000 (761 users) | instant | 0.280219 | 0.280671 | -0.161 [-0.185, -0.141] | 599 / 161 |

Row check (check users re-scored by the 1,500 ticket vs their 732-set rows): {"s1": {"users": 6, "max_abs_logloss_diff": 0.0}, "s4": {"users": 6, "max_abs_logloss_diff": 0.0}, "release": {"users": 6, "max_abs_logloss_diff": 0.0}}

