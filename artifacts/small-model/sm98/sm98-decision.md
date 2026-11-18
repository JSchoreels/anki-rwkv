# The show-time instant-only model, stage 2 (SM98): decision

Verdict: **no go**. The show-time instant-only model, stage 2 (SM98) minus the show-time instant-only small model (SM96), instant head, show-time inputs, 722 users: -0.035 [-0.041, -0.029] points (% of the big model (1 leak-free epoch) normal-input instant LogLoss 0.275757), 489 better / 232 worse; go needs the CI below 0 and <= -0.05.

New 0.276191 vs the show-time instant-only small model (SM96) 0.276287 (instant LogLoss, show-time inputs).

## Secondary (732 set, instant head, points)

| new minus | inputs | new | ref | points [95% CI] | better / worse |
|---|---|---|---|---|---|
| the show-time retrain of the release (SM92) | show-time | 0.276191 | 0.276876 | -0.248 [-0.270, -0.228] | 658 / 64 |
| the instant-only small model (SM57) stage 4 | show-time | 0.276191 | 0.277236 | -0.379 [-0.465, -0.309] | 600 / 122 |
| the release (SM63) | show-time | 0.276191 | 0.277718 | -0.554 [-0.657, -0.474] | 672 / 50 |
| the show-time instant-only small model (SM96) | normal | 0.27606 | 0.276134 | -0.027 [-0.033, -0.021] | 457 / 265 |
| the show-time retrain of the release (SM92) | normal | 0.27606 | 0.276757 | -0.253 [-0.274, -0.232] | 659 / 62 |
| the instant-only small model (SM57) stage 4 | normal | 0.27606 | 0.275897 | +0.059 [+0.040, +0.080] | 290 / 431 |
| the new model on normal inputs (its show-time cost) | show-time vs normal | 0.276191 | 0.27606 | +0.048 [+0.035, +0.061] | 211 / 496 |

## 1,500 validation users, show-time inputs (1487 users), % of the release's own LogLoss

| comparison | new | ref | % [95% CI] | better / worse |
|---|---|---|---|---|
| new minus the release (SM63) | 0.279162 | 0.2806 | -0.512 [-0.564, -0.469] | 1399 / 88 |
| new minus the show-time instant-only small model (SM96) | 0.279162 | 0.279249 | -0.031 [-0.037, -0.024] | 1010 / 475 |
| new minus the instant-only small model (SM57) stage 4 | 0.279162 | 0.280122 | -0.343 [-0.386, -0.304] | 1267 / 219 |
| new minus the release (SM63), ids < 5000 (the 732 set) (726 users) | 0.278971 | 0.280514 | -0.550 [-0.652, -0.476] | 676 / 50 |
| new minus the release (SM63), ids >= 5000 (761 users) | 0.279321 | 0.280671 | -0.481 [-0.535, -0.433] | 723 / 38 |

Row check: {"users": 6, "max_abs_logloss_diff": 0.0}

The curve head is untrained (curve loss weight 0); curve numbers are in the JSON only and not part of the verdict.
