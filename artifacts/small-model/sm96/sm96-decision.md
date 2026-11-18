# The show-time instant-only small model (SM96): decision

Verdict: **go**. New - the show-time retrain of the release (SM92), instant head, show-time inputs, 722 users: -0.214 [-0.233, -0.195] points (% of the big model (1 leak-free epoch) normal-input instant LogLoss 0.275757), 647 better / 75 worse; go needs the CI below 0 and <= -0.05.

New 0.276287 vs SM92 0.276876 (instant LogLoss, show-time inputs).

## Secondary (732 set, instant head, points)

| new minus | inputs | new | ref | points [95% CI] | better / worse |
|---|---|---|---|---|---|
| stage 4 of the instant-only small model (SM57) | show-time | 0.276287 | 0.277236 | -0.344 [-0.429, -0.275] | 584 / 137 |
| the release (SM63) | show-time | 0.276287 | 0.277718 | -0.519 [-0.620, -0.442] | 664 / 58 |
| stage 4 of the instant-only small model (SM57) | normal | 0.276134 | 0.275897 | +0.086 [+0.070, +0.104] | 213 / 506 |
| the show-time retrain of the release (SM92) | normal | 0.276134 | 0.276757 | -0.226 [-0.246, -0.207] | 646 / 75 |
| the new model on normal inputs (its show-time cost) | show-time vs normal | 0.276287 | 0.276134 | +0.055 [+0.042, +0.072] | 199 / 502 |

## 1,500 validation users, show-time inputs (1487 users), % of the release's own LogLoss

| comparison | new | ref | % [95% CI] | better / worse |
|---|---|---|---|---|
| new - the release (SM63) | 0.279249 | 0.2806 | -0.481 [-0.534, -0.438] | 1383 / 104 |
| new - stage 4 of the instant-only small model (SM57) | 0.279249 | 0.280122 | -0.312 [-0.354, -0.274] | 1226 / 257 |
| new - the release, ids < 5000 (the 732 set) (726 users) | 0.279058 | 0.280514 | -0.519 [-0.622, -0.444] | 668 / 58 |
| new - the release, ids >= 5000 (761 users) | 0.279408 | 0.280671 | -0.450 [-0.503, -0.404] | 715 / 46 |

Row check: {"users": 6, "max_abs_logloss_diff": 0.0}

The curve head is untrained (curve loss weight 0); curve numbers are in the JSON only and not part of the verdict.
