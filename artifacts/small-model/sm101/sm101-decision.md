# The smaller instant-only model (SM101): decision

Verdict: **go** (722 users of the 732 set; points = % of the big model (1 leak-free epoch) normal-input instant LogLoss 0.275757; paired user bootstrap 2,000, seed 0, 95% CI).

- A, the bar: the smaller instant-only model (SM101) minus the big model (1 leak-free epoch), instant, normal inputs: +0.298 [+0.268, +0.329] points (0.276578 vs 0.275757); pass needs the CI upper end <= +0.40: **pass**.
- B, what the fork ships for RWKV-Instant: the smaller instant-only model (SM101) minus the show-time retrain of the release (SM92), instant, show-time inputs: -0.057 [-0.079, -0.034] points (0.27672 vs 0.276876), 460 better / 261 worse; pass needs the CI below 0 and <= -0.05: **pass**.

## Secondary (732 set, instant head, points)

| new minus | inputs | new | ref | points [95% CI] | better / worse |
|---|---|---|---|---|---|
| the show-time instant-only small model (SM96) | show-time | 0.27672 | 0.276287 | +0.157 [+0.137, +0.178] | 140 / 580 |
| the show-time instant-only model, stage 2 (SM98) | show-time | 0.27672 | 0.276191 | +0.192 [+0.171, +0.215] | 114 / 607 |
| the release (SM63) | show-time | 0.27672 | 0.277718 | -0.362 [-0.463, -0.284] | 574 / 148 |
| the show-time instant-only small model (SM96) | normal | 0.276578 | 0.276134 | +0.161 [+0.141, +0.182] | 140 / 582 |
| the show-time instant-only model, stage 2 (SM98) | normal | 0.276578 | 0.27606 | +0.188 [+0.167, +0.211] | 119 / 603 |
| the show-time retrain of the release (SM92) | normal | 0.276578 | 0.276757 | -0.065 [-0.088, -0.042] | 472 / 249 |
| the big model (1 leak-free epoch) on normal inputs | show-time vs normal | 0.27672 | 0.275757 | +0.349 [+0.314, +0.387] | 108 / 614 |
| the new model on normal inputs (its show-time cost) | show-time vs normal | 0.27672 | 0.276578 | +0.051 [+0.039, +0.065] | 186 / 511 |

## 1,500 validation users, show-time inputs (1487 users), % of the release's own LogLoss

| comparison | new | ref | % [95% CI] | better / worse |
|---|---|---|---|---|
| new minus the release (SM63) | 0.279693 | 0.2806 | -0.323 [-0.374, -0.282] | 1216 / 271 |
| new minus the show-time instant-only small model (SM96) | 0.279693 | 0.279249 | +0.159 [+0.141, +0.180] | 333 / 1152 |
| new minus the show-time instant-only model, stage 2 (SM98) | 0.279693 | 0.279162 | +0.190 [+0.171, +0.212] | 244 / 1240 |

Row check: {"users": 6, "max_abs_logloss_diff": 0.0}

The curve head is untrained (curve loss weight 0); curve numbers are in the JSON only and not part of the verdict.
