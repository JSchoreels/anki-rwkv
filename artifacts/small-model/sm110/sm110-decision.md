# The big instant-only teacher (SM110): decision

Verdict: **no go** (722 users of the 732 set; % of the big model (1 leak-free epoch) normal-input instant LogLoss 0.275757; paired user bootstrap 2,000, seed 0, 95% CI).

- Primary: the big instant-only teacher (SM110) minus the Small Instant-Only model (SM98), instant, show-time inputs: -0.023% [-0.028, -0.019] (0.276128 vs 0.276191), 493 better / 225 worse. Go needs the CI below 0 and <= -0.05%.

- Share of the teacher gap closed: 0.213 (the show-time instant-only big model (SM99) minus SM98 is -0.108% [-0.133, -0.084] on the same users).

## Read-out (732 set, instant head, %)

| minus | inputs | new | ref | % [95% CI] | better / worse |
|---|---|---|---|---|---|
| the show-time instant-only big model (SM99) (the teacher) | show-time | 0.276128 | 0.275894 | +0.085% [+0.061, +0.110] | 257 / 464 |
| the show-time instant-only small model (SM96) | show-time | 0.276128 | 0.276287 | -0.058% [-0.065, -0.051] | 551 / 171 |
| the show-time retrain of the release (SM92) | show-time | 0.276128 | 0.276876 | -0.271% [-0.294, -0.249] | 657 / 64 |
| the release (SM63) | show-time | 0.276128 | 0.277718 | -0.577% [-0.681, -0.495] | 673 / 48 |
| the Small Instant-Only model (SM98) | normal | 0.276024 | 0.27606 | -0.013% [-0.019, -0.008] | 434 / 287 |
| the big model (1 leak-free epoch) (the smaller-model bar: CI upper end <= +0.40%) | normal | 0.276024 | 0.275757 | +0.097% [+0.066, +0.127] | 249 / 473 |
| the new model on normal inputs (its show-time cost) | show-time vs normal | 0.276128 | 0.276024 | +0.038% [+0.028, +0.050] | 183 / 509 |
| mid-check: SM110 snapshot users-04000 minus SM98 snapshot users-04000 | show-time | 0.276753 | 0.276755 | -0.001% [-0.010, +0.007] | 375 / 343 |

## 1,500 validation users, show-time inputs (1487 users), % of the release's own LogLoss

| comparison | new | ref | % [95% CI] | better / worse |
|---|---|---|---|---|
| new minus the Small Instant-Only model (SM98) | 0.279102 | 0.279162 | -0.021% [-0.025, -0.018] | 1047 / 433 |
| new minus the show-time instant-only big model (SM99) | 0.279102 | 0.278406 | +0.250% [+0.214, +0.291] | 316 / 1170 |
| new minus the release (SM63) | 0.279102 | 0.2806 | -0.534% [-0.586, -0.489] | 1404 / 82 |
| new minus the Small Instant-Only model (SM98), ids < 5000 (the 732 set) (726 users) | 0.278915 | 0.278971 | -0.020% [-0.025, -0.015] | 494 / 228 |
| new minus the Small Instant-Only model (SM98), ids >= 5000 (SM99's training users) (761 users) | 0.279259 | 0.279321 | -0.022% [-0.026, -0.019] | 553 / 205 |

Users with rows of the show-time instant-only big model (SM99): 1487. Its ids >= 5000 are its own training users, so new minus SM99 there is not a held-out comparison. Row check: {"users": 6, "max_abs_logloss_diff": 0.0}

The curve head is untrained (curve loss weight 0); curve numbers are in the JSON only and not part of the verdict.
