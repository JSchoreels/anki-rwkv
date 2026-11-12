# The show-time instant-only big model (SM99): decision

Verdict: **ceiling above SM98** (final model: epoch 2; 722 users of the 732 set; % of the big model (1 leak-free epoch) normal-input instant LogLoss 0.275757; paired user bootstrap 2,000, seed 0, 95% CI).

- Primary: the show-time instant-only big model (SM99) minus the show-time instant-only model, stage 2 (SM98), instant, show-time inputs: -0.108% [-0.133, -0.084] (0.275894 vs 0.276191), 492 better / 230 worse. "Ceiling above SM98" needs <= -0.05% with the CI below 0 (then a teacher experiment is worth it); otherwise the small instant-only model is at the big-model ceiling.

- With it: the show-time instant-only big model (SM99) minus the show-time big model (SM91 step 2), instant, show-time inputs: -0.105% [-0.118, -0.093] (0.275894 vs 0.276183).

## Read-out (732 set, instant head, %)

| minus | inputs | new | ref | % [95% CI] | better / worse |
|---|---|---|---|---|---|
| the show-time big model (SM91 step 2) | show-time | 0.275894 | 0.276183 | -0.105% [-0.118, -0.093] | 578 / 143 |
| the show-time big model (SM91 step 2) | normal | 0.275748 | 0.276028 | -0.102% [-0.115, -0.090] | 581 / 139 |
| the show-time instant-only small model (SM96) | show-time | 0.275894 | 0.276287 | -0.143% [-0.168, -0.119] | 514 / 207 |
| the show-time retrain of the release (SM92) | show-time | 0.275894 | 0.276876 | -0.356% [-0.392, -0.322] | 630 / 91 |
| the release (SM63) | show-time | 0.275894 | 0.277718 | -0.662% [-0.765, -0.577] | 662 / 59 |
| the show-time instant-only model, stage 2 (SM98) | normal | 0.275748 | 0.27606 | -0.113% [-0.138, -0.089] | 490 / 230 |
| the big model (1 leak-free epoch) | normal | 0.275748 | 0.275757 | -0.004% [-0.024, +0.016] | 392 / 329 |
| the final model on normal inputs (its show-time cost) | show-time vs normal | 0.275894 | 0.275748 | +0.053% [+0.042, +0.067] | 170 / 534 |
| epoch 1 minus the show-time instant-only model, stage 2 (SM98) | show-time | 0.276002 | 0.276191 | -0.069% [-0.094, -0.044] | 444 / 277 |
| epoch 2 minus epoch 1 | show-time | 0.275894 | 0.276002 | -0.039% [-0.046, -0.033] | 512 / 208 |

## 1,500 validation users, show-time inputs (1487 users), % of the release's own LogLoss

| comparison | new | ref | % [95% CI] | better / worse |
|---|---|---|---|---|
| new minus the show-time instant-only model, stage 2 (SM98) | 0.278406 | 0.279162 | -0.271% [-0.311, -0.235] | 1203 / 284 |
| new minus the show-time big model (SM91 step 2) | 0.278406 | 0.278727 | -0.115% [-0.124, -0.107] | 1240 / 246 |
| new minus the show-time instant-only small model (SM96) | 0.278406 | 0.279249 | -0.302% [-0.340, -0.268] | 1234 / 252 |
| new minus the release (SM63) | 0.278406 | 0.2806 | -0.782% [-0.849, -0.719] | 1413 / 73 |

Users with rows of the show-time big model (SM91 step 2): 1487. Row check: {"new": {"users": 6, "max_abs_logloss_diff": 0.0}, "sm91s2": {"users": 6, "max_abs_logloss_diff": 0.0}}

The curve head is untrained (curve loss weight 0); curve numbers are in the JSON only and not part of the verdict.
