# The longer big instant-only teacher (SM113): decision

Verdict: **go** (722 users of the 732 set; % of the big model (1 leak-free epoch) normal-input instant LogLoss 0.275757; paired user bootstrap 2,000, seed 0, 95% CI).

- Primary: the longer big instant-only teacher (SM113) minus the Small Instant-Only model (SM98), instant, show-time inputs: -0.094% [-0.103, -0.085] (0.275933 vs 0.276191), 628 better / 94 worse. Go needs the CI below 0 and <= -0.05%.

- Share of the teacher gap closed: 0.87 (the show-time instant-only big model (SM99) minus the Small Instant-Only model (SM98) is -0.108% [-0.133, -0.084] on the same users).

## Read-out (732 set, instant head, %)

| minus | inputs | new | ref | % [95% CI] | better / worse |
|---|---|---|---|---|---|
| the big instant-only teacher (SM110) (the same targets, one-epoch cool-down) | show-time | 0.275933 | 0.276128 | -0.071% [-0.078, -0.064] | 604 / 117 |
| the show-time instant-only big model (SM99) (the teacher) | show-time | 0.275933 | 0.275894 | +0.014% [-0.010, +0.038] | 321 / 400 |
| the show-time instant-only small model (SM96) | show-time | 0.275933 | 0.276287 | -0.129% [-0.140, -0.118] | 614 / 107 |
| the show-time retrain of the release (SM92) | show-time | 0.275933 | 0.276876 | -0.342% [-0.369, -0.316] | 675 / 47 |
| the release (SM63) | show-time | 0.275933 | 0.277718 | -0.648% [-0.754, -0.566] | 684 / 38 |
| the Small Instant-Only model (SM98) | normal | 0.275844 | 0.27606 | -0.078% [-0.090, -0.068] | 581 / 139 |
| the big model (1 leak-free epoch) (the smaller-model bar: CI upper end <= +0.40%) | normal | 0.275844 | 0.275757 | +0.031% [-0.002, +0.063] | 321 / 401 |
| the new model on normal inputs (its show-time cost) | show-time vs normal | 0.275933 | 0.275844 | +0.032% [+0.023, +0.043] | 188 / 508 |
| early stop: SM113 after pass 1 minus SM98's snapshot after 4,000 users | show-time | 0.276357 | 0.276755 | -0.144% [-0.166, -0.125] | 558 / 164 |
| early stop read-out: SM113 after pass 1 minus SM110's snapshot after 4,000 users | show-time | 0.276357 | 0.276753 | -0.143% [-0.163, -0.124] | 551 / 171 |

## 1,500 validation users, show-time inputs (1487 users), % of the release's own LogLoss

| comparison | new | ref | % [95% CI] | better / worse |
|---|---|---|---|---|
| new minus the Small Instant-Only model (SM98) | 0.278892 | 0.279162 | -0.097% [-0.109, -0.088] | 1291 / 196 |
| new minus the big instant-only teacher (SM110) | 0.278892 | 0.279102 | -0.075% [-0.088, -0.067] | 1233 / 252 |
| new minus the show-time instant-only big model (SM99) | 0.278892 | 0.278406 | +0.175% [+0.141, +0.208] | 405 / 1080 |
| new minus the release (SM63) | 0.278892 | 0.2806 | -0.609% [-0.664, -0.561] | 1421 / 66 |
| new minus the Small Instant-Only model (SM98), ids < 5000 (the 732 set) (726 users) | 0.278684 | 0.278971 | -0.103% [-0.127, -0.087] | 632 / 94 |
| new minus the Small Instant-Only model (SM98), ids >= 5000 (SM99's training users) (761 users) | 0.279065 | 0.279321 | -0.092% [-0.101, -0.083] | 659 / 102 |

Users with rows of the show-time instant-only big model (SM99): 1487. Its ids >= 5000 are its own training users, so new minus SM99 there is not a held-out comparison. Row check: {"users": 6, "max_abs_logloss_diff": 0.0}

The curve head is untrained (curve loss weight 0); curve numbers are in the JSON only and not part of the verdict.
