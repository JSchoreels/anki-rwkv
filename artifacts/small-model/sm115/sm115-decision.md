# The longer cool-down for Tiny (SM115): decision

Verdict: **go** (722 users of the 732 set; % of the big model (1 leak-free epoch) normal-input instant LogLoss 0.275757; paired user bootstrap 2,000, seed 0, 95% CI).

- Primary: the longer cool-down for Tiny (SM115) minus the Tiny Instant-Only model (SM101), instant, show-time inputs: -0.099% [-0.108, -0.091] (0.276446 vs 0.27672), 621 better / 101 worse. Go needs the CI below 0 and <= -0.05%.

- Share of the teacher gap closed: 0.331 (the show-time instant-only big model (SM99) minus the Tiny Instant-Only model (SM101) is -0.299% [-0.328, -0.272] on the same users).

## Read-out (732 set, instant head, %)

| minus | inputs | new | ref | % [95% CI] | better / worse |
|---|---|---|---|---|---|
| the longer big instant-only teacher (SM113) (Small, the same recipe) | show-time | 0.276446 | 0.275933 | +0.186% [+0.166, +0.208] | 108 / 614 |
| the show-time instant-only big model (SM99) (the teacher) | show-time | 0.276446 | 0.275894 | +0.200% [+0.177, +0.224] | 151 / 571 |
| the Small Instant-Only model (SM98) | show-time | 0.276446 | 0.276191 | +0.092% [+0.073, +0.113] | 244 / 478 |
| the show-time retrain of the release (SM92) | show-time | 0.276446 | 0.276876 | -0.156% [-0.181, -0.131] | 547 / 175 |
| the release (SM63) | show-time | 0.276446 | 0.277718 | -0.462% [-0.565, -0.382] | 611 / 110 |
| the Tiny Instant-Only model (SM101) | normal | 0.276343 | 0.276578 | -0.085% [-0.095, -0.076] | 577 / 144 |
| the big model (1 leak-free epoch) (the smaller-model instant bar: CI upper end <= +0.40%) | normal | 0.276343 | 0.275757 | +0.212% [+0.182, +0.245] | 174 / 548 |
| the new model on normal inputs (its show-time cost) | show-time vs normal | 0.276446 | 0.276343 | +0.037% [+0.027, +0.050] | 174 / 520 |
| early stop: SM115 after pass 1 minus SM101's snapshot after 4,094 users | show-time | 0.276874 | 0.27718 | -0.111% [-0.129, -0.094] | 510 / 211 |
| early stop read-out: SM115 after pass 1 minus SM113 after its pass 1 | show-time | 0.276874 | 0.276357 | +0.188% [+0.165, +0.211] | 105 / 616 |

## 1,500 validation users, show-time inputs (1487 users), % of the release's own LogLoss

| comparison | new | ref | % [95% CI] | better / worse |
|---|---|---|---|---|
| new minus the Tiny Instant-Only model (SM101) | 0.279406 | 0.279693 | -0.103% [-0.110, -0.096] | 1259 / 226 |
| new minus the longer big instant-only teacher (SM113) | 0.279406 | 0.278892 | +0.184% [+0.167, +0.204] | 237 / 1250 |
| new minus the show-time instant-only big model (SM99) | 0.279406 | 0.278406 | +0.359% [+0.325, +0.398] | 180 / 1307 |
| new minus the release (SM63) | 0.279406 | 0.2806 | -0.425% [-0.476, -0.383] | 1304 / 182 |
| new minus the Tiny Instant-Only model (SM101), ids < 5000 (the 732 set) (726 users) | 0.279235 | 0.279522 | -0.103% [-0.115, -0.092] | 625 / 101 |
| new minus the Tiny Instant-Only model (SM101), ids >= 5000 (SM99's training users) (761 users) | 0.279549 | 0.279836 | -0.102% [-0.112, -0.094] | 634 / 125 |

Users with rows of the show-time instant-only big model (SM99): 1487. Its ids >= 5000 are its own training users, so new minus SM99 there is not a held-out comparison. Row check: {"users": 6, "max_abs_logloss_diff": 0.0}

The curve head is untrained (curve loss weight 0); curve numbers are in the JSON only and not part of the verdict.
