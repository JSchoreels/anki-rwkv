# The show-time two-head cool-down for Small (SM118): decision

Verdict: **go** (gain on: curve, instant; regression on: no head; 722 users of the 732 set; % of the big model (1 leak-free epoch) normal-input LogLoss of each head, curve 0.298966, instant 0.275757; paired user bootstrap 2,000, seed 0, 95% CI).

Rule (pre-registered): per head, gain = CI below 0 and at most -0.05%; regression = +0.05% or more, or CI above 0; go = a gain on at least one head and a regression on neither.

| the show-time two-head cool-down for Small (SM118) minus the show-time retrain of the release (SM92), show-time | new | ref | % [95% CI] | better / worse | teacher gap closed |
|---|---|---|---|---|---|
| curve | 0.300016 | 0.300356 | -0.114% [-0.177, -0.062] | 518 / 204 | 0.362 (the show-time big model (SM91s2) minus SM92: -0.315% [-0.491, -0.149]) |
| instant | 0.276599 | 0.276876 | -0.100% [-0.110, -0.091] | 607 / 114 | 0.398 (the show-time big model (SM91s2) minus SM92: -0.251% [-0.280, -0.222]) |

## Read-out (732 set, %)

| minus | inputs | curve % [95% CI] | instant % [95% CI] | instant better / worse |
|---|---|---|---|---|
| the show-time big model (SM91s2) (the teacher) | show-time | +0.201% [+0.051, +0.359] | +0.151% [+0.126, +0.176] | 204 / 518 |
| the release (SM63) | show-time | -0.133% [-0.209, -0.036] | -0.406% [-0.506, -0.331] | 641 / 81 |
| the big model (1 leak-free epoch) (smaller-model bar, show-time: curve <= +0.5%, instant <= +0.4%) | show-time | +0.146% [-0.030, +0.341] | -0.202% [-0.317, -0.117] | 403 / 319 |
| the Small Instant-Only model (SM98) (instant only) | show-time | (not trained) | +0.148% [+0.133, +0.164] | 144 / 578 |
| the longer big instant-only teacher (SM113) (instant only) | show-time | (not trained) | +0.242% [+0.222, +0.262] | 70 / 652 |
| the show-time retrain of the release (SM92) | normal | -0.108% [-0.172, -0.056] | -0.086% [-0.097, -0.076] | 570 / 151 |
| the big model (1 leak-free epoch) (the smaller-model bar: curve <= +0.5%, instant <= +0.4%) | normal | +0.318% [+0.151, +0.515] | +0.276% [+0.243, +0.309] | 144 / 578 |
| the new model on normal inputs (its show-time cost) | show-time vs normal | +0.033% [+0.026, +0.042] | +0.029% [+0.020, +0.040] | 204 / 489 |
| early stop: SM118 after pass 1 minus SM92's snapshot after 4,056 users | show-time | -0.147% [-0.239, -0.062] | -0.093% [-0.112, -0.076] | 480 / 241 |
| early stop read-out: SM118 after pass 1 minus SM92's final model | show-time | +0.141% [+0.030, +0.230] | +0.119% [+0.106, +0.132] | 143 / 578 |

## 1,500 validation users, show-time inputs (1487 users), % of the release's own LogLoss

| comparison | users | curve % [95% CI] | instant % [95% CI] |
|---|---|---|---|
| new minus the show-time retrain of the release (SM92) | 1487 | -0.102% [-0.146, -0.065] | -0.099% [-0.108, -0.091] |
| new minus the show-time big model (SM91s2) | 1487 | +0.890% [+0.712, +1.083] | +0.305% [+0.269, +0.344] |
| new minus the release (SM63) | 1487 | -0.143% [-0.192, -0.093] | -0.365% [-0.414, -0.324] |
| new minus the show-time retrain of the release (SM92), ids < 5000 (the 732 set) | 726 | -0.146% [-0.242, -0.075] | -0.107% [-0.125, -0.093] |
| new minus the show-time retrain of the release (SM92), ids >= 5000 (training users of the big models) | 761 | -0.065% [-0.089, -0.038] | -0.092% [-0.103, -0.083] |

Users with rows of the show-time big model (SM91s2): 1487. Its ids >= 5000 are its own training users, so new minus SM91s2 there is not a held-out comparison. Row checks (732 rows vs 1,500-pool rows of the same users): {"new": {"users": 6, "max_abs_logloss_diff": 0.0}, "sm92": {"users": 6, "max_abs_logloss_diff": 0.0}}
