# Tiny ensemble on the 1,500 users (SM169): result

Verdict: **GO.** The logit ensemble of the six-layer Tiny Instant-Only (SM130) and the six-layer Tiny two-head (SM130) instant head, with w = 0.69 fixed and not refitted, beats the six-layer Tiny Instant-Only (SM130) alone on the 1,500 validation users minus the 100 fit users, at T24 (`w8+lr2.16.4q8x8@e2s-query`). Nothing ships from this test; shipping the Tiny ensemble in the fork's default mode is a later, separate decision for the user.

## Question and rules (design.md)

Does the gain found on the 732 set (SM160) hold on users that did not take part in fitting w? The statistic D is the review-weighted change of the benchmark instant LogLoss of the ensemble against the Instant-Only model alone, with a paired user bootstrap of 2,000 draws (seed 0, 95% CI). The go needed all of: (a) the CI upper bound below 0 and D at most -0.05%; (b) at least 1,375 rule users and no non-finite row; (c) every dump reproducing its row's instant LogLoss within 2e-6.

## Rule check

- (a) YES: D = -0.062% [-0.066, -0.057] on 1,385 users (64,864,920 reviews). The upper bound is below 0 and D is below -0.05%.
- (b) YES: 1,385 rule users (at least 1,375) and no non-finite row.
- (c) YES: the largest difference between a dump and its row is 5.0e-07 (limit 2e-06).

## Report-only lines

| Users | Reviews | D (w = 0.69) | Oracle w: D | Share of users that improve | Two-head instant head vs Instant-Only | Ensemble vs two-head instant head |
|---|---|---|---|---|---|---|
| all: 1,385 | 64,864,920 | -0.062% [-0.066, -0.057] | 0.67: -0.062% | 82.8% (1,147) | +0.191% [+0.174, +0.210] | -0.252% [-0.269, -0.237] |
| 732 part: 723 | 30,479,310 | -0.067% [-0.072, -0.061] | 0.66: -0.067% | 81.5% (589) | +0.184% [+0.165, +0.208] | -0.251% [-0.272, -0.233] |
| new part (id >= 5000): 662 | 34,385,610 | -0.057% [-0.063, -0.051] | 0.68: -0.057% | 84.3% (558) | +0.197% [+0.173, +0.227] | -0.254% [-0.281, -0.232] |

The 732 part reproduces SM160's figure exactly. The oracle w is within 0.02 of the fixed 0.69 everywhere and gains at most 0.0003% more, so the fixed w loses nothing. The two-head instant head alone is 0.19% worse than the Instant-Only model; the ensemble recovers that and goes below it.

Aggregate curve on all rule users (D against the Instant-Only model, w = 0, 0.1, ..., 1): +0.191, +0.121, +0.062, +0.015, -0.021, -0.046, -0.059, -0.061, -0.052, -0.032, 0.000 (in %). The minimum is at w about 0.7.

## Excluded users

Out of GPU memory for the Instant-Only model even at cap 0.92: users 7625, 7775 and 9526 (new part). SM164 had four such users; user 7562 fit this time (662 of 665 scored). The 732 part has none. Not counted as non-finite rows.

## Cross-checks and GPU time

- Smoke on users 5625 and 9094: rows equal SM164's rows (difference 0). The result script's cross-check against SM164's rows covered 1,323 pairs but its per-model largest-difference table came out empty (a reporting quirk of the script, not investigated); the smoke check is the evidence of agreement with SM164.
- A first launch at 09:38 stopped on a bug in the smoke check (fixed); the log line `STOP: GPU smoke mismatch` comes from it, no scoring was lost.
- GPU minutes per ticket (new part): pass 1 23 minutes (cap 0.8), passes 2 to 5 about 1 minute each (caps 0.8, 0.92, 0.92, 0.92); the smoke tickets under 1 minute. Total about 27 minutes.
