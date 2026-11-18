# Training-free state check (SM116): decision

**Verdict: starting point.** both heads have the upper 95% CI bound below +0.05% (curve +0.015% [-0.007, +0.032]; instant +0.032% [+0.025, +0.038]).

The show-time retrain (SM92) with its card and note state kept at 70% of the dimensions (per block PCA from 100 users outside the scored pool, no training) minus the unmodified show-time retrain (SM92); show-time inputs, w8+s8, leak-free scoring, 629 users (the 722-user rule pool with <= 125,000 reviews); % of the big model (1 leak-free epoch) show-time LogLoss; paired user bootstrap 2,000, seed 0, 95% CI. Rule: starting point iff both upper bounds < +0.05%.

| head | unmodified | 70% state | difference | % (CI) | % of own LogLoss (CI) | users better / worse |
|---|---|---|---|---|---|---|
| curve | 0.303053 | 0.303099 | +0.0000462 | +0.015% [-0.007, +0.032] | +0.015% [-0.007, +0.032] | 248 / 380 |
| instant | 0.274177 | 0.274265 | +0.0000878 | +0.032% [+0.025, +0.038] | +0.032% [+0.025, +0.038] | 203 / 426 |
