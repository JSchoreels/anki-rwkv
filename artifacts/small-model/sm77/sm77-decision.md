# Tiny architecture additions (SM77) stage 1: key extras + learned initial state vs the same recipe without them

Leak-free scoring, w8+s8, review-weighted LogLoss, user bootstrap 2,000 seed 0, 95% CI; deltas in points of % of the big model (1 leak-free epoch).

| pool | users | head | arm | reference | delta | points [95% CI] | users better | arm vs big | reference vs big |
|---|---|---|---|---|---|---|---|---|---|
| 732 (rule) | 722 | curve | 0.30092 | 0.30073 | +0.000189 | +0.06 [-0.01, +0.13] | 294 | +0.65% | +0.59% |
| 732 (rule) | 722 | instant | 0.27742 | 0.27722 | +0.000208 | +0.08 [+0.06, +0.09] | 204 | +0.60% | +0.53% |
| 732+long | 725 | curve | 0.30300 | 0.30296 | +0.000036 | +0.01 [-0.13, +0.11] | 296 | +0.59% | +0.58% |
| 732+long | 725 | instant | 0.27835 | 0.27812 | +0.000235 | +0.09 [+0.07, +0.11] | 204 | +0.64% | +0.56% |

Size: 331,511 -> 338,231 parameters; scorer w8 weight bytes 410,459 -> 437,339 (+26,880 B); card / note state bytes unchanged.

Verdict: **stop** (both heads gain under 0.10 points).
