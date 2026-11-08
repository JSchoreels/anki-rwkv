# The grade order by construction (SM95) stage 1: rectified cool-down vs the same cool-down without it

Accuracy: leak-free scoring, w8+s8, review-weighted LogLoss on 699 users, points of % of the big model (1 leak-free epoch), paired user bootstrap 95% CI. "Rectified" = the curve of each review pooled with the other three grades' what-if curves at the same time so that Again <= Hard <= Good <= Easy (the scheduler's curve).

| comparison | head | a | b | delta | points [95% CI] | users better |
|---|---|---|---|---|---|---|
| primary: arm rectified vs reference raw | curve | 0.29978 | 0.29840 | +0.001372 | +0.46 [+0.34, +0.62] | 255 |
| primary: arm rectified vs reference raw | instant | 0.27233 | 0.27232 | +0.000004 | +0.00 [-0.00, +0.01] | 313 |
| arm rectified vs reference rectified (training effect, both scored as the scheduler uses them) | curve | 0.29978 | 0.30117 | -0.001395 | -0.47 [-0.67, -0.33] | 510 |
| arm rectified vs reference rectified (training effect, both scored as the scheduler uses them) | instant | 0.27233 | 0.27232 | +0.000004 | +0.00 [-0.00, +0.01] | 313 |
| reference rectified vs reference raw (rectifying today's model without training) | curve | 0.30117 | 0.29840 | +0.002767 | +0.93 [+0.67, +1.27] | 187 |
| reference rectified vs reference raw (rectifying today's model without training) | instant | 0.27232 | 0.27232 | -0.000000 | -0.00 [-0.00, +0.00] | 29 |
| arm raw vs reference raw (the arm's unpooled curves) | curve | 0.29840 | 0.29840 | +0.000001 | +0.00 [-0.01, +0.02] | 369 |
| arm raw vs reference raw (the arm's unpooled curves) | instant | 0.27233 | 0.27232 | +0.000004 | +0.00 [-0.00, +0.01] | 313 |
| grade-order hinge (SM33) arm vs reference raw | curve | 0.29883 | 0.29840 | +0.000427 | +0.14 [+0.10, +0.19] | 228 |
| grade-order hinge (SM33) arm vs reference raw | instant | 0.27233 | 0.27232 | +0.000009 | +0.00 [-0.00, +0.01] | 341 |

Order: 722 users x 3,000 sampled reviews (2,166,000 reviews, the same in every measure); share of reviews with an inverted pair (review-weighted) and, for the arm, the count.

| measure | arm rectified (count) | arm raw | reference rectified | reference raw | grade-order hinge (SM33) raw |
|---|---|---|---|---|---|
| raw_any@0.80 | 0.00% (0) | 22.99% | 0.00% | 28.42% | 3.51% |
| displayed_any@0.80 | 0.00% (0) | 19.68% | 0.00% | 24.60% | 1.12% |
| raw_any@0.90 | 0.00% (0) | 20.14% | 0.00% | 27.50% | 6.27% |
| displayed_any@0.90 | 0.00% (0) | 14.48% | 0.00% | 19.08% | 0.94% |
| raw_any@0.95 | 0.00% (0) | 21.38% | 0.00% | 31.40% | 10.59% |
| displayed_any@0.95 | 0.00% (0) | 11.79% | 0.00% | 15.55% | 0.60% |
| Again>Hard@0.90 | 0.00% (0) | 6.03% | 0.00% | 9.59% | 0.40% |
| Hard>Good@0.90 | 0.00% (0) | 4.26% | 0.00% | 6.01% | 0.41% |
| Good>Easy@0.90 | 0.00% (0) | 5.41% | 0.00% | 5.41% | 0.15% |
| good_changed_by_fork_pava@0.90 | 0.00% (0) | 8.55% | 0.00% | 10.52% | 0.29% |
| any_changed_by_fork_pava@0.90 | 0.00% (0) | 14.48% | 0.00% | 19.08% | 0.94% |

Verdict: **cost** (go = zero breaks and both heads <= +0.05 points vs the reference raw).
