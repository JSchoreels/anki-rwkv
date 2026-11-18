# Grade-order sanity (SM33) stage 2: the grade-order hinge vs the same cool-down without it

Accuracy: leak-free scoring, w8+s8, review-weighted LogLoss on 722 users, points of % of the big model (1 leak-free epoch), user bootstrap 95% CI. Order: displayed intervals at desired retention 0.90 before the fork's PAVA, share of reviews with an inverted pair, review-weighted, 722 users x 3,000 sampled reviews (the same reviews in both).

| head | arm | reference | delta | points [95% CI] | users better |
|---|---|---|---|---|---|
| curve | 0.30115 | 0.30073 | +0.000418 | +0.14 [+0.10, +0.19] | 232 |
| instant | 0.27722 | 0.27722 | +0.000007 | +0.00 [-0.00, +0.01] | 354 |

| order measure (0.90) | arm | reference | delta [95% CI] | users better / worse |
|---|---|---|---|---|
| any | 0.94% | 19.08% | -18.13% [-20.88%, -15.56%] | 682 / 0 |
| Again>Hard | 0.40% | 9.59% | -9.19% [-11.03%, -7.54%] | 565 / 0 |
| Hard>Good | 0.41% | 6.01% | -5.61% [-6.46%, -4.84%] | 626 / 0 |
| Good>Easy | 0.15% | 5.41% | -5.26% [-6.32%, -4.32%] | 591 / 1 |
| good_pava_changed | 0.29% | 10.52% | -10.23% [-11.95%, -8.65%] | 645 / 1 |
| raw_any | 6.27% | 27.50% | -21.23% [-23.59%, -19.01%] | 722 / 0 |

Verdict: **cost**.
