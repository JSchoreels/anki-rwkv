# Grade-order sanity (SM33): write-up

Question: the four answer buttons must satisfy interval(Again) <= interval(Hard) <= interval(Good) <= interval(Easy). The fork computes each button from its own what-if answer row (rating replaced, state not committed). By default it hides inversions with PAVA (pooling the four margins R_g(t) - target to be non-decreasing in g at each point of the interval search grid), so a pooled button shows a blend of two grades' curves instead of its own. Stage 1 measures how often the raw curves are inverted; stage 2 tests whether a training-time grade-order hinge removes the inversions at no accuracy cost.

Files: `sm33-stage1.json`, `sm33-stage2-arm.json`, `sm33-stage2-reference.json`, `sm33-decision.{json,md}`, `check-cpu-grade-order.{json,out}` (17 CPU checks, PASS), logs and raw outputs in `gpu/` (`sm33-neighbour.jsonl` = per-sample what-if rows). Protocol and reasoning: `HANDOFF.md`.

## Stage 1: how often the released small model's buttons are inverted

Setup: 722 users of the 732 pool, 3,000 sampled reviews each (2,166,000 probed reviews, representing 42,280,165 reviews), leak-free inputs. Each sampled review gets four probe rows (rating 1..4) inserted right before the real row; unrounded intervals are taken from each grade's curve at desired retention 0.80 / 0.90 / 0.95, and the fork's PAVA display is computed at 0.90. "Raw" = the unrounded curve intervals; "displayed" = rounded to days, before PAVA. "Review-weighted" weights users by their review count; "user-averaged" weights users equally.

### Break rate per adjacent pair and desired retention

| desired retention | pair | raw, review-weighted | raw, user-averaged | displayed, review-weighted | displayed, user-averaged |
|---|---|---|---|---|---|
| 0.80 | Again>Hard | 12.95% | 13.25% | 11.89% | 12.39% |
| 0.80 | Hard>Good | 10.00% | 12.15% | 8.29% | 10.21% |
| 0.80 | Good>Easy | 9.71% | 12.86% | 7.82% | 10.68% |
| 0.80 | any pair | 27.81% | 32.39% | 24.18% | 28.54% |
| 0.90 | Again>Hard | 10.95% | 10.00% | 7.93% | 7.41% |
| 0.90 | Hard>Good | 9.23% | 11.06% | 6.48% | 8.02% |
| 0.90 | Good>Easy | 10.79% | 13.40% | 6.54% | 8.45% |
| 0.90 | any pair | 26.78% | 29.80% | 18.58% | 21.21% |
| 0.95 | Again>Hard | 12.39% | 10.51% | 6.19% | 5.19% |
| 0.95 | Hard>Good | 8.53% | 9.90% | 4.96% | 5.95% |
| 0.95 | Good>Easy | 14.38% | 16.99% | 5.62% | 6.86% |
| 0.95 | any pair | 30.99% | 32.64% | 15.28% | 16.36% |

At 0.90, how large the raw inversions are (review-weighted share of reviews where the lower grade's interval exceeds the higher grade's by more than the given ratio):

| pair | any | ratio > 1.05 | ratio > 1.25 | ratio > 2 |
|---|---|---|---|---|
| Again>Hard | 10.95% | 10.65% | 9.69% | 7.48% |
| Hard>Good | 9.23% | 8.81% | 7.66% | 5.90% |
| Good>Easy | 10.79% | 10.10% | 8.36% | 5.92% |

When broken at 0.90, the median interval ratio is 3.37 (Again>Hard), 3.52 (Hard>Good) and 2.60 (Good>Easy); the 90th percentile ratio is 231, 659 and 52. The median gap is 9.6, 14.0 and 23.6 days. Median intervals at 0.90 (days): Again 0.011, Hard 2.36, Good 7.63, Easy 21.0.

The curves are also inverted at fixed horizons (review-weighted share; the median retrievability gap when broken is between 0.007 and 0.067):

| horizon | Again>Hard | Hard>Good | Good>Easy |
|---|---|---|---|
| 1 day | 12.74% | 9.79% | 8.61% |
| 7 days | 11.68% | 10.88% | 6.90% |
| 30 days | 17.14% | 14.37% | 6.93% |
| 180 days | 33.56% | 22.61% | 8.23% |

### PAVA effect at 0.90

The fork's displayed order is monotone on 100% of reviews (PAVA does its job), but it changes the buttons:

| button | interval changed by PAVA (> 0.5%) | changed by PAVA (> 5%) | displayed days changed | median relative change when changed |
|---|---|---|---|---|
| Again | 12.72% | 12.18% | 9.28% | 56.0% |
| Hard | 20.88% | 19.73% | 14.38% | 61.2% |
| Good | 19.29% | 17.39% | 11.42% | 48.5% |
| Easy | 12.96% | 11.77% | 7.52% | 62.8% |

PAVA also creates ties between displayed buttons: Hard = Good on 40.2% of reviews after PAVA (30.1% raw), Good = Easy on 29.7% after PAVA (21.7% raw).

### Breakdowns at 0.90 (raw break share per pair, review-weighted)

| group | share of reviews | Again>Hard | Hard>Good | Good>Easy |
|---|---|---|---|---|
| first review of the card | 14.45% | 1.86% | 6.94% | 24.48% |
| not first review | 85.55% | 12.49% | 9.62% | 8.48% |
| real rating Again | 12.41% | 13.41% | 5.48% | 9.93% |
| real rating Hard | 14.77% | 19.58% | 16.82% | 18.56% |
| real rating Good | 51.13% | 9.99% | 8.72% | 5.60% |
| real rating Easy | 7.25% | 14.09% | 8.35% | 5.75% |
| elapsed same day | 25.42% | 2.53% | 22.48% | 17.52% |
| elapsed 1-6 d | 29.79% | 16.20% | 4.96% | 4.18% |
| elapsed 7-30 d | 17.88% | 18.39% | 3.38% | 3.77% |
| elapsed 31-180 d | 9.81% | 15.90% | 3.27% | 6.00% |
| elapsed > 180 d | 2.65% | 13.75% | 4.27% | 11.12% |
| Good interval < 1 d | 30.18% | 8.81% | 14.48% | 13.47% |
| Good interval 1-6 d | 18.01% | 13.35% | 7.42% | 2.45% |
| Good interval 7-30 d | 17.70% | 13.42% | 5.54% | 2.91% |
| Good interval 31-180 d | 17.43% | 13.01% | 6.16% | 6.68% |
| Good interval > 180 d | 16.69% | 7.47% | 8.83% | 27.61% |
| user history quartile 1 (shortest) | 3.54% | 9.50% | 12.56% | 16.54% |
| user history quartile 2 | 8.56% | 7.63% | 10.80% | 13.35% |
| user history quartile 3 | 19.36% | 10.92% | 10.68% | 12.26% |
| user history quartile 4 (longest) | 68.53% | 11.45% | 8.46% | 9.76% |

### Per-user distribution at 0.90

| pair | p10 | median | p90 | p99 |
|---|---|---|---|---|
| Again>Hard | 0.07% | 3.35% | 31.88% | 60.92% |
| Hard>Good | 0.30% | 5.08% | 29.55% | 62.13% |
| Good>Easy | 0.87% | 7.65% | 34.56% | 64.03% |

410 of 722 users have at least one pair with a break rate above 10%. The worst users (user id: Again>Hard / Hard>Good / Good>Easy): 111 (102,867 reviews): 80.7% / 61.2% / 34.6%; 44: 16.6% / 79.0% / 63.5%; 2957: 5.7% / 72.3% / 79.0%; 4635: 10.9% / 66.8% / 78.1%; 1894: 3.4% / 23.7% / 77.3%.

## Stage 2: does a training-time grade-order hinge fix it, and at what cost

Arm: the leak-free small model (SM72s) epoch 21 plus the quarter-epoch 32k Muon cool-down with the grade-order hinge. Reference: the same cool-down without it (identical except `grade_order` and paths; 1,991 users, 145,578,259 reviews, same start weights). Both scored leak-free (w8+s8, review-weighted LogLoss) on 722 users; points are of % of the big model (1 leak-free epoch), with a user bootstrap 95% CI. Order is measured on the same 722 x 3,000 sampled reviews in both. Rule (pre-registered): go if both heads are <= +0.03 points and the displayed break share drops with a CI clear of zero; cost if it drops but a head is above +0.03.

### Accuracy

| head | arm | reference | delta | points [95% CI] | users better |
|---|---|---|---|---|---|
| curve | 0.30115 | 0.30073 | +0.000418 | +0.14 [+0.10, +0.19] | 232 |
| instant | 0.27722 | 0.27722 | +0.000007 | +0.00 [-0.00, +0.01] | 354 |

Against the big model (1 leak-free epoch) (curve 0.298966, instant 0.275757): the arm is +0.73% on the curve and +0.53% on instant; the reference is +0.59% and +0.53%.

### Order at desired retention 0.90 (displayed intervals before the fork's PAVA)

| order measure | arm | reference | delta [95% CI] | users better / worse |
|---|---|---|---|---|
| any pair | 0.94% | 19.08% | -18.13% [-20.88%, -15.56%] | 682 / 0 |
| Again>Hard | 0.40% | 9.59% | -9.19% [-11.03%, -7.54%] | 565 / 0 |
| Hard>Good | 0.41% | 6.01% | -5.61% [-6.46%, -4.84%] | 626 / 0 |
| Good>Easy | 0.15% | 5.41% | -5.26% [-6.32%, -4.32%] | 591 / 1 |
| Good changed by PAVA | 0.29% | 10.52% | -10.23% [-11.95%, -8.65%] | 645 / 1 |
| raw (unrounded) any pair | 6.27% | 27.50% | -21.23% [-23.59%, -19.01%] | 722 / 0 |

### Reading

The hinge removes almost all inversions: displayed breaks fall from 19.08% to 0.94% of reviews and raw breaks from 27.50% to 6.27%, with the arm better for 682 users and worse for none; the share of reviews where PAVA changes Good's interval falls from 10.52% to 0.29%. The instant head is unchanged (+0.00 points). The curve head is +0.14 points of % worse than the reference measured against the big model (1 leak-free epoch), with a CI of [+0.10, +0.19] entirely above the +0.03 bar. The order gain is real but the curve accuracy price exceeds the pre-registered bar, so the verdict is **cost**.

### Run log

The arm's cool-down was paused and resumed once from a snapshot at 800 users (the modified `../sm33-arm-e21-dec-q32k-muon/protocol.json` is that resume bookkeeping). The stage 2 chain finished 2026-11-02 02:33 (`gpu/launch-sm33-stage2.log`).
