# Session position effect on answer time (SM85): warm-up or selection?

Users: 150 (the response-time prediction (SM82) stage 2 sample of the 732 set); 4,141,508 test reviews of seen cards on a later day than the card's previous review. Times capped at the user's cap (60 s) and floored at 0.5 s for ln(time); fast Good = Good under 3.9 s. Position = number of earlier reviews that day in the revlog (new cards and same-day steps count), active minutes = recorded time of the earlier reviews that day (breaks are invisible). Effects are relative to the first 10 reviews of the day; intervals are user-bootstrap 95% (2,000 resamples, seed 0).

## Takeaways

- Most of the position effect survives the controls, so it is mostly within-card, not selection: the same card is answered about 26-28% faster at positions 201-400 than in the first 10 reviews of the day (card and day effects + controls, S6: -27.6% [-31.6, -23.7] pooled, -26.6% [-30.1, -22.8] users weighted equally), and the chance of a fast Good (Good under 3.9 s) is +6.2 pp higher (pooled [+4.6, +7.7]; equal-user +5.1 pp [+3.7, +6.5]). The unadjusted gap is bigger (-43.7% time, +11.5 pp), so selection (which users, which cards) explains roughly 40-45% of the raw gradient and a within-card speed-up the rest. It is monotone (positions 11-25: -6%, 51-100: -15%, 101-200: -21%) and present for 95% of users with 5,000+ reviews.
- Selection is real but secondary. User mix (heavy users answer faster) is the biggest part (user effects alone: -31%); the card mix adds little (card effects: -26%); late positions hold slightly older, less often lapsed, previously faster cards (descriptive table). Day-level effects (a good or bad day, long or short days) change almost nothing between card effects (S4) and card and day effects (S6), so users who stop early on bad days are not what drives it.
- Short days do not start clearly slower: for the first 20 reviews of a day, days of 100-399 reviews vs under 30 reviews are +3% to +6% slower with 95% intervals including 0 (card effects + controls), and fast Good is 1-2 pp lower; with user effects only the difference is about 0. At most a small part of the gradient is 'users who stop early answer slower from the start'.
- It tracks the number of reviews already done, not elapsed active minutes: once position and active minutes are both in with card and day effects (S8), the position effects stay (-23.7% at 201-400) and the minutes drop to about 0 (-4.7% at 40+ min); without day effects minutes proxy the day's slowness and the pair is not interpretable (S9). So it behaves like getting into the groove (practice, rhythm), not like a clock effect or fatigue.
- Still confounded: (1) 'warm-up' here only means 'faster later in the day for the same card': the data cannot separate genuine warm-up from rushing or a lower reading depth late in the day (the Again share rises only slightly, 0.123 at 11-25 to 0.139 at 401+, so recall does not obviously suffer, but grade and speed choices are the user's); (2) queue order within a day (learning cards and relearn steps vs reviews, deck order) is not observed beyond the position; (3) time of day and breaks are invisible (active minutes ignore breaks); (4) the card effect only removes what is constant for a card, not its state on that day (the previous grade, predicted recall, interval and review count are controlled linearly); (5) 150 users of the 732 set (the stage 2 sample), their test reviews only.
- Practical bound: if the effect is a pure multiplicative position factor and a scheduler could freely re-order a day's reviews, putting the slowest (hardest) cards late and the quickest ones early would save about 55 s per day on average per user (95% [48, 63]; user quartiles 20 / 42 / 86 s), 4.2% of the 1,222 s of review time in the tested reviews. This is an upper-bound-style figure: it assumes full freedom to re-order, ignores that the learning order carries its own effects (interleaving, learning steps) and that the card difficulty cannot be known perfectly in advance. Less than 5% of daily time, so a modest gain.


## 1. Who is answered where (selection, descriptive)

| position | reviews | fast Good | mean time (s) | predicted recall | mean interval (d) | card reviews before | card lapses before | previous time of the card (s) | day length (reviews) |
|---|---|---|---|---|---|---|---|---|---|
| 1-10 | 386,692 | 0.098 | 18.9 | 0.865 | 41.2 | 8.2 | 1.29 | 16.8 | 148 |
| 11-25 | 442,371 | 0.109 | 16.9 | 0.876 | 40.2 | 7.7 | 1.20 | 15.1 | 172 |
| 26-50 | 546,062 | 0.123 | 15.4 | 0.876 | 35.6 | 6.6 | 1.11 | 14.1 | 208 |
| 51-100 | 742,623 | 0.141 | 14.1 | 0.874 | 31.7 | 6.2 | 1.07 | 13.2 | 267 |
| 101-200 | 843,417 | 0.166 | 12.8 | 0.871 | 28.9 | 6.1 | 1.07 | 12.2 | 369 |
| 201-400 | 698,379 | 0.213 | 11.2 | 0.862 | 28.2 | 5.9 | 1.12 | 10.9 | 556 |
| 401+ | 481,964 | 0.269 | 8.6 | 0.858 | 32.4 | 5.2 | 0.96 | 8.8 | 1103 |

Pooled over reviews (heavy users dominate); the later positions only exist on long days, which is itself selection.

## 2. Position effect on answer time (percent change): controlled estimates (users weighted equally)

Specs: S1 = user effects, position buckets only; S2 = user effects + card-state controls; S3 = card effects, position buckets only; S4 = card effects + controls; S5 = day effects + controls (within one day); S6 = card and day effects + controls; S7 = card and day effects + controls, minutes of active time instead of position; S8 = card and day effects + controls, position and active time together; S9 = card effects + controls, position and active time together.

| position | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 |
|---|---|---|---|---|---|---|---|---|---|
| 11-25 | -8.0% [-10.2, -5.9] | -6.0% [-7.9, -4.2] | -5.1% [-7.0, -3.5] | -5.1% [-6.9, -3.6] | -6.4% [-8.0, -4.9] | -5.8% [-7.3, -4.5] | - | -6.6% [-8.4, -4.9] | -28.2% [-32.6, -23.2] |
| 26-50 | -12.2% [-15.2, -9.3] | -9.8% [-12.2, -7.5] | -8.5% [-10.9, -6.2] | -8.5% [-10.8, -6.5] | -10.8% [-12.8, -8.7] | -9.6% [-11.5, -7.8] | - | -10.4% [-13.1, -7.8] | -45.1% [-51.3, -37.7] |
| 51-100 | -16.9% [-20.4, -13.3] | -14.2% [-17.1, -11.2] | -12.8% [-15.8, -9.7] | -13.0% [-15.7, -10.3] | -15.5% [-17.9, -13.1] | -14.7% [-17.0, -12.2] | - | -14.5% [-17.8, -11.1] | -57.3% [-64.2, -48.4] |
| 101-200 | -22.5% [-26.3, -18.3] | -19.4% [-22.7, -16.0] | -18.2% [-21.9, -14.6] | -18.4% [-21.7, -15.2] | -20.5% [-23.1, -17.7] | -20.3% [-22.9, -17.5] | - | -18.7% [-22.6, -14.4] | -66.5% [-73.6, -56.7] |
| 201-400 | -29.4% [-34.2, -24.2] | -26.1% [-30.4, -21.6] | -25.7% [-31.2, -20.5] | -25.9% [-30.5, -21.1] | -25.4% [-29.1, -21.6] | -26.6% [-30.1, -22.8] | - | -23.7% [-28.3, -18.3] | -73.2% [-80.2, -63.5] |
| 401+ | -41.4% [-48.0, -35.1] | -37.9% [-44.3, -32.1] | -38.1% [-47.3, -29.9] | -37.8% [-46.2, -29.8] | -31.3% [-35.2, -27.1] | -33.0% [-36.7, -29.0] | - | -29.7% [-34.4, -24.2] | -79.0% [-85.6, -69.1] |

## 2. Position effect on share of fast Goods (percentage points): controlled estimates (users weighted equally)

Specs: S1 = user effects, position buckets only; S2 = user effects + card-state controls; S3 = card effects, position buckets only; S4 = card effects + controls; S5 = day effects + controls (within one day); S6 = card and day effects + controls; S7 = card and day effects + controls, minutes of active time instead of position; S8 = card and day effects + controls, position and active time together; S9 = card effects + controls, position and active time together.

| position | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 |
|---|---|---|---|---|---|---|---|---|---|
| 11-25 | +1.1 pp [+0.5, +1.8] | +0.7 pp [+0.1, +1.4] | +0.6 pp [+0.1, +1.3] | +0.7 pp [+0.2, +1.4] | +0.9 pp [+0.4, +1.5] | +1.1 pp [+0.6, +1.7] | - | +1.2 pp [+0.6, +2.0] | +3.4 pp [+1.4, +5.5] |
| 26-50 | +1.8 pp [+1.0, +2.7] | +1.2 pp [+0.5, +2.1] | +1.0 pp [+0.3, +1.9] | +1.2 pp [+0.5, +2.0] | +1.6 pp [+0.8, +2.3] | +1.6 pp [+0.9, +2.5] | - | +1.9 pp [+0.9, +3.0] | +6.2 pp [+2.9, +9.7] |
| 51-100 | +2.5 pp [+1.4, +3.7] | +1.9 pp [+0.8, +3.0] | +1.6 pp [+0.7, +2.6] | +1.9 pp [+1.0, +2.9] | +2.2 pp [+1.3, +3.2] | +2.4 pp [+1.5, +3.4] | - | +2.5 pp [+1.2, +4.0] | +9.0 pp [+4.5, +13.6] |
| 101-200 | +3.5 pp [+2.2, +5.1] | +2.8 pp [+1.6, +4.2] | +2.6 pp [+1.4, +3.8] | +2.9 pp [+1.8, +4.1] | +3.1 pp [+2.1, +4.3] | +3.5 pp [+2.4, +4.7] | - | +3.5 pp [+1.9, +5.5] | +12.2 pp [+6.8, +17.8] |
| 201-400 | +5.3 pp [+3.5, +7.1] | +4.6 pp [+2.9, +6.2] | +4.0 pp [+2.5, +5.5] | +4.4 pp [+2.9, +5.9] | +4.4 pp [+3.1, +6.0] | +5.1 pp [+3.7, +6.5] | - | +5.0 pp [+3.0, +7.2] | +15.2 pp [+9.0, +21.6] |
| 401+ | +8.3 pp [+5.6, +11.0] | +7.4 pp [+4.8, +10.0] | +7.4 pp [+5.0, +10.0] | +7.8 pp [+5.3, +10.4] | +6.8 pp [+4.7, +8.8] | +7.8 pp [+5.7, +10.0] | - | +7.6 pp [+5.1, +10.3] | +19.6 pp [+12.4, +26.3] |

Active-minutes coefficients (vs 0-2 min) in the specs that include them:

| active time | S7 ln time | S8 ln time | S9 ln time | S7 fast Good | S8 fast Good | S9 fast Good |
|---|---|---|---|---|---|---|
| 2-5 min | -3.9% [-5.6, -2.5] | +1.4% [-0.3, +3.1] | +35.6% [+26.9, +43.8] | +0.8 pp [+0.2, +1.4] | -0.1 pp [-0.6, +0.3] | -2.7 pp [-4.8, -0.7] |
| 5-10 min | -7.6% [-9.6, -5.5] | +2.0% [-0.5, +4.7] | +73.1% [+52.7, +94.2] | +1.2 pp [+0.4, +2.0] | -0.5 pp [-1.1, +0.2] | -5.2 pp [-8.8, -1.6] |
| 10-20 min | -11.8% [-14.0, -9.4] | +1.5% [-1.7, +4.8] | +115.0% [+78.8, +155.1] | +1.9 pp [+1.1, +2.9] | -0.4 pp [-1.2, +0.4] | -7.6 pp [-12.2, -2.9] |
| 20-40 min | -18.2% [-20.9, -15.3] | -1.2% [-5.2, +2.7] | +162.2% [+104.5, +230.6] | +3.1 pp [+2.2, +4.2] | -0.1 pp [-1.2, +1.0] | -9.6 pp [-15.4, -4.0] |
| 40+ min | -25.9% [-29.0, -22.9] | -4.7% [-9.2, -0.6] | +229.2% [+137.8, +346.9] | +4.8 pp [+3.5, +6.1] | +0.1 pp [-1.4, +1.4] | -12.8 pp [-19.4, -6.3] |

Pooled over reviews (every review equal), unadjusted vs adjusted. Unadjusted = bucket mean vs the 1-10 mean (ln time for the percent change):

| position | unadjusted time | S1 time | S4 time | S6 time | unadjusted fast Good | S1 fast Good | S4 fast Good | S6 fast Good | Again share (unadjusted) |
|---|---|---|---|---|---|---|---|---|---|
| 11-25 | -10.4% | -7.3% [-10.3, -4.4] | -5.2% [-7.0, -3.4] | -5.9% [-7.5, -4.2] | +1.1 pp | +0.6 pp [+0.0, +1.1] | +0.5 pp [+0.2, +0.9] | +1.0 pp [+0.6, +1.4] | 0.123 (first 10: 0.135) |
| 26-50 | -18.9% | -13.0% [-17.5, -8.8] | -9.1% [-11.5, -6.6] | -10.2% [-12.5, -7.9] | +2.5 pp | +1.2 pp [+0.4, +2.1] | +1.1 pp [+0.5, +1.8] | +1.7 pp [+1.1, +2.4] | 0.122 |
| 51-100 | -27.0% | -18.1% [-23.3, -12.9] | -13.3% [-16.5, -10.1] | -15.3% [-18.3, -12.4] | +4.3 pp | +2.2 pp [+0.9, +3.5] | +2.0 pp [+1.1, +2.9] | +2.8 pp [+1.9, +3.7] | 0.124 |
| 101-200 | -35.3% | -24.4% [-29.3, -19.2] | -18.7% [-22.3, -14.8] | -21.4% [-24.6, -17.9] | +6.8 pp | +3.6 pp [+1.9, +5.3] | +3.1 pp [+1.8, +4.4] | +4.2 pp [+2.9, +5.4] | 0.128 |
| 201-400 | -43.7% | -31.3% [-36.3, -25.7] | -26.2% [-30.8, -21.5] | -27.6% [-31.6, -23.7] | +11.5 pp | +5.5 pp [+3.3, +7.7] | +5.1 pp [+3.2, +6.7] | +6.2 pp [+4.6, +7.7] | 0.137 |
| 401+ | -56.0% | -39.9% [-46.8, -33.3] | -36.6% [-43.2, -30.2] | -34.0% [-37.9, -29.8] | +17.1 pp | +7.4 pp [+3.6, +11.1] | +8.2 pp [+5.1, +10.8] | +9.3 pp [+6.5, +11.6] | 0.139 |

Heterogeneity (S6 per user, users with at least 5,000 reviews, 125 users): the 101-200 coefficient on ln time is negative for 95% of users; quartiles -25.4% / -18.3% / -10.0%.

## 3. Does a short day start slower? (first 20 reviews of the day, effect of the day's length vs under 30 reviews)

| day length | user effects: time | user effects: fast Good | card effects + controls: time | card effects + controls: fast Good |
|---|---|---|---|---|
| 30-99 | -0.4% [-6.5, +4.5] | -0.6 pp [-1.5, +0.5] | +3.1% [-3.2, +7.7] | -1.3 pp [-2.2, -0.4] |
| 100-199 | +0.5% [-5.8, +5.9] | -0.7 pp [-2.0, +0.7] | +5.0% [-2.2, +10.3] | -1.9 pp [-3.0, -0.5] |
| 200-399 | -2.2% [-8.8, +4.1] | -0.1 pp [-1.4, +1.3] | +6.1% [-1.4, +12.5] | -1.6 pp [-2.7, -0.4] |
| 400+ | -6.7% [-13.7, +0.3] | -0.4 pp [-1.9, +1.3] | +4.7% [-3.9, +13.0] | -1.0 pp [-2.7, +0.9] |

## 4. First 10 reviews vs the 201st on (reviews at positions 1-10 or 201+ only)

| model | time | fast Good |
|---|---|---|
| same card (card effects) | -28.5% [-36.6, -21.0] | +3.6 pp [+2.0, +5.4] |
| same card, + controls | -28.1% [-35.0, -21.6] | +4.3 pp [+2.5, +6.2] |
| same card and day (card and day effects) | -24.7% [-29.4, -19.9] | +3.3 pp [+1.9, +4.9] |
| same card and day, + controls | -25.9% [-30.3, -21.1] | +4.1 pp [+2.6, +5.6] |

## 5. Rough bound: what a warm-up-friendly order could save

Take the multiplicative position factors of S6 (pooled): 1-10 x1.000, 11-25 x0.941, 26-50 x0.898, 51-100 x0.847, 101-200 x0.786, 201-400 x0.724, 401+ x0.660. For every day with at least 2 tested reviews, keep the same set of reviews and the same set of positions, divide each time by the factor of its position (the card's own speed) and re-assign: the slowest cards to the positions with the lowest factor, the quickest cards to the highest. Result over 46,789 user-days: **55 s per day** (user average, 95% [48, 63]; user quartiles 20 / 42 / 86 s) out of 1222 s of tested review time per day, i.e. 4.2% (pooled 4.5%, 95% [4.3, 4.7]). Ignores that the factors are estimates (the interval is over users only).
