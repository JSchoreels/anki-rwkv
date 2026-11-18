# Time-drift diagnostic for fluency vs desired retention (SM127): 12 users, days 0-149

Pre-registered in design.md 12.2. Later-day reviews only. Answer time = the whole time on the card (answered within S s means reading the back included). Ratios are ratios of totals over users; CIs paired user bootstrap (10,000 resamples, seed 127).

Identity with the pilot rows (lanes also in the pilot, 15-day buckets of time and load): max relative difference 1.85e-07 over 12 users (OK).

## Seconds per later-day review by 30-day month, and the model's expected seconds (bin midpoints)

| lane | month 1 | month 2 | month 3 | month 4 | month 5 | all | model expected | share >= 60 s |
|---|---|---|---|---|---|---|---|---|
| real schedule, real times | 14.3 | 14.5 | 12.0 | 10.7 | 10.3 | 12.48 | 12.60 | 0.039 |
| real schedule, sampled times fed back (re-anchored) | 15.3 | 14.7 | 12.9 | 12.1 | 11.7 | 13.47 | 13.45 | 0.051 |
| real schedule, sampled times fed back, no re-anchoring | 14.3 | 13.4 | 12.6 | 13.2 | 13.3 | 13.46 | 13.41 | 0.051 |
| real schedule, sampled times, median time in the trunk (re-anchored) | 14.9 | 14.0 | 11.3 | 11.7 | 11.2 | 12.82 | 12.78 | 0.043 |
| desired retention 0.90, feedback | 17.6 | 17.8 | 17.7 | 17.5 | 18.0 | 17.69 | 17.64 | 0.089 |
| desired retention 0.90, no feedback | 16.0 | 15.7 | 14.4 | 13.9 | 14.0 | 14.74 | 14.68 | 0.061 |
| desired retention 0.98, feedback | 15.9 | 15.1 | 14.5 | 13.8 | 13.5 | 14.53 | 14.48 | 0.065 |
| desired retention 0.98, no feedback | 15.0 | 13.5 | 12.8 | 12.0 | 12.3 | 13.06 | 13.01 | 0.054 |

## Ratios of seconds per later-day review (months 1, 5 and all)

| comparison | month 1 | month 5 | all [CI] |
|---|---|---|---|
| real schedule, sampled times fed back (re-anchored) vs real schedule, real times | 1.069 | 1.133 | 1.079 [1.011, 1.154] |
| real schedule, sampled times, median time in the trunk (re-anchored) vs real schedule, real times | 1.046 | 1.087 | 1.027 [0.905, 1.129] |
| real schedule, sampled times fed back, no re-anchoring vs real schedule, real times | 1.004 | 1.293 | 1.078 [0.950, 1.241] |
| real schedule, sampled times fed back, no re-anchoring vs real schedule, sampled times fed back (re-anchored) | 0.939 | 1.142 | 0.999 [0.913, 1.108] |
| desired retention 0.90, feedback vs desired retention 0.90, no feedback | 1.097 | 1.279 | 1.200 [1.127, 1.280] |
| desired retention 0.98, feedback vs desired retention 0.98, no feedback | 1.058 | 1.094 | 1.112 [1.011, 1.212] |

## Persistence: within-user correlation of log answer time, consecutive later-day reviews of a card

| lane | days 0-74 | days 75-149 | both passed, 0-74 | both passed, 75-149 | pairs |
|---|---|---|---|---|---|
| real schedule, real times | 0.420 | 0.329 | 0.441 | 0.368 | 56,490 |
| real schedule, sampled times fed back (re-anchored) | 0.362 | 0.300 | 0.367 | 0.294 | 56,496 |
| real schedule, sampled times fed back, no re-anchoring | 0.333 | 0.278 | 0.340 | 0.285 | 56,496 |
| real schedule, sampled times, median time in the trunk (re-anchored) | 0.163 | 0.130 | 0.148 | 0.096 | 56,496 |
| desired retention 0.90, feedback | 0.201 | 0.220 | 0.218 | 0.238 | 50,832 |
| desired retention 0.90, no feedback | 0.097 | 0.073 | 0.096 | 0.084 | 58,086 |
| desired retention 0.98, feedback | 0.251 | 0.256 | 0.256 | 0.260 | 79,382 |
| desired retention 0.98, no feedback | 0.130 | 0.089 | 0.121 | 0.083 | 92,267 |

Persistence real schedule, sampled times fed back (re-anchored) minus real schedule, real times (whole run): -0.050 [-0.109, +0.022] (2,000 resamples).

Persistence desired retention 0.90, feedback minus desired retention 0.90, no feedback (whole run): +0.124 [+0.097, +0.153] (2,000 resamples).

Persistence real schedule, sampled times, median time in the trunk (re-anchored) minus real schedule, real times (whole run): -0.234 [-0.314, -0.148] (2,000 resamples).

## Twin decomposition of the seconds-per-review ratio

Within = the same unit reviewed in both lanes, slower or faster; unit = card, or card x interval bucket (<= 3, 4-15, 16-63, >= 64 days) x pass/fail (the cell, the pre-registered measure). Within share = mean of the two weightings' log within-ratio / log total ratio; the rest is selection (which cards, intervals and grades are reviewed, including units reviewed in one lane only).

| comparison | ratio | within (card) | within share (card) | within (cell, fb / ref weights) | within share (cell) [CI] | reviews in shared cells (fb / ref) |
|---|---|---|---|---|---|---|
| desired retention 0.90, feedback vs desired retention 0.90, no feedback | 1.200 | 1.125 | 0.65 | 1.120 / 1.097 | 0.57 [0.42, 0.70] | 0.92 / 0.79 |
| desired retention 0.98, feedback vs desired retention 0.98, no feedback | 1.112 | 1.025 | 0.23 | 1.022 / 1.009 | 0.15 [-2.26, 0.63] | 0.93 / 0.90 |
| real schedule, sampled times fed back (re-anchored) vs real schedule, real times | 1.079 | 1.079 | 1.00 | 1.068 / 1.080 | 0.94 [0.64, 1.22] | 0.89 / 0.87 |
| real schedule, sampled times, median time in the trunk (re-anchored) vs real schedule, real times | 1.027 | 1.027 | 1.00 | 1.013 / 1.024 | 0.71 [-1.05, 2.82] | 0.89 / 0.86 |
| real schedule, sampled times fed back, no re-anchoring vs real schedule, real times | 1.078 | 1.078 | 1.00 | 1.080 / 1.090 | 1.08 [0.25, 2.04] | 0.88 / 0.86 |
| real schedule, sampled times fed back (re-anchored) vs real schedule, sampled times, median time in the trunk (re-anchored) | 1.051 | 1.051 | 1.00 | 1.051 / 1.053 | 1.01 [0.78, 1.16] | 0.98 / 0.99 |

## desired retention 0.90, feedback vs desired retention 0.90, no feedback: by interval (days) and grade

| interval | share of reviews (fb / ref) | Again s (fb / ref) | Good s (fb / ref) | all s (fb / ref) |
|---|---|---|---|---|
| 1 | 0.083 / 0.098 | 20.9 / 18.6 | 15.2 / 12.8 | 15.7 / 13.5 |
| 2-3 | 0.106 / 0.104 | 18.9 / 17.3 | 14.7 / 12.0 | 14.8 / 12.3 |
| 4-7 | 0.164 / 0.165 | 22.0 / 18.3 | 15.6 / 12.5 | 16.5 / 13.3 |
| 8-15 | 0.222 / 0.223 | 24.1 / 21.8 | 17.1 / 13.4 | 19.0 / 15.5 |
| 16-31 | 0.200 / 0.203 | 27.3 / 22.9 | 17.0 / 13.6 | 20.2 / 16.8 |
| 32-63 | 0.119 / 0.113 | 24.3 / 20.1 | 15.2 / 12.3 | 17.8 / 15.1 |
| 64-127 | 0.059 / 0.055 | 20.3 / 18.9 | 13.9 / 12.2 | 15.8 / 13.9 |
| >=128 | 0.047 / 0.041 | 23.8 / 23.5 | 12.1 / 11.3 | 16.6 / 16.0 |

Ratio raw 1.200; at the reference's interval x grade mix 1.192.

## real schedule, sampled times fed back (re-anchored) vs real schedule, real times: by interval (days) and grade

| interval | share of reviews (fb / ref) | Again s (fb / ref) | Good s (fb / ref) | all s (fb / ref) |
|---|---|---|---|---|
| 1 | 0.197 / 0.197 | 22.6 / 19.1 | 12.4 / 12.0 | 13.9 / 13.1 |
| 2-3 | 0.133 / 0.133 | 20.5 / 17.6 | 11.9 / 10.9 | 13.2 / 12.0 |
| 4-7 | 0.181 / 0.181 | 20.7 / 20.3 | 12.5 / 11.6 | 14.1 / 13.4 |
| 8-15 | 0.153 / 0.153 | 20.7 / 20.0 | 12.4 / 11.4 | 14.3 / 13.4 |
| 16-31 | 0.138 / 0.138 | 18.4 / 17.5 | 10.9 / 9.6 | 12.3 / 10.8 |
| 32-63 | 0.096 / 0.096 | 18.1 / 16.8 | 10.7 / 9.6 | 12.2 / 11.0 |
| 64-127 | 0.062 / 0.062 | 19.0 / 17.2 | 11.2 / 10.3 | 13.3 / 12.3 |
| >=128 | 0.040 / 0.040 | 22.4 / 21.2 | 9.8 / 10.1 | 13.7 / 13.6 |

Ratio raw 1.079; at the reference's interval x grade mix 1.071.

**Reading (design.md 12.2): (a) drift: a self-feeding loop of sampled answer times.** Within-cell share of the desired retention 0.90 gap 0.57; persistence excess of fed-back sampled times over real on the real schedule -0.050; desired retention 0.90 feedback minus no feedback +0.124. No re-anchoring vs re-anchoring, last month: 1.142 (user-level drift flag).

Cost: 950,973 lane reviews, 1.25 CPU process-hours.
