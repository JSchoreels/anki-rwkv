# Time to fluency (SM84): how many reviews and seconds until a card is answered Good fast

Extension of response-time prediction (SM82) stage 2 (CPU only). Users: the same 150 users of the 732 set (150 with their own FSRS-6 fit, 0 default). Fluency = the card's first Good answered under 3.9 s on a later-day review (the answer-time head's bins 0-6, ending at 3,891 ms); also under 5.5 s (bins 0-7) and two fast Goods (3.9 s) in a row. Simulation: 2,000 cards per user learned on day 0, followed 365 days with no daily limit, the stage 2 FSRS-6 simulator (user's own fit, one same-day relearning step after a lapse, learning = a fixed first-review cost). Run time 72 s.

## Takeaways

- What a scheduler could do: fluency (a card's first Good answered under 3.9 s) is mostly a property of the user, and second of how high recall is when the card comes back. At desired retention 0.90 (cost model C), the simulation makes 37.5% of a median user's cards fluent by day 365, after about 4.9 reviews and 99 s for the cards that get there; counting the time on cards that never do, it costs 312 s per fluent card. The fastest third of users reach 67% (98 s per fluent card), the slowest third 17% (2,106 s): their chance of a Good under 3.9 s is 0.45 vs 0.05. Under 5.5 s (observed curve): 61% fluent, 153 s per fluent card. Two fast Goods in a row: 2.4% (4.5% when the head's persistence is kept).
- Desired retention: for fluency, higher is cheaper per fluent card, the opposite of memorized cards per second. Relative to 0.90, seconds per fluent card are 6.5x at 0.50, 2.6x at 0.70, 1.6x at 0.80, 0.76x at 0.94 and 0.45x at 0.98, and 89% of users have their cheapest fluency at the 0.98 bound. Two reasons: the head's chance of a fast Good rises steeply near the top (0.12 of Goods at recall 0.90, 0.26 at 0.98; the observed answers agree: 0.12 and 0.24), and high retention means more reviews and more Goods (even with a flat speed curve 0.98 costs 0.67x of 0.90). The price: 0.98 spends 304 s per card in the year vs 158 s at 0.90.
- Interval growth: a fixed growth factor of 1.3-2x reaches fluency more cheaply than one desired retention at the same yearly time (x2: 63% fluent, 205 s per fluent card, 190 s per card; desired retention 0.94: 54%, 224 s, 193 s per card) because it front-loads short, high-recall reviews; but it memorizes less at day 365 (0.944 vs 0.971), and 3-4x is worse than 2x for fluency. This is model-based: in the simulation a fast answer depends on recall only, so the stage 2 link between 1.7-4x growth and fast Goods (the scheduler gives big growth to easy cards) is not a lever here.
- Observed check: on the users' own cards (136 users with at least 50 cards first seen in the log and a year of follow-up), 17% became fluent within a year, after only 2.6 reviews for those that did; the simulation at the matching desired retention says 33% after 4.7. Speed is card-specific: after a fast Good the next review is a fast Good 31% of the time vs 7% otherwise (the head predicts 28% vs 7%, so the model knows it). The simulation spreads fast answers evenly over cards, so its absolute shares are about 2x optimistic; keeping the head's persistence fixes two in a row (3.5% simulated vs 3.2% observed) but not the share ever fluent (29%). Read the policy comparisons, not the absolute shares.
- Cost models: B (fixed pass mix) and C give the same picture (37.5-38.0% fluent at 0.90; 298 vs 312 s per fluent card).

## What is model-based and what is observed

- Model-based: the memory (the user's FSRS-6 fit, not the RWKV model), the per-grade costs over recall (the answer-time head on richer frozen features (SM82c) via stage 2's cost curves: B fixed pass mix, C the grade head's pass mix over recall) and the model 3.9 s fast curve: the head's P(under 3.9 s | Good) over recall, pooled over every curve-grid point of the user (stage 2's counterfactual interval queries, including stretched intervals).
- Observed: the 3.9 s and 5.5 s fast curves from the user's own recorded Good answers at the real intervals, binned by the head's predicted recall (the stored head outputs stop at bin 6, so a 5.5 s model curve needs a GPU re-query that was not done); and the observed reference: cards first seen in the user's revlog with at least 365 days of follow-up, counted the same way on the recorded reviews (the user's real scheduler, not the simulated one).
- In the simulation each Good is fast independently given recall: no card-level speed (some cards are always fast, some never) and no practice effect beyond what recall carries. The persistence check below measures how much this misses.

## 1. The fast curves (median over users)

P(fast | Good) at a given recall:

| recall | model, under 3.9 s | observed, under 3.9 s | observed, under 5.5 s |
|---|---|---|---|
| 0.60 | 0.084 | 0.068 | 0.169 |
| 0.70 | 0.089 | 0.076 | 0.190 |
| 0.80 | 0.097 | 0.096 | 0.222 |
| 0.90 | 0.123 | 0.120 | 0.249 |
| 0.95 | 0.159 | 0.156 | 0.300 |
| 0.98 | 0.257 | 0.238 | 0.422 |

One constant per user (flat model, P(under 3.9 s | Good) at the observed intervals): median 0.173; observed share of Goods under 3.9 s 0.175, under 5.5 s 0.319.

Persistence (every later-day review with an earlier later-day review of the card, median over users): observed P(fast Good) after a fast Good 0.305 vs after any other answer 0.070; the head on the test reviews predicts 0.277 vs 0.069 (actual on the same test reviews after a fast Good 0.291). The plain simulation has no such memory and spreads fast answers over cards evenly, which overstates the share of cards that ever become fluent and understates two in a row. The persistent variant multiplies the model curve by the head's own ratio after a fast Good (median 1.96x) and after any other answer (median 0.89x), per user (1 when fewer than 20 test reviews follow a fast Good): a two-state approximation of card-level speed, still not true card heterogeneity.

## 2. Desired retention, cost model C (median over users)

Share = cards fluent by day 365; reviews / seconds = up to and including the fluent review (seconds include the first learning review), over the cards that got there (users with at least 1% fluent); seconds per fluent card = all seconds spent on every card until it became fluent (or the whole year if it never did) / fluent cards, the honest cost of fluency; memorized = mean recall at day 365; seconds per card = the whole year.

| desired retention | recall at review | reviews per card | seconds per card | memorized | fluent 3.9 s | reviews to 3.9 s | seconds to 3.9 s | seconds per fluent card 3.9 s | fluent 3.9 s, persistent | seconds per fluent card 3.9 s, persistent | fluent 5.5 s (obs curve) | reviews to 5.5 s | seconds per fluent card 5.5 s | two in a row | two in a row, persistent | seconds per fluent card two in a row, persistent |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.50 | 0.500 | 1.241 | 41.700 | 0.637 | 0.019 | 2.450 | 65.224 | 2040.210 | 0.016 | 2184.174 | 0.033 | 2.360 | 950.574 | 0.000 | 0.000 | 29330.031 |
| 0.60 | 0.600 | 2.415 | 68.181 | 0.733 | 0.057 | 2.869 | 82.715 | 1397.752 | 0.051 | 1548.778 | 0.094 | 2.723 | 675.458 | 0.000 | 0.001 | 23121.442 |
| 0.70 | 0.700 | 3.553 | 90.578 | 0.813 | 0.107 | 3.370 | 88.743 | 810.544 | 0.090 | 1089.629 | 0.196 | 3.105 | 377.015 | 0.002 | 0.004 | 9794.371 |
| 0.80 | 0.798 | 5.141 | 115.004 | 0.887 | 0.208 | 3.950 | 91.456 | 527.715 | 0.184 | 569.825 | 0.358 | 3.526 | 236.881 | 0.006 | 0.010 | 6859.254 |
| 0.86 | 0.856 | 7.062 | 140.122 | 0.926 | 0.294 | 4.527 | 100.421 | 374.957 | 0.267 | 431.817 | 0.482 | 3.687 | 178.121 | 0.013 | 0.026 | 3441.956 |
| 0.90 | 0.895 | 8.989 | 158.102 | 0.949 | 0.375 | 4.871 | 99.285 | 311.964 | 0.326 | 374.795 | 0.611 | 3.869 | 153.185 | 0.024 | 0.045 | 2671.239 |
| 0.94 | 0.933 | 12.655 | 193.269 | 0.971 | 0.539 | 5.311 | 99.510 | 224.307 | 0.485 | 269.263 | 0.788 | 4.066 | 121.558 | 0.050 | 0.096 | 1616.877 |
| 0.98 | 0.973 | 25.141 | 303.903 | 0.990 | 0.927 | 5.978 | 93.377 | 122.826 | 0.899 | 148.113 | 0.990 | 4.270 | 91.230 | 0.248 | 0.468 | 480.880 |

## 2b. Desired retention, cost model B (fixed pass mix; median over users)

| desired retention | recall at review | reviews per card | seconds per card | memorized | fluent 3.9 s | reviews to 3.9 s | seconds to 3.9 s | seconds per fluent card 3.9 s | fluent 3.9 s, persistent | seconds per fluent card 3.9 s, persistent | fluent 5.5 s (obs curve) | reviews to 5.5 s | seconds per fluent card 5.5 s | two in a row | two in a row, persistent | seconds per fluent card two in a row, persistent |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.50 | 0.500 | 1.212 | 40.499 | 0.639 | 0.022 | 2.448 | 68.297 | 1739.226 | 0.022 | 2119.794 | 0.040 | 2.338 | 792.878 | 0.000 | 0.000 | 35588.175 |
| 0.60 | 0.600 | 2.282 | 63.391 | 0.738 | 0.058 | 2.758 | 78.045 | 1216.592 | 0.053 | 1387.385 | 0.106 | 2.594 | 585.812 | 0.001 | 0.001 | 19837.768 |
| 0.70 | 0.700 | 3.406 | 87.090 | 0.816 | 0.111 | 3.061 | 81.771 | 790.774 | 0.099 | 926.138 | 0.199 | 2.890 | 322.563 | 0.002 | 0.003 | 10656.663 |
| 0.80 | 0.798 | 4.825 | 108.069 | 0.888 | 0.224 | 3.591 | 88.011 | 489.860 | 0.187 | 585.511 | 0.370 | 3.463 | 235.708 | 0.006 | 0.011 | 6754.073 |
| 0.86 | 0.856 | 6.726 | 131.859 | 0.927 | 0.323 | 4.260 | 89.533 | 357.150 | 0.285 | 457.786 | 0.489 | 3.489 | 167.030 | 0.013 | 0.029 | 3545.671 |
| 0.90 | 0.894 | 8.679 | 148.006 | 0.949 | 0.380 | 4.721 | 91.258 | 298.205 | 0.359 | 355.419 | 0.599 | 3.745 | 142.215 | 0.024 | 0.045 | 2484.369 |
| 0.94 | 0.933 | 12.655 | 188.609 | 0.971 | 0.535 | 5.183 | 99.998 | 210.986 | 0.491 | 270.820 | 0.807 | 4.120 | 117.314 | 0.051 | 0.098 | 1520.348 |
| 0.98 | 0.973 | 29.265 | 332.186 | 0.991 | 0.949 | 6.190 | 95.045 | 119.869 | 0.913 | 144.659 | 0.994 | 4.287 | 87.674 | 0.276 | 0.525 | 446.107 |

## 3. Interval-growth policies (cost model C, median over users)

Next interval = previous x g after a pass (1 day after learning and after a lapse); memory still the user's FSRS-6 fit, fast answers still from recall only.

| policy | recall at review | reviews per card | seconds per card | memorized | fluent 3.9 s | reviews to 3.9 s | seconds per fluent card 3.9 s | fluent 3.9 s, persistent | fluent 5.5 s (obs curve) | seconds per fluent card 5.5 s | two in a row | two in a row, persistent | seconds per fluent card two in a row, persistent |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| x1.3 | 0.957 | 20.028 | 278.256 | 0.991 | 0.906 | 7.123 | 151.674 | 0.861 | 0.988 | 94.925 | 0.227 | 0.379 | 530.006 |
| x1.7 | 0.938 | 13.591 | 213.778 | 0.964 | 0.729 | 5.678 | 184.516 | 0.654 | 0.922 | 102.256 | 0.115 | 0.213 | 892.558 |
| x2 | 0.922 | 11.164 | 189.762 | 0.944 | 0.626 | 4.940 | 205.388 | 0.551 | 0.844 | 109.693 | 0.079 | 0.154 | 1113.935 |
| x2.5 | 0.909 | 9.907 | 172.171 | 0.959 | 0.564 | 4.546 | 221.777 | 0.509 | 0.792 | 122.171 | 0.065 | 0.125 | 1179.639 |
| x3 | 0.890 | 7.962 | 150.641 | 0.946 | 0.476 | 3.961 | 262.561 | 0.412 | 0.692 | 137.117 | 0.040 | 0.082 | 1626.596 |
| x4 | 0.874 | 6.995 | 138.624 | 0.942 | 0.413 | 3.694 | 289.134 | 0.369 | 0.640 | 159.540 | 0.025 | 0.060 | 1973.793 |
| 0.80 | 0.798 | 5.141 | 115.004 | 0.887 | 0.208 | 3.950 | 527.715 | 0.184 | 0.358 | 236.881 | 0.006 | 0.010 | 6859.254 |
| 0.90 | 0.895 | 8.989 | 158.102 | 0.949 | 0.375 | 4.871 | 311.964 | 0.326 | 0.611 | 153.185 | 0.024 | 0.045 | 2671.239 |

Seconds per fluent card relative to desired retention 0.90 (cost model C, median of the user's ratio):

| fluency curve | DR 0.50 | DR 0.70 | DR 0.80 | DR 0.94 | DR 0.98 | growth x1.3 | growth x1.7 | growth x2 | growth x2.5 | growth x3 | growth x4 | share fluent, DR 0.70 minus 0.90 | share fluent, DR 0.98 minus 0.90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model, under 3.9 s | 6.510 | 2.551 | 1.627 | 0.757 | 0.454 | 0.578 | 0.676 | 0.747 | 0.795 | 0.866 | 0.931 | -0.221 | 0.334 |
| model flat, under 3.9 s | 4.206 | 1.877 | 1.385 | 0.854 | 0.665 | 0.751 | 0.814 | 0.866 | 0.890 | 0.943 | 0.983 | -0.224 | 0.223 |
| observed, under 3.9 s | 6.546 | 2.605 | 1.676 | 0.739 | 0.433 | 0.561 | 0.651 | 0.727 | 0.763 | 0.864 | 0.928 | -0.213 | 0.344 |
| observed, under 5.5 s | 5.707 | 2.377 | 1.542 | 0.783 | 0.567 | 0.662 | 0.728 | 0.798 | 0.811 | 0.885 | 0.949 | -0.281 | 0.212 |
| model persistent, under 3.9 s | 6.856 | 2.555 | 1.658 | 0.754 | 0.436 | 0.558 | 0.664 | 0.743 | 0.792 | 0.863 | 0.928 | -0.208 | 0.333 |
| two in a row (model) | 27.636 | 6.018 | 2.724 | 0.567 | 0.192 | 0.301 | 0.417 | 0.534 | 0.637 | 0.773 | 0.951 | -0.022 | 0.179 |
| two in a row (model persistent) | 35.745 | 6.276 | 2.778 | 0.573 | 0.183 | 0.296 | 0.428 | 0.562 | 0.630 | 0.773 | 0.950 | -0.041 | 0.291 |

Per user, the desired retention with the fewest seconds per fluent card (cost model C): model, under 3.9 s: median 0.98 (quartiles 0.98-0.98, at 0.50 0%, at 0.98 89%); model persistent, under 3.9 s: median 0.98 (quartiles 0.98-0.98, at 0.50 0%, at 0.98 88%); observed, under 5.5 s: median 0.98 (quartiles 0.98-0.98, at 0.50 1%, at 0.98 89%); two in a row (model): median 0.98 (quartiles 0.98-0.98, at 0.50 3%, at 0.98 87%); two in a row (model persistent): median 0.98 (quartiles 0.98-0.98, at 0.50 3%, at 0.98 87%).

## 4. By user speed (desired retention 0.90, cost model C)

| median recorded time | users | range (s) | P(under 3.9 s given Good), head | Goods under 5.5 s, observed | fluent 3.9 s (sim) | reviews to 3.9 s | seconds per fluent card 3.9 s | fluent 3.9 s, persistent (sim) | fluent 5.5 s (sim, obs curve) | two in a row, persistent (sim) | fluent 3.9 s (observed cards) | fluent 5.5 s (observed cards) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fastest third | 48 | 0.4-6.0 | 0.448 | 0.638 | 0.671 | 3.076 | 98.315 | 0.616 | 0.817 | 0.165 | 0.426 | 0.577 |
| middle third | 52 | 6.0-11.8 | 0.176 | 0.324 | 0.505 | 4.785 | 248.011 | 0.436 | 0.694 | 0.071 | 0.188 | 0.320 |
| slowest third | 50 | 11.8-45.0 | 0.046 | 0.110 | 0.172 | 6.518 | 2106.191 | 0.158 | 0.255 | 0.008 | 0.050 | 0.100 |

## 5. Observed reference vs the simulation

Observed: the user's cards first seen in the revlog with at least 365 days of follow-up (median 2886 cards per user; 136 users with at least 50), counted on the recorded reviews within 365 days of the first one. Simulated: cost model C at the desired retention closest to the user's observed later-day pass rate (median 0.87). The real scheduler is not FSRS-6 at that target and real cards can be suspended or deleted (counted as never fluent), so this is a sanity check, not a calibration.

| definition | users | fluent by day 365, observed | simulated | reviews to fluency, observed | simulated | seconds to fluency, observed | simulated | seconds per fluent card, observed | simulated |
|---|---|---|---|---|---|---|---|---|---|
| under 3.9 s vs simulated observed, under 3.9 s | 136 | 0.170 | 0.336 | 2.567 | 4.626 | 48.161 | 103.391 | 442.958 | 343.565 |
| under 3.9 s vs simulated model, under 3.9 s | 136 | 0.170 | 0.330 | 2.567 | 4.694 | 48.161 | 102.811 | 442.958 | 350.580 |
| under 3.9 s vs simulated model persistent, under 3.9 s | 136 | 0.170 | 0.294 | 2.567 | 4.869 | 48.161 | 104.433 | 442.958 | 414.068 |
| under 5.5 s vs simulated observed, under 5.5 s | 136 | 0.284 | 0.565 | 2.298 | 3.867 | 52.010 | 91.681 | 295.270 | 195.837 |
| two in a row under 3.9 s vs simulated two in a row (model) | 136 | 0.032 | 0.019 | 3.660 | 6.500 | 40.522 | 137.904 | 1832.387 | 4610.778 |
| two in a row under 3.9 s vs simulated two in a row (model persistent) | 136 | 0.032 | 0.035 | 3.660 | 6.500 | 40.522 | 143.451 | 1832.387 | 3041.970 |

## Limits

- Counterfactual: the model 3.9 s curve comes from stage 2's interval queries, where only the review's query row moves (session counters, other reviews and running statistics stay as observed). The recall bins pool different cards, so part of the slope is card difficulty, not the same card getting faster.
- Independence: in the simulation a Good is fast with a probability that depends on the user and the recall only. Real speed is card-specific and persistent (section 1), so the simulated share of cards ever fluent is too even across cards; read the policy comparisons (relative numbers) rather than the absolute shares.
- No practice effect: the head sees the card's history, but the curve over recall does not carry it; with recall-only fast answers, longer intervals can only slow fluency down (lower recall). The stage 2 link between interval growth 1.7-4x and fast Goods is descriptive (the scheduler gives big growth to easy cards) and cannot be reproduced here.
- The memory model is FSRS-6 (user fits), not the RWKV model; learning is a fixed cost per new card, a lapse gets one relearning step, fluency counts later-day reviews only. 5.5 s uses observed curves only. 150 users of the 732 set.

