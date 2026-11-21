# Fluency vs desired retention (SM127): full, 150 of 150 users

Answer time = the whole time on the card, question shown to grade press, capped ("answered within the user's own S s, reading the back included (S = the 90th percentile of the user's own pre-split passed-review times)"). Equal daily time: every policy lane has the user's real mean review seconds per day. Memorised / fluent = expected cards at day 365 (sums of P).

| policy | min/day | reviews/day | memorised | fluent | P>=0.9 | fluent>=0.8 | both | new cards shown | retention | passed within own S (all / last 65 d) | floor share | budget-hit days |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tf | 13.5 | 48 | 2568 | 2180 | 1496 | 1333 | 1165 | 2011 | 0.847 | 0.717 / 0.614 | 0.000 | -1 |
| fr-shared1 | 13.1 | 48 | 2531 | 2133 | 1432 | 1257 | 1121 | 2011 | 0.841 | 0.703 / 0.605 | 0.000 | -1 |
| ihr-0.90 | 11.8 | 40 | 2171 | 1781 | 1660 | 1229 | 1101 | 1115 | 0.854 | 0.656 / 0.661 | 0.000 | 249 |
| ihr-0.95 | 12.5 | 46 | 2030 | 1685 | 1799 | 1321 | 1270 | 896 | 0.892 | 0.713 / 0.716 | 0.000 | 280 |
| ihr-0.98 | 13.2 | 58 | 1930 | 1639 | 1720 | 1366 | 1320 | 768 | 0.918 | 0.770 / 0.773 | 0.000 | 304 |
| flu-0.50 | 11.1 | 28 | 2126 | 1685 | 927 | 575 | 478 | 1374 | 0.774 | 0.437 / 0.432 | 0.225 | 200 |
| flu-0.70 | 13.3 | 35 | 1846 | 1474 | 1068 | 661 | 546 | 830 | 0.858 | 0.558 / 0.559 | 0.384 | 299 |
| flu-0.80 | 13.6 | 40 | 1683 | 1375 | 1210 | 869 | 744 | 590 | 0.884 | 0.632 / 0.632 | 0.476 | 321 |
| flu-0.80-fb95 | 12.5 | 39 | 1995 | 1627 | 1493 | 1086 | 944 | 920 | 0.870 | 0.648 / 0.649 | 0.428 | 277 |
| ihr-0.90-nofb | 11.6 | 43 | 2232 | 1901 | 1778 | 1492 | 1319 | 1156 | 0.859 | 0.705 / 0.722 | 0.000 | 243 |
| ihr-0.98-nofb | 12.9 | 55 | 1741 | 1502 | 1614 | 1338 | 1306 | 543 | 0.918 | 0.787 / 0.799 | 0.000 | 309 |
| flu-0.80-nofb | 13.1 | 45 | 2099 | 1852 | 1487 | 1479 | 1209 | 1033 | 0.869 | 0.711 / 0.729 | 0.275 | 277 |

Best desired-retention arm by fluent cards (population): ihr-0.90.

Primary (design.md amendment 11-18): ratio of population totals (sum over users of a / sum of b - 1), 95% paired user-bootstrap CI (10,000 resamples, seed 127); then the median per-user ratio.

| comparison | fluent: totals [CI] | fluent: median user | memorised: totals [CI] | memorised: median user | minutes: totals [CI] | minutes: median user |
|---|---|---|---|---|---|---|
| flu-0.80 vs ihr-0.90 | -22.78% [-26.91, -18.78] | -19.77% | -22.49% [-26.50, -18.54] | -17.64% | +15.14% [+8.81, +23.12] | +1.07% |
| flu-0.80 vs ihr-0.95 | -18.39% [-22.73, -14.36] | -15.41% | -17.11% [-21.25, -13.13] | -11.81% | +8.89% [+4.51, +14.27] | +0.07% |
| flu-0.80 vs ihr-0.98 | -16.07% [-20.58, -11.95] | -12.71% | -12.84% [-17.19, -8.69] | -7.22% | +2.83% [-0.07, +6.33] | +0.03% |
| flu-0.70 vs ihr-0.90 | -17.23% [-21.65, -12.85] | -14.66% | -14.96% [-19.04, -10.94] | -11.27% | +12.23% [+6.33, +19.57] | +0.03% |
| flu-0.80-fb95 vs ihr-0.90 | -8.66% [-10.86, -6.52] | -7.71% | -8.11% [-10.26, -6.17] | -5.63% | +5.57% [+3.06, +8.53] | +1.08% |
| ihr-0.98 vs ihr-0.90 | -7.99% [-11.60, -4.37] | -5.20% | -11.07% [-14.39, -7.87] | -5.19% | +11.96% [+7.02, +17.99] | +0.93% |
| flu-0.80-nofb vs ihr-0.98-nofb | +23.34% [+16.97, +30.68] | +20.11% | +20.59% [+13.52, +28.31] | +20.30% | +1.70% [-2.14, +5.89] | +0.24% |
| flu-0.80-nofb vs ihr-0.90-nofb | -2.58% [-5.80, +0.28] | -0.02% | -5.94% [-9.83, -2.45] | -0.91% | +12.46% [+5.95, +20.50] | +0.71% |
| ihr-0.98-nofb vs ihr-0.90-nofb | -21.02% [-25.21, -17.08] | -21.80% | -22.01% [-26.19, -18.15] | -22.44% | +10.58% [+5.88, +16.34] | +0.97% |
| flu-0.80 vs flu-0.80-nofb | -25.75% [-30.80, -20.62] | -21.63% | -19.85% [-24.60, -15.13] | -14.08% | +3.90% [+1.57, +6.52] | +0.95% |
| ihr-0.98 vs ihr-0.98-nofb | +9.12% [+3.70, +15.67] | +4.06% | +10.89% [+6.72, +16.10] | +4.34% | +2.76% [-0.17, +6.72] | +0.76% |
| ihr-0.90 vs ihr-0.90-nofb | -6.33% [-9.43, -3.31] | -3.01% | -2.75% [-4.93, -0.65] | -0.67% | +1.49% [-0.49, +3.95] | +0.52% |

Secondary (the pilot report's statistic, unchanged): mean of per-user ratios (a / b - 1), 95% user bootstrap CI. It explodes when a user's b is near 0.

| comparison | fluent | memorised | minutes |
|---|---|---|---|
| flu-0.80 vs ihr-0.90 | -21.10% [-27.50, -11.93] | -22.66% [-26.24, -19.03] | +39.65% [+22.13, +62.20] |
| flu-0.80 vs ihr-0.95 | -18.01% [-22.38, -13.45] | -16.73% [-20.14, -13.37] | +19.86% [+9.35, +32.42] |
| flu-0.80 vs ihr-0.98 | -14.23% [-19.57, -7.39] | -9.45% [-14.63, -2.89] | +5.85% [+1.59, +10.67] |
| flu-0.70 vs ihr-0.90 | -18.40% [-23.43, -12.76] | -15.92% [-19.90, -11.98] | +31.32% [+15.64, +50.19] |
| flu-0.80-fb95 vs ihr-0.90 | -10.56% [-13.71, -6.97] | -10.13% [-12.41, -7.95] | +11.20% [+7.41, +15.35] |
| ihr-0.98 vs ihr-0.90 | -3.43% [-10.89, +6.89] | -10.74% [-14.62, -6.86] | +25.38% [+15.99, +36.87] |
| flu-0.80-nofb vs ihr-0.98-nofb | +43.20% [+32.13, +55.79] | +43.95% [+32.61, +56.32] | +0.82% [-4.00, +6.60] |
| flu-0.80-nofb vs ihr-0.90-nofb | -3.30% [-6.53, -0.17] | -5.02% [-8.26, -1.95] | +31.07% [+14.99, +51.33] |
| ihr-0.98-nofb vs ihr-0.90-nofb | -23.46% [-27.46, -19.49] | -25.14% [-29.04, -21.41] | +26.81% [+17.68, +37.51] |
| flu-0.80 vs flu-0.80-nofb | +14.98% [-25.35, +74.80] | -10.69% [-21.01, +3.87] | +10.69% [+6.33, +15.66] |
| ihr-0.98 vs ihr-0.98-nofb | +43.31% [+19.87, +76.05] | +29.04% [+18.48, +41.85] | +2.65% [+0.10, +5.44] |
| ihr-0.90 vs ihr-0.90-nofb | -3.66% [-7.70, +0.46] | -1.26% [-3.80, +1.37] | +3.75% [+1.14, +6.84] |

**Verdict (decision rule of design.md 7 on the primary statistic, the ratio of totals; feedback on): a high desired retention is enough (the fluency scheduler adds less than 2% fluent cards).** flu-0.80 vs ihr-0.90: fluent -22.78% [-26.91, -18.78] | -19.77%; memorised -22.49% [-26.50, -18.54] | -17.64%.
On the secondary statistic (mean of per-user ratios), as in the pilot report: a high desired retention is enough (the fluency scheduler adds less than 2% fluent cards).
Without answer-time feedback (ratio of totals): flu-0.80-nofb vs ihr-0.90-nofb fluent -2.58% [-5.80, +0.28] | -0.02%; memorised -5.94% [-9.83, -2.45] | -0.91% -> a high desired retention is enough (the fluency scheduler adds less than 2% fluent cards).
Without answer-time feedback (secondary, mean of per-user ratios): flu-0.80-nofb vs ihr-0.90-nofb fluent -3.30% [-6.69, -0.24] -> enough.

Drift check (daily shared-stream re-anchoring (fr-shared1) minus teacher forcing, first-touch predicted pass, whole year): -0.75% [-2.28, +0.47].

Level check (design.md 13.5): seconds per later-day review, daily re-anchoring (fr-shared1) over teacher forcing (real times), ratio of totals: 0.960 [0.902, 1.005] (tf 16.84 s, fr-shared1 16.17 s); accepted band [0.95, 1.05]: accepted.

Per-user S (s): median 31.9, quartiles 17.1-45.0, range 5.8-45.0. Final answer-time level scale: median 0.983, quartiles 0.917-1.029, range 0.774-1.478.

Cost: 52,169,776 lane reviews, 41.47 CPU process-hours (400.6 lane reviews/s per process).
