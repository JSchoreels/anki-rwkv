# Fluency vs desired retention (SM127): pilot, 12 of 12 users

Answer time = the whole time on the card, question shown to grade press, capped ("answered within 8 s, reading the back included"). Equal daily time: every policy lane has the user's real mean review seconds per day. Memorised / fluent = expected cards at day 365 (sums of P).

| policy | min/day | reviews/day | memorised | fluent | P>=0.9 | fluent>=0.8 | both | new cards shown | retention | passed within 8 s (all / last 65 d) | floor share | budget-hit days |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tf | 14.6 | 59 | 3045 | 1452 | 1097 | 215 | 202 | 3078 | 0.845 | 0.410 / 0.312 | 0.000 | -1 |
| fr-shared1 | 15.2 | 59 | 3052 | 1370 | 1120 | 208 | 202 | 3078 | 0.832 | 0.387 / 0.304 | 0.000 | -1 |
| ihr-0.90 | 14.8 | 48 | 1887 | 907 | 1512 | 165 | 161 | 888 | 0.868 | 0.311 / 0.302 | 0.000 | 316 |
| ihr-0.95 | 14.9 | 53 | 1542 | 782 | 1470 | 259 | 257 | 451 | 0.901 | 0.358 / 0.345 | 0.000 | 342 |
| ihr-0.98 | 15.3 | 67 | 1435 | 867 | 1406 | 474 | 474 | 286 | 0.924 | 0.451 / 0.477 | 0.000 | 352 |
| flu-0.50 | 15.4 | 51 | 1421 | 681 | 926 | 44 | 29 | 406 | 0.898 | 0.270 / 0.287 | 0.646 | 348 |
| flu-0.70 | 15.5 | 60 | 1361 | 652 | 1174 | 43 | 42 | 259 | 0.916 | 0.383 / 0.391 | 0.713 | 352 |
| flu-0.80 | 15.6 | 70 | 1418 | 772 | 1259 | 132 | 127 | 316 | 0.926 | 0.466 / 0.479 | 0.741 | 352 |
| flu-0.80-fb95 | 14.9 | 53 | 1505 | 747 | 1410 | 176 | 172 | 416 | 0.902 | 0.363 / 0.363 | 0.783 | 344 |
| ihr-0.90-nofb | 14.6 | 56 | 2184 | 1186 | 1867 | 215 | 211 | 1200 | 0.872 | 0.376 / 0.379 | 0.000 | 288 |
| ihr-0.98-nofb | 15.2 | 68 | 1344 | 788 | 1354 | 284 | 284 | 184 | 0.933 | 0.443 / 0.451 | 0.000 | 352 |
| flu-0.80-nofb | 15.7 | 68 | 1253 | 768 | 1201 | 217 | 210 | 112 | 0.931 | 0.443 / 0.453 | 0.690 | 348 |

Best desired-retention arm by fluent cards (population): ihr-0.90.

| comparison | fluent | memorised | minutes |
|---|---|---|---|
| flu-0.80 vs ihr-0.90 | +54.59% [-29.31, +173.49] | -23.95% [-38.05, -8.32] | +5.39% [-1.80, +14.69] |
| flu-0.80 vs ihr-0.95 | +147.76% [-8.95, +406.87] | +2.74% [-20.77, +38.50] | +2.61% [-0.85, +8.35] |
| flu-0.80 vs ihr-0.98 | +59.13% [-13.72, +159.26] | +9.70% [-12.35, +42.40] | +1.04% [-0.45, +3.05] |
| flu-0.70 vs ihr-0.90 | +21.51% [-41.07, +107.43] | -27.99% [-39.54, -16.55] | +5.20% [-2.09, +14.22] |
| flu-0.80-fb95 vs ihr-0.90 | -22.30% [-36.87, -9.05] | -23.97% [-32.79, -15.23] | +2.63% [-1.81, +8.19] |
| ihr-0.98 vs ihr-0.90 | +8.97% [-30.25, +69.90] | -26.83% [-38.14, -16.46] | +4.19% [-2.29, +12.35] |
| flu-0.80-nofb vs ihr-0.98-nofb | +3.02% [-22.81, +36.89] | -4.02% [-34.61, +33.26] | +0.14% [-6.31, +6.40] |
| flu-0.80-nofb vs ihr-0.90-nofb | -41.39% [-58.02, -25.69] | -45.00% [-61.01, -30.09] | +7.41% [-2.57, +20.56] |
| ihr-0.98-nofb vs ihr-0.90-nofb | -38.88% [-54.35, -23.91] | -42.41% [-52.74, -32.46] | +7.89% [-2.23, +19.57] |
| flu-0.80 vs flu-0.80-nofb | +1012.54% [+21.62, +2801.63] | +231.92% [-2.95, +615.58] | +2.35% [-2.78, +9.13] |
| ihr-0.98 vs ihr-0.98-nofb | +247.53% [+1.90, +613.26] | +21.14% [-2.70, +47.45] | +0.33% [-0.46, +1.14] |

**Verdict (decision rule of design.md 7, feedback on): inconclusive.**
Without answer-time feedback: flu-0.80-nofb vs ihr-0.90-nofb fluent -41.39% [-57.80, -26.07] -> enough.

Drift check (daily shared-stream re-anchoring (fr-shared1) minus teacher forcing, first-touch predicted pass, whole year): -0.28% [-3.22, +2.51].

Cost: 5,738,545 lane reviews, 4.64 CPU process-hours (383.2 lane reviews/s per process).
