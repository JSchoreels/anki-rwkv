# The changing-learner simulation (SM41g): tables

20 users of part A (1 split each), 60 days, seeds 0-3. Gaps: cost of the policy / FSRS-7 - 1, paired per user, mean ± 95% CI over users (%). Change vs no drift = the scenario's gap minus the no-drift gap on the same user (difference in differences, points). No drift = the full run + screen 3 rows.

## Gaps vs FSRS-7

| scenario | policy | users | matched retention (plan) | change vs no drift | matched memorised, days 30-59 | change | matched memorised, 60 days | change |
|---|---|---|---|---|---|---|---|---|
| no drift | instant head | 20 | +5.6 ± 5.0 | — | +0.6 ± 6.3 | — | +5.2 ± 15.4 | — |
| no drift | time-aware beta = 2 | 20 | -18.7 ± 8.4 | — | -7.5 ± 7.9 | — | +0.8 ± 9.5 | — |
| improve | instant head | 20 | +2.7 ± 4.7 | -2.9 ± 1.5 | -2.4 ± 7.3 | -2.9 ± 5.6 | +1.5 ± 8.7 | -3.7 ± 10.6 |
| improve | time-aware beta = 2 | 20 | -15.1 ± 8.3 | +5.7 ± 4.2 | -7.8 ± 8.1 | -0.3 ± 5.0 | -1.0 ± 10.2 | -1.7 ± 7.5 |
| decline | instant head | 20 | +29.7 ± 27.0 | +23.0 ± 25.4 | +8.9 ± 10.7 | +8.3 ± 7.8 | +10.7 ± 14.4 | +5.4 ± 7.7 |
| decline | time-aware beta = 2 | 20 | -32.7 ± 9.9 | -13.4 ± 9.2 | -19.0 ± 8.6 | -11.6 ± 8.7 | -5.6 ± 12.0 | -6.4 ± 10.1 |
| none | instant head | 20 | +5.6 ± 5.0 | +0.0 ± 0.1 | +0.6 ± 6.3 | +0.0 ± 0.1 | +5.3 ± 15.4 | +0.0 ± 0.1 |
| none | time-aware beta = 2 | 20 | -18.7 ± 8.4 | -0.0 ± 0.0 | -7.5 ± 7.9 | -0.0 ± 0.0 | +0.8 ± 9.5 | -0.0 ± 0.1 |

## Does it schedule less? (days 30-59)

Per user and target (0.80 / 0.85 / 0.90): ln(scenario / no drift) of the policy minus the same for FSRS-7 at the same target, averaged over the 3 targets, then mean ± 95% CI over users (in %, as 100 x the log difference). Negative = the policy's reviews/day fell more (or rose less) than FSRS-7's. Intervals: the no-drift histogram comes from the `none` rerun.

| scenario | policy | reviews/day vs FSRS-7, change | median later-day interval vs FSRS-7, change | seconds per later-day review vs FSRS-7, change |
|---|---|---|---|---|
| improve | instant head | -7.1 ± 3.1 | +4.1 ± 6.9 | +7.1 ± 5.5 |
| improve | time-aware beta = 2 | +17.8 ± 9.2 | -27.1 ± 10.2 | -9.1 ± 8.4 |
| decline | instant head | +9.9 ± 4.5 | -2.0 ± 14.6 | -4.4 ± 4.7 |
| decline | time-aware beta = 2 | -30.6 ± 14.9 | +38.4 ± 15.2 | +7.4 ± 7.0 |

## Per policy and target (days 30-59, mean over users)

| scenario | lane | realised retention | reviews/day | min/day | s per later-day review | median later-day interval (days) | memorised at day 60 |
|---|---|---|---|---|---|---|---|
| no drift | ihr-0.80 | 0.780 | 68.1 | 20.0 | 19.1 | 18.3 | 1040 |
| no drift | ihr-0.85 | 0.824 | 78.3 | 22.4 | 17.7 | 15.6 | 1070 |
| no drift | ihr-0.90 | 0.869 | 94.2 | 25.5 | 16.1 | 11.1 | 1102 |
| no drift | ihr-0.80-s2 | 0.799 | 64.2 | 15.6 | 13.6 | 15.2 | 997 |
| no drift | ihr-0.85-s2 | 0.842 | 79.5 | 17.3 | 11.9 | 12.5 | 1020 |
| no drift | ihr-0.90-s2 | 0.879 | 100.3 | 19.7 | 10.6 | 9.4 | 1049 |
| no drift | fsrs7-0.70 | 0.684 | 50.7 | 14.3 | 20.1 | 14.8 | 949 |
| no drift | fsrs7-0.75 | 0.710 | 57.6 | 16.3 | 20.1 | 20.5 | 976 |
| no drift | fsrs7-0.80 | 0.748 | 65.3 | 18.1 | 18.7 | 19.1 | 1014 |
| no drift | fsrs7-0.85 | 0.814 | 77.8 | 20.9 | 17.1 | 12.1 | 1052 |
| no drift | fsrs7-0.90 | 0.873 | 97.9 | 23.4 | 14.7 | 9.9 | 1090 |
| no drift | fsrs7-0.95 | 0.930 | 149.3 | 27.1 | 11.1 | 6.7 | 1131 |
| no drift | ihr-0.90-free | 0.876 | 172.5 | 54.3 | 18.5 | 12.2 | 1081 |
| no drift | fsrs7-0.90-free | 0.868 | 95.2 | 26.7 | 16.5 | 9.8 | 1081 |
| improve | ihr-0.80 | 0.845 | 52.5 | 11.4 | 13.9 | 20.4 | 1099 |
| improve | ihr-0.85 | 0.879 | 59.7 | 12.4 | 12.5 | 17.5 | 1113 |
| improve | ihr-0.90 | 0.912 | 69.2 | 13.3 | 11.2 | 14.0 | 1132 |
| improve | ihr-0.80-s2 | 0.893 | 66.7 | 10.1 | 8.1 | 14.1 | 1080 |
| improve | ihr-0.85-s2 | 0.916 | 78.8 | 10.8 | 7.1 | 11.7 | 1092 |
| improve | ihr-0.90-s2 | 0.940 | 91.2 | 11.2 | 6.4 | 9.0 | 1112 |
| improve | fsrs7-0.70 | 0.766 | 41.2 | 8.8 | 15.0 | 16.0 | 1031 |
| improve | fsrs7-0.75 | 0.788 | 46.6 | 9.8 | 14.7 | 21.8 | 1051 |
| improve | fsrs7-0.80 | 0.820 | 53.1 | 10.7 | 13.3 | 20.5 | 1077 |
| improve | fsrs7-0.85 | 0.877 | 63.4 | 11.7 | 11.5 | 13.6 | 1102 |
| improve | fsrs7-0.90 | 0.922 | 80.5 | 12.4 | 9.5 | 11.5 | 1128 |
| improve | fsrs7-0.95 | 0.961 | 134.2 | 13.3 | 6.4 | 7.7 | 1152 |
| improve | ihr-0.90-free | 0.952 | 46.5 | 8.3 | 10.9 | 18.2 | 1157 |
| improve | fsrs7-0.90-free | 0.959 | 67.1 | 10.1 | 8.7 | 13.2 | 1158 |
| decline | ihr-0.80 | 0.674 | 126.0 | 53.8 | 26.8 | 15.6 | 902 |
| decline | ihr-0.85 | 0.721 | 147.8 | 62.8 | 25.6 | 12.1 | 940 |
| decline | ihr-0.90 | 0.771 | 179.1 | 75.6 | 24.6 | 8.8 | 943 |
| decline | ihr-0.80-s2 | 0.664 | 73.5 | 27.0 | 21.4 | 16.0 | 864 |
| decline | ihr-0.85-s2 | 0.709 | 82.7 | 29.5 | 20.0 | 13.3 | 887 |
| decline | ihr-0.90-s2 | 0.766 | 103.0 | 35.8 | 18.4 | 10.0 | 917 |
| decline | fsrs7-0.70 | 0.571 | 81.1 | 31.7 | 27.5 | 12.6 | 816 |
| decline | fsrs7-0.75 | 0.603 | 90.8 | 35.6 | 27.5 | 18.9 | 843 |
| decline | fsrs7-0.80 | 0.642 | 105.7 | 41.8 | 26.9 | 16.3 | 874 |
| decline | fsrs7-0.85 | 0.711 | 126.0 | 50.8 | 25.5 | 10.8 | 929 |
| decline | fsrs7-0.90 | 0.779 | 158.1 | 62.3 | 23.6 | 7.7 | 971 |
| decline | fsrs7-0.95 | 0.844 | 222.5 | 81.0 | 20.2 | 5.4 | 1009 |
| decline | ihr-0.90-free | 0.282 | 1957.1 | 780.9 | 32.2 | 2.3 | 172 |
| decline | fsrs7-0.90-free | 0.355 | 853.7 | 328.0 | 29.9 | 4.7 | 341 |

## Re-anchoring check: cost gap instant head / FSRS-7 at target 0.90 (days 30-59 minutes per memorised card)

| scenario | anchored | change vs no drift | `-free` | change vs no drift |
|---|---|---|---|---|
| no drift | +8.3 ± 9.6 | — | +577.2 ± 1197.6 | — |
| improve | +6.5 ± 8.2 | -1.9 ± 4.8 | -11.0 ± 9.7 | -588.3 ± 1191.7 |
| decline | +34.2 ± 32.8 | +25.9 ± 27.2 | +14027.8 ± 22870.5 | +13450.6 ± 21707.8 |

## Speed (CPU, 1 thread per process)

| scenario | lane reviews | engine process-hours | lane reviews/s per process | warm-up process-hours |
|---|---|---|---|---|
| improve | 4,769,515 | 1.81 | 732 | 0.44 |
| decline | 13,401,350 | 7.92 | 470 | 0.44 |
| none | 5,688,921 | 2.66 | 595 | 0.43 |
