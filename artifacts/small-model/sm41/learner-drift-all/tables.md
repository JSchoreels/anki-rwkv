# The changing-learner simulation (SM41g): tables

Users of the run (splits of a user averaged first), 60 days, seeds 0-3. Gaps: cost of the policy / FSRS-7 - 1, paired per user, mean ± 95% CI over users (%). Change vs no drift = the scenario's gap minus the no-drift gap on the same user (difference in differences, points). No drift = the full run + screen 3 rows.

## Gaps vs FSRS-7

| scenario | policy | users | matched retention (plan) | change vs no drift | matched memorised, days 30-59 | change | matched memorised, 60 days | change |
|---|---|---|---|---|---|---|---|---|
| no drift | instant head | 117 | +9.0 ± 3.3 | — | +3.1 ± 4.3 | — | +1.8 ± 2.8 | — |
| no drift | time-aware beta = 2 | 117 | -22.6 ± 3.2 | — | -6.2 ± 3.7 | — | +1.8 ± 4.8 | — |
| improve | instant head | 117 | +10.4 ± 4.2 | +0.8 ± 4.0 | +2.6 ± 2.4 | -0.7 ± 3.0 | +3.2 ± 2.5 | +1.5 ± 1.7 |
| improve | time-aware beta = 2 | 117 | -13.5 ± 4.3 | +9.3 ± 5.0 | -5.5 ± 2.2 | +0.5 ± 3.5 | -0.8 ± 2.4 | -2.5 ± 4.5 |
| decline | instant head | 117 | +35.0 ± 12.4 | +27.0 ± 12.6 | +18.3 ± 8.7 | +14.0 ± 10.6 | +13.7 ± 6.7 | +10.8 ± 7.8 |
| decline | time-aware beta = 2 | 117 | -25.6 ± 6.7 | -9.3 ± 3.7 | -13.9 ± 5.4 | -7.5 ± 5.4 | -2.3 ± 6.9 | -4.0 ± 7.0 |

## Does it schedule less? (days 30-59)

Per user and target (0.80 / 0.85 / 0.90): ln(scenario / no drift) of the policy minus the same for FSRS-7 at the same target, averaged over the 3 targets, then mean ± 95% CI over users (in %, as 100 x the log difference). Negative = the policy's reviews/day fell more (or rose less) than FSRS-7's. Intervals: the no-drift histogram comes from the `none` rerun.

| scenario | policy | reviews/day vs FSRS-7, change | median later-day interval vs FSRS-7, change | seconds per later-day review vs FSRS-7, change |
|---|---|---|---|---|
| improve | instant head | -6.2 ± 1.4 | +2.9 ± 2.2 | +7.3 ± 3.0 |
| improve | time-aware beta = 2 | +16.9 ± 2.7 | -30.2 ± 3.1 | -9.3 ± 2.5 |
| decline | instant head | +11.9 ± 2.6 | -4.4 ± 2.7 | -3.7 ± 2.1 |
| decline | time-aware beta = 2 | -30.3 ± 5.0 | +35.3 ± 4.3 | +7.4 ± 2.6 |

## Per policy and target (days 30-59, mean over users)

| scenario | lane | realised retention | reviews/day | min/day | s per later-day review | median later-day interval (days) | memorised at day 60 |
|---|---|---|---|---|---|---|---|
| no drift | ihr-0.80 | 0.801 | 62.2 | 20.0 | 22.1 | 13.7 | 1188 |
| no drift | ihr-0.85 | 0.837 | 74.1 | 23.1 | 21.1 | 12.4 | 1217 |
| no drift | ihr-0.90 | 0.880 | 94.0 | 27.8 | 19.7 | 10.5 | 1244 |
| no drift | ihr-0.80-s2 | 0.825 | 61.5 | 16.0 | 15.9 | 16.9 | 1142 |
| no drift | ihr-0.85-s2 | 0.857 | 76.4 | 18.1 | 14.6 | 11.7 | 1166 |
| no drift | ihr-0.90-s2 | 0.893 | 101.0 | 21.2 | 12.9 | 9.2 | 1192 |
| no drift | fsrs7-0.70 | 0.680 | 44.2 | 14.0 | 23.2 | 16.1 | 1101 |
| no drift | fsrs7-0.75 | 0.722 | 53.3 | 16.3 | 22.2 | 14.7 | 1128 |
| no drift | fsrs7-0.80 | 0.765 | 66.2 | 19.1 | 21.1 | 13.9 | 1157 |
| no drift | fsrs7-0.85 | 0.815 | 87.8 | 23.2 | 20.0 | 13.2 | 1193 |
| no drift | fsrs7-0.90 | 0.873 | 123.8 | 28.9 | 17.8 | 10.5 | 1231 |
| no drift | fsrs7-0.95 | 0.929 | 196.7 | 37.4 | 13.9 | 8.0 | 1275 |
| improve | ihr-0.80 | 0.862 | 48.0 | 12.3 | 16.6 | 16.0 | 1252 |
| improve | ihr-0.85 | 0.889 | 55.7 | 13.6 | 15.6 | 14.6 | 1270 |
| improve | ihr-0.90 | 0.921 | 67.7 | 15.3 | 14.0 | 12.5 | 1286 |
| improve | ihr-0.80-s2 | 0.903 | 60.7 | 10.9 | 10.4 | 15.9 | 1234 |
| improve | ihr-0.85-s2 | 0.924 | 74.1 | 11.8 | 9.3 | 10.7 | 1250 |
| improve | ihr-0.90-s2 | 0.946 | 93.0 | 12.9 | 8.0 | 8.4 | 1266 |
| improve | fsrs7-0.70 | 0.762 | 36.4 | 9.3 | 17.8 | 17.7 | 1192 |
| improve | fsrs7-0.75 | 0.798 | 42.3 | 10.3 | 16.7 | 16.8 | 1213 |
| improve | fsrs7-0.80 | 0.836 | 50.9 | 11.6 | 15.6 | 15.9 | 1233 |
| improve | fsrs7-0.85 | 0.876 | 64.4 | 12.9 | 14.4 | 15.0 | 1257 |
| improve | fsrs7-0.90 | 0.921 | 91.4 | 14.5 | 12.1 | 12.2 | 1282 |
| improve | fsrs7-0.95 | 0.962 | 158.5 | 16.1 | 8.4 | 9.2 | 1306 |
| decline | ihr-0.80 | 0.692 | 170.0 | 74.6 | 30.1 | 11.0 | 998 |
| decline | ihr-0.85 | 0.728 | 229.3 | 99.8 | 29.4 | 9.9 | 997 |
| decline | ihr-0.90 | 0.766 | 322.9 | 137.8 | 28.4 | 8.4 | 993 |
| decline | ihr-0.80-s2 | 0.707 | 85.1 | 29.6 | 23.6 | 17.3 | 971 |
| decline | ihr-0.85-s2 | 0.742 | 102.0 | 34.2 | 22.6 | 12.4 | 989 |
| decline | ihr-0.90-s2 | 0.780 | 140.1 | 45.9 | 21.1 | 9.9 | 1001 |
| decline | fsrs7-0.70 | 0.574 | 90.4 | 36.9 | 30.9 | 13.2 | 931 |
| decline | fsrs7-0.75 | 0.613 | 127.1 | 52.7 | 30.0 | 12.6 | 945 |
| decline | fsrs7-0.80 | 0.654 | 173.1 | 71.2 | 29.1 | 11.6 | 961 |
| decline | fsrs7-0.85 | 0.704 | 244.4 | 101.6 | 28.4 | 10.8 | 981 |
| decline | fsrs7-0.90 | 0.762 | 339.0 | 136.4 | 26.8 | 8.7 | 998 |
| decline | fsrs7-0.95 | 0.816 | 490.2 | 191.7 | 23.8 | 6.5 | 1008 |

## Speed (CPU, 1 thread per process)

| scenario | lane reviews | engine process-hours | lane reviews/s per process | warm-up process-hours |
|---|---|---|---|---|
| fused (all 3 scenarios, one run) | 277,816,034 | 89.33 | 864 | 3.49 |
