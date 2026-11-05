Rows: `C:/Users/jonat/workspace/wt/sm41/artifacts/small-model/sm41/policy/confirm/rows/*.jsonl`; 117 users, 304 splits.

| policy | users | gap vs FSRS-7, % (95% CI) | median | users cheaper than FSRS-7 | gap vs instant head (0.80-0.90), % (95% CI) | gap vs FSRS-7, 60-day cost |
|---|---|---|---|---|---|---|
| instant-head desired retention, targets 0.80 / 0.85 / 0.90 (the reference, full-run rows) | 115 | +9.0 ± 3.3 | +7.1 | 20% | — | +8.7 ± 2.8 |
| instant-head desired retention, target noise s = 0.5 (ihr-*-n0.5) | 116 | +5.3 ± 2.8 | +4.6 | 31% | -3.2 ± 1.1 | +6.5 ± 2.2 |
| instant-head desired retention, time-aware target beta = 2 (ihr-*-s2) | 110 | -22.6 ± 3.2 | -21.7 | 92% | -26.1 ± 2.9 | -20.3 ± 2.8 |

| policy | target | realised retention 30-59 | min/day 30-59 | memorised | cost |
|---|---|---|---|---|---|
| ihr-0.80 (full run) | 0.80 | 0.801 | 20.2 | 1195 | 0.626 |
| ihr-0.85 (full run) | 0.85 | 0.838 | 23.4 | 1225 | 0.691 |
| ihr-0.90 (full run) | 0.90 | 0.879 | 28.1 | 1252 | 0.793 |
| fsrs7-0.70 (full run) | 0.70 | 0.681 | 14.3 | 1111 | 0.496 |
| fsrs7-0.75 (full run) | 0.75 | 0.726 | 16.7 | 1138 | 0.540 |
| fsrs7-0.80 (full run) | 0.80 | 0.769 | 19.6 | 1167 | 0.594 |
| fsrs7-0.85 (full run) | 0.85 | 0.820 | 23.9 | 1203 | 0.673 |
| fsrs7-0.90 (full run) | 0.90 | 0.875 | 29.9 | 1240 | 0.784 |
| fsrs7-0.95 (full run) | 0.95 | 0.929 | 38.9 | 1283 | 0.953 |
| ihr-0.80-n0.5 | 0.80 | 0.806 | 19.4 | 1185 | 0.613 |
| ihr-0.85-n0.5 | 0.85 | 0.838 | 22.4 | 1210 | 0.680 |
| ihr-0.90-n0.5 | 0.90 | 0.875 | 27.0 | 1240 | 0.777 |
| ihr-0.80-s2 | 0.80 | 0.825 | 16.2 | 1151 | 0.541 |
| ihr-0.85-s2 | 0.85 | 0.858 | 18.5 | 1174 | 0.584 |
| ihr-0.90-s2 | 0.90 | 0.893 | 21.6 | 1201 | 0.648 |
