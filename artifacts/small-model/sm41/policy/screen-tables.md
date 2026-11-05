Rows: `artifacts/small-model/sm41/policy/screen*/rows/*.jsonl`; 20 users, 80 splits.

| policy | users | gap vs FSRS-7, % (95% CI) | median | users cheaper than FSRS-7 | gap vs instant head (0.80-0.90), % (95% CI) | gap vs FSRS-7, 60-day cost |
|---|---|---|---|---|---|---|
| instant-head desired retention, targets 0.80 / 0.85 / 0.90 (the reference, full-run rows) | 19 | +5.6 ± 5.0 | +3.8 | 26% | — | +9.0 ± 6.6 |
| curve-head desired retention (curve-*) | 19 | +9.9 ± 8.2 | +6.3 | 16% | +1.5 ± 6.7 | +8.6 ± 3.9 |
| hybrid (FSRS-7's interval corrected by the instant head), alpha = 0.5 (hyb-*-a0.5) | 19 | +6.0 ± 5.3 | +2.3 | 26% | -0.5 ± 3.9 | +6.0 ± 5.1 |
| instant-head desired retention, target noise s = 0.5 (ihr-*-n0.5) | 18 | +4.2 ± 4.5 | +1.2 | 33% | -2.2 ± 2.8 | +5.0 ± 4.1 |
| instant-head desired retention, time-aware target beta = 0.5 (ihr-*-s0.5) | 18 | +1.2 ± 3.9 | +0.0 | 44% | -4.9 ± 2.4 | +1.3 ± 3.7 |
| instant-head desired retention, time-aware target beta = 1 (ihr-*-s1) | 19 | -6.9 ± 6.0 | -5.5 | 79% | -10.9 ± 4.2 | -4.9 ± 5.2 |
| instant-head desired retention, time-aware target beta = 2 (ihr-*-s2) | 19 | -18.7 ± 8.4 | -12.0 | 95% | -21.2 ± 6.7 | -16.4 ± 8.8 |

| policy | target | realised retention 30-59 | min/day 30-59 | memorised | cost |
|---|---|---|---|---|---|
| ihr-0.80 (full run) | 0.80 | 0.780 | 20.0 | 1040 | 0.864 |
| ihr-0.85 (full run) | 0.85 | 0.824 | 22.4 | 1070 | 0.934 |
| ihr-0.90 (full run) | 0.90 | 0.869 | 25.5 | 1102 | 1.012 |
| fsrs7-0.70 (full run) | 0.70 | 0.684 | 14.3 | 949 | 0.711 |
| fsrs7-0.75 (full run) | 0.75 | 0.710 | 16.3 | 976 | 0.773 |
| fsrs7-0.80 (full run) | 0.80 | 0.748 | 18.1 | 1014 | 0.816 |
| fsrs7-0.85 (full run) | 0.85 | 0.814 | 20.9 | 1052 | 0.893 |
| fsrs7-0.90 (full run) | 0.90 | 0.873 | 23.4 | 1090 | 0.929 |
| fsrs7-0.95 (full run) | 0.95 | 0.930 | 27.1 | 1131 | 0.991 |
| curve-0.80 | 0.80 | 0.767 | 20.2 | 1036 | 0.876 |
| curve-0.85 | 0.85 | 0.821 | 22.8 | 1061 | 0.945 |
| curve-0.90 | 0.90 | 0.870 | 26.3 | 1102 | 1.035 |
| hyb-0.80-a0.5 | 0.80 | 0.760 | 18.6 | 1022 | 0.839 |
| hyb-0.85-a0.5 | 0.85 | 0.812 | 21.2 | 1062 | 0.895 |
| hyb-0.90-a0.5 | 0.90 | 0.868 | 24.1 | 1097 | 0.953 |
| ihr-0.80-n0.5 | 0.80 | 0.785 | 19.5 | 1031 | 0.854 |
| ihr-0.85-n0.5 | 0.85 | 0.822 | 21.6 | 1056 | 0.923 |
| ihr-0.90-n0.5 | 0.90 | 0.864 | 24.7 | 1093 | 0.988 |
| ihr-0.80-s0.5 | 0.80 | 0.773 | 18.8 | 1029 | 0.830 |
| ihr-0.85-s0.5 | 0.85 | 0.819 | 21.2 | 1062 | 0.912 |
| ihr-0.90-s0.5 | 0.90 | 0.865 | 24.1 | 1095 | 0.958 |
| ihr-0.80-s1 | 0.80 | 0.772 | 17.8 | 1017 | 0.802 |
| ihr-0.85-s1 | 0.85 | 0.817 | 19.8 | 1044 | 0.874 |
| ihr-0.90-s1 | 0.90 | 0.868 | 22.0 | 1084 | 0.888 |
| ihr-0.80-s2 | 0.80 | 0.799 | 15.6 | 997 | 0.761 |
| ihr-0.85-s2 | 0.85 | 0.842 | 17.3 | 1020 | 0.786 |
| ihr-0.90-s2 | 0.90 | 0.879 | 19.7 | 1049 | 0.858 |
