Rows: `artifacts/small-model/sm41/policy/screen1/rows/*.jsonl`; 20 users, 20 splits.

| policy | users | gap vs FSRS-7, % (95% CI) | median | users cheaper than FSRS-7 | gap vs instant head (0.80-0.90), % (95% CI) | gap vs FSRS-7, 60-day cost |
|---|---|---|---|---|---|---|
| instant-head desired retention, targets 0.80 / 0.85 / 0.90 (the reference, full-run rows) | 19 | +5.6 ± 5.0 | +3.8 | 26% | — | +9.0 ± 6.6 |
| instant-head desired retention, target noise s = 0.5 (ihr-*-n0.5) | 18 | +4.2 ± 4.5 | +1.2 | 33% | -2.2 ± 2.8 | +5.0 ± 4.1 |
| instant-head desired retention, time-aware target beta = 1 (ihr-*-s1) | 19 | -6.9 ± 6.0 | -5.5 | 79% | -10.9 ± 4.2 | -4.9 ± 5.2 |

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
| ihr-0.80-n0.5 | 0.80 | 0.785 | 19.5 | 1031 | 0.854 |
| ihr-0.85-n0.5 | 0.85 | 0.822 | 21.6 | 1056 | 0.923 |
| ihr-0.90-n0.5 | 0.90 | 0.864 | 24.7 | 1093 | 0.988 |
| ihr-0.80-s1 | 0.80 | 0.772 | 17.8 | 1017 | 0.802 |
| ihr-0.85-s1 | 0.85 | 0.817 | 19.8 | 1044 | 0.874 |
| ihr-0.90-s1 | 0.90 | 0.868 | 22.0 | 1084 | 0.888 |
