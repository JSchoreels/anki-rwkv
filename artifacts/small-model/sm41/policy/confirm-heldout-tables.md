Rows: `C:/Users/jonat/workspace/wt/sm41/artifacts/small-model/sm41/policy/confirm/rows/*.jsonl` (screen users excluded); 97 users, 251 splits.

| policy | users | gap vs FSRS-7, % (95% CI) | median | users cheaper than FSRS-7 | gap vs instant head (0.80-0.90), % (95% CI) | gap vs FSRS-7, 60-day cost |
|---|---|---|---|---|---|---|
| instant-head desired retention, targets 0.80 / 0.85 / 0.90 (the reference, full-run rows) | 96 | +10.0 ± 3.8 | +8.8 | 17% | — | +9.3 ± 3.2 |
| instant-head desired retention, target noise s = 0.5 (ihr-*-n0.5) | 97 | +5.9 ± 3.2 | +5.0 | 28% | -3.4 ± 1.3 | +7.3 ± 2.5 |
| instant-head desired retention, time-aware target beta = 2 (ihr-*-s2) | 91 | -23.6 ± 3.5 | -21.8 | 92% | -27.3 ± 3.2 | -21.1 ± 3.2 |

| policy | target | realised retention 30-59 | min/day 30-59 | memorised | cost |
|---|---|---|---|---|---|
| ihr-0.80 (full run) | 0.80 | 0.802 | 20.2 | 1227 | 0.583 |
| ihr-0.85 (full run) | 0.85 | 0.839 | 23.5 | 1256 | 0.649 |
| ihr-0.90 (full run) | 0.90 | 0.880 | 28.5 | 1281 | 0.753 |
| fsrs7-0.70 (full run) | 0.70 | 0.681 | 14.3 | 1146 | 0.456 |
| fsrs7-0.75 (full run) | 0.75 | 0.726 | 16.8 | 1172 | 0.504 |
| fsrs7-0.80 (full run) | 0.80 | 0.770 | 20.0 | 1202 | 0.560 |
| fsrs7-0.85 (full run) | 0.85 | 0.820 | 24.6 | 1234 | 0.642 |
| fsrs7-0.90 (full run) | 0.90 | 0.875 | 31.2 | 1272 | 0.761 |
| fsrs7-0.95 (full run) | 0.95 | 0.929 | 40.6 | 1313 | 0.931 |
| ihr-0.80-n0.5 | 0.80 | 0.808 | 19.4 | 1215 | 0.571 |
| ihr-0.85-n0.5 | 0.85 | 0.839 | 22.5 | 1240 | 0.638 |
| ihr-0.90-n0.5 | 0.90 | 0.876 | 27.3 | 1269 | 0.737 |
| ihr-0.80-s2 | 0.80 | 0.827 | 16.2 | 1183 | 0.492 |
| ihr-0.85-s2 | 0.85 | 0.859 | 18.5 | 1206 | 0.537 |
| ihr-0.90-s2 | 0.90 | 0.895 | 21.8 | 1231 | 0.605 |
