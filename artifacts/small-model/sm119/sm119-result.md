# Time heads for the new candidates (SM119): result

**What answer time means.** The answer time is the whole time on the card, front and back together: from the moment the question is shown to the moment a grade button is pressed, capped at the preset's maximum answer seconds. The "Show answer" moment is not stored, so front-only and back-only times cannot be computed. "Recall within S seconds" therefore means recalled (Hard, Good or Easy) and answered within S seconds, reading the back included.

**Run notes.** Each head trained on 1,991 users, skipping one (user 474, out of GPU memory; 1,991 of 1,991 counted as in the family heads, SM117) (about 42 to 43 minutes each); the trunk is unchanged and the grade check was bitwise on 5 users for all three models. All three scored 722 users.

An answer-time head (the answer-time head on richer frozen features (SM82c) recipe) trained alone on each frozen family model, on show-time inputs with same-day reviews included, then scored on the 732 set (722 users), show-time inputs, leak-free. Pre-registered rules: `design.md`.

## Gate per head (grade unchanged bitwise on 5 users; the head beats both answer-time baselines, CI clear of zero)

| model | users | grade unchanged | time LogLoss (head / ablation / card's last / user median) | head vs user median | mean abs ln error (head / user median) | gate |
|---|---|---|---|---|---|---|
| the longer Small Instant-Only model (SM113) | 722 | True | 1.9605 / 2.0420 / 2.4751 / 2.3849 | -0.4245 [-0.4512, -0.4004] (-17.8%) | 0.4732 / 0.6533 | pass |
| the longer cool-down for Tiny (SM115) | 722 | True | 1.9621 / 2.0420 / 2.4751 / 2.3849 | -0.4229 [-0.4500, -0.3989] (-17.7%) | 0.4737 / 0.6533 | pass |
| the show-time two-head cool-down for Small (SM118) | 722 | True | 1.9700 / 2.0420 / 2.4751 / 2.3849 | -0.4149 [-0.4418, -0.3907] (-17.4%) | 0.4758 / 0.6533 | pass |

Same numbers on the later-day subset (the rows of the earlier answer-time heads, SM82 rule), for comparison with the shipped head of the show-time retrain (SM92) (gated at 1.9580 on normal inputs):

| model | rows | head | card's last | user median | shipped head |
|---|---|---|---|---|---|
| the longer Small Instant-Only model (SM113) | 21,585,938 | 1.9469 | 2.4691 | 2.3652 |  |
| the longer cool-down for Tiny (SM115) | 21,585,938 | 1.9483 | 2.4691 | 2.3652 |  |
| the show-time two-head cool-down for Small (SM118) | 21,585,938 | 1.9566 | 2.4691 | 2.3652 |  |

## P(recall within S seconds)

p = sum over Hard, Good, Easy of P(grade) x P(t <= S | grade); label: grade >= Hard and recorded time <= S. Baseline (a): the model's own P(grade) x the user's running share of earlier answers of that grade under S; baseline (b): the user's running rate of the event. LogLoss deltas are head minus baseline, review-weighted, 95% CI from 2,000 user bootstraps (seed 0).

### the longer Small Instant-Only model (SM113)

| S | rows | event rate | LogLoss head | (a) | (b) | head vs (a) | head vs (b) | Brier head | ECE head | ECE (a) | AUC head | AUC (a) | AUC (b) | reliable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 s | 30,093,623 | 0.199 | 0.3014 | 0.4020 | 0.4268 | -0.1006 [-0.1101, -0.0918] (-25.0%) | -0.1254 [-0.1366, -0.1153] (-29.4%) | 0.0934 | 0.0128 | 0.0305 | 0.8988 | 0.8033 | 0.7664 | no |
| 5 s | 30,093,623 | 0.366 | 0.4272 | 0.5415 | 0.5789 | -0.1143 [-0.1222, -0.1066] (-21.1%) | -0.1517 [-0.1599, -0.1434] (-26.2%) | 0.1393 | 0.0088 | 0.0301 | 0.8713 | 0.7755 | 0.7273 | yes |
| 8 s | 30,093,623 | 0.518 | 0.4700 | 0.5703 | 0.6182 | -0.1003 [-0.1074, -0.0932] (-17.6%) | -0.1482 [-0.1554, -0.1408] (-24.0%) | 0.1549 | 0.0037 | 0.0227 | 0.8547 | 0.7714 | 0.7115 | yes |
| 10 s | 30,093,623 | 0.580 | 0.4702 | 0.5579 | 0.6103 | -0.0878 [-0.0947, -0.0810] (-15.7%) | -0.1402 [-0.1472, -0.1330] (-23.0%) | 0.1547 | 0.0034 | 0.0187 | 0.8486 | 0.7735 | 0.7069 | yes |
| 15 s | 30,093,623 | 0.670 | 0.4520 | 0.5151 | 0.5749 | -0.0630 [-0.0691, -0.0572] (-12.2%) | -0.1228 [-0.1297, -0.1160] (-21.4%) | 0.1472 | 0.0034 | 0.0123 | 0.8390 | 0.7800 | 0.6993 | yes |
| 20 s | 30,093,623 | 0.717 | 0.4320 | 0.4795 | 0.5439 | -0.0475 [-0.0530, -0.0425] (-9.9%) | -0.1120 [-0.1187, -0.1054] (-20.6%) | 0.1393 | 0.0023 | 0.0090 | 0.8344 | 0.7866 | 0.6952 | yes |
| 30 s | 30,093,623 | 0.769 | 0.3953 | 0.4229 | 0.4949 | -0.0275 [-0.0323, -0.0232] (-6.5%) | -0.0996 [-0.1065, -0.0927] (-20.1%) | 0.1255 | 0.0020 | 0.0074 | 0.8340 | 0.8037 | 0.6954 | yes |

Reliable at every S (ECE < 0.01 and LogLoss below both baselines, CI clear of zero): **no**.

### the longer cool-down for Tiny (SM115)

| S | rows | event rate | LogLoss head | (a) | (b) | head vs (a) | head vs (b) | Brier head | ECE head | ECE (a) | AUC head | AUC (a) | AUC (b) | reliable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 s | 30,093,623 | 0.199 | 0.3024 | 0.4022 | 0.4268 | -0.0998 [-0.1093, -0.0910] (-24.8%) | -0.1244 [-0.1354, -0.1142] (-29.1%) | 0.0938 | 0.0134 | 0.0304 | 0.8981 | 0.8030 | 0.7664 | no |
| 5 s | 30,093,623 | 0.366 | 0.4281 | 0.5418 | 0.5789 | -0.1137 [-0.1217, -0.1060] (-21.0%) | -0.1509 [-0.1591, -0.1425] (-26.1%) | 0.1396 | 0.0105 | 0.0300 | 0.8708 | 0.7752 | 0.7273 | no |
| 8 s | 30,093,623 | 0.518 | 0.4704 | 0.5706 | 0.6182 | -0.1002 [-0.1075, -0.0930] (-17.6%) | -0.1477 [-0.1550, -0.1403] (-23.9%) | 0.1551 | 0.0048 | 0.0227 | 0.8544 | 0.7710 | 0.7115 | yes |
| 10 s | 30,093,623 | 0.580 | 0.4706 | 0.5583 | 0.6103 | -0.0877 [-0.0947, -0.0808] (-15.7%) | -0.1397 [-0.1467, -0.1326] (-22.9%) | 0.1548 | 0.0041 | 0.0186 | 0.8482 | 0.7732 | 0.7069 | yes |
| 15 s | 30,093,623 | 0.670 | 0.4525 | 0.5155 | 0.5749 | -0.0629 [-0.0690, -0.0570] (-12.2%) | -0.1223 [-0.1291, -0.1155] (-21.3%) | 0.1474 | 0.0034 | 0.0123 | 0.8386 | 0.7796 | 0.6993 | yes |
| 20 s | 30,093,623 | 0.717 | 0.4325 | 0.4799 | 0.5439 | -0.0474 [-0.0529, -0.0424] (-9.9%) | -0.1114 [-0.1181, -0.1049] (-20.5%) | 0.1396 | 0.0025 | 0.0090 | 0.8340 | 0.7862 | 0.6952 | yes |
| 30 s | 30,093,623 | 0.769 | 0.3958 | 0.4233 | 0.4949 | -0.0275 [-0.0321, -0.0232] (-6.5%) | -0.0991 [-0.1060, -0.0922] (-20.0%) | 0.1257 | 0.0006 | 0.0075 | 0.8335 | 0.8032 | 0.6954 | yes |

Reliable at every S (ECE < 0.01 and LogLoss below both baselines, CI clear of zero): **no**.

### the show-time two-head cool-down for Small (SM118)

| S | rows | event rate | LogLoss head | (a) | (b) | head vs (a) | head vs (b) | Brier head | ECE head | ECE (a) | AUC head | AUC (a) | AUC (b) | reliable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 s | 30,093,623 | 0.199 | 0.3036 | 0.4021 | 0.4268 | -0.0985 [-0.1079, -0.0898] (-24.5%) | -0.1232 [-0.1341, -0.1131] (-28.9%) | 0.0942 | 0.0140 | 0.0305 | 0.8971 | 0.8031 | 0.7664 | no |
| 5 s | 30,093,623 | 0.366 | 0.4290 | 0.5417 | 0.5789 | -0.1127 [-0.1205, -0.1051] (-20.8%) | -0.1499 [-0.1581, -0.1415] (-25.9%) | 0.1399 | 0.0109 | 0.0300 | 0.8702 | 0.7753 | 0.7273 | no |
| 8 s | 30,093,623 | 0.518 | 0.4714 | 0.5705 | 0.6182 | -0.0991 [-0.1063, -0.0920] (-17.4%) | -0.1468 [-0.1539, -0.1394] (-23.7%) | 0.1555 | 0.0048 | 0.0227 | 0.8538 | 0.7711 | 0.7115 | yes |
| 10 s | 30,093,623 | 0.580 | 0.4714 | 0.5582 | 0.6103 | -0.0868 [-0.0938, -0.0800] (-15.6%) | -0.1389 [-0.1458, -0.1318] (-22.8%) | 0.1551 | 0.0034 | 0.0187 | 0.8477 | 0.7733 | 0.7069 | yes |
| 15 s | 30,093,623 | 0.670 | 0.4532 | 0.5154 | 0.5749 | -0.0622 [-0.0682, -0.0563] (-12.1%) | -0.1216 [-0.1285, -0.1148] (-21.2%) | 0.1477 | 0.0036 | 0.0122 | 0.8380 | 0.7797 | 0.6993 | yes |
| 20 s | 30,093,623 | 0.717 | 0.4332 | 0.4798 | 0.5439 | -0.0466 [-0.0520, -0.0416] (-9.7%) | -0.1107 [-0.1174, -0.1042] (-20.4%) | 0.1398 | 0.0035 | 0.0088 | 0.8333 | 0.7863 | 0.6952 | yes |
| 30 s | 30,093,623 | 0.769 | 0.3967 | 0.4233 | 0.4949 | -0.0266 [-0.0312, -0.0224] (-6.3%) | -0.0982 [-0.1050, -0.0914] (-19.8%) | 0.1259 | 0.0025 | 0.0070 | 0.8326 | 0.8033 | 0.6954 | yes |

Reliable at every S (ECE < 0.01 and LogLoss below both baselines, CI clear of zero): **no**.

## Reliability tables (head, 20 equal-mass bins: rows, mean predicted, observed)

In `sm117-result.json` (`<model>.rus.<S>.predictors.model.reliability`); the largest bin gap per S is below.

| model | 3 s | 5 s | 8 s | 10 s | 15 s | 20 s | 30 s |
|---|---|---|---|---|---|---|---|
| the longer Small Instant-Only model (SM113) | 0.050 | 0.024 | 0.008 | 0.008 | 0.006 | 0.004 | 0.006 |
| the longer cool-down for Tiny (SM115) | 0.053 | 0.025 | 0.010 | 0.008 | 0.007 | 0.008 | 0.003 |
| the show-time two-head cool-down for Small (SM118) | 0.055 | 0.028 | 0.010 | 0.008 | 0.007 | 0.007 | 0.008 |
