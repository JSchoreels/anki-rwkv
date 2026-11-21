# Time heads for the family (SM117): result

**What answer time means.** The answer time is the whole time on the card, front and back together: from the moment the question is shown to the moment a grade button is pressed, capped at the preset's maximum answer seconds. The "Show answer" moment is not stored, so front-only and back-only times cannot be computed. "Recall within S seconds" therefore means recalled (Hard, Good or Easy) and answered within S seconds, reading the back included.

**Run notes.**

- Users skipped in head training for GPU memory (the heaviest users, 345k to 394k labelled rows each; ids from `train.json`, field `skipped`): the Tiny two-head model (SM106): 2060, 474, 1573; the Small Instant-Only model (SM98): 474, 4162, 7511, 2349, 1573, 1996; the Tiny Instant-Only model (SM101): 2060, 474, 1573; the show-time instant-only big model (SM99): 2060, 1573; the show-time big model (SM91s2): 2060, 1573; the show-time retrain (SM92): 474.
- Three fixes were needed before the big heads ran (details in `design.md`): the CUDA `compress::wkv_q` kernel was not registered in the head scripts, so the first smoke test of the show-time instant-only big model (SM99) failed (42c916ea); the grade check now scores the release and the trunk in two separate scorer processes, because in one process the second big model differs by float32 rounding (about 3e-7), still requiring bitwise equality (44aee13c); and big head training ran out of GPU memory, fixed by catching the TorchScript form of the out-of-memory error and retrying the user once (d2d151e1), keeping the training buffer on the CPU (dee7ce2e) and freeing the previous user's GPU tensors before the next forward (14ad2cc1). The big heads were restarted from scratch after each fix.
- 3 test users are missing for the two big models (719 of 722 scored, after the second scoring pass at cap 0.92): the show-time instant-only big model (SM99) and the show-time big model (SM91s2).

An answer-time head (the answer-time head on richer frozen features (SM82c) recipe) trained alone on each frozen family model, on show-time inputs with same-day reviews included, then scored on the 732 set (722 users), show-time inputs, leak-free. Pre-registered rules: `design.md`.

## Gate per head (grade unchanged bitwise on 5 users; the head beats both answer-time baselines, CI clear of zero)

| model | users | grade unchanged | time LogLoss (head / ablation / card's last / user median) | head vs user median | mean abs ln error (head / user median) | gate |
|---|---|---|---|---|---|---|
| the Tiny two-head model (SM106) | 722 | True | 1.9751 / 2.0432 / 2.4751 / 2.3849 | -0.4098 [-0.4368, -0.3856] (-17.2%) | 0.4778 / 0.6533 | pass |
| the Small Instant-Only model (SM98) | 722 | True | 1.9625 / 2.0434 / 2.4751 / 2.3849 | -0.4225 [-0.4489, -0.3984] (-17.7%) | 0.4731 / 0.6533 | pass |
| the Tiny Instant-Only model (SM101) | 722 | True | 1.9656 / 2.0432 / 2.4751 / 2.3849 | -0.4194 [-0.4462, -0.3953] (-17.6%) | 0.4738 / 0.6533 | pass |
| the show-time instant-only big model (SM99) | 719 | True | 1.9523 / 2.0458 / 2.4766 / 2.3903 | -0.4380 [-0.4646, -0.4138] (-18.3%) | 0.4734 / 0.6558 | pass |
| the show-time big model (SM91s2) | 719 | True | 1.9558 / 2.0458 / 2.4766 / 2.3903 | -0.4345 [-0.4608, -0.4108] (-18.2%) | 0.4744 / 0.6558 | pass |
| the show-time retrain (SM92) | 722 | True | 1.9716 / 2.0420 / 2.4751 / 2.3849 | -0.4134 [-0.4402, -0.3889] (-17.3%) | 0.4767 / 0.6533 | pass |

Same numbers on the later-day subset (the rows of the earlier answer-time heads, SM82 rule), for comparison with the shipped head of the show-time retrain (SM92) (gated at 1.9580 on normal inputs):

| model | rows | head | card's last | user median | shipped head |
|---|---|---|---|---|---|
| the Tiny two-head model (SM106) | 21,585,938 | 1.9619 | 2.4691 | 2.3652 |  |
| the Small Instant-Only model (SM98) | 21,585,938 | 1.9489 | 2.4691 | 2.3652 |  |
| the Tiny Instant-Only model (SM101) | 21,585,938 | 1.9522 | 2.4691 | 2.3652 |  |
| the show-time instant-only big model (SM99) | 20,724,352 | 1.9378 | 2.4703 | 2.3705 |  |
| the show-time big model (SM91s2) | 20,724,352 | 1.9413 | 2.4703 | 2.3705 |  |
| the show-time retrain (SM92) | 21,585,938 | 1.9583 | 2.4691 | 2.3652 | 1.9580 |

## P(recall within S seconds)

p = sum over Hard, Good, Easy of P(grade) x P(t <= S | grade); label: grade >= Hard and recorded time <= S. Baseline (a): the model's own P(grade) x the user's running share of earlier answers of that grade under S; baseline (b): the user's running rate of the event. LogLoss deltas are head minus baseline, review-weighted, 95% CI from 2,000 user bootstraps (seed 0).

### the Tiny two-head model (SM106)

| S | rows | event rate | LogLoss head | (a) | (b) | head vs (a) | head vs (b) | Brier head | ECE head | ECE (a) | AUC head | AUC (a) | AUC (b) | reliable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 s | 30,093,623 | 0.199 | 0.3051 | 0.4023 | 0.4268 | -0.0972 [-0.1065, -0.0886] (-24.2%) | -0.1217 [-0.1324, -0.1116] (-28.5%) | 0.0947 | 0.0163 | 0.0305 | 0.8964 | 0.8029 | 0.7664 | no |
| 5 s | 30,093,623 | 0.366 | 0.4304 | 0.5420 | 0.5789 | -0.1116 [-0.1194, -0.1041] (-20.6%) | -0.1486 [-0.1568, -0.1402] (-25.7%) | 0.1404 | 0.0109 | 0.0301 | 0.8693 | 0.7750 | 0.7273 | no |
| 8 s | 30,093,623 | 0.518 | 0.4728 | 0.5708 | 0.6182 | -0.0980 [-0.1051, -0.0909] (-17.2%) | -0.1454 [-0.1525, -0.1380] (-23.5%) | 0.1560 | 0.0038 | 0.0228 | 0.8528 | 0.7708 | 0.7115 | yes |
| 10 s | 30,093,623 | 0.580 | 0.4731 | 0.5586 | 0.6103 | -0.0855 [-0.0924, -0.0789] (-15.3%) | -0.1372 [-0.1441, -0.1302] (-22.5%) | 0.1557 | 0.0028 | 0.0189 | 0.8465 | 0.7729 | 0.7069 | yes |
| 15 s | 30,093,623 | 0.670 | 0.4550 | 0.5159 | 0.5749 | -0.0608 [-0.0669, -0.0550] (-11.8%) | -0.1198 [-0.1265, -0.1130] (-20.9%) | 0.1483 | 0.0052 | 0.0123 | 0.8367 | 0.7792 | 0.6993 | yes |
| 20 s | 30,093,623 | 0.717 | 0.4348 | 0.4803 | 0.5439 | -0.0456 [-0.0509, -0.0406] (-9.5%) | -0.1091 [-0.1158, -0.1027] (-20.1%) | 0.1403 | 0.0039 | 0.0089 | 0.8320 | 0.7857 | 0.6952 | yes |
| 30 s | 30,093,623 | 0.769 | 0.3978 | 0.4239 | 0.4949 | -0.0260 [-0.0306, -0.0217] (-6.1%) | -0.0971 [-0.1039, -0.0901] (-19.6%) | 0.1263 | 0.0024 | 0.0070 | 0.8316 | 0.8026 | 0.6954 | yes |

Reliable at every S (ECE < 0.01 and LogLoss below both baselines, CI clear of zero): **no**.

### the Small Instant-Only model (SM98)

| S | rows | event rate | LogLoss head | (a) | (b) | head vs (a) | head vs (b) | Brier head | ECE head | ECE (a) | AUC head | AUC (a) | AUC (b) | reliable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 s | 30,093,623 | 0.199 | 0.3016 | 0.4020 | 0.4268 | -0.1005 [-0.1100, -0.0917] (-25.0%) | -0.1252 [-0.1361, -0.1149] (-29.3%) | 0.0935 | 0.0130 | 0.0305 | 0.8987 | 0.8032 | 0.7664 | no |
| 5 s | 30,093,623 | 0.366 | 0.4272 | 0.5416 | 0.5789 | -0.1144 [-0.1223, -0.1066] (-21.1%) | -0.1517 [-0.1599, -0.1433] (-26.2%) | 0.1393 | 0.0078 | 0.0301 | 0.8713 | 0.7754 | 0.7273 | yes |
| 8 s | 30,093,623 | 0.518 | 0.4702 | 0.5703 | 0.6182 | -0.1001 [-0.1072, -0.0928] (-17.6%) | -0.1479 [-0.1552, -0.1404] (-23.9%) | 0.1550 | 0.0043 | 0.0227 | 0.8546 | 0.7713 | 0.7115 | yes |
| 10 s | 30,093,623 | 0.580 | 0.4705 | 0.5580 | 0.6103 | -0.0875 [-0.0945, -0.0807] (-15.7%) | -0.1398 [-0.1469, -0.1327] (-22.9%) | 0.1548 | 0.0037 | 0.0188 | 0.8483 | 0.7735 | 0.7069 | yes |
| 15 s | 30,093,623 | 0.670 | 0.4524 | 0.5152 | 0.5749 | -0.0627 [-0.0688, -0.0569] (-12.2%) | -0.1225 [-0.1293, -0.1157] (-21.3%) | 0.1474 | 0.0041 | 0.0124 | 0.8387 | 0.7800 | 0.6993 | yes |
| 20 s | 30,093,623 | 0.717 | 0.4323 | 0.4796 | 0.5439 | -0.0472 [-0.0527, -0.0422] (-9.8%) | -0.1116 [-0.1183, -0.1051] (-20.5%) | 0.1395 | 0.0026 | 0.0090 | 0.8341 | 0.7865 | 0.6952 | yes |
| 30 s | 30,093,623 | 0.769 | 0.3956 | 0.4230 | 0.4949 | -0.0273 [-0.0320, -0.0231] (-6.5%) | -0.0992 [-0.1060, -0.0924] (-20.1%) | 0.1256 | 0.0013 | 0.0073 | 0.8337 | 0.8036 | 0.6954 | yes |

Reliable at every S (ECE < 0.01 and LogLoss below both baselines, CI clear of zero): **no**.

### the Tiny Instant-Only model (SM101)

| S | rows | event rate | LogLoss head | (a) | (b) | head vs (a) | head vs (b) | Brier head | ECE head | ECE (a) | AUC head | AUC (a) | AUC (b) | reliable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 s | 30,093,623 | 0.199 | 0.3028 | 0.4022 | 0.4268 | -0.0994 [-0.1089, -0.0906] (-24.7%) | -0.1240 [-0.1349, -0.1137] (-29.1%) | 0.0939 | 0.0158 | 0.0304 | 0.8981 | 0.8030 | 0.7664 | no |
| 5 s | 30,093,623 | 0.366 | 0.4283 | 0.5418 | 0.5789 | -0.1135 [-0.1215, -0.1059] (-20.9%) | -0.1506 [-0.1589, -0.1422] (-26.0%) | 0.1397 | 0.0099 | 0.0300 | 0.8707 | 0.7752 | 0.7273 | yes |
| 8 s | 30,093,623 | 0.518 | 0.4709 | 0.5706 | 0.6182 | -0.0998 [-0.1070, -0.0925] (-17.5%) | -0.1473 [-0.1545, -0.1398] (-23.8%) | 0.1553 | 0.0043 | 0.0226 | 0.8541 | 0.7710 | 0.7115 | yes |
| 10 s | 30,093,623 | 0.580 | 0.4711 | 0.5583 | 0.6103 | -0.0872 [-0.0942, -0.0804] (-15.6%) | -0.1392 [-0.1462, -0.1321] (-22.8%) | 0.1550 | 0.0038 | 0.0187 | 0.8479 | 0.7731 | 0.7069 | yes |
| 15 s | 30,093,623 | 0.670 | 0.4532 | 0.5155 | 0.5749 | -0.0624 [-0.0685, -0.0565] (-12.1%) | -0.1217 [-0.1285, -0.1149] (-21.2%) | 0.1476 | 0.0059 | 0.0123 | 0.8382 | 0.7795 | 0.6993 | yes |
| 20 s | 30,093,623 | 0.717 | 0.4332 | 0.4799 | 0.5439 | -0.0468 [-0.0521, -0.0418] (-9.7%) | -0.1107 [-0.1174, -0.1042] (-20.4%) | 0.1398 | 0.0048 | 0.0089 | 0.8334 | 0.7861 | 0.6952 | yes |
| 30 s | 30,093,623 | 0.769 | 0.3965 | 0.4234 | 0.4949 | -0.0269 [-0.0314, -0.0226] (-6.3%) | -0.0983 [-0.1052, -0.0914] (-19.9%) | 0.1259 | 0.0020 | 0.0073 | 0.8328 | 0.8031 | 0.6954 | yes |

Reliable at every S (ECE < 0.01 and LogLoss below both baselines, CI clear of zero): **no**.

### the show-time instant-only big model (SM99)

| S | rows | event rate | LogLoss head | (a) | (b) | head vs (a) | head vs (b) | Brier head | ECE head | ECE (a) | AUC head | AUC (a) | AUC (b) | reliable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 s | 28,995,499 | 0.195 | 0.2996 | 0.4017 | 0.4263 | -0.1021 [-0.1115, -0.0933] (-25.4%) | -0.1267 [-0.1377, -0.1163] (-29.7%) | 0.0930 | 0.0127 | 0.0302 | 0.8980 | 0.7986 | 0.7600 | no |
| 5 s | 28,995,499 | 0.363 | 0.4263 | 0.5426 | 0.5799 | -0.1163 [-0.1246, -0.1088] (-21.4%) | -0.1535 [-0.1624, -0.1454] (-26.5%) | 0.1392 | 0.0093 | 0.0302 | 0.8711 | 0.7731 | 0.7239 | yes |
| 8 s | 28,995,499 | 0.515 | 0.4688 | 0.5715 | 0.6193 | -0.1028 [-0.1106, -0.0955] (-18.0%) | -0.1505 [-0.1584, -0.1434] (-24.3%) | 0.1546 | 0.0055 | 0.0227 | 0.8554 | 0.7705 | 0.7102 | yes |
| 10 s | 28,995,499 | 0.576 | 0.4689 | 0.5593 | 0.6116 | -0.0904 [-0.0979, -0.0835] (-16.2%) | -0.1426 [-0.1502, -0.1357] (-23.3%) | 0.1543 | 0.0044 | 0.0189 | 0.8497 | 0.7731 | 0.7065 | yes |
| 15 s | 28,995,499 | 0.666 | 0.4516 | 0.5168 | 0.5767 | -0.0653 [-0.0720, -0.0595] (-12.6%) | -0.1251 [-0.1324, -0.1185] (-21.7%) | 0.1471 | 0.0020 | 0.0124 | 0.8404 | 0.7800 | 0.6997 | yes |
| 20 s | 28,995,499 | 0.714 | 0.4320 | 0.4814 | 0.5460 | -0.0494 [-0.0552, -0.0443] (-10.3%) | -0.1140 [-0.1212, -0.1078] (-20.9%) | 0.1394 | 0.0019 | 0.0089 | 0.8359 | 0.7868 | 0.6960 | yes |
| 30 s | 28,995,499 | 0.767 | 0.3956 | 0.4248 | 0.4971 | -0.0292 [-0.0341, -0.0246] (-6.9%) | -0.1015 [-0.1086, -0.0946] (-20.4%) | 0.1257 | 0.0011 | 0.0068 | 0.8356 | 0.8040 | 0.6965 | yes |

Reliable at every S (ECE < 0.01 and LogLoss below both baselines, CI clear of zero): **no**.

### the show-time big model (SM91s2)

| S | rows | event rate | LogLoss head | (a) | (b) | head vs (a) | head vs (b) | Brier head | ECE head | ECE (a) | AUC head | AUC (a) | AUC (b) | reliable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 s | 28,995,499 | 0.195 | 0.3002 | 0.4017 | 0.4263 | -0.1015 [-0.1110, -0.0927] (-25.3%) | -0.1261 [-0.1371, -0.1156] (-29.6%) | 0.0931 | 0.0128 | 0.0303 | 0.8975 | 0.7986 | 0.7600 | no |
| 5 s | 28,995,499 | 0.363 | 0.4273 | 0.5427 | 0.5799 | -0.1154 [-0.1237, -0.1079] (-21.3%) | -0.1526 [-0.1614, -0.1444] (-26.3%) | 0.1395 | 0.0095 | 0.0303 | 0.8705 | 0.7730 | 0.7239 | yes |
| 8 s | 28,995,499 | 0.515 | 0.4697 | 0.5716 | 0.6193 | -0.1019 [-0.1097, -0.0948] (-17.8%) | -0.1496 [-0.1574, -0.1424] (-24.1%) | 0.1549 | 0.0048 | 0.0230 | 0.8548 | 0.7704 | 0.7102 | yes |
| 10 s | 28,995,499 | 0.576 | 0.4698 | 0.5594 | 0.6116 | -0.0896 [-0.0971, -0.0828] (-16.0%) | -0.1418 [-0.1493, -0.1349] (-23.2%) | 0.1546 | 0.0036 | 0.0192 | 0.8491 | 0.7730 | 0.7065 | yes |
| 15 s | 28,995,499 | 0.666 | 0.4524 | 0.5170 | 0.5767 | -0.0646 [-0.0713, -0.0589] (-12.5%) | -0.1243 [-0.1316, -0.1177] (-21.6%) | 0.1474 | 0.0039 | 0.0126 | 0.8398 | 0.7799 | 0.6997 | yes |
| 20 s | 28,995,499 | 0.714 | 0.4328 | 0.4815 | 0.5460 | -0.0488 [-0.0545, -0.0437] (-10.1%) | -0.1133 [-0.1204, -0.1070] (-20.7%) | 0.1397 | 0.0032 | 0.0091 | 0.8353 | 0.7867 | 0.6960 | yes |
| 30 s | 28,995,499 | 0.767 | 0.3963 | 0.4250 | 0.4971 | -0.0287 [-0.0336, -0.0241] (-6.7%) | -0.1008 [-0.1078, -0.0939] (-20.3%) | 0.1259 | 0.0022 | 0.0070 | 0.8349 | 0.8038 | 0.6965 | yes |

Reliable at every S (ECE < 0.01 and LogLoss below both baselines, CI clear of zero): **no**.

### the show-time retrain (SM92)

| S | rows | event rate | LogLoss head | (a) | (b) | head vs (a) | head vs (b) | Brier head | ECE head | ECE (a) | AUC head | AUC (a) | AUC (b) | reliable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 s | 30,093,623 | 0.199 | 0.3042 | 0.4021 | 0.4268 | -0.0979 [-0.1074, -0.0893] (-24.4%) | -0.1226 [-0.1335, -0.1126] (-28.7%) | 0.0943 | 0.0155 | 0.0306 | 0.8968 | 0.8031 | 0.7664 | no |
| 5 s | 30,093,623 | 0.366 | 0.4296 | 0.5418 | 0.5789 | -0.1121 [-0.1199, -0.1045] (-20.7%) | -0.1493 [-0.1575, -0.1410] (-25.8%) | 0.1402 | 0.0129 | 0.0302 | 0.8700 | 0.7752 | 0.7273 | no |
| 8 s | 30,093,623 | 0.518 | 0.4717 | 0.5706 | 0.6182 | -0.0989 [-0.1060, -0.0917] (-17.3%) | -0.1465 [-0.1536, -0.1391] (-23.7%) | 0.1556 | 0.0056 | 0.0229 | 0.8536 | 0.7711 | 0.7115 | yes |
| 10 s | 30,093,623 | 0.580 | 0.4718 | 0.5583 | 0.6103 | -0.0865 [-0.0935, -0.0797] (-15.5%) | -0.1385 [-0.1454, -0.1314] (-22.7%) | 0.1553 | 0.0039 | 0.0190 | 0.8474 | 0.7732 | 0.7069 | yes |
| 15 s | 30,093,623 | 0.670 | 0.4538 | 0.5155 | 0.5749 | -0.0618 [-0.0678, -0.0560] (-12.0%) | -0.1211 [-0.1279, -0.1143] (-21.1%) | 0.1478 | 0.0030 | 0.0125 | 0.8376 | 0.7796 | 0.6993 | yes |
| 20 s | 30,093,623 | 0.717 | 0.4337 | 0.4800 | 0.5439 | -0.0463 [-0.0517, -0.0413] (-9.6%) | -0.1102 [-0.1169, -0.1037] (-20.3%) | 0.1399 | 0.0032 | 0.0091 | 0.8329 | 0.7861 | 0.6952 | yes |
| 30 s | 30,093,623 | 0.769 | 0.3970 | 0.4235 | 0.4949 | -0.0265 [-0.0311, -0.0223] (-6.3%) | -0.0979 [-0.1047, -0.0911] (-19.8%) | 0.1260 | 0.0023 | 0.0072 | 0.8325 | 0.8031 | 0.6954 | yes |

Reliable at every S (ECE < 0.01 and LogLoss below both baselines, CI clear of zero): **no**.

## Reliability tables (head, 20 equal-mass bins: rows, mean predicted, observed)

In `sm117-result.json` (`<model>.rus.<S>.predictors.model.reliability`); the largest bin gap per S is below.

| model | 3 s | 5 s | 8 s | 10 s | 15 s | 20 s | 30 s |
|---|---|---|---|---|---|---|---|
| the Tiny two-head model (SM106) | 0.066 | 0.030 | 0.010 | 0.006 | 0.009 | 0.007 | 0.010 |
| the Small Instant-Only model (SM98) | 0.057 | 0.017 | 0.008 | 0.009 | 0.006 | 0.005 | 0.004 |
| the Tiny Instant-Only model (SM101) | 0.064 | 0.025 | 0.007 | 0.010 | 0.009 | 0.010 | 0.003 |
| the show-time instant-only big model (SM99) | 0.048 | 0.024 | 0.011 | 0.009 | 0.005 | 0.003 | 0.004 |
| the show-time big model (SM91s2) | 0.047 | 0.028 | 0.010 | 0.006 | 0.008 | 0.006 | 0.006 |
| the show-time retrain (SM92) | 0.060 | 0.036 | 0.010 | 0.009 | 0.006 | 0.005 | 0.012 |
