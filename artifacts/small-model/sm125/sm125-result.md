# Rolling recalibration on the 732 set (SM125): result

**What answer time means.** The answer time is the whole time on the card, front and back together: from the moment the question is shown to the moment a grade button is pressed, capped at the preset's maximum answer seconds. Every "within S s" below means answered within S s, reading the back included.

Scoring only (no training), CPU recalibration of per-review outputs re-dumped on the GPU with fast Good/Easy prediction (SM120)'s code, on the 732 set (723 users, show-time inputs, leak-free, SM117's time-labelled test reviews). Pre-registered in `design.md`. Scheme: per user, Platt scaling per event refitted every 7 days on the user's latest 30,000 earlier reviews (earlier days only); raw model until 1,000 earlier reviews; repair (non-decreasing in S, Easy within Good or Easy within recalled, each timed event at most the raw P(set)). Grade-only P(set) is not recalibrated. Reliable = ECE (20 equal-mass bins) below 0.01 and LogLoss below every baseline of SM120 (not recalibrated), each difference with the 95% CI (2,000 user bootstraps, seed 0, review-weighted) below 0 and at least 0.05%.

## Verdict per model (pre-registered rules)

| model | go | 2-10 s cells reliable raw / recalibrated (of 21) | Good or Easy within 4 s reliable raw / recalibrated | G1 all 21 | G2 4 s better than raw | G3 no cell worse | G4 P(Good or Easy), P(Easy) | rows recalibrated |
|---|---|---|---|---|---|---|---|---|
| the longer Small Instant-Only model (SM113) | go | 15 / 21 | no / yes | yes | yes | yes | yes | 96.2% |
| the longer cool-down for Tiny (SM115) | go | 14 / 21 | no / yes | yes | yes | yes | yes | 96.2% |
| the show-time two-head cool-down for Small (SM118) | go | 14 / 21 | no / yes | yes | yes | yes | yes | 96.2% |
| the Small Instant-Only model (SM98) | go | 16 / 21 | yes / yes | yes | yes | yes | yes | 96.2% |
| the Tiny two-head model (SM106) | go | 14 / 21 | no / yes | yes | yes | yes | yes | 96.2% |

## Good or Easy within 4 s (answered within 4 s, reading the back included)

| model | ECE raw / recalibrated | LogLoss raw / recalibrated | recalibrated vs raw | recalibrated vs (a) | recalibrated vs (b) | raw vs (a) | raw vs (b) |
|---|---|---|---|---|---|---|---|
| the longer Small Instant-Only model (SM113) | 0.0105 / 0.0016 | 0.3439 / 0.3410 | -0.0029 [-0.0034, -0.0025] (-0.85%) | -0.0930 [-0.1004, -0.0859] (-21.43%) | -0.1448 [-0.1539, -0.1363] (-29.81%) | -0.0901 [-0.0973, -0.0832] (-20.75%) | -0.1419 [-0.1510, -0.1334] (-29.21%) |
| the longer cool-down for Tiny (SM115) | 0.0111 / 0.0016 | 0.3447 / 0.3419 | -0.0029 [-0.0033, -0.0024] (-0.84%) | -0.0923 [-0.0998, -0.0853] (-21.27%) | -0.1439 [-0.1530, -0.1353] (-29.63%) | -0.0895 [-0.0968, -0.0827] (-20.60%) | -0.1411 [-0.1503, -0.1325] (-29.03%) |
| the show-time two-head cool-down for Small (SM118) | 0.0128 / 0.0014 | 0.3457 / 0.3426 | -0.0032 [-0.0036, -0.0027] (-0.92%) | -0.0917 [-0.0990, -0.0847] (-21.11%) | -0.1432 [-0.1523, -0.1348] (-29.48%) | -0.0885 [-0.0957, -0.0818] (-20.38%) | -0.1401 [-0.1491, -0.1316] (-28.83%) |
| the Small Instant-Only model (SM98) | 0.0097 / 0.0013 | 0.3440 / 0.3412 | -0.0028 [-0.0034, -0.0024] (-0.82%) | -0.0929 [-0.1004, -0.0858] (-21.40%) | -0.1446 [-0.1538, -0.1361] (-29.77%) | -0.0901 [-0.0974, -0.0832] (-20.75%) | -0.1418 [-0.1511, -0.1333] (-29.19%) |
| the Tiny two-head model (SM106) | 0.0134 / 0.0016 | 0.3469 / 0.3436 | -0.0033 [-0.0038, -0.0029] (-0.96%) | -0.0909 [-0.0983, -0.0840] (-20.92%) | -0.1422 [-0.1512, -0.1338] (-29.27%) | -0.0876 [-0.0948, -0.0809] (-20.16%) | -0.1389 [-0.1479, -0.1304] (-28.58%) |

## Where each timed event is reliable (values of S), raw -> recalibrated

| model | Good or Easy within S s | recalled within S s | Easy within S s |
|---|---|---|---|
| the longer Small Instant-Only model (SM113) | 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s -> 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s | 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s -> 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s | 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s -> 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s |
| the longer cool-down for Tiny (SM115) | 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s -> 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s | 6 s, 8 s, 10 s, 15 s, 20 s, 30 s -> 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s | 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s -> 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s |
| the show-time two-head cool-down for Small (SM118) | 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s -> 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s | 6 s, 8 s, 10 s, 15 s, 20 s, 30 s -> 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s | 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s -> 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s |
| the Small Instant-Only model (SM98) | 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s -> 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s | 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s -> 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s | 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s -> 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s |
| the Tiny two-head model (SM106) | 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s -> 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s | 6 s, 8 s, 10 s, 15 s, 20 s, 30 s -> 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s | 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s -> 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s |

## ECE per S, raw -> recalibrated (Good or Easy within S s)

| model | 2 s | 3 s | 4 s | 5 s | 6 s | 8 s | 10 s | 15 s | 20 s | 30 s |
|---|---|---|---|---|---|---|---|---|---|---|
| the longer Small Instant-Only model (SM113) | 0.0106 -> 0.0017 | 0.0110 -> 0.0021 | 0.0105 -> 0.0016 | 0.0069 -> 0.0011 | 0.0056 -> 0.0015 | 0.0028 -> 0.0014 | 0.0028 -> 0.0015 | 0.0028 -> 0.0009 | 0.0019 -> 0.0010 | 0.0022 -> 0.0013 |
| the longer cool-down for Tiny (SM115) | 0.0103 -> 0.0019 | 0.0113 -> 0.0018 | 0.0111 -> 0.0016 | 0.0081 -> 0.0014 | 0.0069 -> 0.0020 | 0.0037 -> 0.0016 | 0.0033 -> 0.0014 | 0.0023 -> 0.0011 | 0.0019 -> 0.0013 | 0.0014 -> 0.0014 |
| the show-time two-head cool-down for Small (SM118) | 0.0115 -> 0.0018 | 0.0122 -> 0.0015 | 0.0128 -> 0.0014 | 0.0087 -> 0.0015 | 0.0070 -> 0.0016 | 0.0034 -> 0.0013 | 0.0024 -> 0.0013 | 0.0032 -> 0.0010 | 0.0031 -> 0.0012 | 0.0031 -> 0.0014 |
| the Small Instant-Only model (SM98) | 0.0134 -> 0.0021 | 0.0111 -> 0.0016 | 0.0097 -> 0.0013 | 0.0064 -> 0.0011 | 0.0053 -> 0.0017 | 0.0031 -> 0.0016 | 0.0029 -> 0.0016 | 0.0022 -> 0.0011 | 0.0018 -> 0.0010 | 0.0025 -> 0.0014 |
| the Tiny two-head model (SM106) | 0.0139 -> 0.0024 | 0.0144 -> 0.0021 | 0.0134 -> 0.0016 | 0.0087 -> 0.0013 | 0.0068 -> 0.0015 | 0.0026 -> 0.0014 | 0.0022 -> 0.0014 | 0.0036 -> 0.0008 | 0.0026 -> 0.0011 | 0.0026 -> 0.0013 |

## Grade only: P(Good or Easy) and P(Easy) (shipped raw; diagnostic: recalibrated too)

| model | event | ECE raw | raw vs (b) | reliable raw | ECE recalibrated (diagnostic) | recalibrated vs raw (diagnostic) |
|---|---|---|---|---|---|---|
| the longer Small Instant-Only model (SM113) | Good or Easy | 0.0015 | -0.1802 [-0.1949, -0.1660] (-33.03%) | yes | 0.0003 | +0.0000 [+0.0000, +0.0001] (+0.01%) |
| the longer Small Instant-Only model (SM113) | Easy | 0.0003 | -0.0998 [-0.1122, -0.0879] (-49.92%) | yes | 0.0003 | +0.0000 [+0.0000, +0.0001] (+0.04%) |
| the longer cool-down for Tiny (SM115) | Good or Easy | 0.0019 | -0.1795 [-0.1941, -0.1653] (-32.90%) | yes | 0.0004 | +0.0000 [+0.0000, +0.0000] (+0.01%) |
| the longer cool-down for Tiny (SM115) | Easy | 0.0002 | -0.0995 [-0.1119, -0.0877] (-49.79%) | yes | 0.0003 | +0.0000 [+0.0000, +0.0001] (+0.04%) |
| the show-time two-head cool-down for Small (SM118) | Good or Easy | 0.0010 | -0.1791 [-0.1938, -0.1650] (-32.84%) | yes | 0.0003 | +0.0000 [+0.0000, +0.0001] (+0.01%) |
| the show-time two-head cool-down for Small (SM118) | Easy | 0.0004 | -0.0993 [-0.1116, -0.0875] (-49.68%) | yes | 0.0003 | +0.0000 [+0.0000, +0.0000] (+0.04%) |
| the Small Instant-Only model (SM98) | Good or Easy | 0.0014 | -0.1798 [-0.1945, -0.1657] (-32.96%) | yes | 0.0003 | +0.0000 [+0.0000, +0.0001] (+0.01%) |
| the Small Instant-Only model (SM98) | Easy | 0.0003 | -0.0996 [-0.1120, -0.0878] (-49.85%) | yes | 0.0003 | +0.0000 [+0.0000, +0.0001] (+0.04%) |
| the Tiny two-head model (SM106) | Good or Easy | 0.0018 | -0.1781 [-0.1927, -0.1639] (-32.65%) | yes | 0.0004 | +0.0000 [+0.0000, +0.0000] (+0.01%) |
| the Tiny two-head model (SM106) | Easy | 0.0003 | -0.0989 [-0.1113, -0.0871] (-49.50%) | yes | 0.0003 | +0.0000 [+0.0000, +0.0000] (+0.04%) |

## Coverage and fits

| model | users | rows | recalibrated rows | users with recalibration | refits | identity fits | clamped fits | repair changed | dump equals SM120 |
|---|---|---|---|---|---|---|---|---|---|
| the longer Small Instant-Only model (SM113) | 722 | 30,093,623 | 28,949,746 (96.2%) | 722 | 49,203 | 185,862 | 120 | 18.60% | yes |
| the longer cool-down for Tiny (SM115) | 722 | 30,093,623 | 28,949,746 (96.2%) | 722 | 49,203 | 185,862 | 89 | 18.66% | yes |
| the show-time two-head cool-down for Small (SM118) | 722 | 30,093,623 | 28,949,746 (96.2%) | 722 | 49,203 | 185,862 | 104 | 18.86% | yes |
| the Small Instant-Only model (SM98) | 722 | 30,093,623 | 28,949,746 (96.2%) | 722 | 49,203 | 185,862 | 45 | 18.78% | yes |
| the Tiny two-head model (SM106) | 722 | 30,093,623 | 28,949,746 (96.2%) | 722 | 49,203 | 185,862 | 63 | 18.87% | yes |

Every per-S number, the sensitivity arm (5,000-review minimum), the arm before the repair and the recalibrated-rows-only comparison are in `sm125-result.json`.
