# Rolling recalibration for the Big models (SM128): result

**What answer time means.** The answer time is the whole time on the card, front and back together: from the moment the question is shown to the moment a grade button is pressed, capped at the preset's maximum answer seconds. Every "within S s" below means answered within S s, reading the back included.

Scoring only (no training), CPU recalibration of per-review outputs re-dumped on the GPU with fast Good/Easy prediction (SM120)'s code, on the 732 set (723 users, show-time inputs, leak-free, SM117's time-labelled test reviews). Pre-registered in `design.md`. Scheme: per user, Platt scaling per event refitted every 7 days on the user's latest 30,000 earlier reviews (earlier days only); raw model until 1,000 earlier reviews; repair (non-decreasing in S, Easy within Good or Easy within recalled, each timed event at most the raw P(set)). Grade-only P(set) is not recalibrated. Reliable = ECE (20 equal-mass bins) below 0.01 and LogLoss below every baseline of SM120 (not recalibrated), each difference with the 95% CI (2,000 user bootstraps, seed 0, review-weighted) below 0 and at least 0.05%.

## Verdict per model (pre-registered rules)

| model | go | 2-10 s cells reliable raw / recalibrated (of 21) | Good or Easy within 4 s reliable raw / recalibrated | G1 all 21 | G2 4 s better than raw | G3 no cell worse | G4 P(Good or Easy), P(Easy) | rows recalibrated |
|---|---|---|---|---|---|---|---|---|
| the show-time big model (SM91s2) | go | 15 / 21 | no / yes | yes | yes | yes | yes | 96.1% |
| the show-time instant-only big model (SM99) | go | 15 / 21 | no / yes | yes | yes | yes | yes | 96.1% |

## Good or Easy within 4 s (answered within 4 s, reading the back included)

| model | ECE raw / recalibrated | LogLoss raw / recalibrated | recalibrated vs raw | recalibrated vs (a) | recalibrated vs (b) | raw vs (a) | raw vs (b) |
|---|---|---|---|---|---|---|---|
| the show-time big model (SM91s2) | 0.0116 / 0.0019 | 0.3435 / 0.3410 | -0.0025 [-0.0028, -0.0021] (-0.71%) | -0.0931 [-0.1005, -0.0862] (-21.45%) | -0.1449 [-0.1544, -0.1360] (-29.82%) | -0.0907 [-0.0981, -0.0840] (-20.89%) | -0.1425 [-0.1518, -0.1334] (-29.32%) |
| the show-time instant-only big model (SM99) | 0.0108 / 0.0018 | 0.3427 / 0.3404 | -0.0023 [-0.0027, -0.0020] (-0.68%) | -0.0937 [-0.1012, -0.0867] (-21.59%) | -0.1456 [-0.1550, -0.1366] (-29.96%) | -0.0914 [-0.0988, -0.0845] (-21.05%) | -0.1433 [-0.1527, -0.1341] (-29.48%) |

## Where each timed event is reliable (values of S), raw -> recalibrated

| model | Good or Easy within S s | recalled within S s | Easy within S s |
|---|---|---|---|
| the show-time big model (SM91s2) | 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s -> 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s | 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s -> 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s | 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s -> 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s |
| the show-time instant-only big model (SM99) | 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s -> 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s | 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s -> 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s | 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s -> 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s |

## ECE per S, raw -> recalibrated (Good or Easy within S s)

| model | 2 s | 3 s | 4 s | 5 s | 6 s | 8 s | 10 s | 15 s | 20 s | 30 s |
|---|---|---|---|---|---|---|---|---|---|---|
| the show-time big model (SM91s2) | 0.0110 -> 0.0013 | 0.0113 -> 0.0020 | 0.0116 -> 0.0019 | 0.0080 -> 0.0015 | 0.0069 -> 0.0018 | 0.0035 -> 0.0013 | 0.0028 -> 0.0014 | 0.0035 -> 0.0013 | 0.0029 -> 0.0010 | 0.0026 -> 0.0012 |
| the show-time instant-only big model (SM99) | 0.0106 -> 0.0014 | 0.0106 -> 0.0018 | 0.0108 -> 0.0018 | 0.0072 -> 0.0018 | 0.0064 -> 0.0022 | 0.0039 -> 0.0015 | 0.0032 -> 0.0016 | 0.0021 -> 0.0009 | 0.0019 -> 0.0008 | 0.0018 -> 0.0012 |

## Grade only: P(Good or Easy) and P(Easy) (shipped raw; diagnostic: recalibrated too)

| model | event | ECE raw | raw vs (b) | reliable raw | ECE recalibrated (diagnostic) | recalibrated vs raw (diagnostic) |
|---|---|---|---|---|---|---|
| the show-time big model (SM91s2) | Good or Easy | 0.0011 | -0.1809 [-0.1949, -0.1680] (-33.04%) | yes | 0.0004 | +0.0000 [+0.0000, +0.0000] (+0.01%) |
| the show-time big model (SM91s2) | Easy | 0.0003 | -0.1005 [-0.1132, -0.0885] (-49.93%) | yes | 0.0003 | +0.0000 [+0.0000, +0.0001] (+0.04%) |
| the show-time instant-only big model (SM99) | Good or Easy | 0.0010 | -0.1814 [-0.1954, -0.1684] (-33.12%) | yes | 0.0005 | +0.0001 [+0.0000, +0.0001] (+0.01%) |
| the show-time instant-only big model (SM99) | Easy | 0.0003 | -0.1006 [-0.1134, -0.0886] (-50.02%) | yes | 0.0003 | +0.0000 [+0.0000, +0.0001] (+0.04%) |

## Coverage and fits

| model | users | rows | recalibrated rows | users with recalibration | refits | identity fits | clamped fits | repair changed | dump equals SM120 |
|---|---|---|---|---|---|---|---|---|---|
| the show-time big model (SM91s2) | 719 | 28,995,499 | 27,866,512 (96.1%) | 719 | 48,670 | 185,588 | 76 | 19.18% | yes |
| the show-time instant-only big model (SM99) | 719 | 28,995,499 | 27,866,512 (96.1%) | 719 | 48,670 | 185,588 | 187 | 19.45% | yes |

Every per-S number, the sensitivity arm (5,000-review minimum), the arm before the repair and the recalibrated-rows-only comparison are in `sm125-result.json`.
