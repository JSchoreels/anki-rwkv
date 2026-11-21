# Rolling recalibration for the six-layer Tiny (SM140): result

**In plain words.** Go for both models. After per-user recalibration, all 21 timed cells of 2 to 10 s are reliable for the six-layer Tiny two-head model (SM130) and for the six-layer Tiny Instant-Only model (SM130) (raw: 13 and 14 of 21). Good or Easy answered within 4 s, reading the back included, becomes reliable: ECE 0.0143 to 0.0016 and 0.0131 to 0.0022, LogLoss -1.24% and -1.09% vs raw (CI below 0), about -20% and -28.5% to -28.8% vs the two baselines of fast Good/Easy prediction (SM120). 96.2% of the rows are recalibrated; grade-only P(Good or Easy) and P(Easy) stay reliable raw. The answer-time heads come from the answer-time heads for the six-layer Tiny (SM138). 722 of the 723 users have rows in the recalibration for both models.

**What answer time means.** The answer time is the whole time on the card, front and back together: from the moment the question is shown to the moment a grade button is pressed, capped at the preset's maximum answer seconds. Every "within S s" below means answered within S s, reading the back included.

Scoring only (no training), CPU recalibration of per-review outputs re-dumped on the GPU with fast Good/Easy prediction (SM120)'s code, on the 732 set (723 users, show-time inputs, leak-free, SM117's time-labelled test reviews). Pre-registered in `design.md`. Scheme: per user, Platt scaling per event refitted every 7 days on the user's latest 30,000 earlier reviews (earlier days only); raw model until 1,000 earlier reviews; repair (non-decreasing in S, Easy within Good or Easy within recalled, each timed event at most the raw P(set)). Grade-only P(set) is not recalibrated. Reliable = ECE (20 equal-mass bins) below 0.01 and LogLoss below every baseline of SM120 (not recalibrated), each difference with the 95% CI (2,000 user bootstraps, seed 0, review-weighted) below 0 and at least 0.05%.

## Verdict per model (pre-registered rules)

| model | go | 2-10 s cells reliable raw / recalibrated (of 21) | Good or Easy within 4 s reliable raw / recalibrated | G1 all 21 | G2 4 s better than raw | G3 no cell worse | G4 P(Good or Easy), P(Easy) | rows recalibrated |
|---|---|---|---|---|---|---|---|---|
| the six-layer Tiny two-head model (SM130) | go | 13 / 21 | no / yes | yes | yes | yes | yes | 96.2% |
| the six-layer Tiny Instant-Only model (SM130) | go | 14 / 21 | no / yes | yes | yes | yes | yes | 96.2% |

## Good or Easy within 4 s (answered within 4 s, reading the back included)

| model | ECE raw / recalibrated | LogLoss raw / recalibrated | recalibrated vs raw | recalibrated vs (a) | recalibrated vs (b) | raw vs (a) | raw vs (b) |
|---|---|---|---|---|---|---|---|
| the six-layer Tiny two-head model (SM130) | 0.0143 / 0.0016 | 0.3517 / 0.3474 | -0.0044 [-0.0049, -0.0038] (-1.24%) | -0.0875 [-0.0946, -0.0807] (-20.11%) | -0.1384 [-0.1474, -0.1301] (-28.50%) | -0.0831 [-0.0899, -0.0766] (-19.11%) | -0.1341 [-0.1430, -0.1258] (-27.60%) |
| the six-layer Tiny Instant-Only model (SM130) | 0.0131 / 0.0022 | 0.3497 / 0.3459 | -0.0038 [-0.0044, -0.0033] (-1.09%) | -0.0887 [-0.0959, -0.0819] (-20.41%) | -0.1399 [-0.1489, -0.1315] (-28.80%) | -0.0849 [-0.0919, -0.0783] (-19.54%) | -0.1361 [-0.1451, -0.1278] (-28.01%) |

## Where each timed event is reliable (values of S), raw -> recalibrated

| model | Good or Easy within S s | recalled within S s | Easy within S s |
|---|---|---|---|
| the six-layer Tiny two-head model (SM130) | 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s -> 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s | 8 s, 10 s, 15 s, 20 s, 30 s -> 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s | 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s -> 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s |
| the six-layer Tiny Instant-Only model (SM130) | 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s -> 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s | 6 s, 8 s, 10 s, 15 s, 20 s, 30 s -> 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s | 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s -> 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s |

## ECE per S, raw -> recalibrated (Good or Easy within S s)

| model | 2 s | 3 s | 4 s | 5 s | 6 s | 8 s | 10 s | 15 s | 20 s | 30 s |
|---|---|---|---|---|---|---|---|---|---|---|
| the six-layer Tiny two-head model (SM130) | 0.0103 -> 0.0022 | 0.0144 -> 0.0021 | 0.0143 -> 0.0016 | 0.0100 -> 0.0013 | 0.0077 -> 0.0011 | 0.0030 -> 0.0007 | 0.0024 -> 0.0007 | 0.0024 -> 0.0007 | 0.0022 -> 0.0009 | 0.0028 -> 0.0014 |
| the six-layer Tiny Instant-Only model (SM130) | 0.0108 -> 0.0022 | 0.0143 -> 0.0024 | 0.0131 -> 0.0022 | 0.0092 -> 0.0015 | 0.0067 -> 0.0017 | 0.0026 -> 0.0012 | 0.0027 -> 0.0012 | 0.0045 -> 0.0013 | 0.0031 -> 0.0014 | 0.0016 -> 0.0014 |

## Grade only: P(Good or Easy) and P(Easy) (shipped raw; diagnostic: recalibrated too)

| model | event | ECE raw | raw vs (b) | reliable raw | ECE recalibrated (diagnostic) | recalibrated vs raw (diagnostic) |
|---|---|---|---|---|---|---|
| the six-layer Tiny two-head model (SM130) | Good or Easy | 0.0016 | -0.1770 [-0.1916, -0.1629] (-32.44%) | yes | 0.0004 | +0.0000 [-0.0000, +0.0000] (+0.00%) |
| the six-layer Tiny two-head model (SM130) | Easy | 0.0005 | -0.0983 [-0.1106, -0.0866] (-49.22%) | yes | 0.0003 | +0.0000 [+0.0000, +0.0000] (+0.02%) |
| the six-layer Tiny Instant-Only model (SM130) | Good or Easy | 0.0022 | -0.1780 [-0.1927, -0.1639] (-32.63%) | yes | 0.0004 | +0.0000 [-0.0000, +0.0000] (+0.00%) |
| the six-layer Tiny Instant-Only model (SM130) | Easy | 0.0003 | -0.0988 [-0.1111, -0.0870] (-49.42%) | yes | 0.0003 | +0.0000 [+0.0000, +0.0000] (+0.03%) |

## Coverage and fits

| model | users | rows | recalibrated rows | users with recalibration | refits | identity fits | clamped fits | repair changed | dump equals SM120 |
|---|---|---|---|---|---|---|---|---|---|
| the six-layer Tiny two-head model (SM130) | 722 | 30,093,623 | 28,949,746 (96.2%) | 722 | 49,203 | 185,862 | 119 | 19.30% | yes |
| the six-layer Tiny Instant-Only model (SM130) | 722 | 30,093,623 | 28,949,746 (96.2%) | 722 | 49,203 | 185,862 | 79 | 19.44% | yes |

Every per-S number, the sensitivity arm (5,000-review minimum), the arm before the repair and the recalibrated-rows-only comparison are in `sm140-result.json`.
