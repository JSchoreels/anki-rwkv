# Rolling recalibration for the show-time two-head cool-down for Tiny (SM124) (SM126): result

**What answer time means.** The answer time is the whole time on the card, front and back together: from the moment the question is shown to the moment a grade button is pressed, capped at the preset's maximum answer seconds. Every "within S s" below means answered within S s, reading the back included.

Scoring only (no training), CPU recalibration of per-review outputs re-dumped on the GPU with fast Good/Easy prediction (SM120)'s code, on the 732 set (723 users, show-time inputs, leak-free, SM117's time-labelled test reviews). Pre-registered in `design.md`. Scheme: per user, Platt scaling per event refitted every 7 days on the user's latest 30,000 earlier reviews (earlier days only); raw model until 1,000 earlier reviews; repair (non-decreasing in S, Easy within Good or Easy within recalled, each timed event at most the raw P(set)). Grade-only P(set) is not recalibrated. Reliable = ECE (20 equal-mass bins) below 0.01 and LogLoss below every baseline of SM120 (not recalibrated), each difference with the 95% CI (2,000 user bootstraps, seed 0, review-weighted) below 0 and at least 0.05%.

## Verdict per model (pre-registered rules)

| model | go | 2-10 s cells reliable raw / recalibrated (of 21) | Good or Easy within 4 s reliable raw / recalibrated | G1 all 21 | G2 4 s better than raw | G3 no cell worse | G4 P(Good or Easy), P(Easy) | rows recalibrated |
|---|---|---|---|---|---|---|---|---|
| the show-time two-head cool-down for Tiny (SM124) | go | 13 / 21 | no / yes | yes | yes | yes | yes | 96.2% |

## Good or Easy within 4 s (answered within 4 s, reading the back included)

| model | ECE raw / recalibrated | LogLoss raw / recalibrated | recalibrated vs raw | recalibrated vs (a) | recalibrated vs (b) | raw vs (a) | raw vs (b) |
|---|---|---|---|---|---|---|---|
| the show-time two-head cool-down for Tiny (SM124) | 0.0137 / 0.0018 | 0.3469 / 0.3434 | -0.0035 [-0.0040, -0.0030] (-1.00%) | -0.0910 [-0.0983, -0.0840] (-20.94%) | -0.1423 [-0.1514, -0.1339] (-29.30%) | -0.0875 [-0.0947, -0.0808] (-20.14%) | -0.1389 [-0.1479, -0.1305] (-28.59%) |

## Where each timed event is reliable (values of S), raw -> recalibrated

| model | Good or Easy within S s | recalled within S s | Easy within S s |
|---|---|---|---|
| the show-time two-head cool-down for Tiny (SM124) | 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s -> 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s | 8 s, 10 s, 15 s, 20 s, 30 s -> 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s | 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s -> 2 s, 3 s, 4 s, 5 s, 6 s, 8 s, 10 s, 15 s, 20 s, 30 s |

## ECE per S, raw -> recalibrated (Good or Easy within S s)

| model | 2 s | 3 s | 4 s | 5 s | 6 s | 8 s | 10 s | 15 s | 20 s | 30 s |
|---|---|---|---|---|---|---|---|---|---|---|
| the show-time two-head cool-down for Tiny (SM124) | 0.0102 -> 0.0023 | 0.0136 -> 0.0021 | 0.0137 -> 0.0018 | 0.0098 -> 0.0017 | 0.0079 -> 0.0020 | 0.0038 -> 0.0018 | 0.0031 -> 0.0018 | 0.0028 -> 0.0011 | 0.0023 -> 0.0013 | 0.0024 -> 0.0013 |

## Grade only: P(Good or Easy) and P(Easy) (shipped raw; diagnostic: recalibrated too)

| model | event | ECE raw | raw vs (b) | reliable raw | ECE recalibrated (diagnostic) | recalibrated vs raw (diagnostic) |
|---|---|---|---|---|---|---|
| the show-time two-head cool-down for Tiny (SM124) | Good or Easy | 0.0014 | -0.1785 [-0.1931, -0.1643] (-32.71%) | yes | 0.0003 | +0.0000 [+0.0000, +0.0001] (+0.01%) |
| the show-time two-head cool-down for Tiny (SM124) | Easy | 0.0004 | -0.0990 [-0.1114, -0.0872] (-49.56%) | yes | 0.0003 | +0.0000 [+0.0000, +0.0000] (+0.04%) |

## Coverage and fits

| model | users | rows | recalibrated rows | users with recalibration | refits | identity fits | clamped fits | repair changed | dump equals SM120 |
|---|---|---|---|---|---|---|---|---|---|
| the show-time two-head cool-down for Tiny (SM124) | 722 | 30,093,623 | 28,949,746 (96.2%) | 722 | 49,203 | 185,862 | 108 | 18.85% | yes |

Every per-S number, the sensitivity arm (5,000-review minimum), the arm before the repair and the recalibrated-rows-only comparison are in `sm126-recal-result.json`.
