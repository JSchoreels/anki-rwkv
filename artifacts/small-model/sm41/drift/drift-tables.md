# The small model in the review simulator (SM41): drift study tables

## Horizon 60 days: 50 users, 133 split points

Lanes per split: 32 (tf, tf-answer, fr#0, fr#1, ...). Drift = lane minus the real user (the teacher-forced lane for predictions) on the same reviews, in points; mean over users (splits of a user averaged first) ± 95% CI over users. 'first touch' = each old card's first later-day review in the window (its day was set from real pre-split outcomes, so it does not depend on the lane's samples). 'state' = the model's own predicted pass rate on those reviews (no outcome noise): the cleanest drift signal.

### First-touch state drift (predicted pass rate, lane minus teacher forcing), points

| arm | days 0-29 | days 30-59 | mean abs, last block |
|---|---|---|---|
| tf | +0.0 ± 0.0 | +0.0 ± 0.0 | 0.00 |
| tf-answer | -0.0 ± 0.0 | +0.0 ± 0.0 | 0.00 |
| fr | -1.1 ± 0.9 | -5.1 ± 3.6 | 9.07 |
| fr-answer | -1.2 ± 0.9 | -4.5 ± 3.4 | 8.72 |
| fr-treal | -1.5 ± 0.8 | -4.7 ± 3.3 | 9.20 |
| fr-texp | -1.6 ± 0.9 | -4.5 ± 3.0 | 8.36 |
| fr-tt0.5 | -1.0 ± 0.9 | -3.7 ± 3.1 | 8.31 |
| fr-gt0.8 | +1.5 ± 1.6 | +5.5 ± 2.8 | 10.15 |
| fr-anchor7 | -0.7 ± 0.8 | -2.5 ± 2.2 | 5.90 |
| fr-shared1 | -0.6 ± 0.4 | -0.6 ± 1.0 | 2.19 |
| fr-shared7 | -0.7 ± 0.7 | -1.7 ± 2.1 | 4.10 |
| fr-texp-shared1 | -0.8 ± 0.4 | -0.8 ± 1.0 | 2.23 |

### First-touch realised retention drift (lane minus real), points

| arm | days 0-29 | days 30-59 |
|---|---|---|
| tf | +0.0 ± 0.0 | +0.0 ± 0.0 |
| tf-answer | +0.0 ± 0.0 | +0.0 ± 0.0 |
| fr | -1.3 ± 1.2 | -5.8 ± 3.9 |
| fr-answer | -1.3 ± 1.2 | -5.4 ± 4.0 |
| fr-treal | -1.7 ± 1.1 | -5.4 ± 3.9 |
| fr-texp | -1.7 ± 1.3 | -4.9 ± 3.4 |
| fr-tt0.5 | -1.1 ± 1.3 | -4.0 ± 3.5 |
| fr-gt0.8 | +2.6 ± 1.9 | +6.1 ± 3.3 |
| fr-anchor7 | -0.8 ± 1.1 | -2.6 ± 2.5 |
| fr-shared1 | -0.8 ± 0.7 | -1.0 ± 1.8 |
| fr-shared7 | -0.8 ± 1.0 | -2.1 ± 2.6 |
| fr-texp-shared1 | -1.0 ± 0.8 | -1.2 ± 1.7 |

### All later-day reviews on the real schedule (old cards / new cards), realised retention drift, points

| arm | old days 0-29 | old days 30-59 | new days 0-29 | new days 30-59 |
|---|---|---|---|---|
| tf | +0.0 ± 0.0 | +0.0 ± 0.0 | +0.0 ± 0.0 | +0.0 ± 0.0 |
| tf-answer | +0.0 ± 0.0 | +0.0 ± 0.0 | +0.0 ± 0.0 | +0.0 ± 0.0 |
| fr | -1.0 ± 1.1 | -4.0 ± 2.5 | +0.3 ± 1.7 | -1.2 ± 2.0 |
| fr-answer | -1.1 ± 1.1 | -3.8 ± 2.3 | -0.8 ± 1.8 | -2.4 ± 1.9 |
| fr-treal | -1.5 ± 1.1 | -4.1 ± 2.1 | -0.2 ± 1.9 | -1.8 ± 1.8 |
| fr-texp | -1.2 ± 1.1 | -3.8 ± 2.1 | +1.2 ± 1.9 | -0.2 ± 1.7 |
| fr-tt0.5 | -0.8 ± 1.2 | -2.8 ± 2.3 | +0.6 ± 1.8 | -0.4 ± 1.9 |
| fr-gt0.8 | +3.8 ± 1.7 | +5.5 ± 3.0 | +6.3 ± 2.4 | +8.8 ± 2.6 |
| fr-anchor7 | -0.8 ± 1.0 | -2.6 ± 1.4 | -0.2 ± 1.2 | -1.6 ± 1.3 |
| fr-shared1 | -0.5 ± 0.5 | -1.1 ± 1.1 | -0.4 ± 1.1 | -0.9 ± 1.1 |
| fr-shared7 | -0.5 ± 0.7 | -1.8 ± 1.6 | +0.1 ± 1.3 | -1.0 ± 1.4 |
| fr-texp-shared1 | -0.6 ± 0.5 | -1.4 ± 1.1 | -0.2 ± 1.1 | -0.8 ± 1.1 |

### Grades, answer time and implied same-day load (days 30-59), lane minus real

Again = share of non-first reviews; Easy = share of later-day passes; seconds = mean answer time of later-day reviews as fed back (fr-texp feeds exp E[ln t], a geometric mean) / the head's arithmetic expectation for the lane's grade; follow-ups = same-day reviews per review implied by the user's own step rules (on the real schedule reviews per day are fixed by construction; this is the load a policy lane would add). Real means: Again 0.148, Easy 0.153, 17.4 s.

| arm | Again, points | Easy of passes, points | seconds fed | seconds expected | follow-ups per 100 reviews |
|---|---|---|---|---|---|
| tf | +0.0 ± 0.0 | +0.0 ± 0.0 | +0.0 ± 0.0 | -0.7 ± 0.8 | +0.0 ± 0.0 |
| tf-answer | +0.0 ± 0.0 | +0.0 ± 0.0 | +0.0 ± 0.0 | -0.7 ± 0.8 | +0.0 ± 0.0 |
| fr | +0.4 ± 1.9 | +5.6 ± 2.5 | -1.4 ± 1.0 | -1.4 ± 1.0 | +0.5 ± 1.6 |
| fr-answer | +1.6 ± 1.7 | +4.4 ± 2.6 | -1.3 ± 1.0 | -1.4 ± 1.0 | +1.6 ± 1.5 |
| fr-treal | +0.7 ± 1.4 | +4.8 ± 2.4 | -0.0 ± 0.0 | -0.5 ± 0.8 | +0.8 ± 1.1 |
| fr-texp | -0.4 ± 1.3 | +7.5 ± 2.9 | -1.1 ± 1.2 | +2.1 ± 1.1 | -0.1 ± 1.0 |
| fr-tt0.5 | -0.3 ± 1.8 | +5.4 ± 2.9 | -1.9 ± 1.5 | -1.4 ± 1.1 | +0.1 ± 1.5 |
| fr-gt0.8 | -7.8 ± 1.5 | +2.9 ± 3.7 | -3.1 ± 1.3 | -3.2 ± 1.3 | -5.7 ± 1.1 |
| fr-anchor7 | +0.4 ± 1.2 | +1.6 ± 1.3 | -1.1 ± 0.9 | -1.2 ± 1.0 | +0.5 ± 1.0 |
| fr-shared1 | -0.9 ± 1.0 | +1.2 ± 0.9 | -0.2 ± 0.9 | -0.2 ± 0.9 | -0.5 ± 0.7 |
| fr-shared7 | -0.5 ± 1.4 | +2.1 ± 1.7 | -0.7 ± 1.0 | -0.8 ± 1.0 | -0.2 ± 1.1 |
| fr-texp-shared1 | -0.9 ± 1.0 | +1.4 ± 0.9 | -1.8 ± 1.0 | +1.5 ± 0.9 | -0.6 ± 0.8 |

### Mitigation vs plain free running, paired per user (days 30-59)

Reduction of the mean absolute first-touch state drift and of the absolute Easy and seconds drift; positive = less drift than fr.

| arm | what it does | state drift reduction, points | Easy | seconds |
|---|---|---|---|---|
| fr-answer | free running, answer-to-answer query rows (stage 1) | +0.35 ± 0.61 | +0.4 ± 1.1 | +0.0 ± 0.1 |
| fr-treal | free running grades, real times (bound for any time fix) | -0.13 ± 1.77 | +0.2 ± 1.3 | +3.0 ± 0.8 |
| fr-texp | expected answer time fed (exp of the expected ln time) | +0.71 ± 2.22 | -1.5 ± 1.5 | -0.4 ± 0.4 |
| fr-tt0.5 | time head sampled at temperature 0.5 | +0.76 ± 0.87 | -0.5 ± 1.0 | -2.0 ± 0.9 |
| fr-gt0.8 | grade head sampled at temperature 0.8 | -1.08 ± 3.13 | -1.8 ± 2.1 | -1.3 ± 0.6 |
| fr-anchor7 | every 7th day teacher forced (real grades and times) | +3.17 ± 2.05 | +5.3 ± 2.1 | +0.6 ± 0.2 |
| fr-shared1 | deck / preset / user states re-anchored to teacher forcing every day | +6.87 ± 2.91 | +6.9 ± 2.4 | +0.8 ± 0.4 |
| fr-shared7 | deck / preset / user states re-anchored every 7 days | +4.97 ± 2.23 | +5.3 ± 2.3 | +0.5 ± 0.3 |
| fr-texp-shared1 | expected time + daily shared-stream re-anchoring | +6.84 ± 3.05 | +6.8 ± 2.5 | +0.3 ± 0.5 |

### The duration leak (duration-leak scoring (SM31)): teacher forcing with show-time vs answer-to-answer query rows

| query rows | grade LogLoss, later-day reviews | grade LogLoss, same-day reviews |
|---|---|---|
| answer-to-answer (training data, stage 1) | 0.65231 | 0.53153 |
| show time (the app) | 0.65231 | 0.53912 |

Speed: 19,881,280 lane reviews in 2.27 process-hours of lanes (2,437 lane reviews/s per process, 1 thread), warm-ups 2.29 process-hours.

## Horizon 120 days: 10 users, 10 split points

Lanes per split: 32 (tf, tf-answer, fr#0, fr#1, ...). Drift = lane minus the real user (the teacher-forced lane for predictions) on the same reviews, in points; mean over users (splits of a user averaged first) ± 95% CI over users. 'first touch' = each old card's first later-day review in the window (its day was set from real pre-split outcomes, so it does not depend on the lane's samples). 'state' = the model's own predicted pass rate on those reviews (no outcome noise): the cleanest drift signal.

### First-touch state drift (predicted pass rate, lane minus teacher forcing), points

| arm | days 0-29 | days 30-59 | days 60-89 | days 90-119 | mean abs, last block |
|---|---|---|---|---|---|
| tf | +0.0 ± 0.0 | +0.0 ± 0.0 | +0.0 ± 0.0 | +0.0 ± 0.0 | 0.00 |
| tf-answer | -0.0 ± 0.0 | +0.0 ± 0.0 | +0.0 ± 0.0 | +0.0 ± 0.0 | 0.00 |
| fr | -1.3 ± 1.7 | -8.9 ± 6.8 | -10.2 ± 9.4 | -0.7 ± 5.8 | 4.96 |
| fr-answer | -1.3 ± 1.9 | -7.9 ± 5.4 | -11.1 ± 10.3 | -7.3 ± 11.5 | 10.66 |
| fr-treal | -1.7 ± 1.9 | -8.8 ± 4.8 | -11.4 ± 9.2 | -1.1 ± 7.9 | 7.96 |
| fr-texp | -2.1 ± 1.8 | -9.4 ± 5.8 | -11.9 ± 8.0 | -3.6 ± 10.8 | 10.89 |
| fr-tt0.5 | -1.1 ± 1.9 | -7.7 ± 6.2 | -9.0 ± 9.1 | -1.0 ± 6.3 | 5.51 |
| fr-gt0.8 | +2.1 ± 1.3 | -0.6 ± 16.7 | +2.1 ± 14.6 | +13.9 ± 10.4 | 13.90 |
| fr-anchor7 | -1.7 ± 1.2 | -6.5 ± 5.2 | -8.0 ± 7.8 | +0.3 ± 4.2 | 3.51 |
| fr-shared1 | -0.4 ± 0.9 | -2.1 ± 2.8 | -3.1 ± 3.2 | +1.0 ± 2.7 | 1.61 |
| fr-shared7 | -1.2 ± 1.2 | -3.6 ± 5.4 | -4.0 ± 4.2 | +1.0 ± 2.5 | 2.05 |
| fr-texp-shared1 | -0.7 ± 0.9 | -2.3 ± 2.8 | -3.1 ± 3.2 | +0.5 ± 2.8 | 1.96 |

### First-touch realised retention drift (lane minus real), points

| arm | days 0-29 | days 30-59 | days 60-89 | days 90-119 |
|---|---|---|---|---|
| tf | +0.0 ± 0.0 | +0.0 ± 0.0 | +0.0 ± 0.0 | +0.0 ± 0.0 |
| tf-answer | +0.0 ± 0.0 | +0.0 ± 0.0 | +0.0 ± 0.0 | +0.0 ± 0.0 |
| fr | -2.2 ± 2.7 | -10.9 ± 8.9 | -17.5 ± 11.5 | -3.2 ± 13.5 |
| fr-answer | -2.2 ± 2.9 | -9.3 ± 7.1 | -23.8 ± 16.5 | -9.9 ± 22.5 |
| fr-treal | -2.6 ± 2.8 | -11.0 ± 6.5 | -18.8 ± 10.4 | -5.2 ± 13.0 |
| fr-texp | -3.1 ± 2.8 | -12.1 ± 7.9 | -22.8 ± 14.8 | -4.5 ± 14.1 |
| fr-tt0.5 | -2.0 ± 2.8 | -10.3 ± 8.8 | -19.7 ± 16.3 | -8.0 ± 23.4 |
| fr-gt0.8 | +3.2 ± 1.8 | -1.7 ± 17.4 | -5.2 ± 15.2 | +8.1 ± 9.5 |
| fr-anchor7 | -2.6 ± 1.8 | -7.6 ± 6.4 | -15.3 ± 9.9 | -5.1 ± 17.7 |
| fr-shared1 | -1.4 ± 1.8 | -4.2 ± 4.1 | -10.9 ± 8.0 | -0.8 ± 10.4 |
| fr-shared7 | -2.0 ± 2.0 | -4.9 ± 5.5 | -10.9 ± 8.1 | -2.6 ± 12.3 |
| fr-texp-shared1 | -1.6 ± 1.8 | -4.7 ± 4.5 | -10.5 ± 7.9 | -2.2 ± 12.7 |

### All later-day reviews on the real schedule (old cards / new cards), realised retention drift, points

| arm | old days 0-29 | old days 30-59 | old days 60-89 | old days 90-119 | new days 0-29 | new days 30-59 | new days 60-89 | new days 90-119 |
|---|---|---|---|---|---|---|---|---|
| tf | +0.0 ± 0.0 | +0.0 ± 0.0 | +0.0 ± 0.0 | +0.0 ± 0.0 | +0.0 ± 0.0 | +0.0 ± 0.0 | +0.0 ± 0.0 | +0.0 ± 0.0 |
| tf-answer | +0.0 ± 0.0 | +0.0 ± 0.0 | +0.0 ± 0.0 | +0.0 ± 0.0 | +0.0 ± 0.0 | +0.0 ± 0.0 | +0.0 ± 0.0 | +0.0 ± 0.0 |
| fr | -2.3 ± 1.9 | -7.2 ± 6.0 | -8.9 ± 5.6 | -4.9 ± 5.6 | +1.5 ± 3.7 | -1.9 ± 3.5 | -2.6 ± 6.9 | +0.7 ± 7.1 |
| fr-answer | -2.7 ± 2.1 | -6.6 ± 5.0 | -9.1 ± 5.8 | -8.0 ± 5.4 | +0.3 ± 4.1 | -3.2 ± 2.0 | -3.8 ± 5.5 | -3.0 ± 5.1 |
| fr-treal | -3.1 ± 2.0 | -8.3 ± 5.6 | -11.1 ± 6.3 | -7.4 ± 6.8 | +1.4 ± 3.4 | -2.6 ± 2.3 | -5.0 ± 6.1 | -1.6 ± 6.9 |
| fr-texp | -3.0 ± 2.0 | -7.5 ± 4.9 | -8.9 ± 5.4 | -6.0 ± 6.6 | +2.9 ± 4.2 | -0.5 ± 2.2 | -2.4 ± 5.9 | -0.6 ± 6.6 |
| fr-tt0.5 | -2.4 ± 2.2 | -5.9 ± 4.9 | -7.1 ± 5.6 | -4.5 ± 6.2 | +2.2 ± 3.4 | -0.7 ± 2.4 | -2.3 ± 6.0 | -0.6 ± 8.1 |
| fr-gt0.8 | +3.4 ± 2.2 | +4.9 ± 7.0 | +6.8 ± 5.5 | +9.6 ± 5.5 | +9.7 ± 4.7 | +12.7 ± 4.4 | +12.9 ± 5.9 | +16.9 ± 5.5 |
| fr-anchor7 | -2.9 ± 1.7 | -4.4 ± 3.5 | -6.1 ± 4.5 | -1.7 ± 3.9 | +0.6 ± 2.0 | -2.1 ± 2.4 | -3.2 ± 4.1 | +0.0 ± 4.4 |
| fr-shared1 | -1.2 ± 1.2 | -2.5 ± 2.9 | -3.5 ± 2.5 | -1.8 ± 3.3 | +0.6 ± 1.8 | -0.7 ± 2.0 | -1.5 ± 3.2 | +0.3 ± 3.4 |
| fr-shared7 | -1.9 ± 1.3 | -3.5 ± 3.4 | -3.9 ± 2.5 | -2.3 ± 3.8 | +1.4 ± 2.1 | -0.9 ± 2.2 | -1.7 ± 3.7 | +1.0 ± 4.0 |
| fr-texp-shared1 | -1.4 ± 1.2 | -2.8 ± 2.7 | -3.9 ± 2.5 | -2.8 ± 3.5 | +1.0 ± 1.8 | -0.2 ± 1.8 | -1.2 ± 3.0 | +0.3 ± 3.3 |

### Grades, answer time and implied same-day load (days 90-119), lane minus real

Again = share of non-first reviews; Easy = share of later-day passes; seconds = mean answer time of later-day reviews as fed back (fr-texp feeds exp E[ln t], a geometric mean) / the head's arithmetic expectation for the lane's grade; follow-ups = same-day reviews per review implied by the user's own step rules (on the real schedule reviews per day are fixed by construction; this is the load a policy lane would add). Real means: Again 0.174, Easy 0.080, 24.0 s.

| arm | Again, points | Easy of passes, points | seconds fed | seconds expected | follow-ups per 100 reviews |
|---|---|---|---|---|---|
| tf | +0.0 ± 0.0 | +0.0 ± 0.0 | +0.0 ± 0.0 | -4.3 ± 7.6 | +0.0 ± 0.0 |
| tf-answer | +0.0 ± 0.0 | +0.0 ± 0.0 | +0.0 ± 0.0 | -4.3 ± 7.6 | +0.0 ± 0.0 |
| fr | +0.6 ± 4.9 | +11.1 ± 12.9 | -7.1 ± 9.1 | -7.1 ± 9.2 | -0.6 ± 3.4 |
| fr-answer | +3.7 ± 4.4 | +8.0 ± 12.8 | -6.4 ± 9.3 | -6.4 ± 9.3 | +2.4 ± 3.3 |
| fr-treal | +1.8 ± 5.0 | +8.9 ± 15.5 | +0.0 ± 0.0 | -4.1 ± 8.0 | +0.4 ± 4.0 |
| fr-texp | +0.7 ± 4.7 | +8.7 ± 19.0 | -6.4 ± 10.1 | -3.0 ± 9.6 | -0.4 ± 3.5 |
| fr-tt0.5 | +1.0 ± 6.4 | +8.8 ± 17.9 | -5.7 ± 5.5 | -6.0 ± 6.4 | -0.3 ± 4.4 |
| fr-gt0.8 | -14.2 ± 3.4 | +5.2 ± 19.6 | -10.6 ± 9.4 | -10.6 ± 9.4 | -11.5 ± 3.5 |
| fr-anchor7 | -0.3 ± 3.4 | +1.6 ± 3.1 | -5.6 ± 8.4 | -6.1 ± 9.3 | +0.3 ± 2.5 |
| fr-shared1 | -0.9 ± 2.1 | +1.1 ± 1.1 | -4.6 ± 9.2 | -4.5 ± 9.1 | -0.3 ± 1.3 |
| fr-shared7 | -1.4 ± 2.4 | +0.5 ± 1.6 | -5.5 ± 9.4 | -5.4 ± 9.4 | -0.7 ± 1.3 |
| fr-texp-shared1 | -0.8 ± 2.1 | +1.7 ± 1.5 | -5.7 ± 9.3 | -2.2 ± 8.6 | -0.4 ± 1.4 |

### Mitigation vs plain free running, paired per user (days 90-119)

Reduction of the mean absolute first-touch state drift and of the absolute Easy and seconds drift; positive = less drift than fr.

| arm | what it does | state drift reduction, points | Easy | seconds |
|---|---|---|---|---|
| fr-answer | free running, answer-to-answer query rows (stage 1) | -5.70 ± 10.91 | +1.8 ± 4.3 | +0.5 ± 0.8 |
| fr-treal | free running grades, real times (bound for any time fix) | -3.00 ± 4.08 | -4.2 ± 7.0 | +8.6 ± 8.5 |
| fr-texp | expected answer time fed (exp of the expected ln time) | -5.93 ± 4.55 | -5.0 ± 8.1 | -1.2 ± 1.7 |
| fr-tt0.5 | time head sampled at temperature 0.5 | -0.55 ± 3.76 | -3.9 ± 6.8 | +1.7 ± 4.6 |
| fr-gt0.8 | grade head sampled at temperature 0.8 | -8.94 ± 11.09 | -1.6 ± 10.9 | -2.4 ± 1.9 |
| fr-anchor7 | every 7th day teacher forced (real grades and times) | +1.45 ± 2.31 | +8.9 ± 10.1 | +1.7 ± 1.4 |
| fr-shared1 | deck / preset / user states re-anchored to teacher forcing every day | +3.34 ± 1.60 | +10.0 ± 12.6 | +1.6 ± 2.0 |
| fr-shared7 | deck / preset / user states re-anchored every 7 days | +2.91 ± 2.51 | +9.6 ± 12.2 | +1.4 ± 2.0 |
| fr-texp-shared1 | expected time + daily shared-stream re-anchoring | +3.00 ± 2.08 | +9.5 ± 12.0 | +0.9 ± 1.9 |

### The duration leak (duration-leak scoring (SM31)): teacher forcing with show-time vs answer-to-answer query rows

| query rows | grade LogLoss, later-day reviews | grade LogLoss, same-day reviews |
|---|---|---|
| answer-to-answer (training data, stage 1) | 0.64621 | 0.52698 |
| show time (the app) | 0.64621 | 0.53548 |

Speed: 2,893,216 lane reviews in 0.34 process-hours of lanes (2,349 lane reviews/s per process, 1 thread), warm-ups 0.17 process-hours.

