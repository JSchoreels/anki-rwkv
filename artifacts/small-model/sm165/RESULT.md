# Big long cards with 6-bit factors (SM165): result

## Question and method

How much do the 6-bit factors of the release format for Big (F6, `w8+lr2.16.4q6x8`) cost on cards with 128 or more earlier
answers? The CPU stream engine scored F6 only, for Big two-head (SM91s2) and Big Instant-Only (SM99), on the 50 long-card
users (3,505,515 reviews); the int8 reference (REF) and 8-bit-factor (T24) rows were reused from the low-rank int8 states
test (SM148) and the bounded-memory long-card scorer (SM155). All 50 users finished (`SM165 DONE (50 of 50 users)`, 05:45 on
11-25); no user failed or was killed. Read-out exit code 0 (decided); T24 vs REF reproduces SM155 exactly.

## Gate

`cpu/gate/compare.txt`: `4/4 pass; worst max|dp| 0.00e+00` (users 2388 and 2883, F6 and T24, stock vs stream).
`cpu/gate/reuse.txt`: `T24 stream rows vs reused rows, 12 metrics, worst 0.0e+00: PASS`.

## Cards with 128+ earlier answers (30,089 reviews, 50 of 50 users for every label)

% = 100 x (LogLoss new / LogLoss base - 1), review-weighted, paired user bootstrap, 2,000 draws, 95% CI.

| model, head | T24 vs REF (SM155) | F6 vs REF | F6 vs T24 |
|---|---|---|---|
| Big two-head (SM91s2), curve | -0.016% [-0.092, +0.086] | +0.022% [-0.117, +0.299] | +0.038% [-0.135, +0.336] |
| Big two-head (SM91s2), instant | +0.026% [-0.084, +0.110] | -0.009% [-0.171, +0.166] | -0.035% [-0.137, +0.122] |
| Big Instant-Only (SM99), instant | +0.006% [-0.105, +0.061] | +0.010% [-0.137, +0.154] | +0.004% [-0.087, +0.142] |

The other review-count buckets are in `sm165-result.md`.

## Tolerance

The user's 11-22 long-card tolerance is about 0.5% (point) and 1% (95% CI upper bound) against int8 on 128+ cards. All
three F6 vs REF cells are inside it (largest point +0.022%, largest CI upper bound +0.299%, both for the curve head of Big
two-head (SM91s2)). It is a warning only, never a blocker, and it does not change the Big F6 go of the release-candidate
re-check (SM164).

## Speed

7.0M review-evaluations (3,505,515 reviews x 2 models) in 160.4 process-minutes from the rows' model seconds, so about 728
review-evals/s per process with 3 processes side by side (Big two-head (SM91s2) 80.0 process-minutes, Big Instant-Only
(SM99) 80.5). The job ran 04:44 to 05:45, 61 minutes of wall time, inside the forecast of 50 to 100 minutes.
