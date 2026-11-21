# Fluency vs desired retention (SM127): full run result (v3, 150 users)

**Short answer.** A high desired retention is enough. The fluency scheduler (flu-0.80) did not add fluent cards; it lost about 23% of them against the best desired-retention arm.

## Design in three lines

- 150 benchmark users, 365 simulated days, 22 lanes per user, every policy lane with the user's real mean review seconds per day (equal daily time).
- Answer-time level anchor: sampled later-day times are scaled per user and per day so that the reference lane matches the real seconds (design.md 13.1).
- Fluent means recalled and answered within the user's own S_u, reading the back included, where S_u is the 90th percentile of the user's own pre-split passed-review time, clipped to 3-45 s (design.md 13.2). The decision rule is the pre-registered one (design.md 13.4); no threshold was changed.

## Run note: three users crashed and were rerun

At 15:26 on 11-21, users 1190, 3339 and 4070 crashed in the engine set-up because they have no old card at the split. The fix is commit 083e5ae9 (`scripts/sm41_engine.py`, design.md 13.8); the changed line runs only for a user with no old card, so the other 147 rows, which come from the original run, could not change. Only those 3 users were rerun (15:30-15:46). Their crash logs are kept as `logs-v3/<user>.log.crash-1526`. No table or report was read before the fix.

## Policy table (population means, expected cards at day 365)

| policy | min/day | reviews/day | memorised | fluent | retention | passed within own S | floor share |
|---|---|---|---|---|---|---|---|
| teacher forcing, real schedule (tf) | 13.5 | 48 | 2568 | 2180 | 0.847 | 0.717 | 0.000 |
| desired retention 0.90 (ihr-0.90) | 11.8 | 40 | 2171 | 1781 | 0.854 | 0.656 | 0.000 |
| desired retention 0.95 (ihr-0.95) | 12.5 | 46 | 2030 | 1685 | 0.892 | 0.713 | 0.000 |
| desired retention 0.98 (ihr-0.98) | 13.2 | 58 | 1930 | 1639 | 0.918 | 0.770 | 0.000 |
| fluency target 0.80 (flu-0.80) | 13.6 | 40 | 1683 | 1375 | 0.884 | 0.632 | 0.476 |
| fluency target 0.80 with fallback at 0.95 (flu-0.80-fb95) | 12.5 | 39 | 1995 | 1627 | 0.870 | 0.648 | 0.428 |
| fluency target 0.70 (flu-0.70) | 13.3 | 35 | 1846 | 1474 | 0.858 | 0.558 | 0.384 |
| fluency target 0.50 (flu-0.50) | 11.1 | 28 | 2126 | 1685 | 0.774 | 0.437 | 0.225 |

The full 12-lane table with all columns is in `sm127-tables-v3.md`. The best desired-retention arm by fluent cards is desired retention 0.90 (ihr-0.90).

## Comparisons (ratio of population totals, 95% paired user-bootstrap CI)

| comparison | fluent | memorised | minutes |
|---|---|---|---|
| flu-0.80 vs ihr-0.90 | -22.78% [-26.91, -18.78] | -22.49% [-26.50, -18.54] | +15.14% [+8.81, +23.12] |
| flu-0.80 vs ihr-0.95 | -18.39% [-22.73, -14.36] | -17.11% [-21.25, -13.13] | +8.89% [+4.51, +14.27] |
| flu-0.80 vs ihr-0.98 | -16.07% [-20.58, -11.95] | -12.84% [-17.19, -8.69] | +2.83% [-0.07, +6.33] |
| flu-0.70 vs ihr-0.90 | -17.23% [-21.65, -12.85] | -14.96% [-19.04, -10.94] | +12.23% [+6.33, +19.57] |
| flu-0.80-fb95 vs ihr-0.90 | -8.66% [-10.86, -6.52] | -8.11% [-10.26, -6.17] | +5.57% [+3.06, +8.53] |

**Verdict (decision rule of design.md 7 on the primary statistic, the ratio of totals; feedback on): a high desired retention is enough (the fluency scheduler adds less than 2% fluent cards).** flu-0.80 vs ihr-0.90: fluent -22.78% [-26.91, -18.78] | -19.77%; memorised -22.49% [-26.50, -18.54] | -17.64%. The secondary statistic (mean of per-user ratios) gives the same verdict.

## Feedback check (arms without answer-time feedback)

Without answer-time feedback, flu-0.80-nofb vs ihr-0.90-nofb: fluent -2.58% [-5.80, +0.28] | -0.02%; memorised -5.94% [-9.83, -2.45] | -0.91%. The rule gives the same verdict: a high desired retention is enough. So the result does not come from the feedback loop alone. Against desired retention 0.98 without feedback (ihr-0.98-nofb), the fluency arm is +23.34% [+16.97, +30.68] on fluent cards, but the best desired-retention arm is 0.90, so this does not change the verdict; feedback makes the 0.98 arm itself gain +9.12% fluent cards over its no-feedback version.

## Level check (design.md 13.5)

Seconds per later-day review, daily re-anchoring over teacher forcing: 0.960 [0.902, 1.005] (16.17 s against 16.84 s); accepted band [0.95, 1.05]: accepted. Drift check on the first-touch predicted pass: -0.75% [-2.28, +0.47]. Final answer-time scale per user: median 0.983, quartiles 0.917-1.029.

## Floor shares and S_u

The floor share is the share of reviews sent to the daily-review floor because no interval reaches the target: 0.476 for the 0.80 target, 0.384 for 0.70 and 0.225 for 0.50. S_u has median 31.9 s, quartiles 17.1-45.0 s, range 5.8-45.0 s.

## Caveat (design.md 13.2)

S_u is lenient for slow users (median about 32 s). The result reads as "a fluency scheduler against desired retention, each user at their own usual slow tail", not as "within 8 s". The global 8 s pilot stays the reference for that wording.

## Pilot next to it

The 12-user pilot (answered within 8 s, reading the back included) was inconclusive (flu-0.80 vs ihr-0.90 fluent -14.83% [-31.31, +20.20] after the diagnostic version, memorised -24.85% [-36.10, -13.49]). The 150-user run has the power the pilot lacked and points the same way, now with a clear negative interval.

## Cost

52,169,776 lane reviews, 41.47 CPU process-hours (about 400 lane reviews/s per process), CPU only, launched 01:28:51 and done 15:46:56 on Sat 11-21 (including the 3-user rerun).
