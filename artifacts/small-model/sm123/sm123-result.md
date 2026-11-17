# Recalibrating timed recall on your own collection (SM123): result

**What answer time means.** The whole time on the card, front and back together, from the question shown to the grade pressed, capped at the preset's maximum answer seconds. Every "within S s" means recalled and answered within S s, reading the back included.

CPU only, scoring only, on your own collection (the copy of 2026-10-14, reviews up to 2026-10-04; aggregate numbers only), reusing the per-review outputs of the timed-recall analysis on your own collection (SM121). Pre-registered in `design.md`. Rows: benchmark rows, whole collection 173,384, benchmark rows, deck subset 153,435, Mac slice, whole collection 35,256, Mac slice, deck subset 26,947. Static fits: 34,727 rows before the benchmark rows (2024-01 to 2024-11) for the benchmark rows, 35,256 eligible rows just before the Mac slice for the Mac slice; rolling: 99 refits, every 7 days, each on the latest 30,000 time-labelled rows of earlier days.

**Reliable after recalibration** = ECE (20 equal-mass bins) below 0.01 and LogLoss below every baseline of SM121's bar with the day-block 95% CI below 0 and each difference at least 0.05%. The baselines are not recalibrated (as in SM121); a sensitivity column compares the Platt-recalibrated head with the Platt-recalibrated strongest baseline.

The deck subset is the subset of decks the Mac replay used (scored with the whole collection's recalibration). Easy is rare (0.1% to 1% of rows), so its ECE is always tiny; where Easy fails, it is because a baseline is as good, not calibration. Hard has only 112 rows in the static benchmark window, below the 200-row minimum, so the static shared transform keeps Hard's time distribution unchanged there.

## Answer

**Yes, with a rolling refit: Good or Easy within 4 s (reading the back included) becomes reliable on your own collection after a 7-day rolling recalibration fitted only on earlier reviews.** For the longer Small Instant-Only model (SM113) with its own head, on the benchmark rows of the whole collection, the calibration error at 4 s drops from 0.0185 to 0.0053 with rolling Platt scaling (0.0046 with rolling isotonic), LogLoss improves by 0.28% against the uncalibrated head (CI below 0), and the lead over the strongest baseline (the Mac's recall-times-card-history baseline) grows from 7.92% to 8.18% (CI clear of zero); on the deck subset the error goes from 0.0175 to 0.0062 and the lead is 6.67%. The current Small pair (grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s head) goes from 0.0240 to 0.0046, the Tiny two-head model (SM106) from 0.0166 to 0.0049, the show-time big model (SM91s2) from 0.0385 to 0.0059.

- Rolling Platt: Good or Easy and recalled reliable at every S from 2 to 10 s on the whole collection for all eleven combinations; on the deck subset 64 of 77 (combination x S) cells, missing mostly 6 and 8 s by ECE 0.0103 to 0.011. Easy reliable at 2 to 8 s; where it fails (often 10 s) a baseline is as good, not calibration.
- Rolling isotonic: Good or Easy reliable in 77 of 77 cells on the whole collection and 76 of 77 on the deck subset; Easy never (its steps put near-zero probabilities on rare Easy answers, LogLoss 2% to 9% worse than uncalibrated).
- Rolling shared transform (consistent across S by construction): reliable at 4 s for 9 of 11 combinations on the whole collection, 7 of 11 on the deck subset; the ECE drop is smaller (0.0185 to 0.0114 for the longer Small Instant-Only model (SM113)).
- Static fits (once, on the 34,727 reviews before the benchmark period, 2024-01 to 2024-11) mostly fail: the early part of the collection is a different regime. Static Platt is reliable at 4 s only for the Tiny two-head model (SM106) and two other combinations.
- Sensitivity: when the strongest baseline is also recalibrated by rolling Platt, the recalibrated head still leads by 8.29% (the longer Small Instant-Only model (SM113)), so the lead is not just calibration.
- Platt and isotonic need a repair to stay non-decreasing in S and nested across sets (about 4% and 14% of row x event values per rolling refit); the shared transform needs none.

## Where it becomes reliable (S values in seconds)

### the longer Small Instant-Only model (SM113) with its own SM119 head

| method | Good or Easy, benchmark rows, whole collection | Good or Easy, benchmark rows, deck subset | Easy, benchmark rows, whole collection | Easy, benchmark rows, deck subset | recalled, benchmark rows, whole collection | recalled, benchmark rows, deck subset |
|---|---|---|---|---|---|---|
| uncalibrated (SM121) | none | none | 2, 3, 4, 5, 6, 8 | 3, 4, 5, 6 | none | none |
| (a) Platt, static | 3 | none | 2, 3, 4, 5, 6 | 2, 3, 4, 5, 6 | 3 | none |
| (b) shared transform, static | 2, 3, 10 | 2, 3 | 2, 3, 4, 5, 6, 8 | 2, 3, 4, 5, 6 | 2, 3 | 2, 3 |
| (c) isotonic, static | none | none | none | none | none | none |
| (a) Platt, rolling 7 days | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 10 | 2, 3, 4, 5, 6, 8 | 3, 4, 5, 6, 8 | 2, 3, 4, 5, 6, 8, 10 | 3, 4, 5, 10 |
| (b) shared transform, rolling 7 days | 2, 3, 10 | 2 | 2, 3, 4, 5, 6, 8 | 3, 4, 5, 6, 8 | 2, 3 | 2 |
| (c) isotonic, rolling 7 days | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 6, 10 | none | none | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 6, 10 |

Mac slice (secondary), Good or Easy: uncalibrated (SM121): whole collection none, deck subset 2; (a) Platt, static: whole collection none, deck subset 2; (b) shared transform, static: whole collection 2, deck subset 2; (c) isotonic, static: whole collection 2, deck subset 2; (a) Platt, rolling 7 days: whole collection 2, 3, 4, 10, deck subset 2, 3, 4, 5, 6, 10; (b) shared transform, rolling 7 days: whole collection 2, 4, deck subset 4; (c) isotonic, rolling 7 days: whole collection 2, 3, 4, 5, 10, deck subset 2, 4, 5.

### Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head

| method | Good or Easy, benchmark rows, whole collection | Good or Easy, benchmark rows, deck subset | Easy, benchmark rows, whole collection | Easy, benchmark rows, deck subset | recalled, benchmark rows, whole collection | recalled, benchmark rows, deck subset |
|---|---|---|---|---|---|---|
| uncalibrated (SM121) | none | none | 2, 3, 4, 5, 6 | 3, 4, 5, 6 | none | none |
| (a) Platt, static | 3 | 3 | 2, 3, 4, 5, 6 | 2, 3, 4, 5, 6 | 3 | 3 |
| (b) shared transform, static | 2, 3 | 2, 3 | 2, 3, 4, 5, 6 | 2, 3, 4, 5, 6 | 2, 3 | 2, 3 |
| (c) isotonic, static | 3 | 3 | none | none | 3 | 3 |
| (a) Platt, rolling 7 days | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 10 | 2, 3, 4, 5, 6 | 3, 4, 5, 6 | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 6, 10 |
| (b) shared transform, rolling 7 days | 2, 3, 4, 8, 10 | 2, 3, 4 | 2, 3, 4, 5, 6 | 3, 4, 5, 6 | 2, 3, 4 | 2, 3, 4 |
| (c) isotonic, rolling 7 days | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 6, 8, 10 | none | none | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 6, 8, 10 |

Mac slice (secondary), Good or Easy: uncalibrated (SM121): whole collection 2, deck subset 2; (a) Platt, static: whole collection 2, deck subset 2; (b) shared transform, static: whole collection 2, 3, deck subset 2; (c) isotonic, static: whole collection 2, deck subset 2; (a) Platt, rolling 7 days: whole collection 2, 3, 4, 10, deck subset 2, 3, 4, 8; (b) shared transform, rolling 7 days: whole collection 2, 4, deck subset 4; (c) isotonic, rolling 7 days: whole collection 2, 3, 4, 5, 6, 8, 10, deck subset 2, 4, 10.

### the Tiny two-head model (SM106) with its own SM117 head

| method | Good or Easy, benchmark rows, whole collection | Good or Easy, benchmark rows, deck subset | Easy, benchmark rows, whole collection | Easy, benchmark rows, deck subset | recalled, benchmark rows, whole collection | recalled, benchmark rows, deck subset |
|---|---|---|---|---|---|---|
| uncalibrated (SM121) | none | none | 2, 3, 4, 5, 6, 8, 10 | 3, 4, 5, 6, 8, 10 | none | none |
| (a) Platt, static | 4 | 4 | 2, 3, 4, 5, 6, 8 | 2, 3, 4, 5, 6 | 4 | 4 |
| (b) shared transform, static | 2, 4, 5, 10 | 2, 4 | 2, 3, 4, 5, 6, 8 | 2, 3, 4, 5, 6, 8 | 2, 4, 5, 6, 8, 10 | 2, 4, 5 |
| (c) isotonic, static | none | none | none | none | none | none |
| (a) Platt, rolling 7 days | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 6, 8 | 3, 4, 5, 6, 8 | 2, 3, 4, 5, 6, 8, 10 | 3, 4, 5, 6, 10 |
| (b) shared transform, rolling 7 days | 2, 3, 4, 6, 8, 10 | 2, 3, 4 | 2, 3, 4, 5, 6, 8 | 3, 4, 5, 6 | 2, 3, 4, 6, 8 | 2, 3, 4 |
| (c) isotonic, rolling 7 days | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 6, 8, 10 | none | none | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 6, 10 |

Mac slice (secondary), Good or Easy: uncalibrated (SM121): whole collection none, deck subset 2; (a) Platt, static: whole collection none, deck subset 2, 3, 4, 5; (b) shared transform, static: whole collection none, deck subset 2, 4; (c) isotonic, static: whole collection 2, deck subset 2; (a) Platt, rolling 7 days: whole collection 2, 3, 4, 5, 10, deck subset 5, 6; (b) shared transform, rolling 7 days: whole collection 2, 4, deck subset none; (c) isotonic, rolling 7 days: whole collection 2, 3, 4, 5, 8, 10, deck subset none.

### the show-time big model (SM91s2) with its own SM117 head

| method | Good or Easy, benchmark rows, whole collection | Good or Easy, benchmark rows, deck subset | Easy, benchmark rows, whole collection | Easy, benchmark rows, deck subset | recalled, benchmark rows, whole collection | recalled, benchmark rows, deck subset |
|---|---|---|---|---|---|---|
| uncalibrated (SM121) | 2 | 2 | 2, 3, 4, 5, 6, 8 | 3, 4, 5, 6 | 2 | none |
| (a) Platt, static | 3 | 3 | 2, 3, 4, 5, 6, 8 | 2, 3, 4, 5, 6, 8, 10 | 3, 4 | 3 |
| (b) shared transform, static | 2, 3, 5, 8, 10 | 3, 10 | 2, 3, 4, 5, 6, 8 | 3, 4, 5, 6 | 2, 3, 5, 8, 10 | 3 |
| (c) isotonic, static | none | none | 3 | 3 | none | none |
| (a) Platt, rolling 7 days | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 6 | 2, 3, 4, 5, 6, 8 | 2, 3, 4, 5, 6, 8, 10 | 3, 4, 5, 6, 8, 10 |
| (b) shared transform, rolling 7 days | 2, 3, 4, 5, 6, 8, 10 | 8, 10 | 2, 3, 4, 5, 6 | 3, 4, 5, 6, 8 | 2, 3, 4, 6, 8, 10 | 6, 8, 10 |
| (c) isotonic, rolling 7 days | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 6, 8, 10 | none | none | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 6, 8, 10 |

Mac slice (secondary), Good or Easy: uncalibrated (SM121): whole collection none, deck subset 2; (a) Platt, static: whole collection 2, 4, 5, deck subset 2; (b) shared transform, static: whole collection 2, deck subset 2; (c) isotonic, static: whole collection none, deck subset 2; (a) Platt, rolling 7 days: whole collection 2, 3, 4, 5, 6, 8, 10, deck subset 2, 3, 4; (b) shared transform, rolling 7 days: whole collection 2, 4, deck subset 2, 4; (c) isotonic, rolling 7 days: whole collection 2, 3, 4, 5, 8, deck subset 2, 3, 4, 5.

### the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92)

| method | Good or Easy, benchmark rows, whole collection | Good or Easy, benchmark rows, deck subset | Easy, benchmark rows, whole collection | Easy, benchmark rows, deck subset | recalled, benchmark rows, whole collection | recalled, benchmark rows, deck subset |
|---|---|---|---|---|---|---|
| uncalibrated (SM121) | 2 | 2 | 2, 3, 4, 5, 6 | 3, 4, 5, 6 | 2 | 2 |
| (a) Platt, static | 2, 3, 4 | 3, 4 | 2, 3, 4, 5, 6, 8 | 2, 3, 4, 5, 6, 8, 10 | 3, 4, 5, 6, 8, 10 | 3, 4 |
| (b) shared transform, static | 2, 10 | 2, 4, 10 | 2, 3, 4, 5, 6, 8 | 2, 3, 4, 5, 6, 8 | 2, 5, 8, 10 | 2, 4 |
| (c) isotonic, static | 2, 3, 4 | 2, 3, 4 | none | none | 2, 3, 4 | 2, 4 |
| (a) Platt, rolling 7 days | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 6 | 2, 3, 4, 5, 6 | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 6, 8, 10 |
| (b) shared transform, rolling 7 days | 2, 4, 8, 10 | 2, 4, 10 | 2, 3, 4, 5, 6 | 3, 4, 5, 6 | 2, 4, 8, 10 | 2, 4 |
| (c) isotonic, rolling 7 days | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 6, 8, 10 | 3 | none | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 6, 8, 10 |

Mac slice (secondary), Good or Easy: uncalibrated (SM121): whole collection none, deck subset 2; (a) Platt, static: whole collection 2, 8, deck subset 2, 8; (b) shared transform, static: whole collection 2, deck subset 2; (c) isotonic, static: whole collection 2, deck subset 2; (a) Platt, rolling 7 days: whole collection 2, 3, 4, 5, 6, 8, deck subset 2, 4, 5, 6, 8; (b) shared transform, rolling 7 days: whole collection 2, 4, deck subset 4; (c) isotonic, rolling 7 days: whole collection 2, 3, 4, 5, 6, 8, 10, deck subset 2, 4, 10.

### Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head

| method | Good or Easy, benchmark rows, whole collection | Good or Easy, benchmark rows, deck subset | Easy, benchmark rows, whole collection | Easy, benchmark rows, deck subset | recalled, benchmark rows, whole collection | recalled, benchmark rows, deck subset |
|---|---|---|---|---|---|---|
| uncalibrated (SM121) | none | none | 2, 3, 4, 5, 6 | 3, 4, 5, 6 | none | none |
| (a) Platt, static | 3 | 3 | 2, 3, 4, 5, 6 | 2, 3, 4, 5, 6 | 3 | 3 |
| (b) shared transform, static | 2, 3 | 2, 3 | 2, 3, 4, 5, 6 | 2, 3, 4, 5, 6 | 2, 3 | 2, 3 |
| (c) isotonic, static | 3 | 3 | none | none | 3 | 3 |
| (a) Platt, rolling 7 days | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 10 | 2, 3, 4, 5, 6 | 3, 4, 5, 6 | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5 |
| (b) shared transform, rolling 7 days | 2, 3, 4, 10 | 2, 3, 4 | 2, 3, 4, 5, 6 | 3, 4, 5, 6 | 2, 3, 4 | 2, 3, 4 |
| (c) isotonic, rolling 7 days | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 6, 8, 10 | none | none | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 6, 10 |

Mac slice (secondary), Good or Easy: uncalibrated (SM121): whole collection 2, deck subset 2; (a) Platt, static: whole collection 2, deck subset 2; (b) shared transform, static: whole collection 2, 3, deck subset 2; (c) isotonic, static: whole collection 2, deck subset 2; (a) Platt, rolling 7 days: whole collection 2, 3, 4, deck subset 2, 3, 4; (b) shared transform, rolling 7 days: whole collection 2, 3, 4, deck subset 4; (c) isotonic, rolling 7 days: whole collection 2, 3, 4, 10, deck subset 2, 4.

### the show-time retrain (SM92) with its own SM117 head

| method | Good or Easy, benchmark rows, whole collection | Good or Easy, benchmark rows, deck subset | Easy, benchmark rows, whole collection | Easy, benchmark rows, deck subset | recalled, benchmark rows, whole collection | recalled, benchmark rows, deck subset |
|---|---|---|---|---|---|---|
| uncalibrated (SM121) | none | none | 2, 3, 4, 5, 6, 8, 10 | 3, 4, 5, 6, 8 | none | none |
| (a) Platt, static | 3 | 3 | 2, 3, 4, 5, 6 | 2, 3, 4, 5, 6 | 3 | 3 |
| (b) shared transform, static | 2, 3 | 2, 3 | 2, 3, 4, 5, 6, 8 | 2, 3, 4, 5, 6, 8 | 2, 3 | 2, 3 |
| (c) isotonic, static | 2, 3 | 3 | none | none | 3 | 3 |
| (a) Platt, rolling 7 days | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 10 | 2, 3, 4, 5, 6 | 3, 4, 5, 6 | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5 |
| (b) shared transform, rolling 7 days | 2, 3, 4, 10 | 2, 3, 4 | 2, 3, 4, 5, 6 | 3, 4, 5, 6 | 2, 3, 4 | 2, 3, 4 |
| (c) isotonic, rolling 7 days | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 6, 8, 10 | none | none | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 6, 10 |

Mac slice (secondary), Good or Easy: uncalibrated (SM121): whole collection 2, deck subset 2; (a) Platt, static: whole collection 2, 8, deck subset 2; (b) shared transform, static: whole collection 2, 3, deck subset 2; (c) isotonic, static: whole collection 2, deck subset 2; (a) Platt, rolling 7 days: whole collection 2, 3, 4, 5, 6, 8, 10, deck subset 2, 3, 4; (b) shared transform, rolling 7 days: whole collection 2, 4, deck subset 4; (c) isotonic, rolling 7 days: whole collection 2, 3, 4, 5, 6, 8, 10, deck subset 2, 10.

### the Small Instant-Only model (SM98) with its own SM117 head

| method | Good or Easy, benchmark rows, whole collection | Good or Easy, benchmark rows, deck subset | Easy, benchmark rows, whole collection | Easy, benchmark rows, deck subset | recalled, benchmark rows, whole collection | recalled, benchmark rows, deck subset |
|---|---|---|---|---|---|---|
| uncalibrated (SM121) | none | none | 2, 3, 4, 5, 6, 8 | 2, 3, 4, 5, 6, 8 | none | none |
| (a) Platt, static | 3 | none | 2, 3, 4, 5, 6, 8 | 2, 3, 4, 5, 6, 8, 10 | none | none |
| (b) shared transform, static | 2, 3, 10 | 2 | 2, 3, 4, 5, 6, 8 | 2, 3, 4, 5, 6, 8 | 2, 3, 10 | 2 |
| (c) isotonic, static | none | none | none | none | none | none |
| (a) Platt, rolling 7 days | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 10 | 2, 3, 4, 5, 6, 8 | 3, 4, 5, 6, 8 | 2, 3, 4, 5, 6, 8, 10 | 3, 4, 5, 10 |
| (b) shared transform, rolling 7 days | 2, 3, 8, 10 | 2 | 2, 3, 4, 5, 6, 8 | 3, 4, 5, 6, 8 | 2, 3 | 2 |
| (c) isotonic, rolling 7 days | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 6, 8, 10 | none | none | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 6, 10 |

Mac slice (secondary), Good or Easy: uncalibrated (SM121): whole collection none, deck subset 2; (a) Platt, static: whole collection 2, deck subset 2, 3, 4; (b) shared transform, static: whole collection 2, 3, deck subset none; (c) isotonic, static: whole collection 2, deck subset 2; (a) Platt, rolling 7 days: whole collection 2, 3, 4, 5, 8, 10, deck subset 4, 5, 10; (b) shared transform, rolling 7 days: whole collection 4, deck subset none; (c) isotonic, rolling 7 days: whole collection 2, 3, 4, 5, 6, 10, deck subset 2, 4, 5.

### Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head

| method | Good or Easy, benchmark rows, whole collection | Good or Easy, benchmark rows, deck subset | Easy, benchmark rows, whole collection | Easy, benchmark rows, deck subset | recalled, benchmark rows, whole collection | recalled, benchmark rows, deck subset |
|---|---|---|---|---|---|---|
| uncalibrated (SM121) | none | none | 2, 3, 4, 5, 6, 8 | 3, 4, 5, 6, 8 | none | none |
| (a) Platt, static | 4 | 4 | 2, 3, 4, 5, 6 | 2, 3, 4, 5, 6 | 4 | none |
| (b) shared transform, static | 2, 4, 5, 10 | 2, 4 | 2, 3, 4, 5, 6 | 2, 3, 4, 5, 6 | 2, 4, 5, 6, 8, 10 | 2, 4 |
| (c) isotonic, static | none | none | none | none | none | none |
| (a) Platt, rolling 7 days | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 6, 8 | 3, 4, 5, 6 | 2, 3, 4, 5, 6, 8, 10 | 3, 4, 5, 6, 10 |
| (b) shared transform, rolling 7 days | 2, 3, 4, 8, 10 | 2, 3, 4 | 2, 3, 4, 5, 6 | 3, 4, 5, 6 | 2, 3, 4, 8 | 2, 4 |
| (c) isotonic, rolling 7 days | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 6, 8, 10 | none | none | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 6, 10 |

Mac slice (secondary), Good or Easy: uncalibrated (SM121): whole collection none, deck subset 2; (a) Platt, static: whole collection 8, deck subset 2, 3, 4, 5; (b) shared transform, static: whole collection none, deck subset 2, 4; (c) isotonic, static: whole collection 2, deck subset 2; (a) Platt, rolling 7 days: whole collection 2, 3, 4, 5, 6, 8, 10, deck subset 5, 8, 10; (b) shared transform, rolling 7 days: whole collection 2, 4, deck subset none; (c) isotonic, rolling 7 days: whole collection 2, 3, 4, 5, 10, deck subset 5.

### the longer cool-down for Tiny (SM115) with its own SM119 head

| method | Good or Easy, benchmark rows, whole collection | Good or Easy, benchmark rows, deck subset | Easy, benchmark rows, whole collection | Easy, benchmark rows, deck subset | recalled, benchmark rows, whole collection | recalled, benchmark rows, deck subset |
|---|---|---|---|---|---|---|
| uncalibrated (SM121) | none | none | 2, 3, 4, 5, 6, 8 | 2, 3, 4, 5, 6, 8, 10 | none | none |
| (a) Platt, static | 3 | 3 | 2, 3, 4, 5, 6 | 2, 3, 4, 5, 6 | 3 | 3 |
| (b) shared transform, static | 2, 10 | 2, 3 | 2, 3, 4, 5, 6 | 2, 3, 4, 5 | 2, 10 | 2, 3 |
| (c) isotonic, static | 2, 3 | 2, 3 | none | none | none | 3 |
| (a) Platt, rolling 7 days | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5 | 2, 3, 4, 5, 6, 8 | 2, 3, 4, 5, 6, 8 | 2, 3, 4, 5, 6, 8, 10 | 3, 4, 5, 6 |
| (b) shared transform, rolling 7 days | 2, 3, 4, 8, 10 | 2, 3, 10 | 2, 3, 4, 5, 6, 8 | 2, 3, 4, 5, 6, 8 | 2, 3, 4, 6, 8, 10 | 2, 3 |
| (c) isotonic, rolling 7 days | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 6, 8, 10 | none | none | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 6, 8, 10 |

Mac slice (secondary), Good or Easy: uncalibrated (SM121): whole collection none, deck subset 2; (a) Platt, static: whole collection 2, 4, 5, deck subset 2, 3, 4, 5; (b) shared transform, static: whole collection 2, 4, 5, deck subset 2, 4, 5; (c) isotonic, static: whole collection 2, 4, deck subset 2, 3; (a) Platt, rolling 7 days: whole collection 2, 3, 4, 5, 10, deck subset 2, 3, 4, 5; (b) shared transform, rolling 7 days: whole collection 2, 4, 5, deck subset 2, 4, 5; (c) isotonic, rolling 7 days: whole collection 2, 3, 4, 5, 6, 8, 10, deck subset 2, 5, 8.

### the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it)

| method | Good or Easy, benchmark rows, whole collection | Good or Easy, benchmark rows, deck subset | Easy, benchmark rows, whole collection | Easy, benchmark rows, deck subset | recalled, benchmark rows, whole collection | recalled, benchmark rows, deck subset |
|---|---|---|---|---|---|---|
| uncalibrated (SM121) | 2 | 2 | 2, 3, 4, 5, 6 | 3, 4, 5, 6 | 2 | 2 |
| (a) Platt, static | 2 | none | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 6, 8, 10 | 4, 5 | none |
| (b) shared transform, static | 2, 3, 6, 8, 10 | 2, 3, 10 | 2, 3, 4, 5, 6, 8, 10 | 3, 4, 5, 6, 8, 10 | 2, 3 | 3 |
| (c) isotonic, static | 2 | 2 | 3 | none | none | none |
| (a) Platt, rolling 7 days | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 6, 8 | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 6, 8, 10 |
| (b) shared transform, rolling 7 days | 2, 3, 4, 8, 10 | 2, 4, 10 | 2, 3, 4, 5, 6, 8, 10 | 3, 4, 5, 6, 8, 10 | 2, 3, 4, 8, 10 | 4 |
| (c) isotonic, rolling 7 days | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 6, 8, 10 | none | none | 2, 3, 4, 5, 6, 8, 10 | 2, 3, 4, 5, 6, 8, 10 |

Mac slice (secondary), Good or Easy: uncalibrated (SM121): whole collection none, deck subset 2; (a) Platt, static: whole collection 2, 4, deck subset 2, 3, 4; (b) shared transform, static: whole collection 2, 4, deck subset 2, 4; (c) isotonic, static: whole collection none, deck subset 2, 3; (a) Platt, rolling 7 days: whole collection 2, 3, 4, 5, 6, 10, deck subset 2, 3, 4, 5; (b) shared transform, rolling 7 days: whole collection 2, 4, deck subset 2, 4; (c) isotonic, rolling 7 days: whole collection 2, 3, 4, 5, 8, 10, deck subset 2, 3, 4, 5.

## Good or Easy within 4 s (reading the back included)

### benchmark rows, whole collection

| combination | method | LogLoss | ECE | vs uncalibrated | strongest baseline (LogLoss) | vs strongest [day CI of the difference] | vs strongest, both Platt | AUC | reliable |
|---|---|---|---|---|---|---|---|---|---|
| the longer Small Instant-Only model (SM113) with its own SM119 head | uncalibrated (SM121) | 0.4964 | 0.0185 | - | Mac R x card (0.5391) | -7.92% [-0.0450, -0.0404] | - | 0.800 | no |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (a) Platt, static | 0.4962 | 0.0128 | -0.05% | Mac R x card (0.5391) | -7.97% [-0.0454, -0.0406] | -10.97% [-0.0643, -0.0582] | 0.800 | no |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (b) shared transform, static | 0.4961 | 0.0147 | -0.06% | Mac R x card (0.5391) | -7.98% [-0.0454, -0.0406] | - | 0.801 | no |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (c) isotonic, static | 0.4984 | 0.0156 | +0.40% | Mac R x card (0.5391) | -7.55% [-0.0433, -0.0382] | - | 0.800 | no |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (a) Platt, rolling 7 days | 0.4950 | 0.0053 | -0.28% | Mac R x card (0.5391) | -8.18% [-0.0464, -0.0418] | -8.29% [-0.0472, -0.0423] | 0.801 | yes |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (b) shared transform, rolling 7 days | 0.4956 | 0.0114 | -0.16% | Mac R x card (0.5391) | -8.07% [-0.0459, -0.0411] | - | 0.801 | no |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (c) isotonic, rolling 7 days | 0.4961 | 0.0046 | -0.06% | Mac R x card (0.5391) | -7.98% [-0.0453, -0.0407] | - | 0.800 | yes |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | uncalibrated (SM121) | 0.4976 | 0.0240 | - | Mac R x card (0.5393) | -7.72% [-0.0439, -0.0394] | - | 0.800 | no |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (a) Platt, static | 0.4963 | 0.0126 | -0.25% | Mac R x card (0.5393) | -7.96% [-0.0452, -0.0406] | -10.95% [-0.0641, -0.0581] | 0.800 | no |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (b) shared transform, static | 0.4968 | 0.0197 | -0.16% | Mac R x card (0.5393) | -7.87% [-0.0447, -0.0401] | - | 0.800 | no |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (c) isotonic, static | 0.4973 | 0.0125 | -0.07% | Mac R x card (0.5393) | -7.79% [-0.0443, -0.0397] | - | 0.800 | no |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (a) Platt, rolling 7 days | 0.4956 | 0.0046 | -0.41% | Mac R x card (0.5393) | -8.10% [-0.0459, -0.0414] | -8.21% [-0.0467, -0.0419] | 0.800 | yes |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (b) shared transform, rolling 7 days | 0.4959 | 0.0080 | -0.34% | Mac R x card (0.5393) | -8.04% [-0.0457, -0.0410] | - | 0.800 | yes |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (c) isotonic, rolling 7 days | 0.4970 | 0.0047 | -0.12% | Mac R x card (0.5393) | -7.84% [-0.0445, -0.0400] | - | 0.799 | yes |
| the Tiny two-head model (SM106) with its own SM117 head | uncalibrated (SM121) | 0.4974 | 0.0166 | - | Mac R x card (0.5397) | -7.84% [-0.0445, -0.0400] | - | 0.799 | no |
| the Tiny two-head model (SM106) with its own SM117 head | (a) Platt, static | 0.4967 | 0.0063 | -0.14% | Mac R x card (0.5397) | -7.97% [-0.0454, -0.0407] | -10.89% [-0.0638, -0.0578] | 0.799 | yes |
| the Tiny two-head model (SM106) with its own SM117 head | (b) shared transform, static | 0.4966 | 0.0088 | -0.15% | Mac R x card (0.5397) | -7.98% [-0.0454, -0.0407] | - | 0.799 | yes |
| the Tiny two-head model (SM106) with its own SM117 head | (c) isotonic, static | 0.4983 | 0.0132 | +0.18% | Mac R x card (0.5397) | -7.67% [-0.0439, -0.0390] | - | 0.799 | no |
| the Tiny two-head model (SM106) with its own SM117 head | (a) Platt, rolling 7 days | 0.4963 | 0.0049 | -0.21% | Mac R x card (0.5397) | -8.03% [-0.0457, -0.0410] | -8.14% [-0.0465, -0.0415] | 0.799 | yes |
| the Tiny two-head model (SM106) with its own SM117 head | (b) shared transform, rolling 7 days | 0.4967 | 0.0070 | -0.14% | Mac R x card (0.5397) | -7.96% [-0.0454, -0.0407] | - | 0.799 | yes |
| the Tiny two-head model (SM106) with its own SM117 head | (c) isotonic, rolling 7 days | 0.4975 | 0.0042 | +0.02% | Mac R x card (0.5397) | -7.82% [-0.0445, -0.0399] | - | 0.799 | yes |
| the show-time big model (SM91s2) with its own SM117 head | uncalibrated (SM121) | 0.5028 | 0.0385 | - | Mac R x card (0.5393) | -6.78% [-0.0389, -0.0343] | - | 0.798 | no |
| the show-time big model (SM91s2) with its own SM117 head | (a) Platt, static | 0.4983 | 0.0104 | -0.89% | Mac R x card (0.5393) | -7.61% [-0.0434, -0.0388] | -10.69% [-0.0627, -0.0567] | 0.798 | no |
| the show-time big model (SM91s2) with its own SM117 head | (b) shared transform, static | 0.4980 | 0.0129 | -0.94% | Mac R x card (0.5393) | -7.66% [-0.0436, -0.0390] | - | 0.798 | no |
| the show-time big model (SM91s2) with its own SM117 head | (c) isotonic, static | 0.5005 | 0.0143 | -0.46% | Mac R x card (0.5393) | -7.21% [-0.0413, -0.0365] | - | 0.797 | no |
| the show-time big model (SM91s2) with its own SM117 head | (a) Platt, rolling 7 days | 0.4983 | 0.0059 | -0.89% | Mac R x card (0.5393) | -7.60% [-0.0433, -0.0387] | -7.70% [-0.0441, -0.0392] | 0.797 | yes |
| the show-time big model (SM91s2) with its own SM117 head | (b) shared transform, rolling 7 days | 0.4979 | 0.0099 | -0.96% | Mac R x card (0.5393) | -7.67% [-0.0438, -0.0391] | - | 0.798 | yes |
| the show-time big model (SM91s2) with its own SM117 head | (c) isotonic, rolling 7 days | 0.4992 | 0.0043 | -0.71% | Mac R x card (0.5393) | -7.44% [-0.0424, -0.0379] | - | 0.797 | yes |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | uncalibrated (SM121) | 0.5068 | 0.0360 | - | Mac R x card (0.5393) | -6.01% [-0.0346, -0.0304] | - | 0.792 | no |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (a) Platt, static | 0.5029 | 0.0069 | -0.77% | Mac R x card (0.5393) | -6.74% [-0.0385, -0.0343] | -9.77% [-0.0574, -0.0517] | 0.792 | yes |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (b) shared transform, static | 0.5032 | 0.0108 | -0.71% | Mac R x card (0.5393) | -6.69% [-0.0382, -0.0340] | - | 0.792 | no |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (c) isotonic, static | 0.5037 | 0.0076 | -0.61% | Mac R x card (0.5393) | -6.58% [-0.0377, -0.0334] | - | 0.792 | yes |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (a) Platt, rolling 7 days | 0.5028 | 0.0054 | -0.79% | Mac R x card (0.5393) | -6.75% [-0.0386, -0.0344] | -6.87% [-0.0393, -0.0349] | 0.792 | yes |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (b) shared transform, rolling 7 days | 0.5032 | 0.0080 | -0.72% | Mac R x card (0.5393) | -6.69% [-0.0383, -0.0340] | - | 0.792 | yes |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (c) isotonic, rolling 7 days | 0.5040 | 0.0045 | -0.55% | Mac R x card (0.5393) | -6.53% [-0.0374, -0.0331] | - | 0.791 | yes |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | uncalibrated (SM121) | 0.4976 | 0.0241 | - | Mac R x card (0.5391) | -7.70% [-0.0437, -0.0392] | - | 0.800 | no |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (a) Platt, static | 0.4964 | 0.0136 | -0.24% | Mac R x card (0.5391) | -7.92% [-0.0450, -0.0404] | -10.92% [-0.0640, -0.0579] | 0.800 | no |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (b) shared transform, static | 0.4970 | 0.0207 | -0.14% | Mac R x card (0.5391) | -7.82% [-0.0445, -0.0399] | - | 0.800 | no |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (c) isotonic, static | 0.4973 | 0.0132 | -0.07% | Mac R x card (0.5391) | -7.76% [-0.0441, -0.0395] | - | 0.800 | no |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (a) Platt, rolling 7 days | 0.4956 | 0.0046 | -0.42% | Mac R x card (0.5391) | -8.08% [-0.0458, -0.0413] | -8.19% [-0.0466, -0.0418] | 0.800 | yes |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (b) shared transform, rolling 7 days | 0.4959 | 0.0078 | -0.35% | Mac R x card (0.5391) | -8.02% [-0.0456, -0.0409] | - | 0.800 | yes |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (c) isotonic, rolling 7 days | 0.4970 | 0.0039 | -0.13% | Mac R x card (0.5391) | -7.82% [-0.0444, -0.0399] | - | 0.799 | yes |
| the show-time retrain (SM92) with its own SM117 head | uncalibrated (SM121) | 0.4980 | 0.0255 | - | Mac R x card (0.5397) | -7.72% [-0.0439, -0.0394] | - | 0.800 | no |
| the show-time retrain (SM92) with its own SM117 head | (a) Platt, static | 0.4967 | 0.0155 | -0.26% | Mac R x card (0.5397) | -7.96% [-0.0452, -0.0407] | -10.83% [-0.0635, -0.0574] | 0.800 | no |
| the show-time retrain (SM92) with its own SM117 head | (b) shared transform, static | 0.4973 | 0.0221 | -0.15% | Mac R x card (0.5397) | -7.86% [-0.0447, -0.0401] | - | 0.800 | no |
| the show-time retrain (SM92) with its own SM117 head | (c) isotonic, static | 0.4977 | 0.0143 | -0.06% | Mac R x card (0.5397) | -7.77% [-0.0443, -0.0396] | - | 0.799 | no |
| the show-time retrain (SM92) with its own SM117 head | (a) Platt, rolling 7 days | 0.4958 | 0.0052 | -0.45% | Mac R x card (0.5397) | -8.13% [-0.0462, -0.0416] | -8.23% [-0.0469, -0.0421] | 0.800 | yes |
| the show-time retrain (SM92) with its own SM117 head | (b) shared transform, rolling 7 days | 0.4961 | 0.0080 | -0.39% | Mac R x card (0.5397) | -8.08% [-0.0460, -0.0413] | - | 0.800 | yes |
| the show-time retrain (SM92) with its own SM117 head | (c) isotonic, rolling 7 days | 0.4970 | 0.0037 | -0.20% | Mac R x card (0.5397) | -7.91% [-0.0450, -0.0404] | - | 0.799 | yes |
| the Small Instant-Only model (SM98) with its own SM117 head | uncalibrated (SM121) | 0.4969 | 0.0177 | - | Mac R x card (0.5393) | -7.85% [-0.0446, -0.0401] | - | 0.800 | no |
| the Small Instant-Only model (SM98) with its own SM117 head | (a) Platt, static | 0.4965 | 0.0112 | -0.08% | Mac R x card (0.5393) | -7.92% [-0.0451, -0.0403] | -10.91% [-0.0640, -0.0579] | 0.800 | no |
| the Small Instant-Only model (SM98) with its own SM117 head | (b) shared transform, static | 0.4966 | 0.0124 | -0.06% | Mac R x card (0.5393) | -7.91% [-0.0450, -0.0403] | - | 0.800 | no |
| the Small Instant-Only model (SM98) with its own SM117 head | (c) isotonic, static | 0.4984 | 0.0142 | +0.31% | Mac R x card (0.5393) | -7.57% [-0.0434, -0.0383] | - | 0.799 | no |
| the Small Instant-Only model (SM98) with its own SM117 head | (a) Platt, rolling 7 days | 0.4955 | 0.0051 | -0.29% | Mac R x card (0.5393) | -8.12% [-0.0461, -0.0415] | -8.23% [-0.0469, -0.0420] | 0.800 | yes |
| the Small Instant-Only model (SM98) with its own SM117 head | (b) shared transform, rolling 7 days | 0.4965 | 0.0136 | -0.07% | Mac R x card (0.5393) | -7.92% [-0.0451, -0.0404] | - | 0.800 | no |
| the Small Instant-Only model (SM98) with its own SM117 head | (c) isotonic, rolling 7 days | 0.4968 | 0.0042 | -0.02% | Mac R x card (0.5393) | -7.87% [-0.0448, -0.0401] | - | 0.799 | yes |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | uncalibrated (SM121) | 0.4971 | 0.0167 | - | Mac R x card (0.5392) | -7.80% [-0.0443, -0.0398] | - | 0.799 | no |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (a) Platt, static | 0.4965 | 0.0081 | -0.12% | Mac R x card (0.5392) | -7.91% [-0.0450, -0.0403] | -11.00% [-0.0645, -0.0585] | 0.799 | yes |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (b) shared transform, static | 0.4964 | 0.0084 | -0.15% | Mac R x card (0.5392) | -7.93% [-0.0451, -0.0404] | - | 0.800 | yes |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (c) isotonic, static | 0.4982 | 0.0120 | +0.22% | Mac R x card (0.5392) | -7.59% [-0.0434, -0.0385] | - | 0.799 | no |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (a) Platt, rolling 7 days | 0.4961 | 0.0054 | -0.22% | Mac R x card (0.5392) | -8.00% [-0.0455, -0.0409] | -8.11% [-0.0463, -0.0413] | 0.800 | yes |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (b) shared transform, rolling 7 days | 0.4964 | 0.0066 | -0.15% | Mac R x card (0.5392) | -7.94% [-0.0452, -0.0405] | - | 0.799 | yes |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (c) isotonic, rolling 7 days | 0.4972 | 0.0039 | +0.01% | Mac R x card (0.5392) | -7.79% [-0.0443, -0.0397] | - | 0.799 | yes |
| the longer cool-down for Tiny (SM115) with its own SM119 head | uncalibrated (SM121) | 0.4968 | 0.0233 | - | Mac R x card (0.5392) | -7.87% [-0.0447, -0.0401] | - | 0.801 | no |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (a) Platt, static | 0.4961 | 0.0161 | -0.13% | Mac R x card (0.5392) | -7.99% [-0.0455, -0.0407] | -11.08% [-0.0649, -0.0589] | 0.801 | no |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (b) shared transform, static | 0.4968 | 0.0220 | +0.01% | Mac R x card (0.5392) | -7.86% [-0.0448, -0.0399] | - | 0.801 | no |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (c) isotonic, static | 0.4980 | 0.0161 | +0.26% | Mac R x card (0.5392) | -7.63% [-0.0436, -0.0388] | - | 0.800 | no |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (a) Platt, rolling 7 days | 0.4948 | 0.0066 | -0.40% | Mac R x card (0.5392) | -8.24% [-0.0468, -0.0421] | -8.35% [-0.0476, -0.0427] | 0.801 | yes |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (b) shared transform, rolling 7 days | 0.4953 | 0.0099 | -0.29% | Mac R x card (0.5392) | -8.14% [-0.0463, -0.0415] | - | 0.801 | yes |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (c) isotonic, rolling 7 days | 0.4962 | 0.0042 | -0.11% | Mac R x card (0.5392) | -7.97% [-0.0454, -0.0406] | - | 0.800 | yes |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | uncalibrated (SM121) | 0.5013 | 0.0381 | - | Mac R x card (0.5391) | -7.01% [-0.0402, -0.0354] | - | 0.799 | no |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (a) Platt, static | 0.4970 | 0.0107 | -0.85% | Mac R x card (0.5391) | -7.80% [-0.0444, -0.0397] | -10.86% [-0.0636, -0.0577] | 0.799 | no |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (b) shared transform, static | 0.4967 | 0.0101 | -0.92% | Mac R x card (0.5391) | -7.86% [-0.0448, -0.0401] | - | 0.799 | no |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (c) isotonic, static | 0.4988 | 0.0165 | -0.49% | Mac R x card (0.5391) | -7.47% [-0.0427, -0.0379] | - | 0.799 | no |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (a) Platt, rolling 7 days | 0.4968 | 0.0049 | -0.90% | Mac R x card (0.5391) | -7.84% [-0.0446, -0.0400] | -7.95% [-0.0455, -0.0405] | 0.799 | yes |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (b) shared transform, rolling 7 days | 0.4966 | 0.0062 | -0.95% | Mac R x card (0.5391) | -7.89% [-0.0449, -0.0402] | - | 0.799 | yes |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (c) isotonic, rolling 7 days | 0.4983 | 0.0036 | -0.60% | Mac R x card (0.5391) | -7.57% [-0.0432, -0.0385] | - | 0.798 | yes |

### benchmark rows, deck subset

| combination | method | LogLoss | ECE | vs uncalibrated | strongest baseline (LogLoss) | vs strongest [day CI of the difference] | vs strongest, both Platt | AUC | reliable |
|---|---|---|---|---|---|---|---|---|---|
| the longer Small Instant-Only model (SM113) with its own SM119 head | uncalibrated (SM121) | 0.5060 | 0.0175 | - | Mac grade x card (0.5412) | -6.50% [-0.0376, -0.0329] | - | 0.782 | no |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (a) Platt, static | 0.5060 | 0.0129 | -0.00% | Mac grade x card (0.5412) | -6.50% [-0.0377, -0.0328] | -11.98% [-0.0734, -0.0647] | 0.782 | no |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (b) shared transform, static | 0.5063 | 0.0151 | +0.05% | Mac grade x card (0.5412) | -6.45% [-0.0374, -0.0325] | - | 0.782 | no |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (c) isotonic, static | 0.5073 | 0.0158 | +0.26% | Mac grade x card (0.5412) | -6.26% [-0.0364, -0.0314] | - | 0.781 | no |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (a) Platt, rolling 7 days | 0.5051 | 0.0062 | -0.18% | Mac grade x card (0.5412) | -6.67% [-0.0385, -0.0338] | -6.87% [-0.0399, -0.0348] | 0.782 | yes |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (b) shared transform, rolling 7 days | 0.5059 | 0.0132 | -0.01% | Mac grade x card (0.5412) | -6.51% [-0.0378, -0.0329] | - | 0.782 | no |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (c) isotonic, rolling 7 days | 0.5057 | 0.0050 | -0.05% | Mac grade x card (0.5412) | -6.55% [-0.0379, -0.0331] | - | 0.781 | yes |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | uncalibrated (SM121) | 0.5076 | 0.0238 | - | Mac grade x card (0.5413) | -6.23% [-0.0360, -0.0315] | - | 0.781 | no |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (a) Platt, static | 0.5064 | 0.0128 | -0.24% | Mac grade x card (0.5413) | -6.45% [-0.0373, -0.0326] | -12.04% [-0.0739, -0.0650] | 0.781 | no |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (b) shared transform, static | 0.5071 | 0.0203 | -0.09% | Mac grade x card (0.5413) | -6.32% [-0.0365, -0.0319] | - | 0.781 | no |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (c) isotonic, static | 0.5069 | 0.0131 | -0.14% | Mac grade x card (0.5413) | -6.36% [-0.0367, -0.0321] | - | 0.781 | no |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (a) Platt, rolling 7 days | 0.5058 | 0.0055 | -0.34% | Mac grade x card (0.5413) | -6.55% [-0.0378, -0.0332] | -6.76% [-0.0392, -0.0343] | 0.781 | yes |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (b) shared transform, rolling 7 days | 0.5063 | 0.0088 | -0.26% | Mac grade x card (0.5413) | -6.47% [-0.0374, -0.0327] | - | 0.781 | yes |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (c) isotonic, rolling 7 days | 0.5067 | 0.0048 | -0.17% | Mac grade x card (0.5413) | -6.39% [-0.0369, -0.0323] | - | 0.780 | yes |
| the Tiny two-head model (SM106) with its own SM117 head | uncalibrated (SM121) | 0.5071 | 0.0140 | - | Mac grade x card (0.5415) | -6.36% [-0.0368, -0.0321] | - | 0.780 | no |
| the Tiny two-head model (SM106) with its own SM117 head | (a) Platt, static | 0.5068 | 0.0075 | -0.06% | Mac grade x card (0.5415) | -6.41% [-0.0372, -0.0323] | -11.86% [-0.0725, -0.0640] | 0.780 | yes |
| the Tiny two-head model (SM106) with its own SM117 head | (b) shared transform, static | 0.5068 | 0.0068 | -0.07% | Mac grade x card (0.5415) | -6.42% [-0.0372, -0.0324] | - | 0.780 | yes |
| the Tiny two-head model (SM106) with its own SM117 head | (c) isotonic, static | 0.5081 | 0.0149 | +0.20% | Mac grade x card (0.5415) | -6.17% [-0.0360, -0.0311] | - | 0.780 | no |
| the Tiny two-head model (SM106) with its own SM117 head | (a) Platt, rolling 7 days | 0.5067 | 0.0066 | -0.08% | Mac grade x card (0.5415) | -6.43% [-0.0373, -0.0324] | -6.66% [-0.0389, -0.0337] | 0.780 | yes |
| the Tiny two-head model (SM106) with its own SM117 head | (b) shared transform, rolling 7 days | 0.5071 | 0.0074 | +0.01% | Mac grade x card (0.5415) | -6.35% [-0.0369, -0.0320] | - | 0.780 | yes |
| the Tiny two-head model (SM106) with its own SM117 head | (c) isotonic, rolling 7 days | 0.5074 | 0.0056 | +0.07% | Mac grade x card (0.5415) | -6.29% [-0.0366, -0.0317] | - | 0.779 | yes |
| the show-time big model (SM91s2) with its own SM117 head | uncalibrated (SM121) | 0.5136 | 0.0412 | - | Mac grade x card (0.5412) | -5.09% [-0.0300, -0.0252] | - | 0.778 | no |
| the show-time big model (SM91s2) with its own SM117 head | (a) Platt, static | 0.5086 | 0.0122 | -0.98% | Mac grade x card (0.5412) | -6.02% [-0.0350, -0.0302] | -11.40% [-0.0699, -0.0614] | 0.778 | no |
| the show-time big model (SM91s2) with its own SM117 head | (b) shared transform, static | 0.5085 | 0.0152 | -1.00% | Mac grade x card (0.5412) | -6.04% [-0.0351, -0.0304] | - | 0.779 | no |
| the show-time big model (SM91s2) with its own SM117 head | (c) isotonic, static | 0.5101 | 0.0164 | -0.70% | Mac grade x card (0.5412) | -5.75% [-0.0337, -0.0286] | - | 0.778 | no |
| the show-time big model (SM91s2) with its own SM117 head | (a) Platt, rolling 7 days | 0.5085 | 0.0064 | -1.00% | Mac grade x card (0.5412) | -6.04% [-0.0352, -0.0304] | -6.25% [-0.0365, -0.0315] | 0.778 | yes |
| the show-time big model (SM91s2) with its own SM117 head | (b) shared transform, rolling 7 days | 0.5084 | 0.0115 | -1.02% | Mac grade x card (0.5412) | -6.05% [-0.0353, -0.0303] | - | 0.779 | no |
| the show-time big model (SM91s2) with its own SM117 head | (c) isotonic, rolling 7 days | 0.5091 | 0.0058 | -0.89% | Mac grade x card (0.5412) | -5.93% [-0.0346, -0.0298] | - | 0.778 | yes |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | uncalibrated (SM121) | 0.5176 | 0.0360 | - | Mac grade x card (0.5413) | -4.38% [-0.0257, -0.0218] | - | 0.771 | no |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (a) Platt, static | 0.5138 | 0.0066 | -0.73% | Mac grade x card (0.5413) | -5.08% [-0.0296, -0.0256] | -10.75% [-0.0662, -0.0578] | 0.771 | yes |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (b) shared transform, static | 0.5141 | 0.0098 | -0.67% | Mac grade x card (0.5413) | -5.02% [-0.0292, -0.0252] | - | 0.771 | yes |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (c) isotonic, static | 0.5142 | 0.0079 | -0.65% | Mac grade x card (0.5413) | -4.99% [-0.0291, -0.0250] | - | 0.771 | yes |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (a) Platt, rolling 7 days | 0.5138 | 0.0045 | -0.73% | Mac grade x card (0.5413) | -5.08% [-0.0296, -0.0255] | -5.29% [-0.0309, -0.0267] | 0.771 | yes |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (b) shared transform, rolling 7 days | 0.5143 | 0.0087 | -0.63% | Mac grade x card (0.5413) | -4.98% [-0.0292, -0.0249] | - | 0.771 | yes |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (c) isotonic, rolling 7 days | 0.5145 | 0.0052 | -0.59% | Mac grade x card (0.5413) | -4.94% [-0.0289, -0.0247] | - | 0.771 | yes |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | uncalibrated (SM121) | 0.5076 | 0.0241 | - | Mac grade x card (0.5412) | -6.20% [-0.0358, -0.0313] | - | 0.781 | no |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (a) Platt, static | 0.5065 | 0.0138 | -0.23% | Mac grade x card (0.5412) | -6.41% [-0.0370, -0.0324] | -11.90% [-0.0729, -0.0641] | 0.781 | no |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (b) shared transform, static | 0.5072 | 0.0213 | -0.07% | Mac grade x card (0.5412) | -6.27% [-0.0363, -0.0316] | - | 0.781 | no |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (c) isotonic, static | 0.5071 | 0.0143 | -0.10% | Mac grade x card (0.5412) | -6.30% [-0.0364, -0.0318] | - | 0.781 | no |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (a) Platt, rolling 7 days | 0.5058 | 0.0052 | -0.35% | Mac grade x card (0.5412) | -6.53% [-0.0376, -0.0331] | -6.74% [-0.0391, -0.0342] | 0.781 | yes |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (b) shared transform, rolling 7 days | 0.5063 | 0.0090 | -0.27% | Mac grade x card (0.5412) | -6.45% [-0.0373, -0.0325] | - | 0.781 | yes |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (c) isotonic, rolling 7 days | 0.5068 | 0.0046 | -0.17% | Mac grade x card (0.5412) | -6.36% [-0.0368, -0.0321] | - | 0.780 | yes |
| the show-time retrain (SM92) with its own SM117 head | uncalibrated (SM121) | 0.5080 | 0.0255 | - | Mac grade x card (0.5416) | -6.22% [-0.0360, -0.0315] | - | 0.781 | no |
| the show-time retrain (SM92) with its own SM117 head | (a) Platt, static | 0.5067 | 0.0158 | -0.24% | Mac grade x card (0.5416) | -6.45% [-0.0373, -0.0326] | -11.74% [-0.0720, -0.0632] | 0.781 | no |
| the show-time retrain (SM92) with its own SM117 head | (b) shared transform, static | 0.5075 | 0.0226 | -0.09% | Mac grade x card (0.5416) | -6.31% [-0.0365, -0.0318] | - | 0.781 | no |
| the show-time retrain (SM92) with its own SM117 head | (c) isotonic, static | 0.5073 | 0.0145 | -0.12% | Mac grade x card (0.5416) | -6.34% [-0.0367, -0.0320] | - | 0.781 | no |
| the show-time retrain (SM92) with its own SM117 head | (a) Platt, rolling 7 days | 0.5060 | 0.0060 | -0.39% | Mac grade x card (0.5416) | -6.59% [-0.0380, -0.0334] | -6.78% [-0.0393, -0.0344] | 0.781 | yes |
| the show-time retrain (SM92) with its own SM117 head | (b) shared transform, rolling 7 days | 0.5064 | 0.0089 | -0.31% | Mac grade x card (0.5416) | -6.51% [-0.0377, -0.0329] | - | 0.781 | yes |
| the show-time retrain (SM92) with its own SM117 head | (c) isotonic, rolling 7 days | 0.5068 | 0.0046 | -0.23% | Mac grade x card (0.5416) | -6.44% [-0.0372, -0.0326] | - | 0.780 | yes |
| the Small Instant-Only model (SM98) with its own SM117 head | uncalibrated (SM121) | 0.5066 | 0.0166 | - | Mac grade x card (0.5413) | -6.41% [-0.0371, -0.0325] | - | 0.781 | no |
| the Small Instant-Only model (SM98) with its own SM117 head | (a) Platt, static | 0.5065 | 0.0115 | -0.02% | Mac grade x card (0.5413) | -6.43% [-0.0373, -0.0325] | -12.02% [-0.0736, -0.0650] | 0.781 | no |
| the Small Instant-Only model (SM98) with its own SM117 head | (b) shared transform, static | 0.5068 | 0.0120 | +0.05% | Mac grade x card (0.5413) | -6.36% [-0.0369, -0.0321] | - | 0.781 | no |
| the Small Instant-Only model (SM98) with its own SM117 head | (c) isotonic, static | 0.5077 | 0.0141 | +0.22% | Mac grade x card (0.5413) | -6.21% [-0.0361, -0.0311] | - | 0.781 | no |
| the Small Instant-Only model (SM98) with its own SM117 head | (a) Platt, rolling 7 days | 0.5057 | 0.0052 | -0.17% | Mac grade x card (0.5413) | -6.57% [-0.0380, -0.0333] | -6.79% [-0.0394, -0.0345] | 0.781 | yes |
| the Small Instant-Only model (SM98) with its own SM117 head | (b) shared transform, rolling 7 days | 0.5070 | 0.0143 | +0.08% | Mac grade x card (0.5413) | -6.34% [-0.0368, -0.0319] | - | 0.781 | no |
| the Small Instant-Only model (SM98) with its own SM117 head | (c) isotonic, rolling 7 days | 0.5066 | 0.0043 | -0.00% | Mac grade x card (0.5413) | -6.42% [-0.0372, -0.0324] | - | 0.780 | yes |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | uncalibrated (SM121) | 0.5069 | 0.0144 | - | Mac grade x card (0.5412) | -6.33% [-0.0367, -0.0320] | - | 0.780 | no |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (a) Platt, static | 0.5068 | 0.0095 | -0.03% | Mac grade x card (0.5412) | -6.36% [-0.0369, -0.0320] | -11.90% [-0.0728, -0.0643] | 0.780 | yes |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (b) shared transform, static | 0.5066 | 0.0073 | -0.06% | Mac grade x card (0.5412) | -6.39% [-0.0370, -0.0322] | - | 0.780 | yes |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (c) isotonic, static | 0.5080 | 0.0144 | +0.22% | Mac grade x card (0.5412) | -6.13% [-0.0357, -0.0307] | - | 0.780 | no |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (a) Platt, rolling 7 days | 0.5065 | 0.0070 | -0.09% | Mac grade x card (0.5412) | -6.42% [-0.0372, -0.0323] | -6.63% [-0.0387, -0.0335] | 0.780 | yes |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (b) shared transform, rolling 7 days | 0.5069 | 0.0082 | +0.00% | Mac grade x card (0.5412) | -6.33% [-0.0368, -0.0318] | - | 0.780 | yes |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (c) isotonic, rolling 7 days | 0.5072 | 0.0052 | +0.04% | Mac grade x card (0.5412) | -6.29% [-0.0366, -0.0317] | - | 0.780 | yes |
| the longer cool-down for Tiny (SM115) with its own SM119 head | uncalibrated (SM121) | 0.5070 | 0.0226 | - | Mac grade x card (0.5412) | -6.32% [-0.0366, -0.0319] | - | 0.781 | no |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (a) Platt, static | 0.5066 | 0.0176 | -0.08% | Mac grade x card (0.5412) | -6.39% [-0.0372, -0.0322] | -11.93% [-0.0732, -0.0643] | 0.781 | no |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (b) shared transform, static | 0.5076 | 0.0227 | +0.12% | Mac grade x card (0.5412) | -6.21% [-0.0362, -0.0312] | - | 0.781 | no |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (c) isotonic, static | 0.5073 | 0.0175 | +0.06% | Mac grade x card (0.5412) | -6.26% [-0.0364, -0.0315] | - | 0.781 | no |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (a) Platt, rolling 7 days | 0.5053 | 0.0070 | -0.34% | Mac grade x card (0.5412) | -6.64% [-0.0384, -0.0336] | -6.85% [-0.0398, -0.0349] | 0.782 | yes |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (b) shared transform, rolling 7 days | 0.5061 | 0.0108 | -0.18% | Mac grade x card (0.5412) | -6.49% [-0.0377, -0.0327] | - | 0.781 | no |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (c) isotonic, rolling 7 days | 0.5063 | 0.0048 | -0.13% | Mac grade x card (0.5412) | -6.44% [-0.0373, -0.0325] | - | 0.781 | yes |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | uncalibrated (SM121) | 0.5123 | 0.0410 | - | Mac grade x card (0.5411) | -5.31% [-0.0312, -0.0264] | - | 0.780 | no |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (a) Platt, static | 0.5076 | 0.0135 | -0.93% | Mac grade x card (0.5411) | -6.19% [-0.0359, -0.0311] | -11.49% [-0.0703, -0.0618] | 0.780 | no |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (b) shared transform, static | 0.5073 | 0.0123 | -0.97% | Mac grade x card (0.5411) | -6.23% [-0.0362, -0.0314] | - | 0.780 | no |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (c) isotonic, static | 0.5089 | 0.0182 | -0.66% | Mac grade x card (0.5411) | -5.94% [-0.0346, -0.0297] | - | 0.779 | no |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (a) Platt, rolling 7 days | 0.5071 | 0.0042 | -1.01% | Mac grade x card (0.5411) | -6.27% [-0.0363, -0.0316] | -6.47% [-0.0377, -0.0327] | 0.780 | yes |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (b) shared transform, rolling 7 days | 0.5072 | 0.0072 | -1.00% | Mac grade x card (0.5411) | -6.26% [-0.0364, -0.0314] | - | 0.780 | yes |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (c) isotonic, rolling 7 days | 0.5084 | 0.0045 | -0.77% | Mac grade x card (0.5411) | -6.04% [-0.0352, -0.0303] | - | 0.779 | yes |

### Mac slice, whole collection

| combination | method | LogLoss | ECE | vs uncalibrated | strongest baseline (LogLoss) | vs strongest [day CI of the difference] | vs strongest, both Platt | AUC | reliable |
|---|---|---|---|---|---|---|---|---|---|
| the longer Small Instant-Only model (SM113) with its own SM119 head | uncalibrated (SM121) | 0.4940 | 0.0345 | - | Mac R x card (0.5278) | -6.41% [-0.0382, -0.0295] | - | 0.795 | no |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (a) Platt, static | 0.4918 | 0.0226 | -0.45% | Mac R x card (0.5278) | -6.83% [-0.0404, -0.0317] | -7.07% [-0.0419, -0.0330] | 0.795 | no |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (b) shared transform, static | 0.4911 | 0.0161 | -0.59% | Mac R x card (0.5278) | -6.96% [-0.0413, -0.0323] | - | 0.795 | no |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (c) isotonic, static | 0.4926 | 0.0232 | -0.27% | Mac R x card (0.5278) | -6.66% [-0.0397, -0.0307] | - | 0.795 | no |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (a) Platt, rolling 7 days | 0.4905 | 0.0093 | -0.71% | Mac R x card (0.5278) | -7.07% [-0.0417, -0.0330] | -7.05% [-0.0414, -0.0330] | 0.795 | yes |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (b) shared transform, rolling 7 days | 0.4904 | 0.0099 | -0.72% | Mac R x card (0.5278) | -7.08% [-0.0418, -0.0330] | - | 0.795 | yes |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (c) isotonic, rolling 7 days | 0.4915 | 0.0098 | -0.51% | Mac R x card (0.5278) | -6.88% [-0.0408, -0.0319] | - | 0.794 | yes |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | uncalibrated (SM121) | 0.4950 | 0.0332 | - | Mac R x card (0.5279) | -6.23% [-0.0374, -0.0284] | - | 0.794 | no |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (a) Platt, static | 0.4923 | 0.0160 | -0.56% | Mac R x card (0.5279) | -6.75% [-0.0402, -0.0312] | -6.98% [-0.0416, -0.0324] | 0.794 | no |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (b) shared transform, static | 0.4921 | 0.0141 | -0.60% | Mac R x card (0.5279) | -6.79% [-0.0404, -0.0314] | - | 0.794 | no |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (c) isotonic, static | 0.4933 | 0.0183 | -0.35% | Mac R x card (0.5279) | -6.56% [-0.0392, -0.0302] | - | 0.793 | no |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (a) Platt, rolling 7 days | 0.4918 | 0.0083 | -0.66% | Mac R x card (0.5279) | -6.84% [-0.0406, -0.0317] | -6.83% [-0.0404, -0.0319] | 0.793 | yes |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (b) shared transform, rolling 7 days | 0.4917 | 0.0088 | -0.68% | Mac R x card (0.5279) | -6.86% [-0.0407, -0.0318] | - | 0.794 | yes |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (c) isotonic, rolling 7 days | 0.4928 | 0.0080 | -0.45% | Mac R x card (0.5279) | -6.64% [-0.0396, -0.0307] | - | 0.792 | yes |
| the Tiny two-head model (SM106) with its own SM117 head | uncalibrated (SM121) | 0.4940 | 0.0255 | - | Mac R x card (0.5286) | -6.55% [-0.0388, -0.0303] | - | 0.793 | no |
| the Tiny two-head model (SM106) with its own SM117 head | (a) Platt, static | 0.4926 | 0.0151 | -0.29% | Mac R x card (0.5286) | -6.81% [-0.0403, -0.0316] | -7.04% [-0.0418, -0.0329] | 0.793 | no |
| the Tiny two-head model (SM106) with its own SM117 head | (b) shared transform, static | 0.4924 | 0.0105 | -0.33% | Mac R x card (0.5286) | -6.86% [-0.0406, -0.0318] | - | 0.793 | no |
| the Tiny two-head model (SM106) with its own SM117 head | (c) isotonic, static | 0.4938 | 0.0174 | -0.04% | Mac R x card (0.5286) | -6.58% [-0.0394, -0.0303] | - | 0.793 | no |
| the Tiny two-head model (SM106) with its own SM117 head | (a) Platt, rolling 7 days | 0.4923 | 0.0086 | -0.35% | Mac R x card (0.5286) | -6.88% [-0.0407, -0.0320] | -6.87% [-0.0406, -0.0322] | 0.793 | yes |
| the Tiny two-head model (SM106) with its own SM117 head | (b) shared transform, rolling 7 days | 0.4922 | 0.0068 | -0.36% | Mac R x card (0.5286) | -6.89% [-0.0408, -0.0320] | - | 0.793 | yes |
| the Tiny two-head model (SM106) with its own SM117 head | (c) isotonic, rolling 7 days | 0.4931 | 0.0076 | -0.19% | Mac R x card (0.5286) | -6.72% [-0.0400, -0.0311] | - | 0.792 | yes |
| the show-time big model (SM91s2) with its own SM117 head | uncalibrated (SM121) | 0.4927 | 0.0287 | - | Mac R x card (0.5277) | -6.64% [-0.0394, -0.0307] | - | 0.796 | no |
| the show-time big model (SM91s2) with its own SM117 head | (a) Platt, static | 0.4899 | 0.0087 | -0.57% | Mac R x card (0.5277) | -7.18% [-0.0421, -0.0337] | -7.48% [-0.0440, -0.0351] | 0.796 | yes |
| the show-time big model (SM91s2) with its own SM117 head | (b) shared transform, static | 0.4901 | 0.0101 | -0.52% | Mac R x card (0.5277) | -7.13% [-0.0420, -0.0334] | - | 0.796 | no |
| the show-time big model (SM91s2) with its own SM117 head | (c) isotonic, static | 0.4911 | 0.0125 | -0.32% | Mac R x card (0.5277) | -6.95% [-0.0412, -0.0321] | - | 0.795 | no |
| the show-time big model (SM91s2) with its own SM117 head | (a) Platt, rolling 7 days | 0.4899 | 0.0079 | -0.57% | Mac R x card (0.5277) | -7.17% [-0.0421, -0.0337] | -7.12% [-0.0416, -0.0335] | 0.796 | yes |
| the show-time big model (SM91s2) with its own SM117 head | (b) shared transform, rolling 7 days | 0.4900 | 0.0080 | -0.55% | Mac R x card (0.5277) | -7.16% [-0.0420, -0.0336] | - | 0.796 | yes |
| the show-time big model (SM91s2) with its own SM117 head | (c) isotonic, rolling 7 days | 0.4911 | 0.0065 | -0.31% | Mac R x card (0.5277) | -6.93% [-0.0411, -0.0320] | - | 0.795 | yes |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | uncalibrated (SM121) | 0.4956 | 0.0365 | - | Mac R x card (0.5279) | -6.11% [-0.0367, -0.0278] | - | 0.793 | no |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (a) Platt, static | 0.4929 | 0.0222 | -0.55% | Mac R x card (0.5279) | -6.63% [-0.0395, -0.0306] | -6.85% [-0.0409, -0.0318] | 0.793 | no |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (b) shared transform, static | 0.4926 | 0.0195 | -0.62% | Mac R x card (0.5279) | -6.69% [-0.0398, -0.0309] | - | 0.793 | no |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (c) isotonic, static | 0.4939 | 0.0239 | -0.36% | Mac R x card (0.5279) | -6.44% [-0.0386, -0.0295] | - | 0.793 | no |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (a) Platt, rolling 7 days | 0.4916 | 0.0068 | -0.81% | Mac R x card (0.5279) | -6.87% [-0.0408, -0.0319] | -6.86% [-0.0405, -0.0319] | 0.793 | yes |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (b) shared transform, rolling 7 days | 0.4916 | 0.0065 | -0.81% | Mac R x card (0.5279) | -6.87% [-0.0408, -0.0320] | - | 0.793 | yes |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (c) isotonic, rolling 7 days | 0.4926 | 0.0082 | -0.61% | Mac R x card (0.5279) | -6.68% [-0.0398, -0.0309] | - | 0.792 | yes |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | uncalibrated (SM121) | 0.4952 | 0.0343 | - | Mac R x card (0.5278) | -6.17% [-0.0371, -0.0281] | - | 0.794 | no |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (a) Platt, static | 0.4923 | 0.0169 | -0.59% | Mac R x card (0.5278) | -6.72% [-0.0400, -0.0311] | -6.96% [-0.0416, -0.0323] | 0.794 | no |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (b) shared transform, static | 0.4920 | 0.0145 | -0.65% | Mac R x card (0.5278) | -6.77% [-0.0403, -0.0313] | - | 0.794 | no |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (c) isotonic, static | 0.4940 | 0.0193 | -0.25% | Mac R x card (0.5278) | -6.40% [-0.0386, -0.0289] | - | 0.793 | no |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (a) Platt, rolling 7 days | 0.4917 | 0.0085 | -0.70% | Mac R x card (0.5278) | -6.83% [-0.0405, -0.0316] | -6.81% [-0.0402, -0.0317] | 0.793 | yes |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (b) shared transform, rolling 7 days | 0.4916 | 0.0080 | -0.73% | Mac R x card (0.5278) | -6.86% [-0.0407, -0.0318] | - | 0.794 | yes |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (c) isotonic, rolling 7 days | 0.4930 | 0.0081 | -0.46% | Mac R x card (0.5278) | -6.60% [-0.0393, -0.0304] | - | 0.792 | yes |
| the show-time retrain (SM92) with its own SM117 head | uncalibrated (SM121) | 0.4961 | 0.0356 | - | Mac R x card (0.5289) | -6.21% [-0.0374, -0.0284] | - | 0.793 | no |
| the show-time retrain (SM92) with its own SM117 head | (a) Platt, static | 0.4929 | 0.0164 | -0.65% | Mac R x card (0.5289) | -6.82% [-0.0406, -0.0317] | -7.05% [-0.0421, -0.0328] | 0.793 | no |
| the show-time retrain (SM92) with its own SM117 head | (b) shared transform, static | 0.4928 | 0.0150 | -0.66% | Mac R x card (0.5289) | -6.84% [-0.0407, -0.0317] | - | 0.793 | no |
| the show-time retrain (SM92) with its own SM117 head | (c) isotonic, static | 0.4940 | 0.0191 | -0.41% | Mac R x card (0.5289) | -6.59% [-0.0394, -0.0305] | - | 0.793 | no |
| the show-time retrain (SM92) with its own SM117 head | (a) Platt, rolling 7 days | 0.4923 | 0.0066 | -0.75% | Mac R x card (0.5289) | -6.92% [-0.0411, -0.0322] | -6.89% [-0.0407, -0.0323] | 0.793 | yes |
| the show-time retrain (SM92) with its own SM117 head | (b) shared transform, rolling 7 days | 0.4922 | 0.0062 | -0.77% | Mac R x card (0.5289) | -6.94% [-0.0411, -0.0324] | - | 0.793 | yes |
| the show-time retrain (SM92) with its own SM117 head | (c) isotonic, rolling 7 days | 0.4935 | 0.0077 | -0.52% | Mac R x card (0.5289) | -6.70% [-0.0400, -0.0310] | - | 0.792 | yes |
| the Small Instant-Only model (SM98) with its own SM117 head | uncalibrated (SM121) | 0.4941 | 0.0315 | - | Mac R x card (0.5279) | -6.40% [-0.0382, -0.0293] | - | 0.794 | no |
| the Small Instant-Only model (SM98) with its own SM117 head | (a) Platt, static | 0.4914 | 0.0131 | -0.55% | Mac R x card (0.5279) | -6.92% [-0.0409, -0.0321] | -7.14% [-0.0424, -0.0333] | 0.794 | no |
| the Small Instant-Only model (SM98) with its own SM117 head | (b) shared transform, static | 0.4914 | 0.0118 | -0.54% | Mac R x card (0.5279) | -6.91% [-0.0410, -0.0319] | - | 0.794 | no |
| the Small Instant-Only model (SM98) with its own SM117 head | (c) isotonic, static | 0.4922 | 0.0167 | -0.38% | Mac R x card (0.5279) | -6.76% [-0.0402, -0.0312] | - | 0.794 | no |
| the Small Instant-Only model (SM98) with its own SM117 head | (a) Platt, rolling 7 days | 0.4910 | 0.0074 | -0.64% | Mac R x card (0.5279) | -7.00% [-0.0413, -0.0325] | -6.99% [-0.0410, -0.0326] | 0.794 | yes |
| the Small Instant-Only model (SM98) with its own SM117 head | (b) shared transform, rolling 7 days | 0.4913 | 0.0097 | -0.57% | Mac R x card (0.5279) | -6.93% [-0.0411, -0.0320] | - | 0.794 | yes |
| the Small Instant-Only model (SM98) with its own SM117 head | (c) isotonic, rolling 7 days | 0.4921 | 0.0062 | -0.41% | Mac R x card (0.5279) | -6.79% [-0.0403, -0.0314] | - | 0.793 | yes |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | uncalibrated (SM121) | 0.4936 | 0.0271 | - | Mac R x card (0.5275) | -6.43% [-0.0382, -0.0296] | - | 0.794 | no |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (a) Platt, static | 0.4921 | 0.0166 | -0.30% | Mac R x card (0.5275) | -6.71% [-0.0397, -0.0310] | -6.96% [-0.0412, -0.0323] | 0.794 | no |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (b) shared transform, static | 0.4916 | 0.0111 | -0.39% | Mac R x card (0.5275) | -6.79% [-0.0403, -0.0314] | - | 0.794 | no |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (c) isotonic, static | 0.4932 | 0.0186 | -0.08% | Mac R x card (0.5275) | -6.50% [-0.0389, -0.0298] | - | 0.793 | no |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (a) Platt, rolling 7 days | 0.4917 | 0.0076 | -0.39% | Mac R x card (0.5275) | -6.79% [-0.0402, -0.0315] | -6.77% [-0.0399, -0.0315] | 0.794 | yes |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (b) shared transform, rolling 7 days | 0.4914 | 0.0064 | -0.43% | Mac R x card (0.5275) | -6.83% [-0.0404, -0.0317] | - | 0.794 | yes |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (c) isotonic, rolling 7 days | 0.4922 | 0.0066 | -0.28% | Mac R x card (0.5275) | -6.69% [-0.0398, -0.0309] | - | 0.793 | yes |
| the longer cool-down for Tiny (SM115) with its own SM119 head | uncalibrated (SM121) | 0.4930 | 0.0314 | - | Mac R x card (0.5275) | -6.53% [-0.0385, -0.0304] | - | 0.795 | no |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (a) Platt, static | 0.4900 | 0.0066 | -0.61% | Mac R x card (0.5275) | -7.10% [-0.0418, -0.0332] | -7.35% [-0.0434, -0.0344] | 0.795 | yes |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (b) shared transform, static | 0.4900 | 0.0083 | -0.61% | Mac R x card (0.5275) | -7.10% [-0.0418, -0.0332] | - | 0.795 | yes |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (c) isotonic, static | 0.4908 | 0.0099 | -0.46% | Mac R x card (0.5275) | -6.96% [-0.0410, -0.0324] | - | 0.795 | yes |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (a) Platt, rolling 7 days | 0.4900 | 0.0067 | -0.60% | Mac R x card (0.5275) | -7.10% [-0.0417, -0.0332] | -7.07% [-0.0413, -0.0332] | 0.795 | yes |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (b) shared transform, rolling 7 days | 0.4899 | 0.0064 | -0.63% | Mac R x card (0.5275) | -7.12% [-0.0418, -0.0333] | - | 0.795 | yes |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (c) isotonic, rolling 7 days | 0.4922 | 0.0051 | -0.17% | Mac R x card (0.5275) | -6.69% [-0.0401, -0.0303] | - | 0.794 | yes |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | uncalibrated (SM121) | 0.4929 | 0.0259 | - | Mac R x card (0.5276) | -6.59% [-0.0393, -0.0302] | - | 0.795 | no |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (a) Platt, static | 0.4906 | 0.0061 | -0.47% | Mac R x card (0.5276) | -7.03% [-0.0416, -0.0326] | -7.32% [-0.0434, -0.0340] | 0.795 | yes |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (b) shared transform, static | 0.4905 | 0.0072 | -0.47% | Mac R x card (0.5276) | -7.03% [-0.0416, -0.0326] | - | 0.795 | yes |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (c) isotonic, static | 0.4938 | 0.0135 | +0.20% | Mac R x card (0.5276) | -6.41% [-0.0394, -0.0277] | - | 0.794 | no |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (a) Platt, rolling 7 days | 0.4905 | 0.0050 | -0.48% | Mac R x card (0.5276) | -7.04% [-0.0416, -0.0328] | -7.02% [-0.0412, -0.0327] | 0.795 | yes |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (b) shared transform, rolling 7 days | 0.4905 | 0.0069 | -0.47% | Mac R x card (0.5276) | -7.04% [-0.0416, -0.0327] | - | 0.795 | yes |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (c) isotonic, rolling 7 days | 0.4927 | 0.0079 | -0.03% | Mac R x card (0.5276) | -6.62% [-0.0398, -0.0301] | - | 0.794 | yes |

### Mac slice, deck subset

| combination | method | LogLoss | ECE | vs uncalibrated | strongest baseline (LogLoss) | vs strongest [day CI of the difference] | vs strongest, both Platt | AUC | reliable |
|---|---|---|---|---|---|---|---|---|---|
| the longer Small Instant-Only model (SM113) with its own SM119 head | uncalibrated (SM121) | 0.4764 | 0.0268 | - | Mac grade x card (0.5084) | -6.28% [-0.0375, -0.0265] | - | 0.780 | no |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (a) Platt, static | 0.4748 | 0.0167 | -0.35% | Mac grade x card (0.5084) | -6.60% [-0.0392, -0.0282] | -6.69% [-0.0398, -0.0286] | 0.780 | no |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (b) shared transform, static | 0.4744 | 0.0106 | -0.43% | Mac grade x card (0.5084) | -6.68% [-0.0398, -0.0284] | - | 0.780 | no |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (c) isotonic, static | 0.4757 | 0.0175 | -0.14% | Mac grade x card (0.5084) | -6.42% [-0.0383, -0.0271] | - | 0.779 | no |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (a) Platt, rolling 7 days | 0.4744 | 0.0073 | -0.42% | Mac grade x card (0.5084) | -6.67% [-0.0396, -0.0285] | -6.91% [-0.0404, -0.0302] | 0.779 | yes |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (b) shared transform, rolling 7 days | 0.4742 | 0.0090 | -0.47% | Mac grade x card (0.5084) | -6.72% [-0.0400, -0.0286] | - | 0.780 | yes |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (c) isotonic, rolling 7 days | 0.4751 | 0.0088 | -0.27% | Mac grade x card (0.5084) | -6.53% [-0.0390, -0.0278] | - | 0.778 | yes |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | uncalibrated (SM121) | 0.4788 | 0.0281 | - | Mac grade x card (0.5083) | -5.80% [-0.0351, -0.0241] | - | 0.777 | no |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (a) Platt, static | 0.4766 | 0.0131 | -0.46% | Mac grade x card (0.5083) | -6.23% [-0.0374, -0.0263] | -6.32% [-0.0380, -0.0267] | 0.777 | no |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (b) shared transform, static | 0.4763 | 0.0102 | -0.52% | Mac grade x card (0.5083) | -6.29% [-0.0378, -0.0265] | - | 0.777 | no |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (c) isotonic, static | 0.4776 | 0.0164 | -0.26% | Mac grade x card (0.5083) | -6.04% [-0.0365, -0.0252] | - | 0.777 | no |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (a) Platt, rolling 7 days | 0.4766 | 0.0063 | -0.46% | Mac grade x card (0.5083) | -6.23% [-0.0375, -0.0263] | -6.50% [-0.0384, -0.0279] | 0.776 | yes |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (b) shared transform, rolling 7 days | 0.4762 | 0.0089 | -0.55% | Mac grade x card (0.5083) | -6.31% [-0.0378, -0.0266] | - | 0.777 | yes |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (c) isotonic, rolling 7 days | 0.4774 | 0.0086 | -0.29% | Mac grade x card (0.5083) | -6.07% [-0.0367, -0.0256] | - | 0.775 | yes |
| the Tiny two-head model (SM106) with its own SM117 head | uncalibrated (SM121) | 0.4770 | 0.0155 | - | Mac grade x card (0.5088) | -6.25% [-0.0374, -0.0265] | - | 0.777 | no |
| the Tiny two-head model (SM106) with its own SM117 head | (a) Platt, static | 0.4764 | 0.0078 | -0.12% | Mac grade x card (0.5088) | -6.37% [-0.0381, -0.0270] | -6.45% [-0.0387, -0.0273] | 0.777 | yes |
| the Tiny two-head model (SM106) with its own SM117 head | (b) shared transform, static | 0.4763 | 0.0082 | -0.13% | Mac grade x card (0.5088) | -6.38% [-0.0382, -0.0269] | - | 0.777 | yes |
| the Tiny two-head model (SM106) with its own SM117 head | (c) isotonic, static | 0.4775 | 0.0124 | +0.11% | Mac grade x card (0.5088) | -6.15% [-0.0371, -0.0258] | - | 0.776 | no |
| the Tiny two-head model (SM106) with its own SM117 head | (a) Platt, rolling 7 days | 0.4771 | 0.0125 | +0.02% | Mac grade x card (0.5088) | -6.23% [-0.0374, -0.0262] | -6.52% [-0.0386, -0.0284] | 0.776 | no |
| the Tiny two-head model (SM106) with its own SM117 head | (b) shared transform, rolling 7 days | 0.4769 | 0.0149 | -0.01% | Mac grade x card (0.5088) | -6.26% [-0.0377, -0.0264] | - | 0.777 | no |
| the Tiny two-head model (SM106) with its own SM117 head | (c) isotonic, rolling 7 days | 0.4779 | 0.0143 | +0.19% | Mac grade x card (0.5088) | -6.07% [-0.0368, -0.0255] | - | 0.775 | no |
| the show-time big model (SM91s2) with its own SM117 head | uncalibrated (SM121) | 0.4787 | 0.0270 | - | Mac grade x card (0.5077) | -5.72% [-0.0344, -0.0239] | - | 0.778 | no |
| the show-time big model (SM91s2) with its own SM117 head | (a) Platt, static | 0.4761 | 0.0110 | -0.53% | Mac grade x card (0.5077) | -6.22% [-0.0371, -0.0262] | -6.29% [-0.0376, -0.0264] | 0.778 | no |
| the show-time big model (SM91s2) with its own SM117 head | (b) shared transform, static | 0.4762 | 0.0109 | -0.51% | Mac grade x card (0.5077) | -6.20% [-0.0370, -0.0261] | - | 0.778 | no |
| the show-time big model (SM91s2) with its own SM117 head | (c) isotonic, static | 0.4772 | 0.0119 | -0.30% | Mac grade x card (0.5077) | -6.01% [-0.0363, -0.0248] | - | 0.777 | no |
| the show-time big model (SM91s2) with its own SM117 head | (a) Platt, rolling 7 days | 0.4762 | 0.0084 | -0.52% | Mac grade x card (0.5077) | -6.21% [-0.0369, -0.0262] | -6.40% [-0.0374, -0.0279] | 0.778 | yes |
| the show-time big model (SM91s2) with its own SM117 head | (b) shared transform, rolling 7 days | 0.4762 | 0.0097 | -0.51% | Mac grade x card (0.5077) | -6.20% [-0.0370, -0.0261] | - | 0.778 | yes |
| the show-time big model (SM91s2) with its own SM117 head | (c) isotonic, rolling 7 days | 0.4776 | 0.0091 | -0.22% | Mac grade x card (0.5077) | -5.92% [-0.0360, -0.0241] | - | 0.776 | yes |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | uncalibrated (SM121) | 0.4792 | 0.0309 | - | Mac grade x card (0.5083) | -5.71% [-0.0346, -0.0237] | - | 0.777 | no |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (a) Platt, static | 0.4770 | 0.0178 | -0.47% | Mac grade x card (0.5083) | -6.15% [-0.0369, -0.0259] | -6.24% [-0.0376, -0.0262] | 0.777 | no |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (b) shared transform, static | 0.4766 | 0.0143 | -0.56% | Mac grade x card (0.5083) | -6.24% [-0.0374, -0.0262] | - | 0.777 | no |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (c) isotonic, static | 0.4781 | 0.0192 | -0.24% | Mac grade x card (0.5083) | -5.94% [-0.0359, -0.0247] | - | 0.777 | no |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (a) Platt, rolling 7 days | 0.4765 | 0.0085 | -0.56% | Mac grade x card (0.5083) | -6.25% [-0.0376, -0.0262] | -6.51% [-0.0386, -0.0279] | 0.776 | yes |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (b) shared transform, rolling 7 days | 0.4761 | 0.0077 | -0.66% | Mac grade x card (0.5083) | -6.33% [-0.0379, -0.0266] | - | 0.777 | yes |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (c) isotonic, rolling 7 days | 0.4773 | 0.0097 | -0.41% | Mac grade x card (0.5083) | -6.09% [-0.0370, -0.0254] | - | 0.775 | yes |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | uncalibrated (SM121) | 0.4791 | 0.0295 | - | Mac grade x card (0.5084) | -5.75% [-0.0349, -0.0238] | - | 0.777 | no |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (a) Platt, static | 0.4768 | 0.0135 | -0.50% | Mac grade x card (0.5084) | -6.22% [-0.0374, -0.0261] | -6.30% [-0.0380, -0.0265] | 0.777 | no |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (b) shared transform, static | 0.4765 | 0.0102 | -0.56% | Mac grade x card (0.5084) | -6.27% [-0.0378, -0.0264] | - | 0.777 | no |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (c) isotonic, static | 0.4786 | 0.0161 | -0.11% | Mac grade x card (0.5084) | -5.85% [-0.0356, -0.0240] | - | 0.777 | no |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (a) Platt, rolling 7 days | 0.4767 | 0.0071 | -0.51% | Mac grade x card (0.5084) | -6.23% [-0.0376, -0.0262] | -6.46% [-0.0382, -0.0277] | 0.776 | yes |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (b) shared transform, rolling 7 days | 0.4763 | 0.0084 | -0.60% | Mac grade x card (0.5084) | -6.31% [-0.0379, -0.0265] | - | 0.777 | yes |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (c) isotonic, rolling 7 days | 0.4776 | 0.0089 | -0.32% | Mac grade x card (0.5084) | -6.04% [-0.0366, -0.0254] | - | 0.775 | yes |
| the show-time retrain (SM92) with its own SM117 head | uncalibrated (SM121) | 0.4797 | 0.0303 | - | Mac grade x card (0.5092) | -5.79% [-0.0351, -0.0241] | - | 0.777 | no |
| the show-time retrain (SM92) with its own SM117 head | (a) Platt, static | 0.4771 | 0.0142 | -0.54% | Mac grade x card (0.5092) | -6.30% [-0.0378, -0.0267] | -6.37% [-0.0384, -0.0269] | 0.777 | no |
| the show-time retrain (SM92) with its own SM117 head | (b) shared transform, static | 0.4769 | 0.0131 | -0.60% | Mac grade x card (0.5092) | -6.35% [-0.0382, -0.0268] | - | 0.777 | no |
| the show-time retrain (SM92) with its own SM117 head | (c) isotonic, static | 0.4784 | 0.0192 | -0.28% | Mac grade x card (0.5092) | -6.05% [-0.0367, -0.0254] | - | 0.776 | no |
| the show-time retrain (SM92) with its own SM117 head | (a) Platt, rolling 7 days | 0.4771 | 0.0097 | -0.54% | Mac grade x card (0.5092) | -6.30% [-0.0379, -0.0266] | -6.54% [-0.0387, -0.0283] | 0.776 | yes |
| the show-time retrain (SM92) with its own SM117 head | (b) shared transform, rolling 7 days | 0.4767 | 0.0087 | -0.64% | Mac grade x card (0.5092) | -6.39% [-0.0383, -0.0269] | - | 0.777 | yes |
| the show-time retrain (SM92) with its own SM117 head | (c) isotonic, rolling 7 days | 0.4780 | 0.0105 | -0.37% | Mac grade x card (0.5092) | -6.14% [-0.0371, -0.0260] | - | 0.775 | no |
| the Small Instant-Only model (SM98) with its own SM117 head | uncalibrated (SM121) | 0.4767 | 0.0247 | - | Mac grade x card (0.5083) | -6.21% [-0.0372, -0.0262] | - | 0.779 | no |
| the Small Instant-Only model (SM98) with its own SM117 head | (a) Platt, static | 0.4747 | 0.0070 | -0.41% | Mac grade x card (0.5083) | -6.60% [-0.0391, -0.0281] | -6.68% [-0.0398, -0.0284] | 0.779 | yes |
| the Small Instant-Only model (SM98) with its own SM117 head | (b) shared transform, static | 0.4749 | 0.0104 | -0.37% | Mac grade x card (0.5083) | -6.56% [-0.0391, -0.0278] | - | 0.779 | no |
| the Small Instant-Only model (SM98) with its own SM117 head | (c) isotonic, static | 0.4756 | 0.0116 | -0.23% | Mac grade x card (0.5083) | -6.43% [-0.0382, -0.0273] | - | 0.779 | no |
| the Small Instant-Only model (SM98) with its own SM117 head | (a) Platt, rolling 7 days | 0.4748 | 0.0058 | -0.39% | Mac grade x card (0.5083) | -6.58% [-0.0392, -0.0280] | -6.84% [-0.0399, -0.0298] | 0.779 | yes |
| the Small Instant-Only model (SM98) with its own SM117 head | (b) shared transform, rolling 7 days | 0.4751 | 0.0116 | -0.33% | Mac grade x card (0.5083) | -6.53% [-0.0389, -0.0275] | - | 0.779 | no |
| the Small Instant-Only model (SM98) with its own SM117 head | (c) isotonic, rolling 7 days | 0.4757 | 0.0079 | -0.20% | Mac grade x card (0.5083) | -6.40% [-0.0382, -0.0272] | - | 0.778 | yes |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | uncalibrated (SM121) | 0.4767 | 0.0166 | - | Mac grade x card (0.5081) | -6.17% [-0.0370, -0.0260] | - | 0.777 | no |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (a) Platt, static | 0.4760 | 0.0084 | -0.14% | Mac grade x card (0.5081) | -6.31% [-0.0378, -0.0266] | -6.39% [-0.0384, -0.0269] | 0.777 | yes |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (b) shared transform, static | 0.4760 | 0.0087 | -0.16% | Mac grade x card (0.5081) | -6.32% [-0.0380, -0.0266] | - | 0.778 | yes |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (c) isotonic, static | 0.4772 | 0.0163 | +0.11% | Mac grade x card (0.5081) | -6.07% [-0.0366, -0.0254] | - | 0.777 | no |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (a) Platt, rolling 7 days | 0.4766 | 0.0126 | -0.02% | Mac grade x card (0.5081) | -6.19% [-0.0373, -0.0259] | -6.42% [-0.0380, -0.0278] | 0.777 | no |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (b) shared transform, rolling 7 days | 0.4764 | 0.0147 | -0.06% | Mac grade x card (0.5081) | -6.22% [-0.0376, -0.0260] | - | 0.777 | no |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (c) isotonic, rolling 7 days | 0.4772 | 0.0133 | +0.11% | Mac grade x card (0.5081) | -6.07% [-0.0368, -0.0253] | - | 0.776 | no |
| the longer cool-down for Tiny (SM115) with its own SM119 head | uncalibrated (SM121) | 0.4778 | 0.0263 | - | Mac grade x card (0.5081) | -5.97% [-0.0355, -0.0252] | - | 0.778 | no |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (a) Platt, static | 0.4755 | 0.0090 | -0.47% | Mac grade x card (0.5081) | -6.41% [-0.0379, -0.0271] | -6.48% [-0.0385, -0.0274] | 0.778 | yes |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (b) shared transform, static | 0.4754 | 0.0082 | -0.49% | Mac grade x card (0.5081) | -6.43% [-0.0381, -0.0271] | - | 0.778 | yes |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (c) isotonic, static | 0.4760 | 0.0102 | -0.36% | Mac grade x card (0.5081) | -6.31% [-0.0375, -0.0267] | - | 0.778 | no |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (a) Platt, rolling 7 days | 0.4756 | 0.0095 | -0.46% | Mac grade x card (0.5081) | -6.40% [-0.0379, -0.0271] | -6.62% [-0.0386, -0.0288] | 0.778 | yes |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (b) shared transform, rolling 7 days | 0.4755 | 0.0092 | -0.47% | Mac grade x card (0.5081) | -6.41% [-0.0381, -0.0271] | - | 0.778 | yes |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (c) isotonic, rolling 7 days | 0.4779 | 0.0105 | +0.02% | Mac grade x card (0.5081) | -5.95% [-0.0363, -0.0239] | - | 0.777 | no |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | uncalibrated (SM121) | 0.4796 | 0.0246 | - | Mac grade x card (0.5079) | -5.58% [-0.0340, -0.0228] | - | 0.776 | no |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (a) Platt, static | 0.4774 | 0.0086 | -0.45% | Mac grade x card (0.5079) | -6.00% [-0.0362, -0.0248] | -6.08% [-0.0366, -0.0251] | 0.776 | yes |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (b) shared transform, static | 0.4774 | 0.0099 | -0.47% | Mac grade x card (0.5079) | -6.02% [-0.0362, -0.0249] | - | 0.776 | yes |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (c) isotonic, static | 0.4811 | 0.0130 | +0.32% | Mac grade x card (0.5079) | -5.27% [-0.0338, -0.0194] | - | 0.775 | no |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (a) Platt, rolling 7 days | 0.4774 | 0.0079 | -0.47% | Mac grade x card (0.5079) | -6.02% [-0.0362, -0.0250] | -6.22% [-0.0366, -0.0269] | 0.776 | yes |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (b) shared transform, rolling 7 days | 0.4774 | 0.0098 | -0.45% | Mac grade x card (0.5079) | -6.01% [-0.0362, -0.0248] | - | 0.776 | yes |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (c) isotonic, rolling 7 days | 0.4801 | 0.0079 | +0.10% | Mac grade x card (0.5079) | -5.49% [-0.0341, -0.0214] | - | 0.774 | yes |

## ECE by S, benchmark rows, whole collection

### Good or Easy

| combination | method | 2 s | 3 s | 4 s | 5 s | 6 s | 8 s | 10 s |
|---|---|---|---|---|---|---|---|---|
| the longer Small Instant-Only model (SM113) with its own SM119 head | uncalibrated (SM121) | 0.0114 | 0.0178 | 0.0185 | 0.0307 | 0.0359 | 0.0353 | 0.0338 |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (a) Platt, static | 0.0148 | 0.0073* | 0.0128 | 0.0213 | 0.0245 | 0.0259 | 0.0254 |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (b) shared transform, static | 0.0066* | 0.0049* | 0.0147 | 0.0152 | 0.0161 | 0.0136 | 0.0089* |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (c) isotonic, static | 0.0122 | 0.0122 | 0.0156 | 0.0219 | 0.0229 | 0.0267 | 0.0260 |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (a) Platt, rolling 7 days | 0.0073* | 0.0032* | 0.0053* | 0.0063* | 0.0065* | 0.0066* | 0.0048* |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (b) shared transform, rolling 7 days | 0.0071* | 0.0073* | 0.0114 | 0.0150 | 0.0133 | 0.0104 | 0.0082* |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (c) isotonic, rolling 7 days | 0.0016* | 0.0033* | 0.0046* | 0.0054* | 0.0072* | 0.0067* | 0.0057* |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | uncalibrated (SM121) | 0.0106 | 0.0179 | 0.0240 | 0.0355 | 0.0397 | 0.0366 | 0.0336 |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (a) Platt, static | 0.0132 | 0.0036* | 0.0126 | 0.0216 | 0.0250 | 0.0265 | 0.0251 |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (b) shared transform, static | 0.0065* | 0.0077* | 0.0197 | 0.0187 | 0.0221 | 0.0173 | 0.0106 |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (c) isotonic, static | 0.0102 | 0.0060* | 0.0125 | 0.0212 | 0.0236 | 0.0254 | 0.0243 |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (a) Platt, rolling 7 days | 0.0066* | 0.0049* | 0.0046* | 0.0064* | 0.0064* | 0.0061* | 0.0051* |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (b) shared transform, rolling 7 days | 0.0063* | 0.0058* | 0.0080* | 0.0116 | 0.0119 | 0.0099* | 0.0087* |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (c) isotonic, rolling 7 days | 0.0012* | 0.0029* | 0.0047* | 0.0048* | 0.0063* | 0.0065* | 0.0052* |
| the Tiny two-head model (SM106) with its own SM117 head | uncalibrated (SM121) | 0.0131 | 0.0209 | 0.0166 | 0.0253 | 0.0290 | 0.0275 | 0.0261 |
| the Tiny two-head model (SM106) with its own SM117 head | (a) Platt, static | 0.0161 | 0.0118 | 0.0063* | 0.0138 | 0.0184 | 0.0222 | 0.0233 |
| the Tiny two-head model (SM106) with its own SM117 head | (b) shared transform, static | 0.0086* | 0.0109 | 0.0088* | 0.0078* | 0.0113 | 0.0111 | 0.0073* |
| the Tiny two-head model (SM106) with its own SM117 head | (c) isotonic, static | 0.0133 | 0.0143 | 0.0132 | 0.0168 | 0.0181 | 0.0214 | 0.0240 |
| the Tiny two-head model (SM106) with its own SM117 head | (a) Platt, rolling 7 days | 0.0076* | 0.0038* | 0.0049* | 0.0061* | 0.0062* | 0.0061* | 0.0053* |
| the Tiny two-head model (SM106) with its own SM117 head | (b) shared transform, rolling 7 days | 0.0084* | 0.0076* | 0.0070* | 0.0118 | 0.0098* | 0.0088* | 0.0067* |
| the Tiny two-head model (SM106) with its own SM117 head | (c) isotonic, rolling 7 days | 0.0019* | 0.0039* | 0.0042* | 0.0046* | 0.0068* | 0.0054* | 0.0055* |
| the show-time big model (SM91s2) with its own SM117 head | uncalibrated (SM121) | 0.0060* | 0.0136 | 0.0385 | 0.0410 | 0.0395 | 0.0311 | 0.0272 |
| the show-time big model (SM91s2) with its own SM117 head | (a) Platt, static | 0.0159 | 0.0083* | 0.0104 | 0.0144 | 0.0160 | 0.0178 | 0.0190 |
| the show-time big model (SM91s2) with its own SM117 head | (b) shared transform, static | 0.0078* | 0.0070* | 0.0129 | 0.0078* | 0.0106 | 0.0070* | 0.0039* |
| the show-time big model (SM91s2) with its own SM117 head | (c) isotonic, static | 0.0124 | 0.0137 | 0.0143 | 0.0166 | 0.0165 | 0.0191 | 0.0197 |
| the show-time big model (SM91s2) with its own SM117 head | (a) Platt, rolling 7 days | 0.0067* | 0.0065* | 0.0059* | 0.0059* | 0.0061* | 0.0042* | 0.0035* |
| the show-time big model (SM91s2) with its own SM117 head | (b) shared transform, rolling 7 days | 0.0077* | 0.0070* | 0.0099* | 0.0097* | 0.0077* | 0.0048* | 0.0041* |
| the show-time big model (SM91s2) with its own SM117 head | (c) isotonic, rolling 7 days | 0.0022* | 0.0035* | 0.0043* | 0.0053* | 0.0060* | 0.0061* | 0.0054* |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | uncalibrated (SM121) | 0.0081* | 0.0158 | 0.0360 | 0.0438 | 0.0450 | 0.0378 | 0.0325 |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (a) Platt, static | 0.0099* | 0.0039* | 0.0069* | 0.0127 | 0.0145 | 0.0136 | 0.0117 |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (b) shared transform, static | 0.0085* | 0.0135 | 0.0108 | 0.0101 | 0.0141 | 0.0112 | 0.0052* |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (c) isotonic, static | 0.0071* | 0.0068* | 0.0076* | 0.0147 | 0.0161 | 0.0144 | 0.0128 |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (a) Platt, rolling 7 days | 0.0060* | 0.0039* | 0.0054* | 0.0057* | 0.0060* | 0.0053* | 0.0047* |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (b) shared transform, rolling 7 days | 0.0069* | 0.0115 | 0.0080* | 0.0123 | 0.0101 | 0.0075* | 0.0054* |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (c) isotonic, rolling 7 days | 0.0014* | 0.0021* | 0.0045* | 0.0049* | 0.0067* | 0.0063* | 0.0050* |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | uncalibrated (SM121) | 0.0105 | 0.0177 | 0.0241 | 0.0360 | 0.0404 | 0.0375 | 0.0347 |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (a) Platt, static | 0.0132 | 0.0039* | 0.0136 | 0.0230 | 0.0266 | 0.0285 | 0.0273 |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (b) shared transform, static | 0.0066* | 0.0075* | 0.0207 | 0.0203 | 0.0240 | 0.0197 | 0.0133 |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (c) isotonic, static | 0.0101 | 0.0076* | 0.0132 | 0.0231 | 0.0257 | 0.0272 | 0.0254 |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (a) Platt, rolling 7 days | 0.0067* | 0.0049* | 0.0046* | 0.0065* | 0.0064* | 0.0059* | 0.0048* |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (b) shared transform, rolling 7 days | 0.0062* | 0.0058* | 0.0078* | 0.0117 | 0.0123 | 0.0104 | 0.0091* |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (c) isotonic, rolling 7 days | 0.0011* | 0.0030* | 0.0039* | 0.0052* | 0.0062* | 0.0062* | 0.0052* |
| the show-time retrain (SM92) with its own SM117 head | uncalibrated (SM121) | 0.0105 | 0.0178 | 0.0255 | 0.0373 | 0.0418 | 0.0389 | 0.0359 |
| the show-time retrain (SM92) with its own SM117 head | (a) Platt, static | 0.0128 | 0.0045* | 0.0155 | 0.0249 | 0.0280 | 0.0299 | 0.0286 |
| the show-time retrain (SM92) with its own SM117 head | (b) shared transform, static | 0.0063* | 0.0080* | 0.0221 | 0.0220 | 0.0258 | 0.0213 | 0.0140 |
| the show-time retrain (SM92) with its own SM117 head | (c) isotonic, static | 0.0097* | 0.0071* | 0.0143 | 0.0234 | 0.0263 | 0.0291 | 0.0255 |
| the show-time retrain (SM92) with its own SM117 head | (a) Platt, rolling 7 days | 0.0066* | 0.0054* | 0.0052* | 0.0064* | 0.0068* | 0.0060* | 0.0051* |
| the show-time retrain (SM92) with its own SM117 head | (b) shared transform, rolling 7 days | 0.0062* | 0.0053* | 0.0080* | 0.0116 | 0.0122 | 0.0106 | 0.0096* |
| the show-time retrain (SM92) with its own SM117 head | (c) isotonic, rolling 7 days | 0.0012* | 0.0031* | 0.0037* | 0.0047* | 0.0067* | 0.0060* | 0.0053* |
| the Small Instant-Only model (SM98) with its own SM117 head | uncalibrated (SM121) | 0.0125 | 0.0211 | 0.0177 | 0.0328 | 0.0394 | 0.0391 | 0.0367 |
| the Small Instant-Only model (SM98) with its own SM117 head | (a) Platt, static | 0.0147 | 0.0096* | 0.0112 | 0.0189 | 0.0230 | 0.0238 | 0.0228 |
| the Small Instant-Only model (SM98) with its own SM117 head | (b) shared transform, static | 0.0076* | 0.0098* | 0.0124 | 0.0167 | 0.0148 | 0.0123 | 0.0071* |
| the Small Instant-Only model (SM98) with its own SM117 head | (c) isotonic, static | 0.0108 | 0.0115 | 0.0142 | 0.0199 | 0.0217 | 0.0243 | 0.0226 |
| the Small Instant-Only model (SM98) with its own SM117 head | (a) Platt, rolling 7 days | 0.0076* | 0.0035* | 0.0051* | 0.0074* | 0.0080* | 0.0060* | 0.0057* |
| the Small Instant-Only model (SM98) with its own SM117 head | (b) shared transform, rolling 7 days | 0.0073* | 0.0099* | 0.0136 | 0.0172 | 0.0149 | 0.0100* | 0.0079* |
| the Small Instant-Only model (SM98) with its own SM117 head | (c) isotonic, rolling 7 days | 0.0021* | 0.0037* | 0.0042* | 0.0046* | 0.0063* | 0.0065* | 0.0055* |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | uncalibrated (SM121) | 0.0132 | 0.0205 | 0.0167 | 0.0257 | 0.0300 | 0.0290 | 0.0278 |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (a) Platt, static | 0.0164 | 0.0124 | 0.0081* | 0.0153 | 0.0194 | 0.0234 | 0.0248 |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (b) shared transform, static | 0.0087* | 0.0109 | 0.0084* | 0.0096* | 0.0116 | 0.0111 | 0.0077* |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (c) isotonic, static | 0.0134 | 0.0151 | 0.0120 | 0.0181 | 0.0204 | 0.0236 | 0.0260 |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (a) Platt, rolling 7 days | 0.0075* | 0.0037* | 0.0054* | 0.0058* | 0.0051* | 0.0056* | 0.0047* |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (b) shared transform, rolling 7 days | 0.0083* | 0.0077* | 0.0066* | 0.0128 | 0.0102 | 0.0086* | 0.0069* |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (c) isotonic, rolling 7 days | 0.0019* | 0.0030* | 0.0039* | 0.0057* | 0.0068* | 0.0060* | 0.0053* |
| the longer cool-down for Tiny (SM115) with its own SM119 head | uncalibrated (SM121) | 0.0120 | 0.0181 | 0.0233 | 0.0328 | 0.0367 | 0.0341 | 0.0318 |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (a) Platt, static | 0.0121 | 0.0073* | 0.0161 | 0.0218 | 0.0243 | 0.0266 | 0.0284 |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (b) shared transform, static | 0.0072* | 0.0104 | 0.0220 | 0.0179 | 0.0185 | 0.0136 | 0.0084* |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (c) isotonic, static | 0.0090* | 0.0095* | 0.0161 | 0.0208 | 0.0219 | 0.0235 | 0.0267 |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (a) Platt, rolling 7 days | 0.0075* | 0.0032* | 0.0066* | 0.0063* | 0.0075* | 0.0072* | 0.0063* |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (b) shared transform, rolling 7 days | 0.0072* | 0.0039* | 0.0099* | 0.0104 | 0.0101 | 0.0071* | 0.0061* |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (c) isotonic, rolling 7 days | 0.0019* | 0.0037* | 0.0042* | 0.0039* | 0.0059* | 0.0053* | 0.0059* |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | uncalibrated (SM121) | 0.0052* | 0.0158 | 0.0381 | 0.0411 | 0.0401 | 0.0327 | 0.0294 |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (a) Platt, static | 0.0099* | 0.0103 | 0.0107 | 0.0125 | 0.0134 | 0.0144 | 0.0164 |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (b) shared transform, static | 0.0063* | 0.0081* | 0.0101 | 0.0113 | 0.0097* | 0.0078* | 0.0053* |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (c) isotonic, static | 0.0085* | 0.0162 | 0.0165 | 0.0157 | 0.0146 | 0.0156 | 0.0164 |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (a) Platt, rolling 7 days | 0.0055* | 0.0060* | 0.0049* | 0.0056* | 0.0057* | 0.0050* | 0.0051* |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (b) shared transform, rolling 7 days | 0.0069* | 0.0090* | 0.0062* | 0.0139 | 0.0109 | 0.0070* | 0.0057* |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (c) isotonic, rolling 7 days | 0.0018* | 0.0032* | 0.0036* | 0.0047* | 0.0052* | 0.0053* | 0.0048* |

### Easy

| combination | method | 2 s | 3 s | 4 s | 5 s | 6 s | 8 s | 10 s |
|---|---|---|---|---|---|---|---|---|
| the longer Small Instant-Only model (SM113) with its own SM119 head | uncalibrated (SM121) | 0.0003* | 0.0004* | 0.0004* | 0.0005* | 0.0005* | 0.0007* | 0.0006 |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (a) Platt, static | 0.0002* | 0.0004* | 0.0007* | 0.0005* | 0.0004* | 0.0008 | 0.0004 |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (b) shared transform, static | 0.0002* | 0.0002* | 0.0003* | 0.0004* | 0.0004* | 0.0005* | 0.0006 |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (c) isotonic, static | 0.0003 | 0.0005 | 0.0006 | 0.0006 | 0.0004 | 0.0005 | 0.0009 |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (a) Platt, rolling 7 days | 0.0002* | 0.0002* | 0.0003* | 0.0003* | 0.0002* | 0.0004* | 0.0003 |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (b) shared transform, rolling 7 days | 0.0002* | 0.0002* | 0.0003* | 0.0003* | 0.0004* | 0.0004* | 0.0004 |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (c) isotonic, rolling 7 days | 0.0003 | 0.0003 | 0.0005 | 0.0005 | 0.0007 | 0.0008 | 0.0009 |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | uncalibrated (SM121) | 0.0003* | 0.0002* | 0.0003* | 0.0005* | 0.0006* | 0.0006 | 0.0006 |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (a) Platt, static | 0.0003* | 0.0006* | 0.0008* | 0.0006* | 0.0005* | 0.0007 | 0.0004 |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (b) shared transform, static | 0.0002* | 0.0004* | 0.0005* | 0.0006* | 0.0005* | 0.0006 | 0.0006 |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (c) isotonic, static | 0.0003 | 0.0007 | 0.0011 | 0.0005 | 0.0005 | 0.0005 | 0.0009 |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (a) Platt, rolling 7 days | 0.0002* | 0.0002* | 0.0003* | 0.0004* | 0.0004* | 0.0003 | 0.0003 |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (b) shared transform, rolling 7 days | 0.0002* | 0.0003* | 0.0004* | 0.0003* | 0.0004* | 0.0004 | 0.0004 |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (c) isotonic, rolling 7 days | 0.0003 | 0.0002 | 0.0004 | 0.0006 | 0.0008 | 0.0008 | 0.0010 |
| the Tiny two-head model (SM106) with its own SM117 head | uncalibrated (SM121) | 0.0003* | 0.0004* | 0.0004* | 0.0004* | 0.0004* | 0.0004* | 0.0004* |
| the Tiny two-head model (SM106) with its own SM117 head | (a) Platt, static | 0.0004* | 0.0006* | 0.0011* | 0.0008* | 0.0008* | 0.0009* | 0.0007 |
| the Tiny two-head model (SM106) with its own SM117 head | (b) shared transform, static | 0.0002* | 0.0005* | 0.0006* | 0.0006* | 0.0007* | 0.0006* | 0.0007 |
| the Tiny two-head model (SM106) with its own SM117 head | (c) isotonic, static | 0.0003 | 0.0005 | 0.0010 | 0.0006 | 0.0008 | 0.0007 | 0.0006 |
| the Tiny two-head model (SM106) with its own SM117 head | (a) Platt, rolling 7 days | 0.0002* | 0.0003* | 0.0003* | 0.0004* | 0.0005* | 0.0004* | 0.0003 |
| the Tiny two-head model (SM106) with its own SM117 head | (b) shared transform, rolling 7 days | 0.0003* | 0.0005* | 0.0006* | 0.0006* | 0.0006* | 0.0005* | 0.0005 |
| the Tiny two-head model (SM106) with its own SM117 head | (c) isotonic, rolling 7 days | 0.0003 | 0.0003 | 0.0004 | 0.0005 | 0.0007 | 0.0008 | 0.0008 |
| the show-time big model (SM91s2) with its own SM117 head | uncalibrated (SM121) | 0.0002* | 0.0002* | 0.0002* | 0.0003* | 0.0004* | 0.0005* | 0.0005 |
| the show-time big model (SM91s2) with its own SM117 head | (a) Platt, static | 0.0003* | 0.0005* | 0.0009* | 0.0005* | 0.0003* | 0.0004* | 0.0004 |
| the show-time big model (SM91s2) with its own SM117 head | (b) shared transform, static | 0.0005* | 0.0004* | 0.0003* | 0.0003* | 0.0003* | 0.0004* | 0.0006 |
| the show-time big model (SM91s2) with its own SM117 head | (c) isotonic, static | 0.0003 | 0.0004* | 0.0005 | 0.0009 | 0.0011 | 0.0010 | 0.0010 |
| the show-time big model (SM91s2) with its own SM117 head | (a) Platt, rolling 7 days | 0.0002* | 0.0002* | 0.0003* | 0.0003* | 0.0003* | 0.0004 | 0.0004 |
| the show-time big model (SM91s2) with its own SM117 head | (b) shared transform, rolling 7 days | 0.0002* | 0.0003* | 0.0003* | 0.0005* | 0.0006* | 0.0007 | 0.0007 |
| the show-time big model (SM91s2) with its own SM117 head | (c) isotonic, rolling 7 days | 0.0004 | 0.0005 | 0.0006 | 0.0007 | 0.0008 | 0.0009 | 0.0010 |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | uncalibrated (SM121) | 0.0004* | 0.0005* | 0.0003* | 0.0004* | 0.0006* | 0.0006 | 0.0006 |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (a) Platt, static | 0.0004* | 0.0007* | 0.0009* | 0.0006* | 0.0005* | 0.0007* | 0.0004 |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (b) shared transform, static | 0.0003* | 0.0006* | 0.0006* | 0.0007* | 0.0005* | 0.0005* | 0.0006 |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (c) isotonic, static | 0.0003 | 0.0009 | 0.0009 | 0.0006 | 0.0006 | 0.0005 | 0.0008 |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (a) Platt, rolling 7 days | 0.0002* | 0.0003* | 0.0003* | 0.0004* | 0.0004* | 0.0003 | 0.0003 |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (b) shared transform, rolling 7 days | 0.0003* | 0.0005* | 0.0005* | 0.0005* | 0.0004* | 0.0004 | 0.0004 |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (c) isotonic, rolling 7 days | 0.0002 | 0.0003* | 0.0004 | 0.0006 | 0.0007 | 0.0008 | 0.0009 |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | uncalibrated (SM121) | 0.0003* | 0.0002* | 0.0003* | 0.0005* | 0.0006* | 0.0007 | 0.0006 |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (a) Platt, static | 0.0004* | 0.0006* | 0.0009* | 0.0006* | 0.0006* | 0.0008 | 0.0005 |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (b) shared transform, static | 0.0002* | 0.0004* | 0.0005* | 0.0006* | 0.0005* | 0.0007 | 0.0007 |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (c) isotonic, static | 0.0003 | 0.0006 | 0.0009 | 0.0006 | 0.0004 | 0.0005 | 0.0008 |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (a) Platt, rolling 7 days | 0.0002* | 0.0003* | 0.0002* | 0.0004* | 0.0004* | 0.0004 | 0.0003 |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (b) shared transform, rolling 7 days | 0.0002* | 0.0003* | 0.0003* | 0.0004* | 0.0004* | 0.0004 | 0.0004 |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (c) isotonic, rolling 7 days | 0.0003 | 0.0003 | 0.0007 | 0.0006 | 0.0008 | 0.0008 | 0.0009 |
| the show-time retrain (SM92) with its own SM117 head | uncalibrated (SM121) | 0.0003* | 0.0004* | 0.0003* | 0.0003* | 0.0003* | 0.0003* | 0.0003* |
| the show-time retrain (SM92) with its own SM117 head | (a) Platt, static | 0.0004* | 0.0007* | 0.0009* | 0.0007* | 0.0005* | 0.0007 | 0.0005 |
| the show-time retrain (SM92) with its own SM117 head | (b) shared transform, static | 0.0002* | 0.0006* | 0.0006* | 0.0006* | 0.0005* | 0.0006* | 0.0006 |
| the show-time retrain (SM92) with its own SM117 head | (c) isotonic, static | 0.0004 | 0.0008 | 0.0010 | 0.0006 | 0.0007 | 0.0006 | 0.0008 |
| the show-time retrain (SM92) with its own SM117 head | (a) Platt, rolling 7 days | 0.0002* | 0.0003* | 0.0002* | 0.0003* | 0.0003* | 0.0003 | 0.0003 |
| the show-time retrain (SM92) with its own SM117 head | (b) shared transform, rolling 7 days | 0.0003* | 0.0004* | 0.0005* | 0.0005* | 0.0004* | 0.0006 | 0.0005 |
| the show-time retrain (SM92) with its own SM117 head | (c) isotonic, rolling 7 days | 0.0003 | 0.0003 | 0.0005 | 0.0005 | 0.0007 | 0.0010 | 0.0011 |
| the Small Instant-Only model (SM98) with its own SM117 head | uncalibrated (SM121) | 0.0002* | 0.0003* | 0.0003* | 0.0005* | 0.0005* | 0.0005* | 0.0005 |
| the Small Instant-Only model (SM98) with its own SM117 head | (a) Platt, static | 0.0002* | 0.0005* | 0.0007* | 0.0005* | 0.0005* | 0.0006* | 0.0003 |
| the Small Instant-Only model (SM98) with its own SM117 head | (b) shared transform, static | 0.0002* | 0.0003* | 0.0003* | 0.0005* | 0.0005* | 0.0005* | 0.0006 |
| the Small Instant-Only model (SM98) with its own SM117 head | (c) isotonic, static | 0.0004 | 0.0007 | 0.0007 | 0.0006 | 0.0006 | 0.0008 | 0.0009 |
| the Small Instant-Only model (SM98) with its own SM117 head | (a) Platt, rolling 7 days | 0.0002* | 0.0002* | 0.0002* | 0.0003* | 0.0004* | 0.0003* | 0.0002 |
| the Small Instant-Only model (SM98) with its own SM117 head | (b) shared transform, rolling 7 days | 0.0002* | 0.0003* | 0.0003* | 0.0004* | 0.0005* | 0.0003* | 0.0003 |
| the Small Instant-Only model (SM98) with its own SM117 head | (c) isotonic, rolling 7 days | 0.0003 | 0.0003 | 0.0005 | 0.0006 | 0.0007 | 0.0008 | 0.0009 |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | uncalibrated (SM121) | 0.0003* | 0.0003* | 0.0003* | 0.0004* | 0.0004* | 0.0005* | 0.0003 |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (a) Platt, static | 0.0003* | 0.0006* | 0.0009* | 0.0006* | 0.0005* | 0.0009 | 0.0006 |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (b) shared transform, static | 0.0002* | 0.0004* | 0.0004* | 0.0006* | 0.0005* | 0.0006 | 0.0008 |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (c) isotonic, static | 0.0003 | 0.0003 | 0.0006 | 0.0007 | 0.0006 | 0.0005 | 0.0009 |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (a) Platt, rolling 7 days | 0.0002* | 0.0003* | 0.0002* | 0.0004* | 0.0005* | 0.0004* | 0.0003 |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (b) shared transform, rolling 7 days | 0.0002* | 0.0004* | 0.0004* | 0.0004* | 0.0004* | 0.0004 | 0.0003 |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (c) isotonic, rolling 7 days | 0.0003 | 0.0003 | 0.0004 | 0.0005 | 0.0008 | 0.0009 | 0.0010 |
| the longer cool-down for Tiny (SM115) with its own SM119 head | uncalibrated (SM121) | 0.0003* | 0.0003* | 0.0004* | 0.0003* | 0.0003* | 0.0005* | 0.0004 |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (a) Platt, static | 0.0004* | 0.0008* | 0.0011* | 0.0007* | 0.0007* | 0.0010 | 0.0007 |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (b) shared transform, static | 0.0002* | 0.0005* | 0.0008* | 0.0007* | 0.0007* | 0.0008 | 0.0010 |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (c) isotonic, static | 0.0005 | 0.0010 | 0.0010 | 0.0006 | 0.0004 | 0.0006 | 0.0007 |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (a) Platt, rolling 7 days | 0.0002* | 0.0003* | 0.0003* | 0.0004* | 0.0004* | 0.0004* | 0.0003 |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (b) shared transform, rolling 7 days | 0.0002* | 0.0004* | 0.0004* | 0.0005* | 0.0004* | 0.0004* | 0.0003 |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (c) isotonic, rolling 7 days | 0.0003 | 0.0003 | 0.0005 | 0.0006 | 0.0008 | 0.0008 | 0.0010 |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | uncalibrated (SM121) | 0.0002* | 0.0003* | 0.0004* | 0.0006* | 0.0005* | 0.0007 | 0.0008 |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (a) Platt, static | 0.0003* | 0.0005* | 0.0007* | 0.0004* | 0.0003* | 0.0004* | 0.0004* |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (b) shared transform, static | 0.0005* | 0.0003* | 0.0003* | 0.0004* | 0.0003* | 0.0004* | 0.0005* |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (c) isotonic, static | 0.0003 | 0.0004* | 0.0006 | 0.0010 | 0.0012 | 0.0010 | 0.0011 |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (a) Platt, rolling 7 days | 0.0002* | 0.0002* | 0.0002* | 0.0003* | 0.0004* | 0.0004* | 0.0004 |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (b) shared transform, rolling 7 days | 0.0002* | 0.0003* | 0.0003* | 0.0003* | 0.0003* | 0.0003* | 0.0003* |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (c) isotonic, rolling 7 days | 0.0003 | 0.0004 | 0.0006 | 0.0006 | 0.0007 | 0.0008 | 0.0009 |

### recalled

| combination | method | 2 s | 3 s | 4 s | 5 s | 6 s | 8 s | 10 s |
|---|---|---|---|---|---|---|---|---|
| the longer Small Instant-Only model (SM113) with its own SM119 head | uncalibrated (SM121) | 0.0113 | 0.0175 | 0.0175 | 0.0273 | 0.0312 | 0.0308 | 0.0315 |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (a) Platt, static | 0.0152 | 0.0082* | 0.0116 | 0.0183 | 0.0214 | 0.0231 | 0.0233 |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (b) shared transform, static | 0.0075* | 0.0052* | 0.0142 | 0.0152 | 0.0141 | 0.0124 | 0.0105 |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (c) isotonic, static | 0.0133 | 0.0135 | 0.0154 | 0.0195 | 0.0188 | 0.0210 | 0.0221 |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (a) Platt, rolling 7 days | 0.0089* | 0.0033* | 0.0057* | 0.0070* | 0.0074* | 0.0066* | 0.0063* |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (b) shared transform, rolling 7 days | 0.0077* | 0.0077* | 0.0114 | 0.0152 | 0.0130 | 0.0108 | 0.0117 |
| the longer Small Instant-Only model (SM113) with its own SM119 head | (c) isotonic, rolling 7 days | 0.0019* | 0.0034* | 0.0044* | 0.0053* | 0.0069* | 0.0063* | 0.0061* |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | uncalibrated (SM121) | 0.0105 | 0.0172 | 0.0236 | 0.0324 | 0.0352 | 0.0321 | 0.0310 |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (a) Platt, static | 0.0136 | 0.0038* | 0.0118 | 0.0181 | 0.0205 | 0.0222 | 0.0227 |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (b) shared transform, static | 0.0073* | 0.0075* | 0.0198 | 0.0171 | 0.0196 | 0.0166 | 0.0128 |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (c) isotonic, static | 0.0114 | 0.0074* | 0.0116 | 0.0178 | 0.0192 | 0.0210 | 0.0207 |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (a) Platt, rolling 7 days | 0.0080* | 0.0045* | 0.0052* | 0.0067* | 0.0067* | 0.0070* | 0.0054* |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (b) shared transform, rolling 7 days | 0.0068* | 0.0060* | 0.0078* | 0.0122 | 0.0117 | 0.0111 | 0.0117 |
| Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head | (c) isotonic, rolling 7 days | 0.0015* | 0.0031* | 0.0037* | 0.0055* | 0.0062* | 0.0069* | 0.0062* |
| the Tiny two-head model (SM106) with its own SM117 head | uncalibrated (SM121) | 0.0128 | 0.0206 | 0.0157 | 0.0218 | 0.0231 | 0.0216 | 0.0221 |
| the Tiny two-head model (SM106) with its own SM117 head | (a) Platt, static | 0.0167 | 0.0127 | 0.0072* | 0.0107 | 0.0142 | 0.0175 | 0.0195 |
| the Tiny two-head model (SM106) with its own SM117 head | (b) shared transform, static | 0.0092* | 0.0109 | 0.0081* | 0.0074* | 0.0076* | 0.0090* | 0.0079* |
| the Tiny two-head model (SM106) with its own SM117 head | (c) isotonic, static | 0.0144 | 0.0160 | 0.0142 | 0.0158 | 0.0160 | 0.0170 | 0.0187 |
| the Tiny two-head model (SM106) with its own SM117 head | (a) Platt, rolling 7 days | 0.0091* | 0.0045* | 0.0056* | 0.0056* | 0.0058* | 0.0070* | 0.0065* |
| the Tiny two-head model (SM106) with its own SM117 head | (b) shared transform, rolling 7 days | 0.0088* | 0.0079* | 0.0072* | 0.0126 | 0.0093* | 0.0090* | 0.0102 |
| the Tiny two-head model (SM106) with its own SM117 head | (c) isotonic, rolling 7 days | 0.0026* | 0.0040* | 0.0057* | 0.0055* | 0.0064* | 0.0067* | 0.0057* |
| the show-time big model (SM91s2) with its own SM117 head | uncalibrated (SM121) | 0.0073* | 0.0140 | 0.0380 | 0.0381 | 0.0351 | 0.0265 | 0.0242 |
| the show-time big model (SM91s2) with its own SM117 head | (a) Platt, static | 0.0172 | 0.0092* | 0.0099* | 0.0121 | 0.0123 | 0.0145 | 0.0160 |
| the show-time big model (SM91s2) with its own SM117 head | (b) shared transform, static | 0.0091* | 0.0064* | 0.0142 | 0.0081* | 0.0105 | 0.0095* | 0.0069* |
| the show-time big model (SM91s2) with its own SM117 head | (c) isotonic, static | 0.0140 | 0.0151 | 0.0136 | 0.0149 | 0.0142 | 0.0156 | 0.0164 |
| the show-time big model (SM91s2) with its own SM117 head | (a) Platt, rolling 7 days | 0.0083* | 0.0066* | 0.0056* | 0.0053* | 0.0050* | 0.0055* | 0.0051* |
| the show-time big model (SM91s2) with its own SM117 head | (b) shared transform, rolling 7 days | 0.0091* | 0.0074* | 0.0096* | 0.0101 | 0.0076* | 0.0062* | 0.0059* |
| the show-time big model (SM91s2) with its own SM117 head | (c) isotonic, rolling 7 days | 0.0031* | 0.0042* | 0.0049* | 0.0052* | 0.0058* | 0.0055* | 0.0054* |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | uncalibrated (SM121) | 0.0086* | 0.0154 | 0.0351 | 0.0404 | 0.0400 | 0.0328 | 0.0294 |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (a) Platt, static | 0.0105 | 0.0043* | 0.0060* | 0.0098* | 0.0099* | 0.0097* | 0.0100* |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (b) shared transform, static | 0.0093* | 0.0134 | 0.0109 | 0.0094* | 0.0116 | 0.0098* | 0.0071* |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (c) isotonic, static | 0.0087* | 0.0078* | 0.0070* | 0.0114 | 0.0129 | 0.0124 | 0.0116 |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (a) Platt, rolling 7 days | 0.0073* | 0.0043* | 0.0054* | 0.0058* | 0.0056* | 0.0056* | 0.0050* |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (b) shared transform, rolling 7 days | 0.0078* | 0.0117 | 0.0087* | 0.0126 | 0.0102 | 0.0082* | 0.0092* |
| the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92) | (c) isotonic, rolling 7 days | 0.0016* | 0.0025* | 0.0048* | 0.0062* | 0.0066* | 0.0074* | 0.0061* |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | uncalibrated (SM121) | 0.0106 | 0.0171 | 0.0238 | 0.0331 | 0.0361 | 0.0334 | 0.0325 |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (a) Platt, static | 0.0137 | 0.0038* | 0.0125 | 0.0197 | 0.0222 | 0.0243 | 0.0250 |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (b) shared transform, static | 0.0073* | 0.0075* | 0.0207 | 0.0186 | 0.0214 | 0.0185 | 0.0151 |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (c) isotonic, static | 0.0113 | 0.0083* | 0.0122 | 0.0193 | 0.0221 | 0.0220 | 0.0241 |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (a) Platt, rolling 7 days | 0.0082* | 0.0044* | 0.0052* | 0.0068* | 0.0071* | 0.0071* | 0.0062* |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (b) shared transform, rolling 7 days | 0.0068* | 0.0061* | 0.0079* | 0.0126 | 0.0118 | 0.0117 | 0.0123 |
| Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head | (c) isotonic, rolling 7 days | 0.0015* | 0.0034* | 0.0042* | 0.0052* | 0.0066* | 0.0068* | 0.0060* |
| the show-time retrain (SM92) with its own SM117 head | uncalibrated (SM121) | 0.0103 | 0.0173 | 0.0249 | 0.0343 | 0.0374 | 0.0345 | 0.0335 |
| the show-time retrain (SM92) with its own SM117 head | (a) Platt, static | 0.0132 | 0.0044* | 0.0146 | 0.0215 | 0.0233 | 0.0253 | 0.0256 |
| the show-time retrain (SM92) with its own SM117 head | (b) shared transform, static | 0.0072* | 0.0078* | 0.0222 | 0.0205 | 0.0234 | 0.0205 | 0.0163 |
| the show-time retrain (SM92) with its own SM117 head | (c) isotonic, static | 0.0107 | 0.0081* | 0.0126 | 0.0200 | 0.0214 | 0.0243 | 0.0230 |
| the show-time retrain (SM92) with its own SM117 head | (a) Platt, rolling 7 days | 0.0080* | 0.0049* | 0.0054* | 0.0074* | 0.0071* | 0.0070* | 0.0054* |
| the show-time retrain (SM92) with its own SM117 head | (b) shared transform, rolling 7 days | 0.0068* | 0.0056* | 0.0080* | 0.0120 | 0.0121 | 0.0119 | 0.0127 |
| the show-time retrain (SM92) with its own SM117 head | (c) isotonic, rolling 7 days | 0.0015* | 0.0034* | 0.0039* | 0.0049* | 0.0065* | 0.0065* | 0.0059* |
| the Small Instant-Only model (SM98) with its own SM117 head | uncalibrated (SM121) | 0.0124 | 0.0207 | 0.0170 | 0.0298 | 0.0350 | 0.0350 | 0.0346 |
| the Small Instant-Only model (SM98) with its own SM117 head | (a) Platt, static | 0.0152 | 0.0104 | 0.0113 | 0.0177 | 0.0206 | 0.0215 | 0.0209 |
| the Small Instant-Only model (SM98) with its own SM117 head | (b) shared transform, static | 0.0088* | 0.0097* | 0.0128 | 0.0170 | 0.0151 | 0.0120 | 0.0090* |
| the Small Instant-Only model (SM98) with its own SM117 head | (c) isotonic, static | 0.0120 | 0.0140 | 0.0141 | 0.0172 | 0.0196 | 0.0212 | 0.0205 |
| the Small Instant-Only model (SM98) with its own SM117 head | (a) Platt, rolling 7 days | 0.0093* | 0.0044* | 0.0055* | 0.0074* | 0.0081* | 0.0068* | 0.0060* |
| the Small Instant-Only model (SM98) with its own SM117 head | (b) shared transform, rolling 7 days | 0.0081* | 0.0100* | 0.0139 | 0.0173 | 0.0146 | 0.0108 | 0.0112 |
| the Small Instant-Only model (SM98) with its own SM117 head | (c) isotonic, rolling 7 days | 0.0024* | 0.0040* | 0.0046* | 0.0055* | 0.0071* | 0.0066* | 0.0062* |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | uncalibrated (SM121) | 0.0128 | 0.0206 | 0.0156 | 0.0223 | 0.0240 | 0.0230 | 0.0237 |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (a) Platt, static | 0.0169 | 0.0133 | 0.0091* | 0.0127 | 0.0160 | 0.0190 | 0.0215 |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (b) shared transform, static | 0.0092* | 0.0108 | 0.0080* | 0.0086* | 0.0085* | 0.0099* | 0.0084* |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (c) isotonic, static | 0.0144 | 0.0170 | 0.0133 | 0.0179 | 0.0175 | 0.0189 | 0.0210 |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (a) Platt, rolling 7 days | 0.0091* | 0.0041* | 0.0060* | 0.0061* | 0.0056* | 0.0070* | 0.0053* |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (b) shared transform, rolling 7 days | 0.0088* | 0.0080* | 0.0070* | 0.0131 | 0.0102 | 0.0094* | 0.0101 |
| Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head | (c) isotonic, rolling 7 days | 0.0026* | 0.0039* | 0.0051* | 0.0061* | 0.0066* | 0.0069* | 0.0064* |
| the longer cool-down for Tiny (SM115) with its own SM119 head | uncalibrated (SM121) | 0.0122 | 0.0180 | 0.0221 | 0.0295 | 0.0314 | 0.0288 | 0.0285 |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (a) Platt, static | 0.0132 | 0.0079* | 0.0148 | 0.0181 | 0.0194 | 0.0227 | 0.0250 |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (b) shared transform, static | 0.0081* | 0.0112 | 0.0215 | 0.0159 | 0.0155 | 0.0113 | 0.0088* |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (c) isotonic, static | 0.0108 | 0.0104 | 0.0149 | 0.0158 | 0.0185 | 0.0183 | 0.0216 |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (a) Platt, rolling 7 days | 0.0092* | 0.0033* | 0.0063* | 0.0064* | 0.0079* | 0.0082* | 0.0067* |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (b) shared transform, rolling 7 days | 0.0077* | 0.0045* | 0.0096* | 0.0105 | 0.0095* | 0.0089* | 0.0087* |
| the longer cool-down for Tiny (SM115) with its own SM119 head | (c) isotonic, rolling 7 days | 0.0022* | 0.0034* | 0.0039* | 0.0038* | 0.0045* | 0.0050* | 0.0059* |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | uncalibrated (SM121) | 0.0065* | 0.0150 | 0.0371 | 0.0377 | 0.0353 | 0.0280 | 0.0264 |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (a) Platt, static | 0.0112 | 0.0106 | 0.0097* | 0.0099* | 0.0111 | 0.0126 | 0.0150 |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (b) shared transform, static | 0.0076* | 0.0075* | 0.0111 | 0.0121 | 0.0107 | 0.0108 | 0.0101 |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (c) isotonic, static | 0.0106 | 0.0167 | 0.0154 | 0.0124 | 0.0112 | 0.0135 | 0.0147 |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (a) Platt, rolling 7 days | 0.0069* | 0.0061* | 0.0043* | 0.0053* | 0.0062* | 0.0061* | 0.0056* |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (b) shared transform, rolling 7 days | 0.0081* | 0.0091* | 0.0066* | 0.0143 | 0.0112 | 0.0088* | 0.0087* |
| the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it) | (c) isotonic, rolling 7 days | 0.0027* | 0.0033* | 0.0042* | 0.0055* | 0.0049* | 0.0060* | 0.0050* |

`*` = reliable after recalibration (ECE below 0.01 and every baseline beaten).

## Fits

- the longer Small Instant-Only model (SM113) with its own SM119 head: static shared transform on the benchmark window: grade temperature 1.009, time scale (Hard, Good, Easy) 1.000, 0.858, 1.086, shift (ln s) +0.000, -0.022, +0.036; Platt at Good or Easy 4 s (a, b) [1.1026, 0.1199]; repair share (rows x events changed) Platt 0.0374, isotonic 0.1818; rolling: 99 refits, grade temperature 0.9768-1.0692; fits checked past-only: 101.
- Small pair: grades of the Small Instant-Only model (SM98), time from the show-time retrain (SM92)'s SM117 head: static shared transform on the benchmark window: grade temperature 1.011, time scale (Hard, Good, Easy) 1.000, 0.895, 1.163, shift (ln s) +0.000, -0.017, +0.029; Platt at Good or Easy 4 s (a, b) [1.0696, 0.1141]; repair share (rows x events changed) Platt 0.0409, isotonic 0.1941; rolling: 99 refits, grade temperature 0.9752-1.0626; fits checked past-only: 101.
- the Tiny two-head model (SM106) with its own SM117 head: static shared transform on the benchmark window: grade temperature 1.017, time scale (Hard, Good, Easy) 1.000, 0.890, 1.112, shift (ln s) +0.000, -0.023, -0.003; Platt at Good or Easy 4 s (a, b) [1.0958, 0.1545]; repair share (rows x events changed) Platt 0.0397, isotonic 0.1927; rolling: 99 refits, grade temperature 0.9604-1.065; fits checked past-only: 101.
- the show-time big model (SM91s2) with its own SM117 head: static shared transform on the benchmark window: grade temperature 1.012, time scale (Hard, Good, Easy) 1.000, 0.947, 1.182, shift (ln s) +0.000, -0.049, -0.053; Platt at Good or Easy 4 s (a, b) [1.0082, 0.2083]; repair share (rows x events changed) Platt 0.0767, isotonic 0.2319; rolling: 99 refits, grade temperature 1.0033-1.1279; fits checked past-only: 101.
- the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92): static shared transform on the benchmark window: grade temperature 1.011, time scale (Hard, Good, Easy) 1.000, 0.921, 1.127, shift (ln s) +0.000, -0.055, +0.016; Platt at Good or Easy 4 s (a, b) [1.0202, 0.1972]; repair share (rows x events changed) Platt 0.0350, isotonic 0.1709; rolling: 99 refits, grade temperature 0.9752-1.0626; fits checked past-only: 101.
- Small pair as the fork runs it today: grades of the longer Small Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head: static shared transform on the benchmark window: grade temperature 1.009, time scale (Hard, Good, Easy) 1.000, 0.895, 1.163, shift (ln s) +0.000, -0.017, +0.029; Platt at Good or Easy 4 s (a, b) [1.0699, 0.1108]; repair share (rows x events changed) Platt 0.0397, isotonic 0.1801; rolling: 99 refits, grade temperature 0.9768-1.0692; fits checked past-only: 101.
- the show-time retrain (SM92) with its own SM117 head: static shared transform on the benchmark window: grade temperature 1.020, time scale (Hard, Good, Easy) 1.000, 0.895, 1.163, shift (ln s) +0.000, -0.017, +0.029; Platt at Good or Easy 4 s (a, b) [1.0606, 0.0991]; repair share (rows x events changed) Platt 0.0355, isotonic 0.1863; rolling: 99 refits, grade temperature 0.954-1.0483; fits checked past-only: 101.
- the Small Instant-Only model (SM98) with its own SM117 head: static shared transform on the benchmark window: grade temperature 1.011, time scale (Hard, Good, Easy) 1.000, 0.848, 1.094, shift (ln s) +0.000, -0.032, +0.010; Platt at Good or Easy 4 s (a, b) [1.1049, 0.1458]; repair share (rows x events changed) Platt 0.0425, isotonic 0.1861; rolling: 99 refits, grade temperature 0.9752-1.0626; fits checked past-only: 101.
- Tiny pair as the fork runs it today: grades of the longer cool-down for Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head: static shared transform on the benchmark window: grade temperature 0.995, time scale (Hard, Good, Easy) 1.000, 0.890, 1.112, shift (ln s) +0.000, -0.023, -0.003; Platt at Good or Easy 4 s (a, b) [1.1059, 0.1664]; repair share (rows x events changed) Platt 0.0511, isotonic 0.1738; rolling: 99 refits, grade temperature 0.9694-1.0586; fits checked past-only: 101.
- the longer cool-down for Tiny (SM115) with its own SM119 head: static shared transform on the benchmark window: grade temperature 0.995, time scale (Hard, Good, Easy) 1.000, 0.849, 1.126, shift (ln s) +0.000, -0.015, +0.045; Platt at Good or Easy 4 s (a, b) [1.1185, 0.118]; repair share (rows x events changed) Platt 0.0376, isotonic 0.1474; rolling: 99 refits, grade temperature 0.9694-1.0586; fits checked past-only: 101.
- the show-time instant-only big model (SM99) with its own SM117 head (the Big pair as the fork runs it): static shared transform on the benchmark window: grade temperature 1.029, time scale (Hard, Good, Easy) 1.000, 0.926, 1.092, shift (ln s) +0.000, -0.056, -0.066; Platt at Good or Easy 4 s (a, b) [0.9941, 0.1697]; repair share (rows x events changed) Platt 0.0711, isotonic 0.2062; rolling: 99 refits, grade temperature 1.0087-1.1394; fits checked past-only: 101.

Run time 19.2 min, CPU time 19.7 min, peak memory 4.83 GB.
