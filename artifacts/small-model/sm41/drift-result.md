# The small model in the review simulator (SM41): drift study result (stage 2 part 1)

Run 2026-11-03 16:27:26 to 19:29:04 (wall clock, including the planned hold), CPU only, 4 processes x 1 thread. Tables: [drift/drift-tables.md](drift/drift-tables.md), raw rows `drift/rows/`, data `drift/drift-result.json`.

## Design

50 users x 2-3 splits (133 split points) at a 60-day horizon, plus 10 users x 1 split at a 120-day horizon. 32 lanes per split: 2 teacher-forced (tf, tf-answer) and 10 free-running arms x 3 seeds (arm list and meanings: DESIGN.md 8.2). Drift = lane minus the real user, mean over users with 95% CI over users. All checks passed: runner done, 60 row files (50 at 60 days, 10 at 120 days), 133 + 10 rows, no Traceback in the logs.

CPU proof (`drift/proof-cpu.json`): PASS. Same seed gives identical results, batched and unbatched lanes agree (max relative gap 8.3e-7, no flipped draws), and stage 1 teacher forcing matches the reference.

## Primary result: first-touch state drift, days 30-59, 60-day horizon

Plain free running (fr): -5.1 +- 3.6 points (mean abs in the last block 9.07).

Best arm: daily shared-stream re-anchoring (fr-shared1): -0.6 +- 1.0 points (mean abs 2.19). Paired reduction versus fr: +6.87 +- 2.91 points of absolute state drift, so most of the drift is removed. Expected-time feeding plus daily re-anchoring (fr-texp-shared1) is equivalent (+6.84 +- 3.05).

Other arms, reduction versus fr: re-anchoring every 7 days (fr-shared7) +4.97 +- 2.23, every 7th day teacher forced (fr-anchor7) +3.17 +- 2.05, time head at temperature 0.5 (fr-tt0.5) +0.76 +- 0.87, expected time alone (fr-texp) +0.71 +- 2.22, answer-to-answer query rows (fr-answer) +0.35 +- 0.61, real times (fr-treal) -0.13 +- 1.77. Grade temperature 0.8 (fr-gt0.8) overshoots to +5.5 and makes it worse (-1.08 +- 3.13).

What is left with fr-shared1 (days 30-59): state drift -0.6 +- 1.0, Easy share of passes +1.2 +- 0.9 points (fr: +5.6), seconds -0.2 +- 0.9 (fed and expected; fr: -1.4), Again -0.9 +- 1.0 points, realised first-touch retention -1.0 +- 1.8.

Caveat: fr-texp feeds a geometric-mean time, so compare its workload with the "seconds expected" column (+2.1 for fr-texp), not "seconds fed".

## Growth with the horizon (120 days, 10 users, wide CIs)

Plain fr state drift by 30-day block: -1.3, -8.9, -10.2, -0.7 (mean abs in the last block 4.96). Drift grows to about -10 points by days 60-89 and the last block is noisy (CI +- 5.8). fr-shared1: -0.4, -2.1, -3.1, +1.0 (last block mean abs 1.61); fr-shared7: -1.2, -3.6, -4.0, +1.0. At days 90-119 the paired reduction versus fr is +3.34 +- 1.60 for fr-shared1. Easy share drift for fr reaches +11.1 +- 12.9 points; for fr-shared1 +1.1 +- 1.1. Re-anchoring still leaves about -3 points of state drift at days 60-89.

## Duration leak (duration-leak scoring (SM31))

Teacher forcing, grade LogLoss of later-day reviews is identical for show-time and answer-to-answer query rows (0.65231 both; 0.64621 at 120 days). Same-day reviews: 0.53153 answer-to-answer versus 0.53912 show-time (120 days: 0.52698 versus 0.53548), so the leak only affects same-day grades, by about +0.008. Free running with answer-to-answer rows (fr-answer) is not better than fr (+0.35 +- 0.61).

## Speed

60-day horizon: 19,881,280 lane reviews in 2.27 process-hours (2,437 lane reviews/s per process), warm-ups 2.29 process-hours. 120-day horizon: 2,893,216 lane reviews in 0.34 process-hours (2,349/s per process), warm-ups 0.17. About 5.07 CPU process-hours in total.
