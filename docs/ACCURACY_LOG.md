# Accuracy log

Chronological record of [accuracy-loop](ACCURACY_LOOP.md) experiments; ideas and their statuses are in [ACCURACY_IDEAS.md](ACCURACY_IDEAS.md). One row per reported experiment (`scripts/accuracy_loop.py report` prints the row). Failed and partial runs are recorded too.

Columns: LogLoss, RMSE(bins) and AUC are upstream benchmark metrics on the label-DB benchmark test reviews, review-weighted, on all evaluation users of the tier; "unseen" repeats LogLoss on users with id < 5000 (not in the fork model's training range). Collection = the user's own collection, full layout, with Δ% against the fork's 2.76M model (curve 0.4417, instant 0.4366; positive is worse). Reviews/s is training review presentations per training second. State = per-card recurrent state bytes, float32. Train and eval minutes are process wall times (eval summed over its jobs). User-averaged values, the R01 all-label losses and the `fork_seen` subset are in each `result.json`. Missing values are `—`.

## Runs

| Reported (UTC) | Experiment | Tier | Seed | Train users | Eval users | Curve LL | Instant LL | Curve RMSE(bins) | Instant RMSE(bins) | Curve AUC | Instant AUC | Unseen curve LL | Unseen instant LL | Reviews/s | Params | State B | Collection curve (Δ%) | Collection instant (Δ%) | Train min | Eval min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

## Noise bands

| Tier | Runs | Metric | Values | Band | Notes |
|---|---|---|---|---|---|

## Tier costs

| Tier | Phase | Users | Reviews | Wall minutes | Rate | Source |
|---|---|---|---|---|---|---|

## Notes

### Full-run reference: per-epoch quick validation (2026-10-14)

The full run (`run-8k-20261013`, width 32, 8,000 training users, constant LR 7e-4) scored its 300 quick-validation users after every epoch. These users are part of the screen evaluation set, so screen numbers can be placed on this curve. Benchmark LogLoss, review-weighted; "unseen" = user id < 5000 (149 users), "seen" = ≥ 5000 (151 users). Source: `epoch-NN/expertium/E89-F65e-F77a/result.json` in the main checkout, summarized with `scripts/accuracy_loop.py`'s `summarize`.

| Epoch | Reviews/s | Curve LL | Instant LL | Unseen curve LL | Unseen instant LL | Seen curve LL | Seen instant LL |
|---|---|---|---|---|---|---|---|
| 1 | 188,882 | 0.3703 | 0.3508 | 0.3588 | 0.3402 | 0.3810 | 0.3607 |
| 2 | 166,636 | 0.3654 | 0.3482 | 0.3530 | 0.3375 | 0.3770 | 0.3582 |
| 3 | 193,127 | 0.3658 | 0.3481 | 0.3534 | 0.3372 | 0.3774 | 0.3583 |
| 4 | 192,661 | 0.3654 | 0.3473 | 0.3536 | 0.3366 | 0.3764 | 0.3574 |
| 5 | 192,103 | 0.3619 | 0.3460 | 0.3492 | 0.3352 | 0.3738 | 0.3561 |
| 6 | 187,359 | 0.3634 | 0.3480 | 0.3500 | 0.3362 | 0.3760 | 0.3589 |
| 7 | 192,453 | 0.3615 | 0.3447 | 0.3485 | 0.3335 | 0.3736 | 0.3552 |
| 8 | 183,055 | 0.3616 | 0.3443 | 0.3482 | 0.3331 | 0.3741 | 0.3548 |

On a constant LR the same users move by 0.002–0.003 instant LogLoss between consecutive epochs, in both directions (epoch 5 → 6 got worse by 0.0020, epoch 6 → 7 better by 0.0033; paired user bootstrap for 6 → 7: −0.0033, 95% interval [−0.0042, −0.0024]). That checkpoint-to-checkpoint jitter is the same order as the gaps the loop wants to resolve. It argues for ending every tier with an LR decay (A03, A08) before comparing, and for reading a screen result against the measured seed band, not against zero.
