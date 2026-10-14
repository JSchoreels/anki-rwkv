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
