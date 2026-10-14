# Accuracy log

Chronological record of [accuracy-loop](ACCURACY_LOOP.md) experiments; ideas and their statuses are in [ACCURACY_IDEAS.md](ACCURACY_IDEAS.md). One row per reported experiment (`scripts/accuracy_loop.py report` prints the row). Failed and partial runs are recorded too.

Columns: LogLoss, RMSE(bins) and AUC are upstream benchmark metrics on the label-DB benchmark test reviews, review-weighted, on all evaluation users of the tier; "unseen" repeats LogLoss on users with id < 5000 (not in the fork model's training range). Collection = the user's own collection, full layout, with Δ% against the fork's 2.76M model (curve 0.4417, instant 0.4366; positive is worse). Reviews/s is training review presentations per training second. State = per-card recurrent state bytes, float32. Train minutes are process wall time; eval minutes are the scoring time of that checkpoint (a shared ticket's total is in `phases.jsonl`). Every evaluation below used `eval: fast (EV01, c6530b9)`. User-averaged values, the R01 all-label losses and the `fork_seen` subset are in each `result.json`. Missing values are `—`.

## Runs

| Reported (UTC) | Experiment | Tier | Seed | Train users | Eval users | Curve LL | Instant LL | Curve RMSE(bins) | Instant RMSE(bins) | Curve AUC | Instant AUC | Unseen curve LL | Unseen instant LL | Reviews/s | Params | State B | Collection curve (Δ%) | Collection instant (Δ%) | Train min | Eval min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-14 14:01 | `screen-w32-s2025` | screen | 2025 | 2000 | 1000/1000 | 0.3526 | 0.3334 | 0.0785 | 0.0694 | 0.7009 | 0.7357 | 0.3493 | 0.3250 | 190,818 | 192800 | 4352 | 0.4643 (+5.12%) | 0.4605 (+5.48%) | 13.2 | 6.3 |
| 2026-10-14 14:02 | `screen-w32-s2026` | screen | 2026 | 2000 | 1000/1000 | 0.3525 | 0.3335 | 0.0774 | 0.0685 | 0.6991 | 0.7349 | 0.3494 | 0.3252 | 185,612 | 192800 | 4352 | 0.4619 (+4.58%) | 0.4656 (+6.64%) | 13.4 | 6.1 |
| 2026-10-14 14:02 | `fullrun-e10` (score only) | screen | 2025 | 8000 × 10 epochs | 1000/1000 | 0.3352 | 0.3185 | 0.0632 | 0.0560 | 0.7335 | 0.7618 | 0.3301 | 0.3097 | — | 192800 | 4352 | 0.4535 (+2.68%) | 0.4525 (+3.64%) | — | 6.0 |
| 2026-10-14 18:14 | `decay-e10-q25` (score only) | screen | 2025 | 8000 × 10 epochs + 0.25-epoch WSD decay | 1000/1000 | 0.3338 | 0.3175 | 0.0609 | 0.0543 | 0.7350 | 0.7629 | 0.3286 | 0.3085 | — | 192800 | 4352 | 0.4536 (+2.70%) | 0.4518 (+3.49%) | — | 6.1 |

### A08 (000317): `decay-e10-q25` vs `fullrun-e10`, 2026-10-14

Paired on the 1,000 screen users (review-weighted, 95% user bootstrap): curve −0.00136 [−0.00192, −0.00081], instant −0.00103 [−0.00129, −0.00080]; better for 676 (curve) and 747 (instant) of 1,000 users. `fork_unseen` (464): −0.00152 / −0.00116; `fork_seen` (536): −0.00122 / −0.00093, all intervals exclude 0. Own collection, full layout (173,690 reviews): fullrun-e10 curve 0.4535 (+2.68%), instant 0.4525 (+3.64%); decay-e10-q25 curve 0.4536 (+2.70%), instant 0.4518 (+3.49%), one user, descriptive only. Raw: `artifacts/accuracy/comparisons/decay-e10-q25--vs--fullrun-e10.json`.

## Noise bands

| Tier | Runs | Metric | Values | Band | Notes |
|---|---|---|---|---|---|
| screen | `screen-w32-s2025`, `screen-w32-s2026` | curve LogLoss, review-weighted (all 1,000) | 0.35258, 0.35248 | 0.00010 | two seeds = one sample of a difference; per-user SD of the paired delta 0.0033 |
| screen | same | instant LogLoss, review-weighted (all 1,000) | 0.33341, 0.33347 | 0.00006 | per-user SD 0.0027 |
| screen | same | curve / instant LogLoss, user-averaged | 0.34426 / 0.32060, 0.34437 / 0.32043 | 0.00011 / 0.00017 | |
| screen | same | `fork_unseen` (464 users) curve / instant, weighted | 0.34934 / 0.32502, 0.34943 / 0.32524 | 0.00009 / 0.00022 | |
| screen | same | `fork_seen` (536 users) curve / instant, weighted | 0.35519 / 0.34018, 0.35494 / 0.34010 | 0.00025 / 0.00008 | |
| screen | same | RMSE(bins) curve / instant; AUC curve / instant | 0.0785 / 0.0694, 0.0774 / 0.0685; 0.7009 / 0.7357, 0.6991 / 0.7349 | 0.0010 / 0.0009; 0.0018 / 0.0008 | |
| screen | same | collection, full layout, curve / instant | 0.4643 / 0.4605, 0.4619 / 0.4656 | 0.0024 / 0.0051 | one user (173,690 reviews): descriptive only, never a decision metric at screen tier |

Decision band until a third seed and an order-seed run (A15) are measured: **0.0004 LogLoss** on review-weighted curve and instant (Expertium's seed noise), not the measured 0.0001–0.0002, because two seeds give a single draw of the difference. Two cautions from this pair: (1) the user-level bootstrap does not contain seed noise; the seeds' user-averaged instant delta has a 95% user-bootstrap interval of [−0.00034, −0.00000], which would look like an effect, so the band must come from seeds, never from the user bootstrap alone; (2) seeds agree to about 0.0001 while the full run's consecutive constant-LR epochs moved 0.002–0.003 on the 300 quick users (see the note below), which points at training order or recency, not initialization, as the larger noise source. Raw: `artifacts/accuracy/noise/screen-w32-s2025--screen-w32-s2026.json`.

## Tier costs

Measured 2026-10-14 on the RTX 5080 host (other sessions' GPU jobs queued but not concurrent with these; CPU collection jobs ran during other sessions' accuracy jobs). Queue waits are separate: the check+train ticket waited 1 h 37 min behind the full run, the evaluation ticket 1 h 27 min behind other sessions.

| Tier | Phase | Users | Reviews | Wall minutes | Rate | Source |
|---|---|---|---|---|---|---|
| screen | train, 1 epoch (seed 2025) | 2,000 | 146,654,211 | 13.2 process (12.8 training) | 190,818 reviews/s, 3,469 updates | `experiments/screen-w32-s2025/phases.jsonl` |
| screen | train, 1 epoch (seed 2026) | 2,000 | 146,654,211 | 13.4 process (13.2 training) | 185,612 reviews/s | `experiments/screen-w32-s2026/phases.jsonl` |
| screen | evaluate, fast (EV01), per checkpoint | 1,000 | 71,444,138 | 6.0–6.3 (374 / 365 / 362 s) | 191k–197k reviews/s | runbook log; three checkpoints in one 18.5-minute ticket |
| screen | evaluate, reference (projected) | 1,000 | 71,444,138 | about 48 | about 25k reviews/s | not run at this size; 12-user check: 24 s for 2 checkpoints |
| screen | collection, full layout, CPU 2 threads | 1 | 223,670 events | 8.4–8.7 | — | `experiments/*/phases.jsonl` |
| screen | **total per candidate** | | | **about 20 GPU minutes + 9 CPU minutes** | | train 13 + fast eval 6 + collection 9 (CPU, can overlap) |
| gate | train, 1 epoch (projected) | 5,000 | 369,477,375 | about 33 | at 189k reviews/s | `tiers.json` review count |
| gate | evaluate, fast (projected) | 1,500 | 107,558,946 | about 9.5 | at 190k reviews/s | |
| final | train, 10 epochs (projected) | 8,000 | 581,942,738 per epoch | about 51 per epoch, 8.5 h for 10 | at 190,818 reviews/s | full run measured 183k–193k |
| final | evaluate, fast, validation + final test | 2,000 | 145,795,912 | 10.2 | 238k reviews/s | EV01 measurement, `docs/EVAL_SPEED.md` |

## Notes

### First measurements (2026-10-14)

- Harness check: the reference scorer (`scripts/accuracy_eval.py`) reproduced the full run's recorded per-user results bitwise for epoch 6 and epoch 7 on 12 quick-validation users, both scored in one shared pass (`artifacts/accuracy/checks/check-eval-check-e0{6,7}.json`). The fast mode on CPU matched epoch 7 within 8e-9 on R01 losses with identical benchmark metrics (`check-fast-cpu`). `scripts/accuracy_collection.py` reproduced the epoch-7 collection scores exactly (curve 0.45456, instant 0.452531) and epoch 10's (0.453526, 0.452503).
- Screen baseline (width 32, accepted R01 recipe `E89-F65e-F77a`, constant LR 7e-4, 2,000 users × 1 epoch, seeds 2025 and 2026; trained at commits `e6895b3` and `de6e9d8`, both clean, identical training source): curve 0.3526 / 0.3525, instant 0.3334 / 0.3335 on the 1,000 screen users. Seed noise above.
- The full run's final checkpoint (8,000 users × 10 epochs) on the same users: curve 0.3352, instant 0.3185. Paired against the screen baseline (seed 2025): curve −0.0174 [−0.0194, −0.0156], instant −0.0149 [−0.0161, −0.0138], better for 954 of 1,000 users on instant; on `fork_unseen` −0.0192 / −0.0154 (`artifacts/accuracy/comparisons/fullrun-e10--vs--screen-w32-s2025.json`). The screen tier sits about 0.015 above the full run, about 60× its seed band, so it has room to rank recipes, but a screen gain does not say how much survives 4× more users and 10 epochs.
- User collection, full layout: screen baseline +4.6% to +5.1% (curve) and +5.5% to +6.6% (instant) worse than the fork's 2.76M model; full run epoch 10 +2.68% / +3.64%.
- 8k/2k 10-epoch ETA at the screen's measured 185,612–190,818 reviews/s: 50.8–52.3 minutes per epoch, 8.5–8.7 h for 10 epochs, plus about 10 minutes of fast final evaluation (the reference path took 97 minutes).

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

### Proposed next jobs (not run; each ticket under 60 minutes)

1. A15 order noise and a third seed (screen, about 26 + 13 GPU minutes of training and 18 minutes of fast evaluation in three tickets):
   ```
   PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe; Q="$PY C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py"; AL=scripts/accuracy_loop.py
   $PY $AL prepare screen-w32-s2025-o1 --tier screen --seed 2025 --order-seed 1 --note "A15 order noise"
   $PY $AL prepare screen-w32-s2025-o2 --tier screen --seed 2025 --order-seed 2 --note "A15 order noise"
   $PY $AL prepare screen-w32-s2027 --tier screen --seed 2027 --note "third seed"
   $Q run --owner accuracy-loop --purpose "A15 order-noise screen trainings" --claim gpu -- bash -c "$PY $AL train screen-w32-s2025-o1 && $PY $AL train screen-w32-s2025-o2"
   $Q run --owner accuracy-loop --purpose "third-seed screen training" --claim gpu -- $PY $AL train screen-w32-s2027
   $Q run --owner accuracy-loop --purpose "screen eval (fast)" --claim gpu -- $PY $AL evaluate screen-w32-s2025-o1 screen-w32-s2025-o2 screen-w32-s2027
   ```
   then `report` each and `noise screen-w32-s2025 screen-w32-s2026 screen-w32-s2027` and `noise screen-w32-s2025 screen-w32-s2025-o1 screen-w32-s2025-o2`.
2. After `expertium-catch-up` merges: re-baseline the screen control with WSD decay (A03), then screen s64h (A17) and width 80 (A01) for cost; both go to the gate (about 33 minutes of training per 5k-user epoch plus about 10 minutes of fast evaluation per checkpoint).
