# Retrieval-effort probe (SM122): design (pre-registered before any result)

Written Tue 2026-11-17 08:30, before any counterfactual or residual number was computed. Only the user selection (`users.json`) had been run.

**Inspiration.** Pyc & Rawson (2009), the retrieval effort hypothesis: harder but successful retrievals produce better later memory. In Anki terms: a successful answer (Good or Easy) that took longer may be followed by a slower forgetting curve than a fast one, or the opposite (a slow success may be a sign of a weak memory).

**What answer time means.** The answer time is front + back together: from the question shown to the grade pressed, capped at the preset's maximum answer seconds. In our models it is an input: feature 8 (`scaled_duration` = (ln(10 + ms) - 8.9) / 1.07) on the history (update) rows, zeroed on the query rows. Because it is an input, the model can have learned any duration effect the data hold.

## Data, models, CPU

- Users: a pre-registered random subsample of the 732-user standard readout set (723 scoreable users in `eval-users-732-fit.json`): `numpy.random.default_rng(122).permutation(users)[:200]` (`users.json`), because the full set would need about 45 h of CPU per model for the eight passes. Each user's first 20,000 reviews (prefix cap; 3,426,011 reviews in total). The show-time big model runs on the first 60 of the 200 (1,012,032 reviews) with four passes only (it is 3.4 times slower on CPU).
- Inputs: the scorer's show-time batch (`e2s-query`, leak-free "neighbour" deleted-card handling, whole capped prefix as one sequence, seed 2025 + user), fp32 weights and state, CPU only (`CUDA_VISIBLE_DEVICES=-1`), below-normal priority, 4 threads, each process under 6 GB.
- Models (all two-head: an instant head and a curve head): the show-time retrain (SM92) (Small two-head), the Tiny two-head model (SM106), the show-time big model (SM91s2) (Big two-head). Masters verified by sha256 against the family release manifests (`scripts/sm117_head.py` `verified_master`).
- Code: `scripts/sm122_effort.py`, which imports read only from `wt/sm121` (`sm117_head.py` loaders, `small_model_heldout.py`, `sm31_inputs.py`, `sm71_leak.py`); both commits are recorded in every output.

## Rows

- **Pairs (t, t').** t' = a review on a later day than the card's previous review (elapsed days >= 1); t = that previous review, which must be Good or Easy, timed (0 < duration < the user's cap), and not the card's first review. t is then the card's last review of its day and its last success before t'. Outcome y = t' recalled (Hard, Good or Easy).
- **Targets (question 1 and the model part of question 3).** The pairs, taken greedily in review order with t at least 100 reviews after the previous target's t, so that the edited reviews are sparse in every stream. The spillover onto the non-target pairs is reported for every arm.

## Read-outs per pair

- Instant head: P(recall) = 1 - P(Again) at the query row of t' (show-time interval).
- Curve head at the update row of t: P(recall) after the show-time interval to t', and after 1 day, 1 week and 30 days.

## Question 1: counterfactual duration of the last success

Arms (each a full pass with only the targets' recorded durations changed in the raw review log; everything else, timestamps and intervals included, is untouched):

| arm | new duration of each target |
|---|---|
| x0.5, x2 | duration x 0.5 or x 2 |
| card's 10th / 90th percentile | the card's own 10th / 90th percentile of its Good/Easy timed answers in the prefix (cards with at least 3 of them; other targets are left unchanged and out of this arm's numbers) |
| user's 10th / 90th percentile | the user's 10th / 90th percentile of Good/Easy timed answers in the prefix |

Every new duration is rounded and clamped to [1 ms, the user's cap]; the mean log2 change actually applied is reported. Big runs x0.5 and x2 only (plus the extra-success arm).

**Effect size.** For each arm and read-out: sum over targets of (R_arm - R_base) / sum of R_base, in % of R. LogLoss-relevant units: the change of the instant (and curve) LogLoss on the real outcomes y when the arm's predictions replace the base ones, in % of the base LogLoss. Strata: grade of t (Good, Easy), t a same-day step or a later-day review, interval to t' (1 d, 2-7 d, 8-30 d, more than 30 d).

**Sign answer.** "The model has learned slower success -> higher later R" if the x2 arm's instant R at the next review goes up with the CI above 0 and size >= 0.05% for the show-time retrain (SM92) (primary), and "slower -> lower" if down; otherwise "no learned effect of this size". The other arms, read-outs and models are reported with the same rule.

## Question 2: residual dependence of real outcomes on the duration

On all pairs of the base pass:
- **Within user:** z = ln(duration of t) - the user's median ln duration of Good/Easy timed answers of the same grade (whole prefix).
- **Within card:** z = ln(duration of t) - the median ln duration of the card's earlier Good/Easy timed answers (at least 2 earlier; causal).
- Quintiles of z (cut points from the pooled pairs of the show-time retrain (SM92)'s users, the same cuts for every model). Per quintile: actual recall, predicted recall, residual (actual - predicted) in % of predicted, user-block CI.
- **Decision statistic:** the cross-fitted LogLoss gain of a recalibration with one logit offset per quintile over a recalibration with a single global offset (2 folds of users, `default_rng(122)` permutation, alternating; fitted on one fold, scored on the other), in % of the global-offset LogLoss. "Residual later recall depends on the previous success's duration" iff this gain has its user-block 95% CI above 0 and is >= 0.05%, for the instant head of the show-time retrain (SM92) (primary); the curve head and the other models are reported with the same rule. Also reported: Q5 - Q1 residual and the slope of the residual on z (clipped to [-2, 2]) per doubling of the duration, both in % of mean predicted recall.

## Question 3: one more same-day success

- **Model (counterfactual):** an extra pass in which each target gets one more review inserted 600 s after it, same card, same grade, same state and same duration (elapsed days 0, elapsed seconds 600; the next review's elapsed seconds reduced by 600). Read-outs from the inserted review's update row (curve after 1 d, 1 w, 30 d and at t') and at the next review's query row (instant), compared with the base pass, same size rule.
- **Real data:** pairs whose day D (the day of t) has no Again for that card, grouped by the number of successes that day (1, 2, 3 or more) and by whether D is the card's first day. Per group: recall, predicted, residual. Decision statistic: the cross-fitted LogLoss gain of per-group offsets over a global offset (same folds and rule as question 2); also the later-day "2 or more successes minus 1 success" residual difference.

## Bootstrap and rule

User blocks (all of a user's pairs together), 1,000 Poisson(1) replicates, seed fixed. The question-2 and question-3 gains also get a user-day block CI (pairs grouped by user and the day of t'); the decision uses the user-block CI (wider). **An effect counts only with the 95% CI excluding 0 and a size of at least 0.05%.**

## CPU check before the run (`check`, a few users, recorded in `check-cpu.json`)

- K1: the curve read-out equals the model's own `get_loss` p_curve on the update rows (max abs error < 1e-4).
- K2: the x2 edit changes only feature 8 on the targets' update rows, plus the interval columns of the targets' own query rows (show time = interval - duration), which never feed later rows.
- K3: every output before the first target's update row is unchanged (< 1e-5), and editing only feature 8 gives the same next-review predictions as the raw-duration edit (< 1e-6).
- K4: batched against exact: for up to 8 targets per check user (6 users), a pass with only that target doubled; the mean delta must agree with the batched pass within 20% (or 0.01% of R) and the Pearson correlation must be >= 0.9. If K4 fails, the spacing is raised so that no target falls inside another target's window and the check is rerun (amendment recorded below).
- K5: two base passes are identical. K6: the extra-success pass adds exactly 2 rows per target (update + query), leaves every earlier output unchanged and gives finite read-outs. Peak memory < 6 GB; the projected run time is recorded.
- K7 (at report time): privacy scan of the result files (aggregates only: no card ids, review indices, day offsets or timestamps, no per-row lists). Per-user arrays stay in `work/` (gitignored, never committed, never `.npz` in git).

## Process

The three model runs take more than 30 min each, so they are launched detached (hidden, WMI) by `launch-sm122.sh`, which runs the three models in parallel, checks quiet.json and the daily 20:00-01:00 break before every user (it sleeps through a window and resumes), then runs `report`. A fresh agent then commits the results and writes the summary (`HANDOFF.md`).

## Amendments (after the CPU check only)

- 11-17 09:00, after the CPU check of the show-time retrain (SM92) only: K3 compared outputs up to the first target's update row, which includes that target's own query row; its show-time interval columns are meant to change (K2: show time = interval - duration) and it feeds no later row (max difference 2.4e-5 there). K3 now compares the rows before that query row. Nothing else changed; every other proof passed (K4: exact -1.15% of R against batched -1.12%, Pearson 0.9994, 48 targets; projected 5.7 h for the Small model alone).
- 11-17 13:00, after the full CPU check of all three models, before the show-time big model (SM91s2) run: the second K3 bound (editing only feature 8 against the raw-duration edit, < 1e-6) failed for the Tiny two-head model (SM106) at 1.43e-6 and the show-time big model (SM91s2) at 1.13e-6 (the show-time retrain (SM92) passed at 9.5e-7). These are float32 rounding differences (about 10 float32 steps at R near 1) between two mathematically identical inputs; every other proof passed for all three models (K4 Pearson 0.9994 / 0.9998 / 0.9994). The bound is raised to 1e-5, the per-user tolerance in AGENT.md and the first K3 bound; all three pass with the recorded `check-cpu.json` numbers, so the check is not rerun. The Small and Tiny runs had already started in parallel with the check and are unchanged; the show-time big model (SM91s2) run starts after this amendment.
