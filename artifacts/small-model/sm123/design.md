# Recalibrating "recalled and answered within S s" on your own collection (SM123): design (pre-registered before any test result)

Written Tue 2026-11-17 07:10, before any recalibrated number on the scored rows was computed.

**Question.** The timed-recall analysis on your own collection (SM121, branch `sm121`, `scripts/sm121_own.py`, `sm121-result.json`) found that at S = 4 s every model and head beats every baseline (the card's own last answers included) with the CI below 0, but the calibration error (ECE) is 0.011 to 0.041, above the 0.01 reliability bar. Can a simple per-collection recalibration, fitted only on reviews available before the scored period, bring ECE below 0.01 at S in {2, 3, 4, 5, 6, 8, 10} for the sets Good or Easy, Easy, and recalled (Hard, Good or Easy), while keeping the lead over every baseline? This decides whether a scheduler can use "Good within 4 s" on a single collection.

**What answer time means.** The whole time on the card, front and back together: from the question shown to the grade pressed, capped at the preset's maximum answer seconds (60 s here). Every "within S s" means recalled and answered within S s, reading the back included.

## Inputs (nothing is rerun)

- The per-review model outputs of SM121: `wt/sm121/artifacts/small-model/sm121/work/trunk-<key>.npz` (`th`, `pg` = P(grade), `pt_<head>` = P(time bin | grade)), the 209,410 time-labelled non-first reviews of the collection copy of 2026-10-14 (reviews up to 2026-10-04), fp32 state, show-time inputs. Read only.
- Labels, row slices, baselines, ECE, day-block bootstrap and privacy scan are imported from `wt/sm121/scripts/sm121_own.py` (commit `4e819746`) and used unchanged: the event is "grade in the set and duration <= S" (SM117 rule); ECE = 20 equal-mass bins on clipped p; 2,000 day-block bootstraps, seed 0; baselines = SM121's bar for timed events (raw P(set), the two recent-history products, the running and rolling 2,000-review rates, the two card-history adjusted products, and the Mac's four at card strength 3).
- CPU only (`CUDA_VISIBLE_DEVICES=-1`), below-normal priority, 4 threads, each process under 6 GB.

## Combinations

Main (the decision rests on these four): the longer Small Instant-Only model (SM113) with its own SM119 head; the current Small pair = grades of the Small Instant-Only model (SM98) with time from the show-time retrain (SM92)'s SM117 head; the Tiny two-head model (SM106) with its own SM117 head; the show-time big model (SM91s2) with its own SM117 head. If the full run of the four takes under 15 min, the other seven SM121 combinations are run with the same code and reported as additional rows (not part of the decision).

## Scored rows and calibration windows

Scored (test) rows, exactly SM121's: `bench/full` (the benchmark test reviews with a recorded time, 173,384 rows, 2024-11 to 2026-10), `bench/yomitan` (the same rows in the deck subset the Mac used), and as secondary sets `mac/full` and `mac/yomitan` (later-day reviews with a positive time, the latest 20%, 35,256 rows from 2026-04). The decision uses `bench/full` and `bench/yomitan`; the `mac` rows are reported alongside.

One recalibration per collection: every fit uses the whole collection's rows (never a deck subset); the deck-subset rows are scored with the same recalibrated probabilities.

- **Static, for the benchmark rows:** fit once on every time-labelled row before the first benchmark row (34,830 rows, 2024-01 to 2024-11; the same eligibility: positive time, not a first review, same-day included). This is the only data strictly before the scored period; it is the early part of the collection, so the static fit is a hard test (the regime drifts).
- **Static, for the Mac slice rows:** fit once on the rows of the slice's own eligibility (later-day, positive time) immediately before the first slice row, as many as the slice holds (35,256).
- **Rolling (online, what a scheduler can do):** refit every 7 days (blocks of 7 Anki days); each fit uses the latest 30,000 time-labelled rows of earlier days only (day < the block's first day), and the fitted map is applied to the rows of that block. Rows with fewer than 5,000 earlier rows stay uncalibrated (no scored row is in that case). The same rolling probabilities are scored on all four row sets.
- Proof, asserted in code for every fit: the fit rows' row index, day and review time are all strictly below those of every row the fit is applied to.

## Methods

The uncalibrated probability is SM121's: p = sum over grades g in the set of P(g) x P(t <= S | g), P(t <= S | g) linear in ln time inside the bin (`cdf_at`).

- **(a) Platt per S and set:** p' = sigmoid(a x logit(p) + b), two parameters per (set, S) fitted by LogLoss (Newton, no regularisation, p clipped to [1e-6, 1 - 1e-6]).
- **(b) Shared transform:** one monotone map of the model's distributions, fitted across all S at once, so the probabilities stay consistent across thresholds and sets. Grade part: P'(g) = softmax(ln P(g) / T + c_g) (temperature T, offsets c for Hard, Good, Easy; 4 parameters, fitted by the 4-class LogLoss of the grade). Time part, per grade g in {Hard, Good, Easy}: a scale and shift in ln time around the row's own predicted median m (ln s): P'(t <= S | g) = CDF_g(exp(m + (ln S - delta_g - m) / tau_g)), tau_g = exp(theta_g) > 0 (6 parameters, fitted on the rows of grade g by the sum over S in {2, 3, 4, 5, 6, 8, 10} of the binary LogLoss of "t <= S"). Then p' = sum over g in the set of P'(g) P'(t <= S | g). Monotone in S and nested across sets (Easy <= Good or Easy <= recalled) by construction.
- **(c) Isotonic per S and set:** isotonic regression of the event on p (clipped to [0, 1], out-of-range p clipped to the fitted range), an upper reference for what a free monotone map can do.
- (a) and (c) are not consistent across S by construction. Repair, applied to both (pre-registered, reported): for each set the running maximum over increasing S, then Good or Easy <- max(Good or Easy, Easy) and recalled <- max(recalled, Good or Easy). The share of repaired rows and the mean change are reported.
- Each method has a static and a rolling variant: six recalibrated predictors per combination.

## Rule (per combination, method, row set, set and S)

**Reliable after recalibration** iff ECE (20 equal-mass bins) < 0.01 **and** the recalibrated LogLoss is below every baseline of SM121's bar with the day-block 95% CI of the paired difference below 0 **and** each difference is at least 0.05% of the baseline's LogLoss. Reported with it: LogLoss, ECE, the % change of LogLoss against the uncalibrated head (with its day-block CI), the strongest baseline, the gap to it in %, and AUC.

**Decision answer.** "Good within 4 s is usable on a single collection" if, for the longer Small Instant-Only model (SM113) or the current Small pair, the rolling variant of some method is reliable for Good or Easy at 4 s on both `bench/full` and `bench/yomitan`. The static result is reported but the rolling one is what a scheduler would run.

**Sensitivity (outside the bar).** The strongest baseline itself (the one SM121 names for that event on `bench/full`) recalibrated by Platt with the same static or rolling windows, compared with the Platt-recalibrated head: does the lead survive when the baseline is also calibrated?

## Process

1. CPU check (`check`): only the first 60,000 rows (static window + the first scored rows), one combination, 200 bootstraps. Proofs: (K1) the vectorised CDF at the identity map equals `cdf_at` (max error < 1e-9) and the uncalibrated head equals SM121's `groups()` head; (K2) every recalibrated probability is finite and in [0, 1]; (K3) after repair all methods are non-decreasing in S and nested across sets, and (b) is so before any repair; (K4) every fit uses only strictly earlier rows (index, day, review time asserted); (K5) on synthetic calibrated data Platt returns a ~ 1, b ~ 0, and (b) at T = 1, c = 0, tau = 1, delta = 0 reproduces the uncalibrated probabilities exactly; (K6) on the full rows, the uncalibrated head's LogLoss and ECE for the four combinations at Good or Easy 4 s on `bench/full` equal `sm121-result.json`.
2. Full run (`run`): all four combinations, six predictors, four row sets, sets x S = 21 events each. If it would take over about 30 min, launched detached with a handoff.
3. Outputs: `sm123-result.md` + `sm123-result.json` (aggregates only, no per-row arrays), `check-cpu.json`; SM121's privacy scan on all three before commit. Per-review intermediates, if any, stay in `work/` (never committed).

## Amendments (before the full run, after the CPU check only)

- The static benchmark window also excludes the pre-benchmark rows that share the first benchmark row's Anki day, so that the "strictly earlier day" proof holds: 34,727 rows instead of 34,830.
- K5: the shared transform at the identity equals the uncalibrated probabilities up to 1.3e-7 (the stored P(grade) is float32 and its row sums are off by up to 2e-7, which the softmax renormalises), so the tolerance is 1e-6 instead of "exactly"; the synthetic Platt check uses 1,000,000 rows (tolerance 0.02).
