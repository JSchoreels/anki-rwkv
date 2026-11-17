# Rolling recalibration on the 732 set (SM125): design (pre-registered before any GPU run)

**Question.** Fast Good/Easy prediction (SM120) found that on the 732 set the probability of "Good or Easy and answered within 4 s, reading the back included" beats both baselines by about 20% and 29% LogLoss for all nine models, but only the Small Instant-Only model (SM98) passes the calibration bar (ECE below 0.01); the others are at 0.0105 to 0.0144. On your own collection, recalibrating timed recall on your own collection (SM123) showed that a rolling Platt scaling refitted every 7 days on the latest 30,000 earlier reviews makes the timed events reliable, while a static fit failed. SM125 asks: does the same causal rolling recalibration, done **per user** (what a deployed Anki client could do with its own history), make the timed events reliable on the 732 set at 2 to 10 s without losing their LogLoss lead over the baselines?

**What answer time means.** The answer time is the whole time on the card, front and back together: from the moment the question is shown to the moment a grade button is pressed, capped at the preset's maximum answer seconds. Every "within S s" means answered within S s, reading the back included.

## Models (five release family candidates)

The longer Small Instant-Only model (SM113), the longer cool-down for Tiny (SM115), the show-time two-head cool-down for Small (SM118), each with its head from the time heads for the new candidates (SM119); the Small Instant-Only model (SM98) and the Tiny two-head model (SM106), each with its head from the time heads for the family (SM117). Same frozen trunks and heads as SM120 (the trunk is checked equal to the release before scoring). The show-time two-head cool-down for Tiny (SM124) has no answer-time head yet and is not included.

## Rows and per-review dump (GPU)

SM120 stored only per-user sums, so one GPU pass re-runs SM120's forward pass and event formulas (SM117's and SM120's code imported, not edited; `scripts/sm125_recal.py dump`) on the same rows: the 732 set (723 users of `sm117_head.pool_users()`), show-time inputs, leak-free (deleted cards: neighbour), fp32, SM117's time-labelled test reviews (duration above 0, not a first review, same-day reviews included). Per review it stores, in `work/dump-<model>/<user>.npz` (gitignored, never committed): the 33 event probabilities (float32), the grade, the duration and the day. Events, for the three grade sets recalled (Hard, Good or Easy), Good or Easy, and Easy: P(set) (grade only) and P(set and answered within S s) for S in SM120's grid {2, 3, 4, 5, 6, 8, 10, 15, 20, 30} s. S counts only below the user's own maximum answer seconds (SM117's rule). Size: about 144 bytes per review, 30.4 million reviews, about 4.4 GB per model, 22 GB for five.

Every user's dump is checked against SM120's recorded per-user model sums (same review sha256, the same row and event counts, LogLoss / Brier / sum of p within 1e-5 relative); a user that fails is listed in the report.

## Recalibration scheme (pre-registered; CPU, `scripts/sm125_recal.py recal`)

**Per user, causal, rolling Platt scaling.** Per user and per event (each grade set and each S separately), the probability is mapped through sigmoid(a logit(p) + b). The user's rows are cut into 7-day blocks counted from the user's first scored review. Each block is predicted with (a, b) fitted on the user's latest 30,000 scored reviews of **strictly earlier days** (never the block itself, never a later review). Before the user has 1,000 earlier scored reviews, the raw model is used (the fallback). A deployed client would do exactly this with its own review history.

- Fit: Newton on the mean LogLoss with step halving (SM123's `fit_platt`, batched over the events), warm-started from the user's previous block. If the fit rows of an event have fewer than 20 positives or 20 negatives, that event keeps the raw model in that block. Safety clamps a in [0.05, 20] and b in [-15, 15] (hits are counted and reported).
- Repair (as SM123, so the shipped numbers stay coherent): for each set, a running maximum over increasing S (non-decreasing in S); then Easy within S at most Good or Easy within S at most recalled within S; then each timed event capped by the raw P(set).
- **Grade-only P(set) is not recalibrated** in the scheme: P(Good or Easy), P(Easy) and P(recalled) stay the model's own (the scheduler's numbers are not touched). "P(Good or Easy) and P(Easy) stay reliable" is therefore checked on these raw values with the bar below; as a diagnostic, the report also gives the grade-only events recalibrated the same way.
- Why per user and not pooled: per-user rolling is what a client can do (no other user's data needed), and the dataset's days are relative per user, so a pooled calendar-time refit has no meaning here.

Reported arms: raw (SM120's numbers), the scheme (primary), a sensitivity arm with a 5,000-review minimum (SM123's minimum), the scheme before the repair, the raw and recalibrated numbers on the recalibrated rows only (rows after the minimum history), and the diagnostic with grade-only events recalibrated too.

## The bar and the go rules (pre-registered)

An event at a given S is **reliable** iff its ECE (20 equal-mass bins over the pooled rows, SM117's fine-histogram method) is below 0.01 **and** its LogLoss is below every baseline of that event, each paired difference review-weighted with the 95% CI (2,000 user bootstraps, seed 0) below 0 **and** at least 0.05% of the baseline's LogLoss. The baselines are SM120's, from its recorded per-user sums on the same rows, not recalibrated: (a) the model's P(grade) times the user's running share of earlier answers of that grade within S s, summed over the set; (b) the user's running rate of the event. Grade-only events have baseline (b) only. The same bar is applied to the raw arm, so raw and recalibrated are compared with the same rule (SM120's own verdicts had no 0.05% minimum and are quoted as such).

The recalibrated vs raw LogLoss difference is reported with its CI for every event and S.

Per model, over the 2 to 10 s range (S in {2, 3, 4, 5, 6, 8, 10}) and the three timed events (Good or Easy, recalled, Easy within S s): 21 cells.

- **G1:** all 21 cells reliable after recalibration.
- **G2:** on "Good or Easy within 4 s", recalibrated LogLoss below raw with the CI below 0 and at least 0.05%.
- **G3:** no cell of the 21 where the recalibrated LogLoss is worse than raw with the CI above 0.
- **G4:** P(Good or Easy) and P(Easy) (grade only, raw) reliable.
- **Go for a model** iff G1, G2, G3 and G4 all hold: per-user rolling recalibration of the timed events is worth shipping for that model. The report also gives, per model, "Good or Easy within 4 s" reliable raw / recalibrated, the number of reliable cells raw / recalibrated, and the S values where each event is reliable over the whole grid.
- **Family verdict:** go if all five models are go; otherwise the models that are go are named.

## CPU proofs (before any GPU run; `check-cpu.json`, PASS)

On 4 small users of the 732 pool (2388, 1952, 3922, 4563; 11,800 rows), the Tiny two-head model (SM106), CPU only:

- P1: the dump reproduces SM120's recorded GPU sums (same review sha256, counts exactly, LogLoss / Brier / sum p within 7e-6 relative, CPU vs GPU); the float32 storage changes the sums by at most 8e-8 relative; days sorted; the raw events nested.
- P2: the batched Newton Platt equals scipy's BFGS optimum (largest parameter difference 2e-8).
- P3: causality: flipping every label and duration from a day D on (user 4563) leaves every earlier recalibrated probability bitwise unchanged and moves later ones.
- P4: without history the scheme is the raw model exactly.
- P5: after the repair every arm is non-decreasing in S, nested (Easy within Good or Easy within recalled) and below P(set).
- P6: a synthetic user whose truth is sigmoid(1.6 logit p + 0.4): the last fit is (1.58, 0.40), ECE 0.064 to 0.005, LogLoss 0.434 to 0.415.
- P7: end to end (dump, recal, evaluate) on the 4 users: every raw LogLoss equals SM120's; 63% of rows recalibrated; (4 users only: a pipeline check, not a result).
- P8: CPU cost: 7,200 rows per second per process, about 1.2 CPU hours per model, run one process per model (1 thread, under 1.5 GB RAM) right after each model's dump.

## Launch

`launch-sm125.sh` waits until the launcher of the show-time two-head cool-down for Tiny (SM124) has ended (the last line of its log is `done` or `STOP`), then queues through `scripts/r01_queue.py` (cap 0.8): a 3-user GPU smoke of the Tiny two-head model (SM106) checked against SM120's sums within 1e-5 (6 min), then one dump ticket per model (30 min budget; SM120 took about 17 min per model), a missing-users pass at 0.92 only if fewer than 723 users dumped. After each dump the CPU recalibration of that model starts in the background; it stops itself 5 min before every break (quiet.json, daily 20:00-01:00, Sunday from 16:00) and resumes after it. Then `sm125-result.md` / `.json` on CPU.
