# The real-data check of the time-aware target (SM41r)

Question (user 11-05): offline, on real benchmark review histories, are slow-answer cards actually cheaper to review
late? The small model in the review simulator (SM41) found that a per-card target
`logit(target) - beta * ln(last later-day answer ms / user median ms)` (beta = 2-3) beats FSRS-7 by about 6-12%
minutes at the same memorised cards once settled ([policy-result.md](policy-result.md)). The simulated learner is the
small model itself, so that gain is inside the model's world. This check asks whether real reviews show the same
structure. Code `scripts/sm41_realcheck.py`, runner `realcheck/run-realcheck.sh`, tables
`realcheck/realcheck-tables.md` (numbers in `realcheck/realcheck.json`).

## Plan (fixed before the run)

### Data

- **Users and windows.** The full run's 117 users and 304 split days (`fullrun/users.json`). Only the real reviews in
  the 60 days after each split day, the part the simulator replaced: 1.39M window reviews. No simulation.
- **Predictions.** The released small model, teacher-forced on the real schedule. This is the engine's `tf` lane
  alone (`scripts/sm41_engine.py`, imported unchanged; the same loader, warm-up and show-time query rows as the full
  run), with its per-review log on. Two read-only hooks in the checking process record what the engine already
  computes for that lane:
  - the instant head's grade probabilities at the real show moment;
  - the answer-time head's distribution per grade on later-day rows.

  Each log row is checked against the real row (card, rating, answer ms, kind), so a mismatch stops the job.
- **Per later-day review:**
  - real grade, answer time (capped at 60 s, as the model sees it) and elapsed days;
  - the same-day steps that follow it on its day, in seconds;
  - the instant head's P(pass) at the show moment, and FSRS-7's from the pre-split refit;
  - the curve head's t90 before and after the review;
  - the card's pre-split later-day Again share (difficulty, shrunk to the user's mean with 4 pseudo-reviews);
  - the card's next later-day review in the window.
- **The rule's input.** x = ln(max(the card's last later-day answer ms, 100) / the user's median later-day answer ms
  over the 60 days before the split). These are the engine's own definitions, and the history before the split
  counts. Reviews without a previous later-day answer are left out.
- **Strata.** Within-user terciles of x (fast / mid / slow), plus fixed x bins for reference. Recall bands of the
  instant head's P(pass): < 0.70, 0.70-0.80, 0.80-0.85, 0.85-0.90, 0.90-0.95, >= 0.95.
- **Statistics.** Every number is computed per user first. Then it is the mean ± 95% t CI over users, paired where two
  strata are compared.

### Metrics and decision rules

1. **Is the instant head right about slow cards?**
   - Realised minus predicted pass rate, per stratum and per recall band.
   - Primary: the recalibration slope delta. It comes from a per-user logistic fit, pass ~ logit(P) + delta * x
     (users with at least 200 reviews).
   - Rule: delta > 0 with the CI excluding 0 means the instant head over-penalises slow previous answers. In that
     case part of the simulator's mechanism is a miscalibration, and a target shift of delta (logit per unit x)
     would only correct it. delta < 0 with the CI excluding 0 means it under-penalises them. Otherwise it is
     calibrated in x.
   - FSRS-7, which is blind to time, is reported on the same reviews.
2. **Is a slow card's review more expensive, and does it stay so?**
   - Primary: real seconds per later-day review, including its same-day steps, slow vs fast. Both strata are
     standardised to the pooled recall-band mix (matched predicted recall).
   - Also reported: answer seconds alone, lapse minutes (same-day steps after a later-day Again), and the
     answer-time head's prediction in the same strata.
   - Persistence: the per-user slope (elasticity gamma) of ln(answer time / median) on x, for this review and for
     the card's next later-day review. Real gamma is compared with the answer-time head's gamma on the same reviews.
   - Rule:
     - "more expensive and stays so" if slow vs fast > 0 and the next-review gamma > 0, both with the CI excluding 0;
     - the head "matches" if its gamma minus the real gamma has a CI that includes 0, or is within ±0.1.
3. **Is reviewing a slow card late cheaper, per unit of memory?**
   - Matching cells: within user and stratum, by the card's state, i.e. curve-head t90 before the review (within-user
     terciles) x difficulty (within-user terciles). Within each cell, late reviews (P < 0.85) are compared with on-time
     reviews (P >= 0.90). Each cell is weighted by its smaller side.
   - Outcomes: pass rate, seconds including steps, lapse minutes, and what the review bought.
     - Primary currency: the curve head's ln t90 growth (after - before). This is the model's reading of the real
       grade.
     - Secondary currency: the real next interval survived, in days (0 if the next review failed; only next reviews
       inside the window, so censored).
   - Cost per unit of memory, E_s(band) = mean seconds / mean ln t90 growth, per user, stratum and band. The late /
     on-time ratio is rho_s = E_s(late) / E_s(on time).
   - Primary Q3 metric: the interaction ln rho_slow - ln rho_fast, paired over users.
     - < 0 with the CI excluding 0 means that late is relatively cheaper for slow cards, as the rule assumes;
     - > 0 with the CI excluding 0 means the opposite;
     - otherwise slow and fast cards have the same shape, and only the cost level differs (point 2).
   - Also reported:
     - the same interaction with a positive currency, seconds per day of t90 after the review. With it, no user
       drops out for a non-positive mean gain;
     - a holding view: X = seconds per card-day saved by reviewing late, per unit of recall lost. Here
       rate = review seconds / the real interval it closed, and mem = (1 + P) / 2. The plan compares ln X between
       slow and fast;
     - the cost-optimal band per stratum (argmin of the pooled E_s(band));
     - its slope vs x (the change in logit of the optimal recall per unit x, slow vs fast). The rule's beta is this
       slope with the sign flipped. The CI is a bootstrap over users.
4. **Verdict and the implied beta.**
   - The rule shifts the target by -beta * x. To first order, its real-data counterpart is
     beta_real ≈ beta_sim x gamma_real / gamma_head + delta, where:
     - beta_sim = 2-3 is the simulator's optimum;
     - gamma_head is the answer-time head's elasticity, which drives the simulated cost;
     - gamma_real is the real elasticity. The simulator's cost lever scales with the cost elasticity;
     - delta is the calibration term of point 1.
   - **Supported** if all of these hold:
     - point 2 passes;
     - the instant head does not under-penalise slow answers (delta not < 0 with the CI excluding 0);
     - point 3's interaction is not > 0 with the CI excluding 0;
     - beta_real's CI includes some value in 2-3 or above it.
   - **Not supported** if any of these holds:
     - the slow cost premium is not real (CI includes 0);
     - it does not persist (next-review gamma CI includes 0);
     - beta_real's CI lies entirely below 1.
   - **Unclear** otherwise.

### What stays uncontrolled (stated before the run)

- **The scheduler chose the intervals.** A low predicted recall at review means either a long interval relative to
  the model's state (the user's scheduler disagreed with the model) or a review that was overdue (a backlog or a
  break). Both can go with other differences, such as tired sessions or neglected decks.
- **Selection.** Slow cards may be harder. Difficulty and stability are matched, but only through the model's state
  and the pre-split Again share.
- **The memory currency.** ln t90 growth is the model's own reading of the real outcome. The real next interval
  survived depends on the user's scheduler and is censored at the window end.
- **One review step.** Cost per unit of memory at a single review ignores what the review does to later intervals.
  The simulator's metric (minutes per memorised card) is a holding metric. So point 3 can disagree with the
  simulator without contradicting it. Point 4's beta_real links the two through the cost elasticity.

### Holds

- CPU only, 1 thread per process, below-normal priority.
- 2 processes while the beta = 3 confirmation runs, 4 after it (`policy/confirm3/runner.log` "done").
- `quiet.json` holds inside each job (next window: 20:00-01:00).

## Results

(to be filled from `realcheck/realcheck-tables.md` after the run; see [HANDOFF-realcheck.md](HANDOFF-realcheck.md))
