# Fluency vs desired retention in the simulator with answer-time feedback (SM127): design (pre-registered)

Written Wed 11-18 ~11:50, before any result of this experiment. Branch `sm127` (from `origin/sm41`), worktree
`C:/Users/jonat/workspace/wt/sm127`, code `scripts/sm127_fluency.py`. Everything below is fixed before the pilot.
The only numbers used to write it are the real-data feasibility counts in section 9 and a 10-day mechanical smoke run
(no policy outcome is read from it).

## 1. Question

User, 11-15: "Is a high desired retention (DR) enough for mastery, or do we need a scheduler that keeps answer time?"
At equal daily review time, one simulated year per user: desired retention 0.90 / 0.95 / 0.98 against a fluency
target scheduler. The fluency scheduler sets the interval where P(recalled AND answered within 8 s, reading the back
included) falls to 0.80. Read-out: memorised cards and fluent cards at day 365.

Answer time = the whole time on the card, from the moment the question is shown to the grade press, capped at the
preset's maximum answer seconds. This is Anki's stored time; there is no front/back split. "Answered within S s" in this
experiment always means answered within S s, reading the back included.

## 2. Simulator and model

- **Engine:** the review simulator hosting the small model (SM41) lane engine (`scripts/sm41_engine.py`), unchanged.
  `sm127_fluency.py` subclasses it.
  - One user, one split day, B lanes advancing together.
  - Daily shared-stream re-anchoring (fr-shared1) to the teacher-forced lane is on for every policy lane.
  - Query rows are at show time; commit rows run answer to answer.
  - First reviews draw their grade and time from the user's own recent pools, and so do same-day times.
  - Later-day grades come from the instant head. Later-day times are sampled from the answer-time head, log-uniform
    inside the bin.
- **Answer-time feedback (required by the duration effect (SM122)).** The sampled time of every review is fed back:
  - as the duration input of its commit row (the trunk's memory reads it);
  - into the 26 explicit time features of the head (the card's last 3 times, the user's medians);
  - into the session clock and the workload.

  The engine already did this in the earlier runs of the review simulator hosting the small model (SM41). The models read a slower success as weaker memory: doubling
  the last success's duration lowers the predicted next recall by about 1.6% for the show-time retrain (SM92). The
  comparison therefore includes that effort effect.
- **Model: the show-time retrain (SM92) with its own head from the time heads for the family (SM117).** The combined
  file is built by `prepare`: the trunk of the show-time retrain (SM92), with its sha256 equal to the family package manifest, plus its head from the time heads for the family (SM117)
  `wt/sm117/artifacts/small-model/sm117-sm92-head/head.pth` under the key prefix of the answer-time head on richer frozen features (SM82c). Only the model table changed.
  Why this model and not the show-time two-head cool-down for Small (SM118) with its head from the time heads for the
  new candidates (SM119):
  1. The duration effect was measured on this exact model (SM122: -1.60% [-1.78, -1.44]), so the feedback arm's
     mechanism is quantified for the learner we simulate.
  2. Its head's reliability at every S was measured (SM117): at S = 8 s the raw head is reliable (ECE 0.001 to 0.006,
     bar 0.01). The result of the time heads for the new candidates (SM119) reports calibration at 10 s, not at 8 s.
  3. The show-time retrain (SM92) is shipped as an option (draft build 112). The show-time two-head cool-down for
     Small (SM118) is held for the 11-28 release decision and may still change.
  4. The engine's interval convention is the show-time (e2s-query) input the show-time retrain (SM92) and its head from the time heads for the family (SM117) were trained and
     scored on.
- **S = 8 s needs no recalibration.** The raw head is already reliable at 8 s on all tested models: the six family
  heads (SM117), the time heads for the new candidates (SM119), and the time head for the show-time two-head cool-down
  for Tiny (SM126), reliable from 8 s upward. The per-user rolling recalibration (SM125) is needed only below about
  5 s. So no recalibration is applied.
- **Training-user overlap: none.** All users come from the 732 set (the 8k split's validation users with id < 5000).
  The show-time retrain (SM92) trunk was trained on the 8,000 train users. Its head from the time heads for the family (SM117) was trained on the 1,991 plan
  users of the recipe of the answer-time head on richer frozen features (SM82c). The review simulator hosting the small model (SM41) checked 0 overlap with the 732 set (DESIGN.md 10.1).

## 3. Users (fixed list, `users.json`, `select`)

- **Order:** the 732 set in numpy `default_rng(4141)` order (the order of the review simulator hosting the small model (SM41)).
- **User filters:**
  - 8k-150k reviews, >= 95% of them timed;
  - later-day retention < 0.99 over the history (Good-only users excluded).
- **One 365-day window per user:** the latest split day D on a 15-day grid, going back from (last day - 364), that
  passes all of:
  - >= 3,000 reviews before D;
  - 3,000-30,000 reviews in the window;
  - >= 150 active days;
  - >= 1,000 later-day reviews, with retention < 0.995 and >= 50 Again;
  - >= 300 new cards;
  - <= 8,000 cards in the window (RAM: about 13.6 KB of card + note state per card and lane).
- **Full design:** the first 150 users that qualify (538 looked at): 2.65M window reviews.
- **Pilot:** the first 12 of those 150: 1901, 4752, 2457, 384, 4337, 2820, 3505, 4503, 2552, 728, 2991, 4227, with
  259k window reviews.
- Outputs are aggregate only: per-user metric sums of benchmark users, no review rows.

## 4. Horizon, new cards, equal time

- **Horizon:** 365 days. Every lane runs the user's real year [D, D + 365); re-anchoring needs the real days.
  Policies are read at day 365. Retention and answer-time shares are read over the whole year and over the last 65 days.
- **New cards:** the user's real new cards, in real order, with their real note, deck and preset.
- **Equal time = a daily time budget, the same for every policy lane:** the user's real mean review seconds per
  calendar day over the window (all real review time in the 365 days / 365).
  - Each day: the due cards (overdue first, ties seeded), then the deferred new cards, then today's new cards, both in
    real order.
  - Once the day's answer seconds reach the budget, no new card and no further due card starts. Same-day steps of
    cards already shown that day still run, as Anki does for learning cards.
  - Unshown due cards stay due and come first the next day. Unshown new cards are deferred.

  This is "scale new cards": an expensive policy shows fewer new cards in the year. Time is matched by construction at
  the cap. A lane that does not need its budget spends less, and the yearly minutes are reported per lane, next to the
  overshoot from same-day steps. Card set: the cards of the real window. New cards a lane never showed count 0 at the
  read-out.

## 5. Arms (22 lanes per user)

| lane | what |
|---|---|
| `tf` | teacher forcing on the real schedule (the re-anchoring source) |
| `fr-shared1#0` | the real schedule with sampled grades and times and daily re-anchoring: the one-year drift check |
| `ihr-0.90/0.95/0.98` | instant-head desired retention (the policies of the review simulator hosting the small model (SM41)) |
| `flu-0.80` | **fluency target (primary)**: first interval on the 16-day candidate grid (1 to 1,000 days) where P(recalled and answered within 8 s) falls to 0.80; 1 day if it is already below at 1 day (a "floor" decision) |
| `flu-0.70`, `flu-0.50` | the same at 0.70 and 0.50 (sensitivity; 0.90 dropped as infeasible, section 9) |
| `flu-0.80-fb95` | `flu-0.80`, but floor cards get the desired retention 0.95 interval ("fluency where reachable") |
| `ihr-0.90-nofb`, `ihr-0.98-nofb`, `flu-0.80-nofb` | **no-feedback sensitivity arm**: later-day commit rows carry the user's pre-split median later-day answer time of the drawn grade, not the sampled one. Everything else is unchanged: sampled times still drive the workload, the clock and the head's explicit features |

- Every policy lane runs with 2 seeds.
- Common random numbers: lanes of the same seed share their draws card by card. A `-nofb` lane therefore draws exactly
  what its feedback twin draws, until the first later-day commit makes them differ (proof K3).
- P(recalled and within S) on a candidate row = sum over Hard, Good and Easy of P(g) x P(t <= S | g). P(t <= S | g) is
  read from the 16 log-time bins, log-uniform inside the bin, exactly as the sampler draws (proof K2).
  - The candidate rows are the instant-head policy's counterfactual rows: elapsed = T days, the card's post-commit
    state.
  - The explicit features use the card's lane history at elapsed T.

## 6. Metrics (per lane, seeds averaged per user)

- **Memorised (primary):** the expected memorised cards at 12:00 of day 365, = the sum over the card set of P(recall)
  from a query row on the lane's final states.
- **Fluent (primary):** the expected fluent cards, = the sum of P(recalled and answered within 8 s, reading the back
  included) on the same row: the expected number of cards answered correctly within 8 s if every card were asked at
  day 365.
- Both count only cards the lane has shown: old cards, plus the new cards it introduced.
- **Secondary:**
  - threshold counts: cards with P(recall) >= 0.9, cards with P(recalled and within 8 s) >= 0.8, and both;
  - minutes per day and reviews per day;
  - new cards shown;
  - later-day retention;
  - the realised share of later-day reviews passed and answered within 8 s, over the whole year and over the last
    65 days, against the head's P of it at the show moment;
  - floor share (fluency lanes);
  - budget-hit days;
  - per-minute efficiency (memorised or fluent cards / yearly minutes).
- **Comparisons:**
  - The relative gap of a over b per user (a / b - 1), averaged over users.
  - 95% CI by user bootstrap (10,000 resamples, seed 127).
  - Paired: same user, same split, same draws.

## 7. Decision rule (fixed now)

- **DR\*** = the desired-retention arm (0.90 / 0.95 / 0.98) with the most fluent cards averaged over the users. G = the
  fluent-card gap of `flu-0.80` over DR\*, with its CI. (Choosing DR\* on the same users favours DR, which is
  conservative for the claim "a fluency scheduler is needed".)
- **"A high desired retention is enough"** if G's 95% CI upper bound is below +2%.
- **"A fluency scheduler is needed"** if G >= +2% with the CI lower bound above 0, and the memorised gap of `flu-0.80`
  over DR\* is no worse than -2%. If the memorised gap is worse than -2%, the result is reported as a trade-off (more
  fluent cards, fewer memorised ones).
- Otherwise: **inconclusive**.
- **Feedback.** The same rule is applied to `flu-0.80-nofb` against the better of `ihr-0.90-nofb` and `ihr-0.98-nofb`.
  If the two verdicts differ, the effort effect changes the answer. The feedback arm is the primary one, since it is
  the realistic one.
- The secondary fluency arms (`flu-0.70`, `flu-0.50`, `flu-0.80-fb95`) are reported against DR\* with the same
  numbers. None of them replaces the primary verdict.
- **Pilot (12 users): not decisive.** It only decides go or no-go for the full run. Go if all of these hold:
  - no mechanical failure;
  - the budget behaves (no rule violation, overshoot reported);
  - the drift check stays within the earlier one-year expectation (section 8).

  It also reports the floor share of `flu-0.80` and whether the full run needs a different fluency definition
  (section 9).

## 8. Drift caveat

- **What re-anchoring removes.** The deck, preset and user states are re-anchored every day to the real user (teacher
  forcing). A lane's own history therefore cannot make the simulated user faster or slower overall. Only the card and
  note states carry the lane's own grades and times.
- **What feedback can and cannot change.** Answer-time feedback acts on the cards' memory (the trunk) and on the head's
  card-level time features. It does not act on the user-level speed.
- **Earlier drift numbers.** Measured in the earlier runs of the review simulator hosting the small model (SM41): -0.6 ± 1.0 at days 30-59, and about -3 at days 60-89
  of a 120-day run. One year was never measured.
  - The `fr-shared1#0` lane measures it here: first-touch predicted pass minus teacher forcing, over the year.
  - The drift is model state, not outcome noise.
- **Reading rule.** A drift beyond -5% at one year is flagged. Absolute numbers (fluent shares in particular) are then
  read as optimistic or pessimistic, and only the paired policy gaps are interpreted.
- **Counterfactual intervals.** Policies outside the real interval range (daily floors, 0.98 intervals) can be read by
  the model as "easy card" or "hard card" (SM41 DESIGN.md 5).

## 9. Feasibility of the plan's target (real data, read before launch)

- **Real pre-split shares, over the 150 users.**
  - Later-day passes answered within 8 s, reading the back included, as a share of passes: median 0.46 (quartiles
    0.27-0.66).
  - P(passed and within 8 s) over all later-day reviews: median 0.40. Only 4.7% of the users reach 0.8, and 9.3%
    reach 0.7.
  - Two pilot users (1901, 384) have a median pass time of 50-60 s, which is the cap.
- **What this means for `flu-0.80`.** At S = 8 s, a target of 0.80 is above what most cards can reach at a 1-day
  interval, so the floor rule will send many cards to daily reviews. Under the budget, that means drilling old cards
  and deferring new ones. This is what the plan's definition does, so it stays the primary arm.
- **Mitigating arms.** `flu-0.80-fb95`, `flu-0.70` and `flu-0.50` are there to show what a feasible fluency
  definition gives.
- **Open decision for the user (before a full run):** if the pilot's `flu-0.80` floor share is above 50%, the fluency
  definition for the full run is the user's choice. The options:
  - keep it (the honest answer is then "a pure 8 s / 0.8 target is not a usable scheduler");
  - use "fluency where reachable" (fb95);
  - use a lower target;
  - use a per-user S (for example the user's own median Good time).

## 10. CPU proof (`check`, `check-cpu.json`), bars fixed now

- **K1 model:**
  - the sha256 of the show-time retrain (SM92) equals the manifest;
  - the combined file equals the trunk and the head bitwise;
  - checks C1-C3 of the review simulator hosting the small model (SM41) pass on the combined model:
    - the builder equals the pipeline;
    - the recurrent path equals the parallel path, including the 448 frozen time-head features (instant and curve
      within 1e-4, features within 1e-3);
    - the explicit features within 1e-6.
  - The checks run on user 722, the user of the own proof of the review simulator hosting the small model (SM41). A first try on user 3876 failed C1b (3.0) and C2
    (about 7e-2). It fails identically with the earlier master of the answer-time head on richer frozen features (SM82c) (diagnostic `check-c123-3876-sm82c.json`), so the
    model swap does not cause it. It was not investigated further; C1 and C3 pass on that user.
- **K2:** the fluent CDF equals the sampler's share at or below 8 s: 64 rows, 200k draws each, max gap below 0.004.
- **K3 feedback path (20 days):**
  - every commit row's duration input equals the scaled sampled time;
  - in `-nofb` lanes, every later-day commit row's duration input equals the scaled user median of its grade;
  - each `-nofb` twin is identical to its feedback lane up to its first later-day review.
- **K4 budget:**
  - no due or new card starts at or beyond the budget;
  - the new cards each lane introduces are a prefix of the real new-card order.
- **K5 targets (3 users, 60 days, no budget, days 30-59):** each instant-head lane's realised later-day retention is
  within 0.02 of its own aim, pooled, and rises with the target. The fluency lanes are reported: realised share passed
  within 8 s vs the aimed P, and the floor share.
- **K6:** the same run twice gives identical output.
- **K7 batched equals alone:** `flu-0.80#1`, `ihr-0.98-nofb#0` and `flu-0.80-fb95#0`, each run alone with `tf`, match
  the 22-lane run: traces identical, counters and read-outs within a relative 1e-4.
- **K8:** speed on the full lane set, which sets the cost below.

## 11. Cost, pilot, what runs

- **Runner:** CPU only, no GPU and no queue ticket.
  - 4 processes x 1 thread at below-normal priority, each under 6 GB (largest pilot window 6.9k cards x 22 lanes,
    about 2 GB of state).
  - Before each user the runner checks quiet.json (scope all / cpu) and the daily 20:00-01:00 break (Sunday from
    16:00). It does not start a user within 60 min of a window. A running user pauses at the next simulated day inside
    a window.
  - The STOP file is checked before every user and every simulated day.
  - Resumable per user: the output file exists = done. A killed user restarts from its warm-up.
- **Cost.** About 22 lane reviews per window review plus candidates, at roughly 600-700 lane reviews/s per process
  (smoke: 688/s; K8 gives the measured number).
  - Pilot: about 5.7M lane reviews, about 3-4 CPU process-hours, about 1.5 h wall.
  - Full design (150 users): about 58M lane reviews, about 30 CPU process-hours, about 8 h wall on 4 processes. That
    is above the ~10 h bar and needs the user's go, so **only the pilot is launched**.

## 12. Amendment, Wed 11-18 ~14:00 (after the pilot; written before the time-drift diagnostic runs)

### 12.1 Comparison statistic (changed after seeing the pilot)

- **What changed and why.** Section 6 fixed the comparison statistic as the mean over users of the per-user relative
  gap (a / b - 1). The pilot showed that this statistic explodes when one user's b is near 0: the fluency scheduler
  at 0.80 with vs without answer-time feedback reads +1012% on fluent cards, while the population means are 772 vs
  768 fluent cards. This change is made **after seeing the pilot**, because of that defect; it is not a pre-registered
  choice.
- **New primary statistic:** the ratio of population totals, sum over users of a / sum over users of b - 1, with a
  95% paired user-bootstrap CI (10,000 resamples, seed 127: one resample matrix shared by all comparisons). Reported
  next to it: the median per-user ratio. The old statistic stays in the report as a secondary table, with its
  numbers unchanged (same rng sequence).
- **Decision rule:** section 7 unchanged, applied to the new statistic (G = the ratio-of-totals fluent gap of the
  fluency scheduler at 0.80 over the best desired retention, with its CI; the memorised gap likewise). The pilot's
  verdict is reported under both statistics. **A full run would use the ratio of totals as the primary statistic.**
- Report code: `report` in `scripts/sm127_fluency.py`; pilot re-read into `report-pilot-v2.out`, `sm127-tables-v2.md`,
  `sm127-result-v2.json`. No re-simulation.

### 12.2 Time-drift diagnostic (pre-registered here, before it runs)

**Question.** With answer-time feedback, desired retention 0.90 costs about 18.4 s per review against 15.5 s without
feedback (later-day reviews: 19.3 s vs 15.7 s). Is this (a) a self-feeding loop of sampled answer times (a sampled slow
time is fed back, the model then predicts the same card slower, and so on), or (b) a real effect of the policy
reviewing different cards (other cards, other intervals, other grades)?

**Already seen in the pilot rows (stated so that nothing below is chosen with hindsight).**
- On the real schedule, sampled-and-fed-back later-day times (`fr-shared1`) are +7% over the real ones over the year
  (pooled), month by month between 0.97x and 1.15x, with no rise over the year (months 1-6 about 1.08x, months 7-12
  about 1.08x).
- The feedback-vs-no-feedback gap in later-day seconds per review grows over the first three months and then stays
  flat, only at desired retention 0.90 (1.10x in month 1, about 1.25x from month 3 on). At desired retention 0.98 and
  for the fluency scheduler at 0.80 the feedback lanes end *faster* than their no-feedback twins (about 0.89x and
  0.85x in month 13). A one-directional "slow begets slow" runaway would push every feedback lane up.

**Run.** `scripts/sm127_timedrift.py`, the 12 pilot users, days 0-149 of the same 365-day set-up (same budget, same
draws), 8 lanes with a per-review log kept in memory (only aggregates are written):
- real schedule: teacher forcing (`tf`, real grades and times); sampled grades and times fed back, daily re-anchoring
  (`fr-shared1#0`); the same without re-anchoring (`fr#0`); the same re-anchored, with the user's median time of the
  grade in the trunk's duration input instead of the sampled one (`fr-shared1-nofb#0`, new);
- the twins: desired retention 0.90 with and without feedback (`ihr-0.90#0`, `ihr-0.90-nofb#0`) and desired retention
  0.98 with and without (`ihr-0.98#0`, `ihr-0.98-nofb#0`), seed 0.
- Identity check: the lanes also in the pilot must reproduce the pilot rows' 15-day buckets (time and load) to 1e-4;
  otherwise the diagnostic is not read.
- Cost: about 1.3-1.6 CPU process-hours on 3 processes (budget 2).

**Measures (later-day reviews; all pooled as ratios of totals over users, paired user bootstrap as in 12.1).**
- **P, persistence:** the within-user correlation of the log answer time of consecutive later-day reviews of the same
  card (pooled within-user covariance over pooled variances), days 0-74 and 75-149. Real answer times give the truth
  (`tf`).
- **T, twin decomposition** at desired retention 0.90 (and 0.98): the log of the feedback / no-feedback ratio of
  seconds per later-day review is split into a **within-cell** part (the same card, in the same interval bucket
  (<= 3, 4-15, 16-63, >= 64 days) and the same pass/fail, slower or faster) and a **selection** part (which cards,
  intervals and grades get reviewed, including cards reviewed in only one lane). Both weightings (feedback's cell
  counts, no-feedback's) are reported; the share is their mean.
- **M, median vs sampled vs real on the real schedule:** seconds per later-day review and the model's own expected
  time per review (bin midpoints, the `time_later` read-out) for real, sampled and median-in-trunk feedback.
- **N, no re-anchoring:** `fr#0` against `fr-shared1#0`, by 30-day month.

**Reading (fixed now).**
- **(a) Drift, a self-feeding loop:** the within-cell share of the desired retention 0.90 gap is >= 1/2 **and** the
  fed-back sampled times persist more than real ones (P of `fr-shared1#0` minus P of `tf` >= +0.05, or P of
  `ihr-0.90#0` minus P of `ihr-0.90-nofb#0` >= +0.10).
- **(b) A real effect of the policy reviewing different cards:** the selection share is >= 1/2 and the persistence
  excess of `fr-shared1#0` over `tf` is below +0.05.
- **Within-card but realistic:** within-cell share >= 1/2 but persistence within +0.05 of real: the same cards are
  slower with feedback, while fed-back times persist no more than real ones. Read as the effort effect acting on the
  lane's own card histories, not as a runaway loop; M then says whether the median-in-trunk arm is the biased one.
- Otherwise: **mixed**, reported as measured.
- **N** is informational: a rise of `fr#0` over `fr-shared1#0` of >= 5% by days 120-149 means user-level drift that the
  daily re-anchoring removes (every policy lane is re-anchored).
- **If (a):** propose the minimal fix with its cost (for example the per-user median of the grade, or the model's
  expected time, as the fed-back duration of later-day reviews; or a daily re-anchoring of the time stream). No full
  re-simulation is run here.
