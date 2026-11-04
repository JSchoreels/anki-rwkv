# The small model in the review simulator (SM41): design note (stage 1)

Goal: a card-level review simulator whose learner is our exported small RWKV model (the released leak-free small
model + the answer-time head on richer frozen features (SM82c)), not FSRS-6 curves, so that scheduling becomes an
offline search an AI agent can run (policy in, memorised cards per time out). Stage 1 = this note + a CPU prototype on
5 users of the 732 set ([prototype-result.md](prototype-result.md)), code `scripts/sm41_sim.py`.

## 1. The simulated learner

The model is a recurrent network with one state per card, note, deck, preset and user (the 5 streams), so a learner
cannot be a set of independent cards: every review moves the user, deck and preset states, and so changes every
later prediction of every card. A simulated learner is therefore:

1. **Warm-up**: one real user's history up to a split day, fed through the recurrent model one review at a time (the
   recurrent form, `expertium/model/variant_rnn.py`, matches the parallel scorer: check C2 below). This gives the
   state of every entity at the split, and for every card the forgetting curve of its last review.
2. **Lanes**: copies of that state, each rolled forward for N days under one schedule. The state tensors are never
   written in place, so a lane only copies the entity maps (cheap); lanes share the warm-up.
3. **Per review** in a lane (2 model steps):
   - the instant (query) row: the card's inputs at the review time with the grade and time columns blank, run
     WITHOUT committing state (exactly how the model is trained and scored) -> P(Again/Hard/Good/Easy) -> **sample the
     grade**; the answer-time head (SM82c) on the same row's frozen features (448: the residual stream after the input
     encoder and after each of the 4 layer rounds + the instant head's hidden layer) and the 26 explicit time features
     -> 16 log-time bins per grade -> **sample the time** for the sampled grade (log-uniform inside the bin, the last
     bin = the 60 s cap);
   - the review row with that grade and time -> **commit** every stream's state; the row's GRU curve head (3 power
     curves, closed form) gives the card's recall R(t) for any next interval -> the scheduler.
4. **What the model does not cover, and what the simulator uses instead** (all from the same user's own recent
   history, the last 60 days before the split, at least 2,000 reviews):
   - first reviews: the model has no instant row for a new card (nothing to predict from), so the first grade is
     drawn from the user's recent first-grade mix and its time from the user's first-review times of that grade;
   - same-day reviews (learning and relearning steps): the grade comes from the model (instant rows exist), the time
     from the user's same-day times of that grade (the time head is trained on later-day reviews only: same-day
     interval inputs contain the previous answer's time, response-time prediction (SM82) stage 0);
   - learning steps: whether a card comes back the same day, and after how many seconds, follows the user's own
     empirical rule per (review kind first / same-day / later day, grade): P(same-day follow-up) and the observed gaps
     (cap 8 reviews per card per day). Only the day-level interval is the scheduler's choice;
   - the clock: the dataset has no timestamps, only each card's elapsed seconds. A simulated day has one session that
     starts at 12:00 + N(0, sigma) (sigma from the spread of the user's real elapsed seconds around whole days,
     clipped to 10 min..6 h); reviews follow one another by their answer times; a card's elapsed seconds = clock now -
     the clock at its last answer (warm-up cards: their last day at 12:00);
   - the inputs: `Builder` reproduces `get_rwkv_data` + `add_queries` + the batch encodings row by row (check C1:
     bitwise on the 24 base columns; all 92 columns to 4.5e-7 with the scorer's id codes). New entities get fresh
     random id codes (training draws them at random too), the same codes in every lane (common random numbers). The
     revlog `state` column needs no simulation: the small model zeroes that input.
5. **New cards** come from the user's real future: each simulated day introduces the real new cards of that day, in
   real order, with their real note / deck / preset (so siblings and deck structure are real). Beyond the real
   horizon (longer runs): continue at the user's recent new-card rate with fresh card ids in the user's most used deck
   and preset (fresh notes), and say so.
6. **Card set**: the cards the real user reviewed in the window (old and new). Old cards the user did not touch in
   the window are left out (suspended, deleted or simply not due under the real schedule cannot be told apart); this
   understates the workload of short-interval policies a little.

## 2. How a scheduler plugs in

A policy sees, after every review of a card, the card's recall curve R(t) (closed form, invertible by bisection), the
card's history, and the user's day so far; it returns the next day. Implemented:

- **desired retention r** (`dr-0.80`, `dr-0.90`, `dr-0.95`): next interval = round(t) days, R(t) = r, at least 1 day; on
  each day all due cards (overdue first), then the day's new cards; same-day steps by the user's rule.

Next (stage 2), same interface:

- **time-aware**: choose the interval that minimises expected answer time per unit of retained memory, using the
  answer-time head at the candidate interval. A candidate interval only changes the instant row's elapsed inputs, and
  instant rows do not commit state, so K candidates = one batched step of K query rows (no extra sequential cost).
- **daily limits / priorities** (fsrs-rs `SimulatorConfig`: max cost per day, review limit, priority functions): the
  session loop already orders due cards; a time budget stops the session and leaves the rest due.
- **the agent search**: the policy is a small parametric function (e.g. r per card difficulty bucket, a time budget,
  an interval cap); the agent proposes parameters, the simulator returns the metrics below on a fixed user set with
  fixed seeds; held-out users judge the winner.

## 3. What to measure

Per lane and user, for the same card set and days: reviews and minutes per day, retention on later-day reviews (by
kind and grade mix), memorised cards at the horizon (sum of each card's predicted recall at the end), minutes per
memorised card, new-card "time to fluency" (answer time spent on a new card until its predicted 90%-recall interval
reaches 21 days; stage 1 reports the share of new cards that reach it by the end and the median time spent per new
card), the backlog on the split day (old cards already overdue under the policy).

## 4. Validity checks

- **C1-C3 (CPU proof, `check`)**: the incremental input builder == the training/scoring pipeline; the recurrent step ==
  the parallel model (instant and curve predictions, and the 448 frozen time-head features); the incremental 26
  explicit features == `sm82c_head.explicit_features`.
- **V1 teacher forcing (`real-tf`)**: the real continuation, real grades and times fed back: predicted vs real
  retention, answer time (the model's ordinary prediction task, on these days).
- **V2 free running on the real schedule (`real-fr`)**: the same cards, days and elapsed seconds, but every grade and
  time sampled and fed back. Should reproduce the real retention, grade mix and minutes per day, but the comparison
  is confounded: the real schedule reacted to the real outcomes, so a sampled lapse keeps the long interval that
  followed a real pass (and a sampled pass the short one after a real lapse). Diagnostic lanes split the loop:
  `real-fr-grade` (grades sampled, times real) and `real-fr-time` (grades real, later-day times sampled).
- **V2' clean drift check (first touch)**: each old card's first later-day review in the window was scheduled from
  real pre-split outcomes, so its day does not depend on the lane's samples; only the user-level state (fed the
  lane's samples since the split) differs. Realised and predicted pass rate on these reviews, real vs free running, per
  30-day block: the drift of the model on its own samples, without the schedule confound.
- **V3 policy retention**: under `dr-r`, the realised later-day retention should be close to r (the curve head chose
  the day; the instant head grades the review: two heads must agree).
- Later: the same users through the FSRS-6 simulator of response-time prediction (SM82) stage 2 (same policies), to
  see where the two learners disagree; held-out real "natural experiments" (reviews whose real interval was far from
  the user's usual interval for that card state) to check the model's calibration off the usual schedule.

## 5. Known risks

- **Drift on its own samples (exposure bias)**: the model was trained on real histories; fed its own samples, its
  user state can drift (e.g. sampled Easy grades or long times make it predict more of the same). V2 measures it;
  mitigations if it grows with the horizon: shorter horizons with many warm-up split points, re-anchoring (periodic
  teacher-forced real days where available), lower sampling temperature for the time head.
- **Counterfactual intervals**: in the training data intervals were chosen by Anki's schedulers from the card's
  past, so the interval itself carries information ("a long interval = an easy card"). A simulated policy that gives
  a hard card a long interval may be read by the model as "easy card", which is optimistic. Check: V3 at extreme r,
  and the natural-experiment calibration above; prefer policies whose intervals stay inside the range real schedulers
  produce for that card.
- **Habit changes and degenerate users**: users change (user 1190 moved to Good-only before its split; the habits
  therefore come from the last 60 days). Good-only users (1190 in the smoke test, 4326 among the 5 prototype users) have no failure signal:
  their retention is ~1.0 under any policy; flag and exclude them from policy conclusions.
- **Clock and same-day steps are empirical**, not modelled: they are the same across lanes, so policy comparisons are
  fair, but absolute minutes per day carry this approximation.
- **Card set**: see 1.6.

## 6. Speed and the full run

Prototype (CPU, 1 thread per process, eager PyTorch, one row per step): measured in
[prototype-result.md](prototype-result.md) (147 simulated reviews/s per process; warm-up 268 reviews/s).
Python overhead dominates (a step is ~13 layer calls on a 64-wide vector), so the full run batches **lanes**: all
learners (users x policies x seeds) advance one review per tick, their states stacked along the batch dimension
(`bench`: one step over B learners). Learners whose day has ended wait; new entities start from a zero state with a
"fresh" mask (the time mixer's first-step token shift = its own input). Warm-up for thousands of users is better done
by the parallel scorer path with a state export at the split (the state-carry machinery already carries the
recurrent state across segments), at scorer speed (31.9M reviews in ~11 min on the GPU), instead of row by row.
Costs and the plan: [prototype-result.md](prototype-result.md) section "Full run".

## 7. Does the srs-simulator repository matter?

Not for stages 1-2: the simulator here is self-contained and the learner must be our stateful model, which no
existing simulator can host without our runtime. fsrs-rs's `simulation.rs` (read locally) is the reference for the
scheduler side (daily cost limits, review limits, priority functions, learning-step transitions, per-state costs);
our `user_habits` and the answer-time head fill the same roles. The open-spaced-repetition srs-simulator becomes
useful only for comparability with published policy results (its scheduler implementations as policies); hosting
our model there needs the fork runtime port (SM08). If wanted, that is a later, separate step and needs the user's
go-ahead to clone it.

## 8. Stage 2 part 1: the drift study (`scripts/sm41_drift.py`)

### 8.1 Interval convention and the duration leak (duration-leak scoring (SM31))

- **Training data.** `elapsed_seconds` runs from the card's previous answer to THIS answer, so it contains this review's own answer time. It feeds columns 2-7 (the interval, its sin / cos, the card's cumulative interval, its sin / cos) of the review (commit) row and of the instant (query) row alike. The query row therefore carries part of the answer it is asked to predict.
- **The app** queries when the card is shown: the interval runs up to the show moment.
- **Stage 1 (`sm41_sim.py`)** differed by lane type:
  - Real-schedule lanes (`real-tf`, `real-fr*`) used the data's answer-to-answer interval in both rows, as in training. In a free-running lane, the query row therefore contained the REAL answer time, not the lane's.
  - Scheduler lanes (`dr-*`) measured both rows from the lane's previous answer to the show moment. The query rows were app-like, but the commit rows were shorter than training's by this review's answer time.
- **Stage 2 (`sm41_drift.py`, default conv `show`):**
  - Query rows are at show time: from the lane's own previous answer of the card to the real show moment, floored and >= 0. The cumulative interval = the earlier intervals + this one. For teacher forcing these are exactly the SM31 `e2s-query` inputs.
  - Commit rows are answer-to-answer with the lane's own answer time (the training convention). For teacher forcing they equal the data bitwise.
  - The sampled answer time therefore feeds the card's next interval twice: the next query starts from the end of this answer, and the next commit row includes the next answer. Both are consistent with training.
  - The arm `fr-answer` (and `tf-answer`) keeps stage 1's convention, so the study measures what the convention changes.
- **Can the leak explain the drift?** Mostly not, by construction:
  - For later-day reviews the answer time is 10-60 s out of an interval of days: about 1e-4 of the log-scaled interval, and well under 0.01 rad of the time-of-day angle.
  - For same-day steps it is a larger share (a 20 s answer in a 10 min step), but stage 1's drift is on later-day reviews.
  - The stage-1 free-running lanes saw the REAL answer time through the leak, which anchors them to reality rather than pushing them away.
  - Measured on one user (3876, 90 days, the proof run): teacher-forced grade LogLoss with answer-to-answer vs show-time query rows is 0.8101 vs 0.8101 on later-day reviews and 0.2545 vs 0.2545 on same-day ones (equal to 4 decimals). The study reports this over all users (report section "The duration leak").
  - Retraining on show-time intervals (SM91) is not part of this study.

### 8.2 Design

- **Users and splits.** 50 users of the 732 set (rule and ids in `drift/users.json`, `select`, numpy `default_rng(4141)` order), with Good-only users excluded. Each user has 2-3 split points, 60 days each, with windows at least 60 days apart: 133 splits, 621k window reviews. The long horizon uses 10 of these users, 1 split each, 120 days (90k reviews).
- **Lanes.** Every split runs on the REAL schedule: same cards, days and review order. Reviews per day are therefore fixed by construction; the load drift is reported as the implied same-day follow-ups. Each split has 2 teacher-forced lanes (`tf`, `tf-answer`) + 10 free-running arms x 3 seeds = 32 lanes, advancing together as one batch (B = 32). Common random numbers: every lane draws the same 3 uniforms per review, whatever it uses, and lanes of the same seed share them.
- **Free running.** Non-first grades are sampled from the instant head, and later-day answer times from the answer-time head. First-review grades and times, and same-day times, stay real (stage 1's `real-fr`).
- **Arms (mitigations):**

  | arm | change |
  |---|---|
  | `fr` | none |
  | `fr-answer` | stage 1's interval convention |
  | `fr-treal` | real times: the bound for any time fix |
  | `fr-texp` | exp of the expected ln time fed back instead of a sample |
  | `fr-tt0.5` | time head sampled at temperature 0.5 |
  | `fr-gt0.8` | grade head sampled at temperature 0.8 |
  | `fr-anchor7` | every 7th day teacher forced (real grades and times) |
  | `fr-shared1` / `fr-shared7` | the deck / preset / user states replaced by the teacher-forced lane's at the start of every day / every 7th day; card and note states stay the lane's own |
  | `fr-texp-shared1` | `fr-texp` and `fr-shared1` together |

- **Metrics.** Per 15-day bucket and lane, counters are kept for:
  - first-touch reviews (each old card's first later-day review in the window): realised and predicted pass rate;
  - all later-day reviews, old cards and new cards separately;
  - grade counts (same-day, later-day);
  - later-day answer time: mean, geometric mean and the head's expectation;
  - the same-day follow-ups implied by the user's step rules;
  - teacher-forced grade LogLoss.
- **Drift** = lane minus real (realised) or minus teacher forcing (predicted), first per split, averaged over a user's splits, then mean ± 95% CI over users.
- **Primary metric, fixed before the run:** the first-touch STATE drift (the predicted pass rate of the free-running lane minus teacher forcing on the same reviews; it has no outcome noise) on days 30-59. Secondary metrics: Easy share of passes, answer time, Again share.
- **Speed.** About 120 batched reviews/s per process for 22 lanes (2,660 lane reviews/s, 1 thread), against 147 reviews/s for one stage-1 lane. The warm-up runs once per user for all its splits.
- **CPU proof (`proof`, `drift/proof-cpu.json`):**
  - the same seed gives identical counters;
  - batched == unbatched: every lane run alone (B = 1, or B = 2 with the teacher-forced lane for the shared-stream arms) against all lanes together, on 2 users;
  - teacher forcing in stage 1's convention reproduces stage 1's `real-tf` lane (user 3876, 90 days: predicted later-day retention 0.8689, 976 first-touch reviews at 0.8407 predicted, 19.79 s predicted per later review, all identical).

## 9. Stage 2 part 2: the lane engine for policies (`scripts/sm41_engine.py`)

Part 1's lock-step batch only replays the real schedule: every lane reviews the same card at the same tick. A policy
moves cards to other days, so here every lane reviews its own card at a tick. Result: [engine-result.md](engine-result.md).

### 9.1 Engine (decisions)

- **One engine = one user, one split, B lanes.** Lane 0 is teacher forcing on the real schedule (`tf`). Days run in step for all lanes. Within a day, every active lane does one review per tick, so a tick is one batched query step and one batched commit step. A lane whose day is over waits. The instant-head policy's candidate rows (9.2) ride along in the next tick's query step: rows of a batch are independent, and a card that left the day is not needed again before the next day (pending decisions are flushed at the end of the day). This saves one model call per tick (+20% throughput).
- **State arenas.**
  - There is one tensor [entities x lanes x ...] per stream, per layer and per state tensor (token shift, wkv state, channel shift), filled from the warm-up.
  - Entities are the card set (every card of the real window, as in stage 1 section 1.6), their notes, decks and presets, and the user.
  - A tick gathers (entity of lane b, lane b) for every active lane and scatters the commit back: about 40 indexing calls per tick, whatever B is.
- **New entities: the fresh mask.**
  - An entity a lane has not reviewed yet must behave like the stock first step (state None): the token shift reads the row's own layer-normed input, and the wkv state is zero.
  - In a batch where only some rows are new, forward hooks on the mixers' layer norms copy that output into the shift buffers of the fresh rows before it is used, and their wkv rows are zeroed.
  - No model code is copied. P2 checks this path against the stock step.
- **Inputs.**
  - `sm41_sim.Builder` is vectorised over lanes (per-lane counters, per lane x card history).
  - The id codes are drawn once, in the real window order, so every lane uses the same codes, and they equal part 1's.
- **Interval convention.** Part 1's `show` (8.1).
  - Query rows run from the lane's previous answer of the card to the show moment.
  - Commit rows run answer to answer, with the lane's own answer time.
- **Clock (policy lanes).** Stage 1's session: the start time is 12:00 + N(0, sigma), reviews follow each other by answer time, and same-day steps come back after the user's own gaps. One rule is added: the clock never runs backwards, so a session that runs past the next day's start delays that start. Without this rule, a long late session gave negative intervals on the next day (found in the smoke test).
- **Re-anchoring by default.** Daily shared-stream re-anchoring (fr-shared1) is on for every lane.
  - At the start of every day, the deck, preset and user states (and their fresh flags) of every anchored lane are copied from lane 0. Card and note states stay the lane's own.
  - The suffix `-free` turns it off.
  - Consequence: a run cannot go past the end of the user's real history.
- **Replay lanes** (`tf`, `fr`, `fr-shared1`, `fr-shared7`) replay the real schedule exactly as part 1 does.
- **Policy lanes** run their own sessions:
  - each day, the due cards (overdue first, ties in a seeded order), then the day's real new cards in real order, all of them (no daily limits yet);
  - same-day steps by the user's own rules;
  - first-review grades and times, and same-day times, from the user's recent pools (stage 1's scheduler lanes);
  - later-day grades from the instant head, later-day times from the answer-time head.
- **Draws.**
  - Policy lanes use counter-based uniforms (a splitmix64 hash) keyed on (user, split, seed, card, the card's review number in the lane, slot). Lanes of the same seed therefore draw the same numbers card by card under every policy (common random numbers). A lane run alone draws exactly what it draws in a batch, so batched == alone can be checked review by review.
  - Replay lanes keep part 1's random stream, so they reproduce part 1's rows (P3).

### 9.2 Policies

- **Instant-head desired retention (`ihr-R`).** This is stage 1's follow-up "choose intervals with the instant head", and the machinery the time-aware policy needs.
  - When a card leaves the day, K = 16 candidate query rows are built, at T = 1, 2, 3, 4, 6, 10, 16, 25, 40, 63, 100, 158, 251, 398, 631 and 1000 days.
  - All 16 rows use the card's post-commit states and go through one batched step (the next tick's query step, 9.1).
  - The interval is the first crossing of P(pass) = R, by log-linear interpolation on the grid, rounded, at least 1 day. If P(pass) is already below R at 1 day, the interval is 1 day; if it never falls to R, it is 1000 days.
  - The candidate rows are counterfactual, so they need rules (decisions):
    - elapsed = T whole days;
    - the day gap since the lane's previous review = 0 (mid-session);
    - the new cards and reviews since the card's last review = the user's recent per-calendar-day rates x T (the 60-day window of `user_habits`, the same for every policy);
    - today's counters at half the user's mean per active day;
    - weekday and day encodings at the candidate day;
    - the shared streams as they are now (their future is unknown).
  - On the split day the same rows are built from the warm-up state for every old card. Cards whose interval has already run out are due on day 0 (the switch backlog).
- **FSRS-7 baseline (`fsrs7-R`).** No FSRS-7 scheduler existed in this repository, so the simplest equivalent was added.
  - It is a numpy port of srs-benchmark's finished 34-parameter FSRS-7 (`models/fsrs_v7.py`: dual stability, the mixture forgetting curve, `--secs` intervals in fractional days).
  - It uses each user's own srs-benchmark fit (`result/FSRS-7-short-secs.jsonl`, read only), or the default parameters when there is none.
  - It is updated at every review of the lane, same-day ones included, with the lane's own grades and answer-to-answer intervals.
  - The interval solves R(t) = R by bisection, rounded, at least 1 day.
  - Caveat: the fit saw most of the user's history, usually including the window. Fixed for the full run: pre-split refits (10.1).
- **Curve-head desired retention (`curve-R`).** Stage 1's policy (the commit row's GRU power curves, `time_at`), kept to measure the gap between the two heads.
- **The time-aware objective is deferred.** On the same candidate grid it changes only the selection rule. But a one-step objective, such as expected answer time per day of retained memory, has no interior optimum with these power curves: like the cost ratio of response-time prediction (SM82) stage 2, it keeps falling as intervals grow. It needs a lapse value (the future cost of a forgotten card) or a daily time budget. Both are full-run design choices, not engine work.

### 9.3 Metrics

Part 1's counters per lane and 15-day bucket, plus:

- workload: the answer seconds of first, same-day and later-day reviews;
- the policy's aim: its own predicted recall at the interval it chose;
- memorised cards at the horizon, by the instant head (a query row at 12:00 of the last day, from the lane's final states) and by the last commit curve (for reference);
- the switch backlog.

Policy metrics are read on days 30-59, after a 30-day burn-in of the switch (stage 1 takeaway 5).

Drift against the real user:

- replay lanes: the first-touch state drift, as in part 1;
- policy lanes: the Easy share of passes and the seconds per later-day review, each against teacher forcing on the real schedule.

### 9.4 Proofs (`proof`, `engine/proof-cpu.json`)

- **P1, batched == alone.**
  - Every lane is run alone (B = 1, or B = 2 with `tf` for re-anchored lanes) and compared with the run of all lanes together.
  - The comparison is review by review: day, card, kind, grade, time and chosen interval identical, P(pass) within 1e-4.
  - The counters must agree within a relative 1e-4.
- **P2, the scalar reference.** One lane at a time against an independent scalar path:
  - stage 1's `Builder`, with the codes preset to the engine's;
  - dict states with None for new entities, and the stock recurrent step;
  - candidate rows built from the Builder's own history;
  - stage 1's `time_at`.

  The two paths share only the session rules, the draws, the FSRS-7 functions and the grid interpolation. P2 also checks the split-day candidate step against scalar steps on 40 old cards.
- **P3, replay.** `tf` and `fr-shared1` x 3 seeds, replayed with part 1's stream, reproduce part 1's row (user 3876, split 225, 60 days): integer counters identical, float sums within a relative 1e-4.

## 10. Stage 2 part 3: the full run (`fullrun/`, `scripts/sm41_fullrun.py`)

Approved by the user on 2026-11-04 ("SM41 : oui on peut lancer la vraie simulation"). Everything in this section was fixed before the launch.

### 10.1 The FSRS-7 baseline: pre-split refits (decision)

- **What the stage 2 part 2 baseline used.** `result/FSRS-7-short-secs.jsonl` of srs-benchmark (`--algo FSRS-7 --short --secs`, no recency weighting). The benchmark fits a 5-fold `TimeSeriesSplit` and saves the LAST fold's parameters (`utils.evaluate`: `w_list[-1]`). That fold trains on the first ~5/6 of the user's benchmark rows. For most split days of the run, this fit had seen the simulated window; the exact count is in `fsrs7_fits` of the report. That gave FSRS-7 an unfair advantage.
- **Decision: refit per (user, split) on the user's reviews before the split day only** (`scripts/sm41_fsrs7_fit.py`, srs-benchmark venv, the benchmark checkout only read).
  - It uses srs-benchmark's own code: the feature pipeline, `script._fit_trainable_weights` and its Trainer, with the flags of that file, seed 42 and 1 torch thread.
  - It reads the dataset the simulator reads (`anki-revlogs-10k-original-20261011`). Its rows equal the simulator's row for row (card, rating, elapsed seconds and days, day).
  - The training rows are the benchmark rows with `day_offset` < the split day.
  - fsrs-rs's `fit-v7` was not used: it applies recency weighting, which the benchmark's FSRS-7-short-secs fit does not.
  - The engine reads one line per (user, split) (`--fsrs7-fits`); a missing line is an error, never a silent fallback.
  - A warm-up reused for a later split of the same user recomputes every card's FSRS-7 state with that split's parameters (`Warm.refit_fs`).
- **CPU proof** (`fullrun/fsrs7-proof-cpu.json`), on 5 users:
  - (a) the same code on the benchmark's own last fold reproduces the saved parameters;
  - (b) the numpy port (`scripts/sm41_fsrs7.py`, moved out of the engine unchanged) equals the benchmark's torch predictions on the pre-split rows with the refit parameters;
  - (c) the simulator's update rule equals the benchmark's predictions on those rows. That rule is: every review in order, the first one initialises, later ones step by max(0, elapsed seconds) / 86400;
  - (d) LogLoss of the refit, the benchmark fit and the defaults, both pre-split and on the 60-day window.

  Tolerance: 1e-4 on parameters and predictions.
- **The small model is equally clean.** None of its training users is in the 732 set (checked 2026-11-04, overlap 0 each):
  - the trunk (the released leak-free small model) was trained on the 8,000 train users of the final tier (`artifacts/accuracy/tiers.json`, = the 8k split's train users);
  - the answer-time head (SM82c) was trained on the 1,991 users of the SM82 cool-down (`sm82c/plan.json`);
  - the 732 set is the 8k split's validation users with id < 5000.

### 10.2 Design

- **Users.** The drift study's rule (`sm41_drift.py select`, seed 4141), asked for 150 users.
  - Only 117 users of the 732 set qualify (all 732 looked at). The run takes all 117 with all their 2-3 splits: 304 split points, against the proposal's 150 x 2 = 300 (`fullrun/users.json`).
  - The first 50 users and their splits are the drift study's.
  - 1.39M window reviews; the largest window has 8,997 reviews.
- **Horizon:** 60 days. Policies are read on days 30-59, after a 30-day burn-in of the switch.
- **Lanes (59 per split, `fullrun/lanes.txt`):**
  - teacher forcing (`tf`);
  - daily shared-stream re-anchoring (fr-shared1) on the real schedule x 2 seeds (the drift check);
  - instant-head and FSRS-7 desired retention at 0.70 / 0.75 / 0.80 / 0.85 / 0.90 / 0.95 x 4 seeds (48 lanes);
  - `ihr-0.90-free` and `fsrs7-0.90-free` (no re-anchoring) x 4 seeds.
  - Common random numbers across policies (9.1).
- **Runner (`fullrun/run-fullrun.sh`).**
  - 4 processes x 1 thread, below-normal priority, CPU only, largest users first.
  - One job per user: the user's pre-split refits (`fits/<user>.jsonl`), then every split with one warm-up pass (`rows/<user>.jsonl`, `logs/<user>.log`).
  - Resumable per (user, split).
  - Each job re-reads `quiet.json` (scope all / cpu) and holds from 10 min before a window until its end.
  - Each job holds a new split while less than 3 GB of RAM is available, because other jobs' CPU workers share the 31 GB.
  - At the end: `check` + `report`.

### 10.3 Metrics (fixed before launch)

- **Realised retention:** the pass share of later-day reviews on days 30-59.
- **Cost:** minutes of every review (first, same-day, later-day) on days 30-59 / memorised cards at day 60 (the instant head's recall summed over the card set). Seeds are averaged per split: minutes and memorised cards separately, then the ratio. The 60-day cost (all 60 days' minutes) is reported alongside.
- **Primary metric: does the instant-head policy beat FSRS-7 in minutes per memorised card at matched realised retention?**
  - Per split, each family's 6 points (realised retention -> cost) form a curve, linear between the points sorted by realised retention, never extrapolated.
  - The relative gap cost(instant head) / cost(FSRS-7) - 1 is averaged over the common realised-retention range of the two curves (grid step 0.005). Splits whose ranges overlap by less than 0.02 are left out.
  - A user's splits are averaged. Result: the mean over users with a 95% CI over users (t; paired, since both families run on the same user, split and draws).
  - Why a common range and not one point: in the smoke run, the instant head's realised retention sat 2-17 points above its aim at low targets, so a single point such as 0.85 is often outside one of the curves.
  - Secondary: the gap at the fixed points 0.75 / 0.80 / 0.85 / 0.90, over the users whose both curves cover the point.
- **Decision rule** (from the proposal): the instant-head policy moves on to the time-aware objective only if the primary gap is < 0 with the 95% CI excluding 0. Otherwise FSRS-7 stays the scheduler baseline, and the simulator's value is the learner.
- **Each user's best desired retention.**
  - Value(R) = memorised cards at day 60 - minutes on days 30-59 / P, at prices P = 1, 3 and 10 minutes per extra memorised card.
  - A user's best R = argmax over the 6 targets, splits averaged, per family.
  - Reported: the counts per R, the single best fixed R, and the mean gain of a per-user target over the fixed one (cards, with CI, and % of memorised).
- **Re-anchoring:** the relative cost gap between the instant head and FSRS-7 at target 0.90, with and without re-anchoring, paired per user. Also reported: free minus anchored with its CI, the users whose sign flips, and the free lanes' Easy and seconds drift.
- **Drift check:** the first-touch state drift of fr-shared1 on days 30-59 (the drift study: -0.6 ± 1.0 points).
- **Speed:** lane reviews/s per process, CPU process-hours (engine + warm-ups).

### 10.4 Smoke run (2026-11-04, before launch)

- Users 4378 (splits 180, 270) and 1606 (660, 945), all 59 lanes, 60 days, 2 processes.
- No errors; 4 rows; 238,768 lane reviews at 667-744 lane reviews/s per process (29-148 s per split, warm-ups 12-22 s).
- The refits used 2,718-9,709 pre-split rows; the srs-benchmark fit had seen 3 of these 4 split days.
- The report runs end to end.
- The realised retention of the instant-head lanes is above their aim at low targets: 0.849 at aim 0.682 for 0.70, against 0.926 at aim 0.897 for 0.90. FSRS-7's is closer: 0.792 at aim 0.694. Hence the common-range primary metric.
- **Cost estimate:** 0.046 engine-seconds per window review gives about 18 process-hours for 1.39M window reviews, plus about 3 h of warm-ups and the refits. That is about 5-6 h on 4 processes.
