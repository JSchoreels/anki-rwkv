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
     interval inputs contain the previous answer's time, SM82 stage 0);
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
