# The small model in the review simulator (SM41): the changing-learner simulation (SM41g)

User 11-05 14:30: "we should also simulate someone that get better performance with time, I guess RWKV would be more
efficient since it would schedule less". Brief: [drift-brief.md](drift-brief.md). Files: `learner-drift/`. (The
brief named this file `drift-result.md`, but that name already holds the stage 2 drift study's result, so this
study uses `learner-drift-result.md` and `learner-drift/`.)

## Plan (written before the runs)

### Why

Until now the simulated learner is the small model itself and never changes. So the instant head predicts its own
world exactly, and FSRS-7's user-level parameters (fitted before the split day, then frozen) are as right as they
will ever be. A real learner who gets better (or worse) over the weeks breaks both. The user's hypothesis: the small
model reads the trend (more passes, faster answers) across all cards and stretches intervals, so it schedules less
than FSRS-7, which adapts card by card only. This also tests robustness when the learner is no longer the policy's
own model.

### The changing learner (`scripts/sm41_engine.py run --drift A,B`, off by default)

- On simulated day t (0..59), a policy lane's learner draws every model-graded review (same-day and later-day) from
  the instant head's P(pass) at the show moment, shifted by `A * t / 60` on the logit scale. A pass keeps the instant
  head's Hard / Good / Easy mix renormalised; a fail is Again. The same counter-based uniform is used as without
  drift (common random numbers). First reviews keep the user's own first-grade mix (no recall to shift).
- Every answer time of a policy lane (first, same-day and later-day) is multiplied by `exp(-B * t / 60)`.
- Only the learner's draws change. The drawn grade and time go into the commit row (so the small model's own card,
  note and shared-stream states see the drifted answers, as a real user's history would), the explicit time
  features, FSRS-7's state update and the time-aware target's last answer time, exactly like an undrifted draw.
- Memorised cards at day 60: the learner's own recall, the instant head's shifted by `A * 60 / 60` (the undrifted
  sum is kept as `memorised_instant_model`). Realised retention is the drifted learner's pass share.
- Replay lanes (`tf`, the real user) are not drifted. Caveat: daily shared-stream re-anchoring (on by default)
  copies the real user's deck, preset and user states into every lane each morning, so the trend can build up in
  the shared streams within a day only; card and note states keep the full drifted history. The `-free` lanes at
  target 0.90 (no re-anchoring) show the effect without that reset.
- FSRS-7 stays frozen at the full run's pre-split refit, as in Anki between optimisations. A mid-run refit is not
  run (see the result for what it would change).
- Option `--ivl-hist` adds the histogram of elapsed days of later-day reviews (`acc.ivl_later`, bins `ivl_edges`).
- **Bit-identity** (user 1606 split 660, the 9 default lanes of part A, `learner-drift/bitcheck/`): the rows without
  the options, and with `--ivl-hist` minus the histogram, are identical to the rows of the code before the change
  (`policy/bitcheck/after-policies.jsonl`) on every field except the wall-clock ones: **IDENTICAL** (20 fields).

### Scenarios (fixed before the runs)

| scenario | A (logit by day 60) | B | answer time by day 60 |
|---|---|---|---|
| improving | +0.5 | +0.2 | x 0.82 (about 18% faster) |
| declining | -0.5 | -0.2 | x 1.22 |
| no drift | 0 | 0 | the existing rows: full run (instant head, FSRS-7, `-free`) and screen 3 (time-aware beta = 2), same users, splits and seeds |

A third run, `none` (no drift, `--ivl-hist`), adds the interval histogram for the no-drift lanes and must reproduce
the existing rows (counters within a relative 1e-6).

### Lanes (57 per split, `learner-drift/lanes.txt`)

`tf`; instant head `ihr-0.80/0.85/0.90`; FSRS-7 `fsrs7-0.70 ... 0.95` (all 6, for the matched metrics); time-aware
`ihr-0.80/0.85/0.90-s2` (beta = 2, the confirmed setting); `ihr-0.90-free` and `fsrs7-0.90-free`; seeds 0-3.

### Users

The 20 users of part A (`diag/sample.json`, 1 split each), 60 days, the full run's pre-split FSRS-7 refits.

### Metrics (paired per user, mean with a 95% t CI over users)

- **Gap vs FSRS-7 at matched realised retention** (the plan metric, `sm41_policy.overlap_gap`): minutes on days
  30-59 per memorised card at day 60, over the common realised-retention range of the policy's 3 points and FSRS-7's
  6.
- **Gap vs FSRS-7 at matched memorised cards** (`sm41_frontier.gap`): days 30-59 minutes, and 60-day minutes, over
  the common memorised range.
- Both for the instant head (`ihr-*`) and the time-aware rule (`ihr-*-s2`), per scenario.
- **Change vs no drift** (difference in differences): per user, the scenario's gap minus the no-drift gap on the same
  user, split and seeds.
- **"Schedules less", shown directly**, per policy and target, days 30-59: reviews per day, and the median elapsed
  days of later-day reviews (seeds pooled). The change from no drift to the scenario in log reviews/day and log
  median interval, the instant head (and the time-aware rule) minus FSRS-7 at the same target, averaged over the
  targets 0.80 / 0.85 / 0.90.
- Re-anchoring check: the relative cost gap instant head / FSRS-7 at target 0.90, anchored and `-free`, change vs no
  drift.

### Decision rule

- **The user's hypothesis is supported** in the improving scenario if both: (1) the instant head's change vs no drift
  in reviews/day relative to FSRS-7 is < 0 with the CI excluding 0 (it schedules less), and (2) its change vs no drift
  of the gap at matched memorised cards (days 30-59) is < 0 with the CI excluding 0 (it becomes relatively more
  efficient). Only (1): "it schedules less, but not more efficiently". Neither: not supported.
- **Confirmation on all 117 users** of a scenario if it changes the ranking (a gap vs FSRS-7 changes sign with its CI
  excluding 0, or goes from CI-including-0 to CI-excluding-0 or back) or a gap moves by more than its CI (the change
  vs no drift has a CI excluding 0). It takes several hours: hand off first.

## Results

(pending)
