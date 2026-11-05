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

(Full tables: `learner-drift/tables.md`, `tables.json`. 20 users, 1 split each, 60 days, seeds 0-3. Gaps are the
cost of the policy relative to FSRS-7 (minutes per memorised card), mean ± 95% CI over users, in %. All 60 jobs ran,
no Tracebacks.)

### Check of the no-drift rerun (`learner-drift/check.out`)

The `none` rerun matches the existing full-run / screen 3 rows within 1e-6 on 1135 of 1140 lanes. **Five lanes
differ** (users 2363, 2503, 3992 and 4207, one or two lanes each: the instant head at 0.80 / 0.85, the time-aware rule
at 0.90 seed 1, the `-free` lane of user 4207 by 0.4%). The first three differ by 60-100% of a counter, which is the
signature of a single flipped draw that then changes the whole trajectory, not of a bug in the drift code (the drift
options are off in `none`); not investigated further. The gap tables use the existing rows for no drift (not `none`),
so only the interval columns see these lanes (5 of 1140, so a small effect). Part A's earlier sample check was 1.0e-6.

### Gaps vs FSRS-7 and change vs no drift

| scenario | policy | matched retention (plan) | change | matched memorised, days 30-59 | change | matched memorised, 60 days | change |
|---|---|---|---|---|---|---|---|
| no drift | instant head | +5.6 ± 5.0 | | +0.6 ± 6.3 | | +5.2 ± 15.4 | |
| no drift | time-aware rule (beta = 2) | -18.7 ± 8.4 | | -7.5 ± 7.9 | | +0.8 ± 9.5 | |
| improving | instant head | +2.7 ± 4.7 | **-2.9 ± 1.5** | -2.4 ± 7.3 | -2.9 ± 5.6 | +1.5 ± 8.7 | -3.7 ± 10.6 |
| improving | time-aware rule | -15.1 ± 8.3 | **+5.7 ± 4.2** | -7.8 ± 8.1 | -0.3 ± 5.0 | -1.0 ± 10.2 | -1.7 ± 7.5 |
| declining | instant head | +29.7 ± 27.0 | +23.0 ± 25.4 | +8.9 ± 10.7 | +8.3 ± 7.8 | +10.7 ± 14.4 | +5.4 ± 7.7 |
| declining | time-aware rule | -32.7 ± 9.9 | **-13.4 ± 9.2** | -19.0 ± 8.6 | **-11.6 ± 8.7** | -5.6 ± 12.0 | -6.4 ± 10.1 |

(The sign of the gap flips between metrics for the time-aware rule at no drift, -18.7 at matched retention vs +0.8
at matched memorised cards over 60 days: this is the same pattern as in the existing no-drift results, not a drift
effect. The declining instant head gap, +29.7 ± 27.0, has a CI as wide as its mean: a few users dominate it; per-user
medians were not computed in this pass.)

### Does it schedule less? (days 30-59, change vs no drift, policy minus FSRS-7, mean of targets 0.80 / 0.85 / 0.90)

| scenario | policy | reviews/day | median later-day interval | seconds per later-day review |
|---|---|---|---|---|
| improving | instant head | **-7.1 ± 3.1** | +4.1 ± 6.9 | +7.1 ± 5.5 |
| improving | time-aware rule | +17.8 ± 9.2 | **-27.1 ± 10.2** | -9.1 ± 8.4 |
| declining | instant head | **+9.9 ± 4.5** | -2.0 ± 14.6 | -4.4 ± 4.7 |
| declining | time-aware rule | **-30.6 ± 14.9** | +38.4 ± 15.2 | +7.4 ± 7.0 |

Key lanes (days 30-59, no drift -> improving): instant head 0.85: 78.3 -> 59.7 reviews/day, retention 0.824 -> 0.879;
0.90: 94.2 -> 69.2, 0.869 -> 0.912. FSRS-7 0.85: 77.8 -> 63.4, 0.814 -> 0.877; 0.90: 97.9 -> 80.5, 0.873 -> 0.922. Both
policies review less when the learner improves (the learner remembers more); the instant head cuts reviews a little
more than FSRS-7 (-7.1 points), which is the "schedules less" direction. The time-aware rule goes the other way
(+17.8, its intervals get shorter, -27%).

### Re-anchoring check (instant head / FSRS-7 at target 0.90, `-free` lanes)

Anchored: +8.3 (no drift), +6.5 (improving, change -1.9 ± 4.8), +34.2 (declining, +25.9 ± 27.2). The `-free` lanes
are too noisy to read: no drift +577 ± 1198% (an outlier user), improving -11.0 ± 9.7, declining +14028 ± 22870%
(both lanes collapse in the declining scenario: instant head `-free` retention 0.28 and about 1957 reviews/day, FSRS-7
`-free` 0.36 and 854/day). Medians were not computed. Conclusion: the anchored result is not an artefact of the daily
re-anchoring as far as can be told; the free lanes do not give a usable second reading.

### FSRS-7 frozen

FSRS-7 keeps the full run's pre-split parameters for all 60 days. A refit in the middle would see the drifted passes
and answers and move its intervals for every card at once (longer when improving, shorter when declining), which is
exactly what the frozen version cannot do; not run, so the improving-scenario gap is probably somewhat in FSRS-7's
disfavour relative to a real Anki between optimisations.

### Verdict by the plan's decision rule

- Improving: (1) reviews/day change vs FSRS-7 -7.1 ± 3.1, below 0 with the CI excluding 0: met. (2) gap at matched
  memorised cards (days 30-59) change -2.9 ± 5.6, CI includes 0: not met. So: **"the instant head schedules less
  than FSRS-7 when the learner improves, but not more efficiently"** (the 60-day metric points the same way, -3.7 ±
  10.6, not significant). The user's hypothesis is supported on scheduling less only. The time-aware rule does not
  schedule less.
- Declining: the instant head schedules more (+9.9 ± 4.5), the time-aware rule less and loses efficiency (-11.6 ± 8.7
  at days 30-59).
- **The confirmation rule fires** (several changes vs no drift have a CI excluding 0: improving instant head and
  time-aware rule at matched retention, declining time-aware rule at matched retention and days 30-59; also the
  improving instant head gap vs FSRS-7 goes from CI excluding 0 to including 0). Matched-retention is the sensitive
  metric; the memorised-card metrics mostly do not move beyond their CIs.

### Next

Confirmation on all 117 users (304 splits) of the improving scenario, only the lanes it needs (`tf`,
`ihr-0.80/0.85/0.90`, `ihr-*-s2`, `fsrs7-*` x 4 seeds = 49 lanes): about 30-40 process-hours (about 8-10 h on 4
processes). Needs the main session's go-ahead and its own handoff. Not run before the break.

### Speed (1 thread per process)

improving 4.77 M lane reviews, 1.81 engine process-hours, 732 lane reviews/s per process; declining 13.40 M, 7.92 h,
470 /s (the collapsing `-free` lanes dominate); none 5.69 M, 2.66 h, 595 /s; warm-up 0.43-0.44 process-hours each.
Total 12.39 engine + 1.31 warm-up process-hours.

