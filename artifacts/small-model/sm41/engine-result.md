# The small model in the review simulator (SM41): lane engine result (stage 2 part 2)

Run on 2026-11-04 between 06:44 and 06:55, CPU only: 3 validation processes and 1 proof process, 1 thread each, about 0.6 GB each when checked.

- Code: `scripts/sm41_engine.py`. Design: [DESIGN.md](DESIGN.md) section 9.
- Tables: [engine/engine-tables.md](engine/engine-tables.md). Data: `engine/engine-result.json`. Rows: `engine/rows/`.
- Proof: `engine/proof-cpu.json`.

## What was built

- **The vectorised lane engine.** Lanes whose schedules diverge advance together; each lane reviews its own card at each tick.
  - State lives in per-stream arenas [entities x lanes].
  - Rows of entities a lane has not seen yet use a fresh mask that reproduces the stock first step.
  - The input builder is vectorised over lanes.
  - Daily shared-stream re-anchoring (fr-shared1) to the teacher-forced lane is on by default.
- **Instant-head desired retention (ihr-R).** When a card leaves the day, 16 candidate query rows (1-1000 days) go through one batched step. The interval is the first crossing of P(pass) = R. The candidate step runs inside the next tick's query step, so it costs no extra model call.
- **FSRS-7 baseline (fsrs7-R).** The repository had no FSRS-7 scheduler, so the simplest equivalent was added: a numpy port of srs-benchmark's finished 34-parameter FSRS-7, with each user's own srs-benchmark fit (FSRS-7 short secs), updated with the lane's own grades and intervals.
- **Curve-head desired retention (curve-R).** Stage 1's policy, kept for comparison.

## Proofs (CPU): PASS

| check | setup | result |
|---|---|---|
| P1 batched == each lane alone | users 3876 and 2503, 20 days, 8 lanes: teacher forcing (tf), daily shared-stream re-anchoring (fr-shared1) x 2, instant-head desired retention 0.90 and 0.80 (ihr-0.90, ihr-0.80), the same at 0.90 without re-anchoring (ihr-0.90-free), FSRS-7 desired retention 0.90 (fsrs7-0.90), curve-head desired retention 0.90 (curve-0.90) | all 16 lane runs identical review by review (15,278 reviews: day, card, grade, time, chosen interval; no flipped draw). P(pass) max gap 6.1e-7, counters max relative gap 5.8e-7, memorised cards identical |
| P2 engine == scalar reference | stage 1 Builder, stock recurrent step, None states, stage 1 `time_at`; 4 policy lanes + tf x 2 users | identical draws on all 9,009 policy reviews, P(pass) max gap 6.9e-7. Split-day candidate step vs scalar steps (40 cards x 16 intervals): max gap 2.8e-7 |
| P3 real schedule reproduces part 1 | user 3876, split 225, 60 days, teacher forcing (tf) + daily shared-stream re-anchoring (fr-shared1) x 3 seeds vs part 1's row | integer counters identical, float sums within relative 6.1e-8 |

## Validation: 8 users x 1 split x 60 days

Users 89, 1070, 1380, 2207, 2503, 2626, 2941 and 4813, each at its first drift-study split. The 17 lanes are teacher forcing, daily shared-stream re-anchoring (fr-shared1) x 2 seeds, and 7 policies x 2 seeds.

All lanes are re-anchored. Values are the mean over users (seeds averaged):

- retention: the pass share of later-day reviews on days 30-59, after the burn-in of the switch;
- aim: the policy's own predicted recall at the interval it chose;
- minutes: every review over the 60 days;
- memorised: the sum over the card set of the instant head's recall at day 60.

| policy | retention 30-59 | aim | min/day | memorised | min per memorised card | switch backlog |
|---|---|---|---|---|---|---|
| real schedule, teacher forcing (tf) | 0.864 | — | 7.2 | 711 | 0.60 | 0 |
| real schedule, sampled, daily shared-stream re-anchoring (fr-shared1) | 0.860 | — | 7.3 | 684 | 0.65 | 0 |
| instant-head desired retention 0.80 (ihr-0.80) | 0.787 | 0.780 | 7.7 | 694 | 0.71 | 39 |
| instant-head desired retention 0.90 (ihr-0.90) | 0.877 | 0.879 | 11.1 | 743 | 0.95 | 74 |
| instant-head desired retention 0.95 (ihr-0.95) | 0.927 | 0.930 | 15.2 | 768 | 1.24 | 150 |
| FSRS-7 desired retention 0.80 (fsrs7-0.80) | 0.771 | 0.775 | 6.7 | 675 | 0.63 | 35 |
| FSRS-7 desired retention 0.90 (fsrs7-0.90) | 0.872 | 0.872 | 9.8 | 729 | 0.85 | 93 |
| FSRS-7 desired retention 0.95 (fsrs7-0.95) | 0.927 | 0.923 | 13.0 | 763 | 1.07 | 188 |
| curve-head desired retention 0.90 (curve-0.90) | 0.874 | 0.880 | 11.0 | 741 | 0.95 | 79 |

**Retention: realised = aim, within 1 point.** On days 30-59 the instant head predicts 0.788 / 0.878 / 0.927 for its own policy, against 0.787 / 0.877 / 0.927 realised.

- Stage 1's gap of 1-4 points between the two heads is gone here. Even the curve-head policy is within 0.6 points.
- The aim sits about 2 points below 0.90 because of the 1-day floor. On user 2503 over 40 days, 6.3% of the decisions of instant-head desired retention 0.90 (ihr-0.90) are cards whose P(pass) at 1 day is already below 0.90 (aim 0.83). These cards come back daily, so they weigh more on the review-weighted mean. Every other decision aims at exactly 0.90.
- Per user, the realised retention of instant-head desired retention 0.90 (ihr-0.90) runs from 0.814 (user 1070: 591 new cards of 721) to 0.918.

**Workload.** Going from 0.80 to 0.95 multiplies minutes per memorised card by 1.7 (instant head) and 1.7 (FSRS-7), for +11% and +13% memorised cards.

- At the same target FSRS-7 is cheaper: at 0.90 it needs 9.8 instead of 11.1 min/day, for 0.5 points less retention and 2% fewer memorised cards.
- That is a hint only (8 users, no CIs). The full run is what decides it.

**Drift against the real user.**

- On the real schedule, the replay lane with daily shared-stream re-anchoring (fr-shared1) drifts in first-touch state by +0.2 points on days 30-59, with Easy share of passes +1.1 points and seconds per later review -0.1. Part 1 measured -0.6 ± 1.0 on 50 users.
- The policy lanes differ from teacher forcing on the real schedule: Easy -2 to -5 points, seconds +1.3 to +4.6 at the lower targets. They review at other recall levels, so this is mostly the policy's effect, not drift.
- Caveat: with re-anchoring, a policy lane's user, deck and preset states follow the REAL schedule's workload. A policy's own load cannot feed back through those streams.

## Speed (CPU, 1 thread per process)

- Validation, 17 lanes: 576,913 simulated lane reviews in 0.31 process-hours, **517 lane reviews/s per process**. Per user: 454-599/s, 2.7M candidate query rows. Warm-ups: 11-22 s per user.
- The cost is per tick, not per lane. User 2207 with 33 lanes ran 62,640 lane reviews in 72.7 s (862/s) against 32,333 in 69.2 s with 17 lanes (467/s).
- Part 1 replay reached about 2,400/s with 32 lanes. A policy tick costs more: the lanes are not in lock-step, the reviews per day differ by lane, and the instant head adds 4.7 candidate rows per review.
- What would make it faster (not done): more lanes per engine (nearly free up to at least 33), several users per engine (one flat arena), GPU (untested, no GPU here).

## Proposal for the full run (not started; the main session decides)

**Question.** For this learner, does instant-head scheduling buy more memory per minute than FSRS-7? Where is each user's best desired retention?

**Design.**

- **Users and splits:** 150 users of the 732 set, chosen by the drift study's rule (Good-only excluded, 1,500-9,000 window reviews), 2 splits each: 300 split points.
- **Horizon:** 60 days, read on days 30-59. Part 1 measured -0.6 ± 1.0 points of state drift at days 30-59 with daily shared-stream re-anchoring (fr-shared1), and about -3 at days 60-89, so a longer horizon is not trustworthy yet.
- **Lanes (59 per split):**
  - teacher forcing;
  - daily shared-stream re-anchoring (fr-shared1) on the real schedule x 2 seeds (the drift check on every split);
  - instant-head and FSRS-7 desired retention at 0.70 / 0.75 / 0.80 / 0.85 / 0.90 / 0.95, 4 seeds each (common random numbers): 48 lanes;
  - instant-head and FSRS-7 desired retention 0.90 without re-anchoring (ihr-0.90-free, fsrs7-0.90-free) x 4 seeds: does re-anchoring change the ranking?
- The curve-head policy is dropped: it equals the instant head here.

**Cost** (from the measured speed: 139 s per split at 17 lanes, x 1.05 measured for 33 lanes):

- about 160 s of engine time + 20 s of warm-up per split at 59 lanes, so about **15 process-hours: about 4 h on 4 CPU processes**, plus about 20% margin;
- memory: about 14 KB per card per lane, so up to about 3.5 GB per process for the largest users (cap: 9,000 window reviews);
- no GPU.

**What it decides.**

1. Instant head vs FSRS-7, paired per user with common random numbers: minutes per memorised card at matched realised retention (interpolated along each policy's 6-point curve), with a 95% CI over users. Fixed in advance: the instant-head policy moves on to the time-aware objective only if it beats FSRS-7 with the CI excluding 0. Otherwise FSRS-7 stays the scheduler baseline, and the simulator's value is the learner.
2. Each user's best desired retention at a stated price (minutes per extra memorised card), and its spread over users: does a per-user target matter?
3. Whether re-anchoring changes the ranking (the lanes without re-anchoring, `-free`).

**What it cannot decide.**

- Real-world gains: this is the model's own world.
- The counterfactual-interval risk (DESIGN.md 5): intervals outside what real schedulers produce.
- Absolute minutes: same-day steps follow the user's empirical rules.
- Daily limits and the time-aware objective (deferred, DESIGN.md 9.2).
