# The small model in the review simulator (SM41): stage 1 result

Prototype of a card-level review simulator whose learner is the released small model (leak-free) plus the answer-time
head on richer frozen features (SM82c). Code: `scripts/sm41_sim.py`. Design: [DESIGN.md](DESIGN.md). Full tables:
[prototype-tables.md](prototype-tables.md) (data: [prototype-result.json](prototype-result.json), rows
`prototype.jsonl`). Setup: 5 users of the 732 set ([users.json](users.json)), 90 simulated days after a split day,
CPU only, 2 seeds per sampled lane.

## Takeaways

1. **The simulator runs and its parts are exact.** The CPU proof ([check-cpu.json](check-cpu.json), PASS) shows three
   things. The row-by-row input builder matches the training pipeline (bitwise on the 24 base columns, all 92 columns to
   4.5e-7). The recurrent step matches the parallel model (instant 8.9e-7, curve 2.9e-7, the 448 frozen time-head
   features 1.1e-5). The incremental explicit time features match SM82c's table bitwise.
2. **Teacher forcing is well calibrated on the continuation.** On the 4 users that are not Good-only, predicted
   later-day retention averages 0.889 against a real 0.897. Predicted answer time is 16.7 s per later review against a
   real 16.1 s.
3. **The model drifts when it reads its own samples. This is the main risk, and it is confirmed.** The clean check
   uses each old card's first later-day review in the window. Real outcomes scheduled those reviews, so their days do
   not depend on the samples. On these reviews the free-running learner moves away from the real user:
   - Days 0-29: the gap ranges from -2.6 to +21 points (+7 on average). Real users have slumps that the model picks up
     online from real grades. Sampled grades cannot contain them.
   - Days 60-89, on the two users with enough reviews (3817: 181, 3876: 137): -6 and -22 points. The model's own
     predictions fall too (3876: 0.913 teacher-forced vs 0.706 free running).

   The decomposition lanes show that sampled grades alone are enough to drift. Sampled times add an Easy inflation:
   user 3486's Easy share of passes goes from 0.22 to 0.47 when times are also sampled, and stays at 0.25 when only
   grades are. Sampled times are +2 to +4 s above the real ones for 3 of the 4 users (-1.7 s for the fourth).
4. **The desired-retention scheduler is self-consistent.** Realised retention equals the instant head's prediction, as
   it should when grades are sampled from it. It lands 1-4 points below the target the curve head aimed at: 0.763 /
   0.888 / 0.936 for targets 0.80 / 0.90 / 0.95. In the simulation the two heads disagree by that much.
5. **The cost shape is plausible.** Going from 0.80 to 0.95 multiplies minutes per memorised card by 2.2 (1.09 to
   2.36) for +10% memorised cards at day 90. Switching an existing deck to the policy creates a split-day backlog of
   42-1,101 reviews, which a full run should keep out of its metrics.
6. **Good-only users are useless for policy search.** User 4326 presses Good on every review, so retention is 1.0 at
   every target. Users like this must be flagged and excluded (rule used here: real later-day retention >= 0.995).

## Speed (measured, CPU)

- Unbatched prototype, 1 thread per process: warm-up 268 reviews/s and simulation 147 reviews/s per process.
  - A simulated review takes 2 model steps (the instant row, then the review row) plus the time head.
  - Total: 382,498 lane reviews in 55 process-minutes, with 4 processes at below-normal priority.
- Batched step (`bench`: B learners advance together, states stacked along the batch dimension):

  | threads | B | reviews/s |
  |---|---|---|
  | 1 | 16 | 1,992 |
  | 1 | 256 | 5,947 |
  | 4 | 256 | 9,428 |
  | 4 | 2,048 | 11,937 |

  The model step is about 30x faster batched. Outside the model, the Python side (feature builder, running medians,
  curve inversion, sampling) costs about 1.6 ms per review today, so it has to be vectorised too.
- GPU: not measured. The queue is full until the afternoon (grade-order sanity (SM33) and the frozen-model grade side
  head (SM87) smoke, about 2.7 h ahead of us), so the short benchmark ticket was withdrawn. The command is in
  [HANDOFF.md](HANDOFF.md).

## Full run: what is needed and what it costs

Fixes before a full run (from the findings above):

- **Drift.** Measure it on 50 users with 2-3 split points each before trusting any absolute number. Keep policy
  comparisons within a user, with common random numbers (the comparison is relative). Prefer horizons of 30-60 days
  with many split points over one long roll.
  - Test the obvious mitigations, each with the first-touch check: sampling temperature below 1 for the time head;
    feeding the expected time instead of a sampled one; periodic re-anchoring on real days.
- **Head mismatch.** Choose intervals with the instant head instead of the curve head: K candidate intervals = one
  batched step of K query rows, which do not commit state. This removes the 1-4 point gap, and the same machinery gives
  the time-aware policy (expected answer time per retained card at each candidate).
- **Backlog and user exclusion.** Treat the first 30 days after the policy switch as burn-in and leave them out of the
  metrics. Exclude Good-only users.

Engineering (about 1-2 days): a vectorised lane engine.

- States live in per-stream, per-layer arenas: one slot per entity of each lane, copy-on-write from the shared
  warm-up, plus a fresh-entity mask.
- Features, explicit time features and curve inversion are computed in torch. Running medians become 64-bin log
  histograms, checked against the exact medians.
- Proof: batched lanes == unbatched lanes for the same seeds, on 2 users.

Compute for one plausible full run:

- 200 users, 1 split each, 365 days.
- 23 lanes per user: teacher forcing, 2 free running, and 2 seeds of each of 8 desired retentions and 2 time-aware
  variants. That is 4,600 lanes.
- At about 60 reviews per day, that is about 100M simulated reviews.
- Run time: on CPU (4 threads, batched model at about 12k reviews/s) about 2.3 h of model time, so a 4-6 h budget with
  the vectorised Python side. GPU likely faster but untested.
- Memory: card and note state is about 14 KB per card per lane (fp32), so lanes run in waves of 250-500 (4-7 GB each).
- New cards: beyond a user's real history they must be synthesised: the user's recent new-card rate, in the user's
  main deck and preset.

## Is the srs-simulator repository needed?

No. The learner has to be our stateful model, and this simulator hosts it directly. The scheduler-side reference is
fsrs-rs `simulation.rs` (local): daily cost and review limits, priority functions, learning-step transitions and
per-state costs. Our per-user habits and the answer-time head play those roles. srs-simulator would only matter for
comparing against published policy results (its schedulers used as policies). Hosting our model there needs the fork
runtime port (SM08). Cloning it is a later decision for the user.
