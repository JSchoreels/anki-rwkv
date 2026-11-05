# Faster engine for the changing-learner simulation (SM41g): scenarios as lanes of one run

Done 11-05 18:20. CPU only. The running engine (`scripts/sm41_engine.py`) was not touched: the changing-learner run
re-imports it at every split. The change was built and checked in an uncommitted copy. Applied 11-05 19:35, after
that run ended: the patched `scripts/sm41_engine.py` is byte-identical to the checked copy, so the checks below hold for it.

## Where the time goes (cProfile)

The profiled job is user 3992, split 840, declining learner, the 57 lanes of `learner-drift/lanes.txt`. It took 431 s.

- **Small-model forward passes: 315 s (73%), 876,889 block calls.**
  - PyTorch module-call overhead (`_call_impl`, `__getattr__`, the tracing check) is about 48 s (11%).
  - `single_timestep` is 88 s of small tensor operations.
- **Batches are tiny.** A tick averages about 7 reviews plus about 34 candidate rows. A tick takes one review from each
  lane still in its day, so the lane with the most reviews sets the number of ticks, and the other lanes drop out as
  their day ends.
- **Warm-up: 36–200 s per job.** This is the teacher-forced replay of the real history before the split. It is
  identical for the improving, declining and no-drift scenarios, yet each scenario runs it again.
- **Cost grows faster than the review count.** User 1022, declining learner: 968k lane reviews at 235/s. The improving
  learner: 299k at 660/s.

## The change: per-lane drift (lane option `g<a>_<b>`, `m` = minus)

- Example lanes: `ihr-0.90-g0.5_0.2#0` (improving), `fsrs7-0.90-gm0.5_m0.2-free#0` (declining).
- The option goes before `-free`.
- A lane's option overrides the run's `--drift`. Lanes without it use `--drift`, or no drift.
- The per-row arithmetic is the same as the scalar path. Patch: `engine-fused-scenarios.patch`, 113 lines.
- One run then holds all three scenarios: `tf` plus 56 lanes × 3 = 169 lanes (`lanes-fused-3scen.txt`).
- Policy-lane draws are keyed on (user, split, seed, card, review number) and not on the lane name. So each scenario's
  lanes draw exactly what they draw in a separate run.

## Checks (user 3992, split 840, 60 days, `check-fused.py`, `run-check-fused.sh`)

- **Patched engine with a global `--drift 0.5,0.2` vs the current engine.**
  - Every row field is identical.
  - Every per-review log column is bit-identical.
  - The first version differed by 1e-16 on day-0 P(pass), because it skipped a zero shift. It was fixed to mask on the
    lane's `a`.
- **Fused run (tf + 4 lanes × 3 scenarios) vs the three separate runs, review by review.**
  - Day, card, kind, grade, answer ms and chosen interval are identical for every lane.
  - P(pass) agrees within 4e-7. The engine's own proof P1 (batched == alone) allows 1e-4.
  - The counters agree within a relative 2e-7.
  - Memorised cards are identical.

## Speed and memory

- **Full lane set, user 3992:**

  | Run | Time |
  |---|---|
  | Fused, all 3 scenarios | 518 s (36 s warm-up + 482 s simulation; 451k lane reviews, 937/s) |
  | Improving, separate | about 224 s |
  | Declining, separate | about 471 s |
  | No drift, separate | about 240 s, estimated from its review count |
  | Separate total | about 935 s |

  **Fused needs about 45% less CPU.** The improving and no-drift lanes ride in the ticks the declining lanes already
  need.
- **Peak memory, 10-day runs:** 0.60 GB for 169 lanes and 0.61 GB for 57 lanes. Memory is per user, not per lane. One
  fused process replaces three, so the same RAM fits about 3× the work. RAM, not cores, is what limits the process
  count today: 16 logical cores, 4 processes of about 2 GB, and as little as 1.7 GB free at times.

**Expected effect:** a 3-scenario confirmation on all 117 users drops from about 8–10 h to about 4.5–5.5 h on 4
processes. More processes would fit in the freed RAM.

## Not done yet (ideas, not measured)

- **Warm-up cache.** Save the warm-up state per (user, split) to disk. Screens and confirmations rerun the same users and
  splits, and the warm-up is deterministic, so the saved state is identical. Expected saving: 50–200 s per job,
  about 10–20%.
- **Module-call overhead (about 11%).** Call the blocks' `forward` directly when no fresh rows need the fill hooks.
  Exactness must be rechecked.
- **Several users per process.** Interleave independent engines to fill the tail ticks.
