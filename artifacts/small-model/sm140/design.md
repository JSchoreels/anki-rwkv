# Rolling recalibration for the six-layer Tiny (SM140): design (pre-registered 11-21 before any GPU ticket)

**Question.** Rolling recalibration on the 732 set (SM125) got a go for five release family candidates, and rolling recalibration for the Big models (SM128) applied the exact same recipe to the two Big models. The six-layer Tiny models (SM130) replace the Tiny models in the next release, with the new answer-time heads of the answer-time heads for the six-layer Tiny (SM138) (346 head inputs each). SM140 applies **exactly the recipe, the rules and the go rule of SM128** (itself SM125's, copied there from `artifacts/small-model/sm128/design.md`) to the six-layer Tiny two-head model (SM130, key `sm130th`) and the six-layer Tiny Instant-Only model (SM130, key `sm130io`), each with its SM138 head. Nothing in the recipe, the rules or the go rule changes.

**What answer time means.** The answer time is the whole time on the card, front and back together: from the moment the question is shown to the moment a grade button is pressed, capped at the preset's maximum answer seconds. Every "within S s" means answered within S s, reading the back included.

## Models

The six-layer Tiny two-head model (SM130) and the six-layer Tiny Instant-Only model (SM130), frozen trunks and the SM138 heads in `wt/sm138/artifacts/small-model/sm138-sm130th-head` and `sm138-sm130io-head` (346 inputs each, model keys `sm130th` / `sm130io` in `scripts/sm117_head.py`, masters sha256-pinned there and checked equal to the head folder's trunk before scoring).

## The one new step: SM120's recorded sums for the two models

SM128's baselines and its "dump equals SM120" check come from SM120's recorded per-user sums (`artifacts/small-model/sm120/gpu/sm120-<key>-rows.jsonl`, the fast Good/Easy prediction scorer `scripts/sm120_fast.py score`). SM120 never scored the six-layer Tiny models (they did not exist), so the launcher first runs **exactly SM120's scoring step for the two models** (same code, same priors file `artifacts/small-model/sm120/priors.json`, same 732 set, 723 users, show-time inputs, leak-free, cap 0.8 then 0.92 for missing users) and writes the rows to the same place, `artifacts/small-model/sm120/gpu/sm120-<key>-rows.jsonl` in this tree. These rows serve as the baselines (a) and (b) and as the dump check, exactly as for the other models. About 17 min of GPU per model.

## Everything else is SM128's (design.md of SM128 and SM125, unchanged)

- **Dump** (GPU, `scripts/sm125_recal.py dump`): per review, the 33 event probabilities (float32), grade, duration, day, on the 732 set (723 users of `sm117_head.pool_users()`), show-time inputs, leak-free, fp32, SM117's time-labelled test reviews (same-day reviews included), into `work/dump-<key>/<user>.npz` (gitignored, never committed, about 4.4 GB per model). Every user's dump is checked against SM120's recorded sums (same review sha256, same counts, LogLoss / Brier / sum of p within 1e-5 relative).
- **Scheme** (CPU, `sm125_recal.py recal`): per user, causal, rolling Platt scaling: 7-day blocks counted from the user's first scored review; each block predicted with (a, b) fitted on the user's latest 30,000 scored reviews of strictly earlier days; raw model before 1,000 earlier reviews; Newton fit warm-started from the previous block; an event with fewer than 20 positives or 20 negatives keeps the raw model; clamps a in [0.05, 20], b in [-15, 15]; repair (non-decreasing in S, Easy within S at most Good or Easy within S at most recalled within S, each timed event capped by the raw P(set)). Grade-only P(set) is not recalibrated. Arms reported: raw, scheme (primary), 5,000-review minimum, before repair, recalibrated rows only, grade-only diagnostic.
- **Reliable** at an event and S: ECE (20 equal-mass bins) below 0.01 and LogLoss below every SM120 baseline of the event (baselines not recalibrated), each paired review-weighted difference with the 95% CI (2,000 user bootstraps, seed 0) below 0 and at least 0.05% of the baseline's LogLoss. Same rule for the raw arm.
- **Go rule, per model, over S in {2, 3, 4, 5, 6, 8, 10} and the three timed events (Good or Easy, recalled, Easy within S s): 21 cells.**
  - **G1:** all 21 cells reliable after recalibration.
  - **G2:** "Good or Easy within 4 s": recalibrated LogLoss below raw with the CI below 0 and at least 0.05%.
  - **G3:** no cell of the 21 where the recalibrated LogLoss is worse than raw with the CI above 0.
  - **G4:** P(Good or Easy) and P(Easy) (grade only, raw) reliable.
  - **Go for a model** iff G1, G2, G3 and G4 all hold. **Verdict:** go if both models are go; otherwise the models that are go are named.
  - The headline of this question is G1 as the user stated it: all 21 timed cells of 2 to 10 s reliable after recalibration; G2 to G4 are SM128's other rules, kept so the verdict reads the same as for the five candidates and the two Big models.

## Source change and CPU proof

One source change, committed before the launch (commit `b928223c`): `scripts/sm120_fast.py` gets the head directories of the two six-layer Tiny heads (`HEAD_DIRS`), so `sm125_recal.py` and `sm120_fast.py` find them when no `--head-dir` is given. The model keys, the sha256 pins and the 346-input width were already in `scripts/sm117_head.py` (SM138); the head width is derived from the model, not hard-coded, in `Frozen`. SM125's CPU proofs P1 to P8 (`artifacts/small-model/sm125/check-cpu.json`, PASS) are inherited unchanged (recipe untouched). A CPU smoke (`cpu-smoke.json`): SM120's scorer and SM125's dump on 2 users (2388, 1952) for each of the two models, on the CPU: the dump's raw sums equal the scorer's within 1e-5 relative (identical), all 33 event probabilities finite. The first GPU tickets are the SM120 scoring of the two models; then a 3-user GPU smoke of the dump of the six-layer Tiny two-head model (SM130) (users 2388, 1952, 3922) against the rows just written (within 1e-5); the launcher stops if it fails.

## Launch

`launch-sm140.sh`: queue owner `sm140`, purposes start with `Rolling recalibration for the six-layer Tiny (SM140): `, GPU cap 0.8 (0.92 only for the pass on missing users). Only quiet.json windows (scope all / gpu) hold it; no planned daily or Sunday break. Steps: SM120 scoring of the two models, GPU smoke, dump of each model, CPU recalibration of both models (2 processes, 1 thread each, under 6 GB RAM each, started right after the first dump, stopping themselves 5 min before a quiet window and resuming after it), `sm125_recal.py report` (CPU, `SM125_TAG=sm140`) writing `sm140-result.md` / `.json`, a final check that every expected output exists, then `done`. STOP file `gpu/STOP`; resume = launch again (scored, dumped and recalibrated users are skipped); DRYRUN=1.

## Expected cost

GPU: two SM120 scorings about 17 min each, smoke 5 min, two dumps about 20 min each: about 1.7 h of GPU if the queue is free. CPU: about 1.2 h per model, in parallel, behind each dump; report about 30 min. Verdict about 3.5 h after the launch.
