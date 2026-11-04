# The small model in the review simulator (SM41): full-run result

## Design
117 users of the 732 set x 304 split points, 60 days, 59 lanes per split; FSRS-7 refit before each split day; details in DESIGN.md section 10. Checks: 304 splits planned and finished, 0 missing, 0 duplicates, 59 lanes per row, 304 fits, 0 tracebacks, FSRS-7 source = pre-split refit only.

## FSRS-7 fit check
The srs-benchmark fit had seen the split day in 275 of 304 splits (footer of report.out). Refit proof: `fullrun/fsrs7-proof-cpu.json`, numbers in DESIGN.md 10.1. The small model's training users do not overlap the 732 set.

## Primary (instant head vs FSRS-7 at matched realised retention, paired per user)
Common range (116 users): relative cost gap +14.5 +/- 7.4 % (median +7.6 %; instant head cheaper for 16 % of users); 60-day cost +15.0 +/- 10.4 %.
Verdict: **FSRS-7 cheaper (CI excludes 0)**.
Decision rule: the instant head would move on to the time-aware objective only if the gap were < 0 with the CI excluding 0. It is > 0, so FSRS-7 stays the scheduler baseline and the time-aware objective (DESIGN.md 9.2) is not started.
Secondary points: 0.80 +9.1 +/- 3.2 %, 0.85 +8.1 +/- 3.4 %, 0.90 +20.6 +/- 11.8 % (0.75 is noisy: +57 +/- 89 %).

## Policies
See `fullrun/fullrun-tables.md` (retention, aim, min/day, memorised, cost per policy). At target 0.90: instant head 0.879 retention, 0.793 min per memorised card; FSRS-7 0.875, 0.784.

## Best desired retention per user
Modal best target (P = 1 / 3 / 10 min per extra memorised card):
- instant head: 0.70 (86 users) / spread, 0.90 most (25) / 0.95 (83)
- FSRS-7: 0.70 (75) / 0.95 (48) / 0.95 (87)
A per-user target matters: gain vs best fixed target, instant head +7.6 / +28.2 / +13.2 cards (+0.85 / +2.75 / +0.88 %); FSRS-7 +19.8 / +56.8 / +22.5 cards (+2.19 / +5.31 / +1.33 %).

## Re-anchoring
Gap at target 0.90: anchored +16.0 +/- 6.3 %, free +66.0 +/- 65.8 %; free minus anchored +49.9 +/- 66.1 % (CI includes 0; sign flips in 27 users). Free lanes drift: instant head Easy share +1.9 points, +2.1 s per review; FSRS-7 +3.8 points, +0.5 s.

## Drift check
Daily shared-stream re-anchoring (fr-shared1) state drift: -0.4 +/- 0.5 points over 115 users (drift study: -0.6 +/- 1.0). Consistent.

## Speed and CPU
682 lane reviews/s per process (1 thread, 4 processes); 96.2M lane reviews in 39.19 process-hours, warm-ups 3.32, 42.51 CPU process-hours in all. Wall time 07:59:52 to 19:39:40 (11 h 40 min, includes the early RAM hold; no quiet-window hold).
