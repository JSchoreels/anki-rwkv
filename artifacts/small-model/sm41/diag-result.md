# The small model in the review simulator (SM41): diagnostic of the full-run result

Question (brief: [diag-brief.md](diag-brief.md)): the full run ([fullrun-result.md](fullrun-result.md)) found that
instant-head desired retention costs +14.5 ± 7.4% more minutes per memorised card than FSRS-7 at matched realised
retention (116 users). The simulated learner is the small model itself, so why does the more accurate model schedule
less efficiently? Tables: [diag/diag-tables.md](diag/diag-tables.md) (numbers in `diag/diag-result.json`), code
`scripts/sm41_diag.py`.

## What was run

- **Engine log.** `scripts/sm41_engine.py run --log-dir DIR` writes a per-review log (one `.npz` per split): lane, day,
  card, kind, elapsed days, grade, answer time, the learner's true recall (the instant head's P(pass) at the show
  moment: the grade is drawn from it), the interval chosen and the policy's aim at it, FSRS-7's recall and the curve
  head's recall at the show moment, the curve head's t90 before and after the review, per card memorised at day 60
  (both heads) and the card's real later-day Again share. Off by default and observation only. **Bit-identity
  proof** (user 1606, split 660, 9 lanes): the rows with logging off and with logging on are identical to the rows of
  the code before the change, except the wall-clock fields (`diag/bitcheck/`).
- **Sample.** 20 users, 1 split each (numpy `default_rng(0)` pick, rule in `diag/sample.json`), 60 days, lanes
  `tf` + `ihr-0.90` and `fsrs7-0.90` x seeds 0-3, the full run's pre-split FSRS-7 refits. 4 processes, 18.5 min wall
  (01:06-01:24). **Sample check:** all 20 splits reproduce the full-run rows for the same lanes (largest relative gap
  1.0e-6; backlog and metadata identical; `diag/check.out`).
- The per-review logs (53 MB) are not committed; `diag/run-diag.sh` regenerates them in about 20 min.
- H2 and H4 use the full-run rows (all 116 users, all 6 targets, the primary metric's matched-retention method).
- At target 0.90 the two policies are matched in the sample: realised retention 0.869 vs 0.873 (FSRS-7 - instant
  head +0.004 ± 0.027).

## Verdicts

| hypothesis | verdict | key numbers |
|---|---|---|
| H1 per-card spread | **spread exists, but it is not the cause** | FSRS-7's per-card true-recall spread is wider (SD over cards 0.082 vs 0.063, +0.019 ± 0.009). But a spread at the same mean buys nothing: Jensen check -0.4 ± 7.4% gain per minute; gain(R) is not convex (quadratic coefficient -1.1 ± 5.2). |
| H2 selection (less knowledge held) | **not supported** | At matched retention the instant head memorises MORE (+2.2 ± 1.3% by the instant head, +1.3 ± 0.3% by the curve head). The cost gap is the same with the curve head's count (+15.1 ± 7.5%). The whole gap is minutes (+16.9 ± 7.7%, median +9.3%). |
| H3 counterfactual miscalibration | **not supported** | The candidate rows are calibrated overall: aim - true recall +0.2 ± 1.6 points (FSRS-7: -0.5 ± 2.7, with a larger mean absolute error of 7.7 vs 5.0). At long intervals the candidates are slightly OPTIMISTIC (+1.4 ± 1.5 points at 16-31 days, +3.3 ± 3.8 at 32-63), which would push intervals longer, not shorter. |
| H4 horizon | **not supported** | The minutes gap does not shrink late: days 30-44 +18.1 ± 6.7% (median +10.3%), days 45-59 +24.6 ± 15.8% (median +10.8%); change +6.5 ± 12.9 points. |
| H5 (user: FSRS-7's errors give lower true recall where it pays) | **not supported as gain per minute; supported in a different form: answer time** | Lower true recall is not more efficient in this learner. Memory gained per minute (ln t90) RISES with true recall, overall and within each stability stratum (t90 3-20 days: -1.6 per minute at 0.80-0.85 vs +3.1 at 0.95-1.00). FSRS-7's overestimate is not concentrated on high-gain cards (Spearman -0.115 ± 0.100). But FSRS-7's errors are correlated with answer time, and that is where the minutes go (next section). |

## What the gap is: seconds per review

- **The size.** Over all 116 users at matched realised retention, a later-day review costs **+7.9 ± 1.6%** more
  seconds under the instant head (median +7.6%). This is the tightest number in the decomposition. The count of
  later-day reviews is also higher (median +8.9%; the mean, +71 ± 37%, is driven by a few users). Same-day steps add
  +8.9 ± 8.1%.
- **The mechanism (sample, days 30-59).** The instant head reads a slow previous answer as weaker memory:
  - Under the instant head, the cards' previous answer averages 17.1 s, against 15.0 s under FSRS-7
    (-2.1 ± 1.9 s for FSRS-7).
  - 44% of the instant head's later-day reviews follow an answer from the user's slowest third, against 39% for
    FSRS-7 (-5.3 ± 3.4 points). Only 25% follow one from the fastest third, against 34% (+8.2 ± 5.0 points).
  - So the instant head brings slow cards back sooner.
- **What FSRS-7 does instead.** FSRS-7 never sees answer times:
  - it reviews fast-answering cards early (true recall 0.908, against 0.888 under the instant head);
  - it reviews slow ones late (0.859, against 0.874).
  - Its mean true recall is the same. Its errors move reviews away from the expensive cards.
- **Not a fixed card property.** Split by the card's real answer-time tercile, the instant head is slower in every
  tercile (by 1.2-1.7 s per review). The selection runs on the card's current state: its recent answer times in the
  lane, which also feed the answer-time head.
- At the same true recall, FSRS-7's reviews are also faster (0.90-0.95: 14.1 vs 15.3 s; 0.95-1.00: 12.0 vs 14.3 s).
- The sample at target 0.90 shows the same: FSRS-7 did +128 ± 315 more later-day reviews, but at -1.5 ± 2.0 s each,
  for -64 ± 99 minutes. The sample's own cost gap is +8.3 ± 9.6% (20 users).

Other findings:

- **The 1-day floor.** 16% of the instant head's later-day reviews (19% for FSRS-7) are at 1 day with an aim below
  the target, because P(pass) at 1 day was already under 0.90. Their true recall is 0.80 (FSRS-7: 0.83).
- **Where FSRS-7 saves, by card difficulty** (the real later-day Again share): the hardest third, -34.9 ± 34.2 minutes
  for the same number of reviews. By interval, the saving is mostly at short elapsed times: 1 day -20 minutes, 2-3
  days -16 minutes.

## What a cost-optimising policy would need

- **Cost in the objective.** "Hit R on every card" equalises recall but ignores what a review costs. A
  cost-optimising policy needs the answer time in its objective:
  - expected minutes per review (from the answer-time head, or from the card's recent answer times);
  - plus the lapse cost (same-day steps).
- **Per-card targets.** It then sets a lower target on cards that are slow to answer, and a higher one on fast cards.
  FSRS-7 does this by accident, because it is blind to time.
- **The minimum test.** A per-card target shifted by the card's recent answer time against the user's median: the
  time-aware objective of DESIGN.md 9.2, reduced to a one-parameter rule.
- **Not supported by this diagnostic:**
  - deliberate noise on the target (no convexity to exploit);
  - lower targets on high gain-per-minute cards (gain per minute rises with recall).

  Part B tests both anyway as controls ([policy-result.md](policy-result.md)).
