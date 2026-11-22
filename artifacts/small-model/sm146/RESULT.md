# Aggressive state tiers for Tiny (SM146): result

Branch `sm146`. Full tables: `sm146-result.md` and `sm146-result.json`; per-review and per-card read-out: `sm146-dr.json` (aggregates only); rules and labels: `design.md`. No training; the two six-layer Tiny release-candidate models, the six-layer Tiny two-head model (SM130) and the six-layer Tiny Instant-Only model (SM130), were scored as they are. Run 11-22 22:41 to 11-23 00:00.

## Question

The user's 11-22 rule lets the Tiny state cost more than the usual +0.05%, as long as Tiny stays within +1.00% of the matching Small model at int8 (95% CI upper bound, per head, paired by user on the 723 users). How far can the young-card tiers go (int3 or ternary int2 for the first answers of a card or note, int4 up to 15 answers, int8 from 16 answers), and how many state bytes does it save? Plus a per-card guard: at most 1% of cards may have a review whose predicted retention moves by more than 0.10 against int8.

## Result

The go pick is I3b (int3 for 1-8 answers, int4 for 9-15, int8 from 16, cards and notes alike): -42.71% state bytes against int8 (the int4 switch at 16 answers, AN16, is -35.34%). It passes every rule for both Tiny models. I3a (int3 for 1-4, int4 for 5-15, int8 from 16; -39.80%) and AN16 pass too. I2a (int2 for 1-4, int4 for 5-15, int8 from 16; -44.27%) fails only the per-card guard, and only on the six-layer Tiny Instant-Only model (SM130). I2b (int2 for 1-4, int3 for 5-15, int8 from 16; -48.64%) fails the per-card guard for both models.

| label | bytes vs int8 | six-layer Tiny two-head model (SM130), cards with a review > 0.10: curve / instant | six-layer Tiny Instant-Only model (SM130), instant | verdict |
|---|---|---|---|---|
| AN16 | -35.34% | 0.0000% / 0.0004% | 0.0008% | passes |
| I3a | -39.80% | 0.0000% / 0.0012% | 0.0086% | passes |
| I3b | -42.71% | 0.0020% / 0.0098% | 0.0407% | passes (go) |
| I2a | -44.27% | 0.2590% / 0.8964% | 1.9875% | fails the guard on the six-layer Tiny Instant-Only model (SM130) |
| I2b | -48.64% | 0.3887% / 1.2490% | 2.9121% | fails the guard on both models |

Cost against Small at int8 (the rule, upper bound at most +1.00%), worst upper bounds for I3b: +0.585% (curve) and +0.631% (instant) for the six-layer Tiny two-head model (SM130) against the show-time two-head cool-down for Small (SM118); +0.702% (instant) for the six-layer Tiny Instant-Only model (SM130) against the longer Small Instant-Only model (SM113). Even the failing labels pass this dataset-wide rule except I2b on the six-layer Tiny Instant-Only model (SM130) (+0.967% [+0.914%, +1.026%]), so the per-card guard is what stops the int2 tiers. Extra dataset-wide cost of I3b over AN16: +0.026% (curve) and +0.042% (instant) for the two-head model, +0.061% for the Instant-Only model. Long cards (128 or more earlier answers, CPU, 50 users): every label passes (point at most +0.5%, upper bound at most +1.0%; the highest upper bound is +0.847%, I2b curve head on the two-head model); I3b's worst upper bound is +0.310%.

## Per-card read-out in plain words

|dR| is how far a review's predicted retention moves against int8 on the same review. For I3b the typical move is tiny (median 0.0018 to 0.0025, 99th percentile 0.022 to 0.035); the largest single move is 0.267 (curve head of the six-layer Tiny two-head model (SM130)) and 0.219 (six-layer Tiny Instant-Only model (SM130)). The share of cards with a review off by more than 0.10 is 0.0020% (curve) and 0.0098% (instant) for the two-head model and 0.0407% for the Instant-Only model, far under 1%. The int2 tiers move predictions much more: for I2a the 99th percentile of reviews is 0.042 to 0.064 and 17.1% of cards on the Instant-Only model have a review above 0.05; the largest move is 0.679.

## Coverage and integrity

723 of 723 GPU users scored for every label, all finite; 50 of 50 long-card users per model and label, none killed or failed. The int8 reference (REF) and AN16 rows equal the earlier SM145 rows exactly (max |difference| 0.0 on 723 GPU users and 50 long-card users, both models); the Small int8 reference rows are the reused ones. The GPU tier-kernel smoke passed (23 checks; int8 labels GPU vs CPU max 1.0e-5, tiered labels max 4.0e-6).

## Cost

GPU: smoke 0 minutes, scoring ticket 001483 49 minutes plus retry passes of 1 and 0 minutes (tickets 001483 to 001485). CPU: the long-card chain took 194 process-minutes over 50 users (3,000 to 4,900 review-evals/s per process), launched 22:41 and finished 23:59 (about 78 minutes wall with 3 processes).

## What it means for the fork

The six-layer Tiny models can go to int3 for cards and notes with 1 to 8 answers (I3b), int4 up to 15 answers and int8 from 16 answers, for -42.71% state bytes against int8 and 7.4% of the int8 bytes more saved than the int4 switch at 16 answers (AN16), within the user's 11-22 Tiny rule. Whether to take I3b over the more conservative I3a (-39.80%, about a fifth of I3b's per-card share on the Instant-Only model) is the user's call; both pass. The int2 tiers are not recommended: they leave the per-card guard.
