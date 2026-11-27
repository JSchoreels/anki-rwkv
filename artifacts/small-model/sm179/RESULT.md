# Session inputs (SM179): result

Read out Fri 2026-11-27 (finished 22:45, `SM179 DONE`). Tables: `sm179-result.md`, `sm179-result.json`.

## Question

Do session and fatigue inputs (arm A, 6 columns), or the card's review kind as 4 one-hot columns (arm B, the information of input 22), lower the LogLoss of six-layer Tiny two-head (SM130) by at least 0.05% on one head, when added in a short cool-down from the release model?

## Rules (pre-registered in `design.md`)

The statistic is the change in review-weighted LogLoss against the control, in %, with a paired user bootstrap (2,000 draws, 95% interval). An arm is a go when on one head (curve or instant) the change is at most -0.05% and the upper bound of its interval is below 0, and on the other head the point estimate is at most +0.05%. The verdict is GO if one arm passes, otherwise NO GO. The control is the control cool-down of the state-energy penalty screen (SM171), rescored by this screen's scorer.

## Verdict: GO (session inputs only, on the instant head)

| arm | curve vs control (95% CI) | instant vs control (95% CI) | go |
|---|---|---|---|
| session inputs (6 columns) | +0.001% [-0.002, +0.003] | -0.065% [-0.073, -0.058] | yes, on the instant head |
| card-state inputs (4 one-hot columns) | -0.003% [-0.006, -0.001] | -0.008% [-0.011, -0.006] | no (both heads far below 0.05%) |

The control's LogLoss (723 users of the 732 set, show-time inputs, int8 weights and states) is 0.30337 on the curve head and 0.28009 on the instant head. The session inputs leave the curve head unchanged (the interval contains 0, the point estimate is within +0.05%) and improve the instant head by 0.065%, which is consistently present in every answer-count bucket up to 128 answers (-0.078% for cards with 1 earlier answer, -0.037% at 128). The card-state inputs are statistically below 0 on both heads but about ten times smaller than the 0.05% minimum effect, so they do not pass. Input 22 therefore stays zeroed, now with a measured reason (no useful gain).

By answer-count bucket (reported only): the session arm's curve head moves by less than 0.003% in every bucket up to 32, and it is worse in the rarest bucket (1000 and more answers, 5,997 reviews, +2.9%), while the card-state arm is better there (-0.81%); these buckets hold under 0.02% of reviews.

## Speed and cost

| model | reviews/s | training minutes | peak allocated / reserved MiB |
|---|---|---|---|
| control (SM171's control cool-down) | 55,723 | 20.1 | 4,216 / 8,354 |
| session inputs (6 columns) | 63,423 | 17.7 | 4,223 / 8,368 |
| card-state inputs (4 one-hot columns) | 63,872 | 17.6 | 4,220 / 8,358 |

The two new arms ran faster than the control (the cause is not established; the extra columns cannot make training faster, so treat it as machine load or GPU sharing); the extra columns cost no measurable memory. GPU minutes: about 35 for the two trainings plus about 10 for each of the scoring passes of the two arms and of the control (the tickets' wall times in the log include queue waits behind SM180), about 1.1 GPU hours in total, well within the 4-hour budget.

## Integrity

The control's rows rescored here equal SM171's rows on all 723 common users (max |LogLoss difference| 0.0). 723 rule users (9 of the 732 did not fit under the memory cap), no non-finite row. CPU proofs C1 to C4 (`check-cpu-sm179.json`) all OK before the screen.

## Next (decision for the main session)

As pre-registered: a follow-up of the session inputs in a full Tiny cool-down (release recipe, all 8,000 users), then Small, then the fork's feature builder with test vectors; arm B alone is not worth it, and A+B together is not required (B did not pass). The true time of day and weekday (A19) become the next inputs to test on donated data. Per the priority re-assessment rule a fresh design agent comes before anything is launched.
