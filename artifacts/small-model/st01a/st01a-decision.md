# Curve consistency on real rapid Good pairs (ST01a): stage 1 decision

**Setup.** Stage 1 measured, on the 723 fit users of the 732 set and every review of their histories, how the forgetting curve moves after a rapid Good: two consecutive reviews a -> b of one card, b answered Good, gap 1-3 days, a passed. Models: the leak-free small model (SM72s, epoch-25 cool-down) and the big model (1 leak-free epoch), both w8+s8. Each user has one valid row per model; the jsonl files have 724 lines because one user per model first failed with an out-of-memory error (user 3735 for the small model, user 4852 for the big model) and was redone in the next pass, so the file holds the error row plus the good row. There was no extra user. The per-user curve LogLoss matches the scorer rows to under 1e-7 (723 users small, 722 big).

## 365-day jump detail (rapid pairs)

| slice | model | median jump | median ratio | down | down by > 0.1 | abs > 0.1 | 90th pct abs |
|---|---|---|---|---|---|---|---|
| rapid (gap 1-3 d, a passed) | small | -0.018 | 0.962 | 64% | 10% | 14% | 0.115 |
| rapid (gap 1-3 d, a passed) | big | -0.018 | 0.965 | 62% | 14% | 18% | 0.135 |

## Tail calibration: which curve was closer at the real next review

| slice | model | days to next review | reviews | recall | mean after | mean before | LogLoss after | LogLoss before | after closer |
|---|---|---|---|---|---|---|---|---|---|
| rapid (gap 1-3 d, a passed) | small | 0-7 | 3,660,463 | 0.921 | 0.919 | 0.831 | 0.2287 | 0.2796 | 90% |
| rapid (gap 1-3 d, a passed) | small | 7-30 | 765,232 | 0.888 | 0.894 | 0.820 | 0.2792 | 0.3114 | 85% |
| rapid (gap 1-3 d, a passed) | small | 30-90 | 70,637 | 0.704 | 0.731 | 0.682 | 0.5005 | 0.5169 | 61% |
| rapid (gap 1-3 d, a passed) | small | 90-180 | 20,184 | 0.618 | 0.631 | 0.618 | 0.5718 | 0.5793 | 52% |
| rapid (gap 1-3 d, a passed) | small | 180- | 27,286 | 0.625 | 0.605 | 0.618 | 0.6103 | 0.6057 | 47% |
| rapid (gap 1-3 d, a passed) | big | 0-7 | 3,660,463 | 0.921 | 0.919 | 0.830 | 0.2284 | 0.2814 | 89% |
| rapid (gap 1-3 d, a passed) | big | 7-30 | 765,232 | 0.888 | 0.892 | 0.813 | 0.2772 | 0.3163 | 85% |
| rapid (gap 1-3 d, a passed) | big | 30-90 | 70,637 | 0.704 | 0.714 | 0.676 | 0.5028 | 0.5250 | 60% |
| rapid (gap 1-3 d, a passed) | big | 90-180 | 20,184 | 0.618 | 0.631 | 0.627 | 0.5826 | 0.5886 | 53% |
| rapid (gap 1-3 d, a passed) | big | 180- | 27,286 | 0.625 | 0.640 | 0.651 | 0.6000 | 0.6063 | 50% |

Pooled over the three 30+ day buckets (118,107 reviews), LogLoss after / before the Good: small 0.5381 / 0.5481, big 0.5389 / 0.5547. The curve after the Good is better in 30-90 and 90-180 days for both models; only the small model's 180+ bucket is slightly worse after (0.6103 vs 0.6057, 47% closer).

## Rule inputs (slice "rapid (gap 1-3 d, a passed)", 365 days)

Part A, is the small model clearly jumpier than the big model? At least one criterion must hold.

| criterion | small | big | threshold | holds |
|---|---|---|---|---|
| share |jump| > 0.1 | 14% | 18% | small >= 1.5x big | no (0.8x) |
| share down by > 0.1 | 10% | 14% | small >= big + 5 points | no (-4 points) |
| mature slice (interval before a >= 21 d) goes down | 56% | 45% | small > 50% while big does not | yes |

Part B, does the tail calibration fail to back the fall (curve after the Good no better than before at 30+ days)? Small model: LogLoss after is lower than before in 30-90 d (0.5005 vs 0.5169), 90-180 d (0.5718 vs 0.5793) and pooled 30+ d (0.5381 vs 0.5481). So the calibration backs the fall: no.

## Verdict

**Stage 2 NOT needed.** Part A holds only through the mature-slice criterion (56% vs 45% going down, small mean -0.012 vs big +0.004); the small model is otherwise less jumpy than the big model (fewer large jumps). Part B fails: the curve after the Good predicts the real later reviews better at 30+ days, so the 365-day fall is a correction, and a penalty pulling the two curves together would hurt. Both conditions are required, so by the pre-registered rule no consistency penalty is trained. Stage 1 closes curve consistency on real rapid Good pairs (ST01a).
