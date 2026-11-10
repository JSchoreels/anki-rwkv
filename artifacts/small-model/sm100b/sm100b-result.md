# SM100b (the show-time error sensitivity test (SM100), follow-up): result

Verdict: **unchanged**. Removing the timeout reviews moves the decision pair by 0.002 points (needs >= 0.05 to count as changed).

Pair: the show-time instant-only small model (SM96) minus the show-time retrain of the release (SM92), instant head, show-time inputs (e2s-query), 722 users of the 732 set, review-weighted, paired user bootstrap 2,000 (seed 0), 95% CI. Points = % of the big model (1 leak-free epoch) normal-input instant LogLoss 0.275757. Negative = the show-time instant-only small model (SM96) is better.

| reviews | review count | users | points [95% CI] |
|---|---|---|---|
| (a) all reviews | 30,126,145 | 722 | -0.214 [-0.232, -0.195] |
| (b) without timeout reviews | 29,115,146 | 722 | -0.212 [-0.230, -0.193] |
| (c) timeout reviews only | 1,010,999 | 551 | -0.272 [-0.344, -0.200] |

Absolute instant LogLoss, show-time inputs:

| model | all reviews | without timeouts | timeouts only |
|---|---|---|---|
| the show-time instant-only small model (SM96) | 0.276287 | 0.274182 | 0.336900 |
| the show-time retrain of the release (SM92) | 0.276876 | 0.274766 | 0.337651 |

Checks: (a) reproduces the earlier -0.214 [-0.233, -0.195]; the new rows of the show-time retrain of the release (SM92) match its old rows bit for bit (dumped per-user LogLoss within 5e-7).

Cost: ticket wall 64 minutes including queue wait (GPU run 12 min 59 s); scorer 57,053 review-evals/s over the 723 users (741 s).

Interpretation: removing the timeout reviews moves the decision pair by about 0.002 points, so the timeout leak does not bias the comparison; on the timeout reviews alone, the show-time instant-only small model (SM96) is still better than the show-time retrain of the release (SM92).
