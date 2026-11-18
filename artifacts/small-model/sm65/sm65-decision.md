# Better teacher (SM65) stage 1: big model + 1 leak-free epoch on 3,969 new users vs the big model (1 leak-free epoch)

Leak-free scoring, w8+s8, review-weighted LogLoss, user bootstrap 2,000 seed 0, 95% CI.

| pool | users | head | new | reference | delta | % vs reference [95% CI] | users better |
|---|---|---|---|---|---|---|---|
| 732 (rule) | 722 | curve | 0.29845 | 0.29897 | -0.000512 | -0.17% [-0.29, -0.05] | 409 |
| 732 (rule) | 722 | instant | 0.27566 | 0.27576 | -0.000095 | -0.03% [-0.04, -0.03] | 428 |
| 732+long | 725 | curve | 0.30045 | 0.30123 | -0.000774 | -0.26% [-0.44, -0.09] | 412 |
| 732+long | 725 | instant | 0.27649 | 0.27658 | -0.000085 | -0.03% [-0.04, -0.02] | 429 |

Verdict: **grey** (between the go and stop bars: report and ask).
