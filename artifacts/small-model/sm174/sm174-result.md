# How many users for MIT weights (SM174): result

Status: **complete**

Every row: six-layer Tiny architecture (`w64-c1-d2p1u1`) trained from scratch under the same review budget (four 8k epochs), scored on the 732 set (723 users), show-time inputs, w8+s8hx8, against the six-layer Tiny two-head release model (SM130). % = review-weighted LogLoss vs the reference, paired user bootstrap 95% CI.

| model | curve vs SM130 | instant vs SM130 | curve LogLoss | instant LogLoss | vs the 8,000-user teacher control (SM174 n8000-kd), curve / instant | epochs (best constant epoch + decay) | reviews trained | GPU h | reviews/s |
|---|---|---|---|---|---|---|---|---|---|
| 500 users, no teacher (SM174 n0500) | +3.97% [+3.56, +4.52] | +3.78% [+3.55, +4.04] | 0.3154 | 0.2906 | +2.69% [+2.39, +3.07] / +2.30% [+2.14, +2.47] | 7 + 1 | 0.58B | 0.9 | 177,401 |
| 1,000 users, no teacher (SM174 n1000) | +2.39% [+2.02, +2.87] | +2.12% [+1.98, +2.28] | 0.3106 | 0.2860 | +1.12% [+0.84, +1.50] / +0.67% [+0.61, +0.73] | 20 + 4 | 2.12B | 3.2 | 182,821 |
| 2,000 users, no teacher (SM174 n2000) | +1.87% [+1.63, +2.20] | +1.88% [+1.77, +2.01] | 0.3091 | 0.2853 | +0.61% [+0.47, +0.79] / +0.43% [+0.39, +0.48] | 8 + 1 | 1.61B | 2.4 | 187,848 |
| 4,000 users, no teacher (SM174 n4000) | +1.70% [+1.50, +1.95] | +1.70% [+1.60, +1.81] | 0.3086 | 0.2848 | +0.45% [+0.34, +0.56] / +0.25% [+0.21, +0.29] | 4 + 1 | 1.44B | 2.1 | 187,067 |
| 8,000 users, no teacher (SM174 n8000) | +1.49% [+1.33, +1.69] | +1.72% [+1.62, +1.84] | 0.3079 | 0.2849 | +0.24% [+0.16, +0.32] / +0.27% [+0.24, +0.31] | 2 + 1 | 2.33B | 3.5 | 184,668 |
| 8,000 users, with the usual teacher (SM174 n8000-kd) | +1.25% [+1.12, +1.42] | +1.45% [+1.36, +1.55] | 0.3072 | 0.2841 | - | 2 + 1 | 2.33B | 3.6 | 180,270 |

## Users needed (no-teacher arms, gap = the worse of the two heads)

| threshold | vs SM130, point estimate | vs SM130, CI upper bound | vs the equal-compute teacher control (SM174 n8000-kd), point |
|---|---|---|---|
| +1.0% | > 8,000 (not reached; fitted floor +1.67% at this budget) | > 8,000 (not reached; fitted floor +1.78% at this budget) | ~1,182 |
| +0.5% | > 8,000 (not reached; fitted floor +1.67% at this budget) | > 8,000 (not reached; fitted floor +1.78% at this budget) | ~3,194 |

## Held-out slice (early stopping; 190 validation users never trained on, not in the 732 set)

- 500 users, no teacher (SM174 n0500): epoch 7: 0.3150 (+0.00%), epoch 14: 0.3153 (+0.09%). early stop: eval point 14 is not 0.05% below the best earlier point; decay 1 epoch(s) from epoch 7.
- 1,000 users, no teacher (SM174 n1000): epoch 4: 0.3151 (+0.00%), epoch 8: 0.3128 (-0.74%), epoch 12: 0.3109 (-1.36%), epoch 16: 0.3105 (-1.47%), epoch 20: 0.3101 (-1.61%), epoch 24: 0.3109 (-1.36%). early stop: eval point 24 is not 0.05% below the best earlier point; decay 4 epoch(s) from epoch 20.
- 2,000 users, no teacher (SM174 n2000): epoch 2: 0.3188 (+0.00%), epoch 4: 0.3176 (-0.37%), epoch 6: 0.3105 (-2.61%), epoch 8: 0.3095 (-2.92%), epoch 10: 0.3102 (-2.71%). early stop: eval point 10 is not 0.05% below the best earlier point; decay 1 epoch(s) from epoch 8.
- 4,000 users, no teacher (SM174 n4000): epoch 1: 0.3167 (+0.00%), epoch 2: 0.3147 (-0.63%), epoch 3: 0.3098 (-2.18%), epoch 4: 0.3096 (-2.23%). early stop: eval point 4 is not 0.05% below the best earlier point; decay 1 epoch(s) from epoch 4.
- 8,000 users, no teacher (SM174 n8000): epoch 1: 0.3137 (+0.00%), epoch 2: 0.3086 (-1.63%), epoch 3: 0.3094 (-1.38%). early stop: eval point 3 is not 0.05% below the best earlier point; decay 1 epoch(s) from epoch 2.
- 8,000 users, with the usual teacher (SM174 n8000-kd): epoch 1: 0.3106 (+0.00%), epoch 2: 0.3076 (-0.96%), epoch 3: 0.3088 (-0.58%). early stop: eval point 3 is not 0.05% below the best earlier point; decay 1 epoch(s) from epoch 2.
