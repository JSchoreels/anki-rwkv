# Six-layer Tiny with longer recovery (SM135): result

Verdict: **no go (early stop after epoch 42)**. The Instant-Only chain ended as pre-registered; the two-head chain did not run.
More constant-LR recovery made the Instant-Only model worse, not better, than the six-layer Tiny (SM130)'s epoch 41.

Early check after epoch 42 (constant LR, show-time inputs, 722 users, w8+s8):

| head | epoch 42 LogLoss | vs the six-layer Tiny (SM130)'s epoch 41 % [95% CI] | better / worse users |
|---|---|---|---|
| instant | 0.279924 | +0.082% [+0.041, +0.121] (worse) | 217 / 505 |

Speed: epoch 42 complete: 581,942,724 reviews, 54.1 min training, 179,160 reviews/s.
Real GPU time: about 65 min (about 02:01 to 03:06 Fri 11-20: trainer smoke, epoch 54.4 min, scoring about 10 min), queue waiting excluded.
