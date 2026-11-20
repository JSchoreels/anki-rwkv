# Six-layer Tiny with longer recovery (SM135), io: early check after epoch 42

Decision: **stop**. Pre-registered: stop / cool down from the previous epoch if any trained head is significantly worse than the previous epoch (CI lower bound > 0); after epoch 43 stop if any trained head gained less than 0.05% vs the six-layer Tiny (SM130)'s epoch 41. Constant-LR checkpoints, show-time inputs, 722 users, w8+s8.

| head | epoch 42 | vs the six-layer Tiny (SM130)'s epoch 41 % [95% CI] | vs the six-layer Tiny (SM130)'s epoch 41 % [95% CI] | better / worse (vs previous) |
|---|---|---|---|---|
| instant | 0.279924 | +0.082% [+0.041, +0.121] | +0.082% [+0.041, +0.121] | 217 / 505 |
