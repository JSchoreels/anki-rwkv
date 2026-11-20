# Six-layer Tiny (SM130), io: early check after epoch 39

Decision: **continue** (pre-registered: stop if any trained head is above +1.50%). Six-layer epoch 39 minus the Tiny Instant-Only model (SM101)'s epoch 38 (uncut, both constant LR), show-time inputs, 722 users, % of the uncut epoch's LogLoss.

| head | six-layer | uncut | % [95% CI] | better / worse |
|---|---|---|---|---|
| instant | 0.281682 | 0.278896 | +0.999% [+0.873, +1.135] | 47 / 675 |
