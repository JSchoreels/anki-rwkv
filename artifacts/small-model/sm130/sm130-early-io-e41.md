# Six-layer Tiny (SM130), io: early check after epoch 41

Decision: **continue** (pre-registered: stop if any trained head is above +1.20%). Six-layer epoch 41 minus the Tiny Instant-Only model (SM101)'s epoch 38 (uncut, both constant LR), show-time inputs, 722 users, % of the uncut epoch's LogLoss.

| head | six-layer | uncut | % [95% CI] | better / worse |
|---|---|---|---|---|
| instant | 0.279695 | 0.278896 | +0.286% [+0.242, +0.333] | 210 / 512 |
