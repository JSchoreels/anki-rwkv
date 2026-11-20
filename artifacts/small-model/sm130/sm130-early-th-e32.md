# Six-layer Tiny (SM130), th: early check after epoch 32

Decision: **continue** (pre-registered: stop if any trained head is above +1.20%). Six-layer epoch 32 minus the Tiny two-head model (SM106)'s epoch 29 (uncut, both constant LR), show-time inputs, 722 users, % of the uncut epoch's LogLoss.

| head | six-layer | uncut | % [95% CI] | better / worse |
|---|---|---|---|---|
| curve | 0.307132 | 0.304025 | +1.022% [+0.710, +1.288] | 140 / 582 |
| instant | 0.281032 | 0.279254 | +0.637% [+0.578, +0.701] | 61 / 661 |
