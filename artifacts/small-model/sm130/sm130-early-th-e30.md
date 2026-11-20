# Six-layer Tiny (SM130), th: early check after epoch 30

Decision: **continue** (pre-registered: stop if any trained head is above +1.50%). Six-layer epoch 30 minus the Tiny two-head model (SM106)'s epoch 29 (uncut, both constant LR), show-time inputs, 722 users, % of the uncut epoch's LogLoss.

| head | six-layer | uncut | % [95% CI] | better / worse |
|---|---|---|---|---|
| curve | 0.308215 | 0.304025 | +1.378% [+1.085, +1.657] | 117 / 605 |
| instant | 0.281818 | 0.279254 | +0.918% [+0.852, +0.990] | 44 / 678 |
