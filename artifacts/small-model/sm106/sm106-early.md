# The Tiny two-head model (SM106): early check

Decision: **stop**. Epoch 27 of the Tiny two-head model (SM106) (card stream 1 layer, constant LR) minus the show-time retrain (SM92)'s epoch 26 (constant LR, the cut's source), show-time inputs, 722 users, % of the big model (1 leak-free epoch)'s show-time LogLoss:

| head | arm | ref | % [95% CI] | better / worse |
|---|---|---|---|---|
| curve | 0.303854 | 0.303079 | +0.259% [+0.131, +0.399] | 301 / 421 |
| instant | 0.279473 | 0.278939 | +0.193% [+0.151, +0.234] | 172 / 550 |

Rule (pre-registered): stop if the point estimate is above +0.25% on either head; otherwise continue to epochs 28-29 and the cool-down.
