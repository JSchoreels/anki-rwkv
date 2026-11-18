# The show-time two-head cool-down for Tiny (SM124): early stop after pass 1

Decision: **continue**. The show-time two-head cool-down for Tiny (SM124)'s model after pass 1 of 2 (learning rate 29.29% of its peak) minus the Tiny two-head model (SM106)'s own cool-down snapshot after 4,000 of 8,000 users (learning rate 30.45% of its peak), show-time inputs, 722 users (% of the big model (1 leak-free epoch) normal-input LogLoss of each head):

| head | pass 1 | SM106 snapshot | % [95% CI] | better / worse |
|---|---|---|---|---|
| curve | 0.301111 | 0.301268 | -0.052% [-0.148, +0.041] | 374 / 348 |
| instant | 0.277656 | 0.278006 | -0.127% [-0.146, -0.111] | 532 / 189 |

Read-out (no verdict): pass 1 minus the Tiny two-head model (SM106)'s final model: curve +0.145% [+0.009, +0.251], instant +0.104% [+0.092, +0.117].

Rule (pre-registered): continue with pass 2 if at least one head's point estimate is below 0 and neither is +0.05% or more; otherwise stop (verdict no go).
