# The show-time two-head cool-down for Small (SM118): early stop after pass 1

Decision: **continue**. The show-time two-head cool-down for Small (SM118)'s model after pass 1 of 2 (learning rate 29.29% of its peak) minus the show-time retrain of the release (SM92)'s own cool-down snapshot after 4,056 of 8,000 users (learning rate 29.77% of its peak), show-time inputs, 722 users (% of the big model (1 leak-free epoch) normal-input LogLoss of each head):

| head | pass 1 | SM92 snapshot | % [95% CI] | better / worse |
|---|---|---|---|---|
| curve | 0.300777 | 0.301217 | -0.147% [-0.239, -0.062] | 419 / 302 |
| instant | 0.277205 | 0.277462 | -0.093% [-0.112, -0.076] | 480 / 241 |

Read-out (no verdict): pass 1 minus the show-time retrain of the release (SM92)'s final model: curve +0.141% [+0.030, +0.230], instant +0.119% [+0.106, +0.132].

Rule (pre-registered): continue with pass 2 if at least one head's point estimate is below 0 and neither is +0.05% or more; otherwise stop (verdict no go).
