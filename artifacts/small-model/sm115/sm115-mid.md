# The longer cool-down for Tiny (SM115): early stop after pass 1

Decision: **continue**. The longer cool-down for Tiny (SM115)'s model after pass 1 of 2 (learning rate 29.29% of its peak) minus the Tiny Instant-Only model (SM101)'s own cool-down snapshot after 4,094 of 8,000 users (learning rate 29.31% of its peak), instant head, show-time inputs, 722 users: -0.111% [-0.129, -0.094] (% of the big model (1 leak-free epoch) normal-input instant LogLoss), 510 better / 211 worse; 0.276874 vs 0.27718.

Read-out (no verdict): minus the longer big instant-only teacher (SM113)'s model after its pass 1 (the Small Instant-Only architecture, the same recipe at the same point): +0.188% [+0.165, +0.211], 105 better / 616 worse.

Rule (pre-registered): continue with pass 2 if the point estimate is below 0; otherwise stop (verdict no go).
