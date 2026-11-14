# The longer big instant-only teacher (SM113): early stop after pass 1

Decision: **continue**. The longer big instant-only teacher (SM113)'s model after pass 1 of 2 (learning rate 29.3% of its peak) minus the Small Instant-Only model (SM98)'s cool-down snapshot after 4,000 of 8,000 users (learning rate 29.9% of its peak), instant head, show-time inputs, 722 users: -0.144% [-0.166, -0.125] (% of the big model (1 leak-free epoch) normal-input instant LogLoss), 558 better / 164 worse; 0.276357 vs 0.276755.

Read-out (no verdict): minus the big instant-only teacher (SM110)'s snapshot after 4,000 users (the same targets at the same learning rate): -0.143% [-0.163, -0.124], 551 better / 171 worse.

Rule (pre-registered): continue with pass 2 if the point estimate is below 0; otherwise stop (verdict no go).
