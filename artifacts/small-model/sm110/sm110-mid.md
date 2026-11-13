# The big instant-only teacher (SM110): mid-check (the early stop)

Decision: **continue**. The big instant-only teacher (SM110)'s cool-down snapshot after 4,000 of the 8,000 users minus the Small Instant-Only model (SM98)'s snapshot after the same 4,000 users, instant head, show-time inputs, 722 users: -0.001% [-0.010, +0.007] (% of the big model (1 leak-free epoch) normal-input instant LogLoss), 375 better / 343 worse; 0.276753 vs 0.276755.

Rule (pre-registered): continue if the point estimate is below 0; otherwise stop (the cool-down is not finished and the verdict is no go).
