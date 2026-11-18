# The smaller instant-only model (SM101): early check

Decision: **continue**. Epoch 36 of the smaller instant-only model (SM101) (card stream 1 layer, constant LR) minus the show-time instant-only model, stage 2 (SM98)'s epoch 35 (constant LR, the cut's source), instant head, show-time inputs, 722 users: +0.051 [-0.004, +0.109] points (% of the big model (1 leak-free epoch) normal-input instant LogLoss), 378 better / 343 worse; arm 0.278963 vs ref 0.278821.

Rule (pre-registered): stop if the point estimate is above +0.25 points; otherwise continue to epochs 37-38 and the cool-down.
