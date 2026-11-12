# The show-time instant-only big model (SM99): early rule

Decision: **continue**. Epoch 1 of the show-time instant-only big model (SM99) minus the show-time instant-only model, stage 2 (SM98), instant head, show-time inputs, 722 users: -0.069% [-0.094, -0.044] (% of the big model (1 leak-free epoch) normal-input instant LogLoss), 444 better / 277 worse; 0.276002 vs 0.276191.

For reference: epoch 1 minus the show-time big model (SM91 step 2), its start, show-time instant: -0.066% [-0.076, -0.056].

Rule (pre-registered): continue to epoch 2 if the point estimate is below 0; otherwise stop, and the final read-out runs on epoch 1.
