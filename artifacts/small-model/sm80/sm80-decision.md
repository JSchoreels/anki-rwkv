# Data-order re-seed (SM80): epoch 21 with a new training_seed vs the same recipe with the usual seed

Arm: the leak-free small model (SM72s) epoch 20 + epoch 21 with another data-order seed + the quarter-epoch 32k Muon cool-down; reference: SM72s's own epoch 21 + the same cool-down (the tiny architecture additions (SM77) reference). Leak-free scoring, w8+s8, review-weighted LogLoss, user bootstrap 2,000 seed 0, 95% CI; deltas in points of % of the big model (1 leak-free epoch).

| pool | users | head | arm | reference | delta | points [95% CI] | users better | arm vs big | reference vs big |
|---|---|---|---|---|---|---|---|---|---|
| 732 (rule) | 722 | curve | 0.30074 | 0.30073 | +0.000010 | +0.00 [-0.03, +0.04] | 379 | +0.59% | +0.59% |
| 732 (rule) | 722 | instant | 0.27720 | 0.27722 | -0.000013 | -0.01 [-0.01, +0.00] | 373 | +0.53% | +0.53% |
| 732+long | 725 | curve | 0.30299 | 0.30296 | +0.000029 | +0.01 [-0.02, +0.04] | 380 | +0.58% | +0.58% |
| 732+long | 725 | instant | 0.27811 | 0.27812 | -0.000006 | -0.00 [-0.01, +0.01] | 374 | +0.56% | +0.56% |

Same architecture and size (w64); only epoch 21's per-window seeds differ.

Verdict: **stop** (both heads gain under 0.10 points).
