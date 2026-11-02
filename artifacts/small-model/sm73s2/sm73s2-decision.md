# Smaller model, card stream 1 layer (SM73 stage 2) early check: 1 card layer vs 2, paired cool-downs

Leak-free scoring, w8+s8, review-weighted LogLoss, user bootstrap 2,000 seed 0, 95% CI. Arm: the leak-free small model (SM72s) epoch 20 cut to 1 card layer + 1 constant-LR epoch + quarter-epoch 32k Muon cool-down. Reference: SM72s epoch 21 + the same cool-down. Points = % of the big model (1 leak-free epoch).

| pool | users | head | arm | reference | delta [95% CI] | points [95% CI] | users better | arm vs big | reference vs big |
|---|---|---|---|---|---|---|---|---|---|
| 732 (rule) | 722 | curve | 0.30122 | 0.30073 | +0.000486 [+0.000382, +0.000600] | +0.16 [+0.13, +0.20] | 192 | +0.75% | +0.59% |
| 732 (rule) | 722 | instant | 0.27781 | 0.27722 | +0.000589 [+0.000536, +0.000648] | +0.21 [+0.19, +0.24] | 94 | +0.74% | +0.53% |
| 732+long | 725 | curve | 0.30350 | 0.30296 | +0.000538 [+0.000417, +0.000662] | +0.18 [+0.14, +0.22] | 192 | +0.75% | +0.58% |
| 732+long | 725 | instant | 0.27878 | 0.27812 | +0.000665 [+0.000560, +0.000791] | +0.24 [+0.20, +0.29] | 94 | +0.80% | +0.56% |

Rule: stop if the delta exceeds +0.00020 curve or +0.00015 instant (point estimate).

Projection (estimate, not the rule): the released small model is +0.36% / +0.28% vs the big model (1 leak-free epoch); + this delta = +0.52% curve / +0.49% instant (bar +0.50% / +0.40%).

Size: 331,511 -> 311,786 parameters; scorer w8 weight bytes 410,459 -> 385,485. Bytes per card + note (fork layout): int8 4,554 -> 3,134, int4 3,018 -> 2,110.

Verdict: **stop** (curve +0.000486 > +0.00020; instant +0.000589 > +0.00015).
