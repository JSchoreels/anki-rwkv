# The instant-only small model (SM57) stage 3: the mirror arm, curve-only vs joint

Curve-only = stage 2's joint arm with the instant loss term at weight 0 (`instant_loss_weight` 0.0, SM26's option; the instant KD targets live in that term), same fresh init, the 2k tier, the leak-free small model (SM72s) recipe, same users, order and seeds; the joint and instant-only arms are stage 2's (not retrained). Constant-LR checkpoints. Leak-free scoring, normal inputs, w8+s8, 722 users of the 732 set; points = % of the big model (1 leak-free epoch) LogLoss, paired user bootstrap 2,000, seed 0, 95% CI.

| epoch | curve-only - joint, curve (the rule) | users better / worse | curve-only - joint, instant | curve-only - big, curve | joint - big, curve | curve-only - big, instant |
|---|---|---|---|---|---|---|
| 1 | +0.135 [-0.063, +0.323] | 287 / 435 | +43.670 [+39.749, +47.869] | +6.872 [+6.329, +7.465] | +6.737 [+6.141, +7.417] | +50.160 [+46.003, +54.582] |
| 3 | +1.017 [+0.838, +1.208] | 159 / 563 | +46.378 [+42.331, +50.657] | +5.326 [+4.883, +5.832] | +4.309 [+3.888, +4.786] | +50.160 [+46.003, +54.582] |
| 5 | +0.873 [+0.655, +1.104] | 219 / 503 | +46.846 [+42.812, +51.140] | +5.402 [+4.888, +5.973] | +4.529 [+4.133, +4.963] | +50.160 [+46.003, +54.582] |

Two dedicated models (instant-only for the instant head, curve-only for the curve head) vs the joint model:

| epoch | instant: instant-only - joint | curve: curve-only - joint | instant-only - big, instant | curve-only - big, curve | joint - big, instant / curve |
|---|---|---|---|---|---|
| 1 | -0.461 [-0.621, -0.329] | +0.135 [-0.063, +0.323] | +6.029 [+5.677, +6.397] | +6.872 [+6.329, +7.465] | +6.490 [+6.079, +6.914] / +6.737 [+6.141, +7.417] |
| 3 | -0.516 [-0.594, -0.445] | +1.017 [+0.838, +1.208] | +3.266 [+3.078, +3.454] | +5.326 [+4.883, +5.832] | +3.782 [+3.555, +4.019] / +4.309 [+3.888, +4.786] |
| 5 | -0.773 [-0.861, -0.685] | +0.873 [+0.655, +1.104] | +2.541 [+2.382, +2.699] | +5.402 [+4.888, +5.973] | +3.315 [+3.104, +3.521] / +4.529 [+4.133, +4.963] |

Training per epoch (curve-only, and stage 2's joint arm for reference):

| arm / epoch | status | segments | wall min | training min | reviews/s | peak allocated / reserved MiB | nvidia-smi max / baseline MiB |
|---|---|---|---|---|---|---|---|
| curve e01 | complete | 1 | 30.3 | 29.6 | 82,522 | 10,350 / 12,746 | 13038 / 911 |
| joint e01 (stage 2) | complete | 1 | 29.8 | 29.4 | 83,234 | 10,350 / 12,746 | 13025 / 842 |
| curve e02 | complete | 1 | 30.3 | 29.9 | 81,706 | 10,354 / 12,726 | 12633 / 828 |
| joint e02 (stage 2) | complete | 1 | 29.3 | 28.8 | 84,907 | 10,354 / 12,726 | 12610 / 852 |
| curve e03 | complete | 1 | 30.3 | 29.5 | 82,829 | 10,354 / 12,768 | 13622 / 857 |
| joint e03 (stage 2) | complete | 1 | 29.8 | 29.6 | 82,671 | 10,354 / 12,768 | 13747 / 909 |
| curve e04 | complete | 1 | 28.3 | 27.7 | 88,083 | 10,336 / 12,966 | 14413 / 851 |
| joint e04 (stage 2) | complete | 1 | 30.3 | 29.8 | 81,905 | 10,336 / 12,966 | 14291 / 948 |
| curve e05 | complete | 1 | 43.4 | 29.5 | 82,847 | 10,357 / 12,460 | 14786 / 11421 |
| joint e05 (stage 2) | complete | 1 | 29.8 | 29.4 | 83,141 | 10,357 / 12,460 | 13660 / 909 |

Same fresh init as stage 2's joint arm (initial_parameter_sha256): True.
Two dedicated models beat the joint model on both heads at epoch 5 (both rules go): False.

Rule: curve-only - joint at epoch 5, curve head: 95% CI upper < 0 and points <= -0.05. Verdict: **stop** (epoch 5: curve +0.873 [+0.655, +1.104]: CI not below zero).
