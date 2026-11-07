# The instant-only small model (SM57) stage 2: from scratch on the 2k tier, instant-only vs joint

Both arms from the same fresh init (training_seed 2025), the leak-free small model (SM72s) recipe on the 2k tier (2,000 users, 2,048-review windows x 16 lanes, leak-free replay KD, AdamW 7e-4 constant), same order and seeds; the instant-only arm trains with every curve loss term at weight 0. Constant-LR checkpoints, no cool-down. Leak-free scoring, normal inputs, w8+s8, 722 users of the 732 set; points = % of the big model (1 leak-free epoch) LogLoss, paired user bootstrap 2,000, seed 0, 95% CI.

| epoch | instant-only - joint, instant | users better / worse | instant-only - big, instant | joint - big, instant | joint - big, curve | joint - the release (SM63), instant / curve |
|---|---|---|---|---|---|---|
| 1 | -0.461 [-0.621, -0.329] | 479 / 243 | +6.029 [+5.677, +6.397] | +6.490 [+6.079, +6.914] | +6.737 [+6.141, +7.417] | +6.207 [+5.810, +6.621] / +6.381 [+5.848, +6.995] |
| 3 | -0.516 [-0.594, -0.445] | 557 / 165 | +3.266 [+3.078, +3.454] | +3.782 [+3.555, +4.019] | +4.309 [+3.888, +4.786] | +3.499 [+3.289, +3.719] / +3.953 [+3.616, +4.330] |
| 5 | -0.773 [-0.861, -0.685] | 661 / 60 | +2.541 [+2.382, +2.699] | +3.315 [+3.104, +3.521] | +4.529 [+4.133, +4.963] | +3.031 [+2.836, +3.226] / +4.173 [+3.825, +4.561] |

Training per arm and epoch:

| arm / epoch | status | segments | wall min | training min | reviews/s | peak allocated / reserved MiB | nvidia-smi max / baseline MiB |
|---|---|---|---|---|---|---|---|
| inst e01 | complete | 1 | 30.3 | 29.8 | 82,038 | 10,350 / 12,746 | 13015 / 643 |
| joint e01 | complete | 1 | 29.8 | 29.4 | 83,234 | 10,350 / 12,746 | 13025 / 842 |
| inst e02 | complete | 1 | 29.3 | 29.0 | 84,209 | 10,354 / 12,726 | 12615 / 843 |
| joint e02 | complete | 1 | 29.3 | 28.8 | 84,907 | 10,354 / 12,726 | 12610 / 852 |
| inst e03 | complete | 1 | 29.8 | 29.3 | 83,303 | 10,354 / 12,768 | 13682 / 844 |
| joint e03 | complete | 1 | 29.8 | 29.6 | 82,671 | 10,354 / 12,768 | 13747 / 909 |
| inst e04 | complete | 1 | 30.3 | 29.7 | 82,227 | 10,336 / 12,966 | 14318 / 932 |
| joint e04 | complete | 1 | 30.3 | 29.8 | 81,905 | 10,336 / 12,966 | 14291 / 948 |
| inst e05 | complete | 1 | 29.8 | 29.5 | 82,985 | 10,356 / 12,460 | 13694 / 907 |
| joint e05 | complete | 1 | 29.8 | 29.4 | 83,141 | 10,357 / 12,460 | 13660 / 909 |

Same fresh init in both arms (initial_parameter_sha256): True.
Epoch-1 reading: instant-only leads at epoch 1.

Rule: instant-only - joint at the final epoch (5), instant head: 95% CI upper < 0 and points <= -0.05. Verdict: **go** (epoch 5: instant -0.773 [-0.861, -0.685]: CI below zero and at least the 0.05-point minimum).
