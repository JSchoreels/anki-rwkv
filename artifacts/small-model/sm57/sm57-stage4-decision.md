# The instant-only small model (SM57) stage 4: more instant-only epochs before the cool-down

Candidate: stage 1's instant-only epoch 26 + instant-only epochs 27-30 (the release's epoch recipe) + the release's one-epoch 32k Muon cool-down, all with every curve loss term at weight 0. Reference: stage 1's instant-only arm (epoch 26 + the same cool-down; the ship candidate). Leak-free scoring, normal inputs, w8+s8, 722 users of the 732 set; points = % of the big model (1 leak-free epoch) LogLoss, paired user bootstrap 2,000, seed 0, 95% CI.

Verdict: **go** (instant -0.071 [-0.079, -0.064]: CI below zero and at least the 0.05-points minimum).

| comparison | head | model | ref | points [95% CI] | users better / worse |
|---|---|---|---|---|---|
| candidate - stage 1's instant-only arm (the rule) | instant | 0.27590 | 0.27609 | -0.071 [-0.079, -0.064] | 553 / 167 |
| candidate - the release (SM63) | instant | 0.27590 | 0.27654 | -0.233 [-0.252, -0.214] | 663 / 58 |
| stage 1's instant-only arm - the release (reference) | instant | 0.27609 | 0.27654 | -0.162 [-0.176, -0.148] | 646 / 75 |
| candidate - the big model (1 leak-free epoch) | instant | 0.27590 | 0.27576 | +0.051 [+0.029, +0.072] | 286 / 436 |
| stage 1's instant-only arm - the big model (1 leak-free epoch) | instant | 0.27609 | 0.27576 | +0.122 [+0.098, +0.145] | 237 / 484 |
| constant-LR epoch 28 - epoch 26 (instant-only, no cool-down) | instant | 0.27817 | 0.27844 | -0.099 [-0.167, -0.040] | 424 / 298 |
| constant-LR epoch 30 - epoch 26 (instant-only, no cool-down) | instant | 0.27884 | 0.27844 | +0.143 [+0.091, +0.196] | 242 / 480 |
| constant-LR epoch 30 - epoch 28 | instant | 0.27884 | 0.27817 | +0.243 [+0.185, +0.303] | 213 / 509 |
| cool-down effect from epoch 30 (candidate - its epoch 30) | instant | 0.27590 | 0.27884 | -1.067 [-1.143, -0.994] | 711 / 11 |
| cool-down effect from epoch 26 (stage 1 - its epoch 26) | instant | 0.27609 | 0.27844 | -0.852 [-0.929, -0.781] | 699 / 23 |
| constant-LR epoch 30 - stage 1's cooled-down arm | instant | 0.27884 | 0.27609 | +0.996 [+0.925, +1.071] | 14 / 708 |

732+long (725 users), instant: candidate - stage 1 -0.068 [-0.078, -0.056], candidate - the release -0.243 [-0.269, -0.220].

Smaller-model bar (instant <= +0.4% of the big model): +0.051% -> passes (the curve head is untrained, never served).

Training (wall minutes over all segments; reviews/s and memory of the last segment's result.json):

| stage | status | segments | wall min | reviews/s | peak allocated / reserved MiB | nvidia-smi max / baseline MiB |
|---|---|---|---|---|---|---|
| epoch 27 | complete | 1 | 118.8 | 81,975 | 10,884 / 13,004 | 14268 / 1533 |
| epoch 28 | complete | 1 | 119.6 | 81,305 | 10,893 / 13,008 | 14928 / 1470 |
| epoch 29 | complete | 1 | 122.8 | 79,409 | 10,892 / 12,974 | 15178 / 1802 |
| epoch 30 | complete | 2 | 115.7 | 87,563 | 10,885 / 12,878 | 15000 / 1656 |
| cool-down | complete | 1 | 478.1 | 20,313 | 7,428 / 12,840 | 14602 / 2145 |

Chain (each epoch's initial parameters == the previous epoch's trained parameters): {27: True, 28: True, 29: True, 30: False}.

Rule: candidate - stage 1's instant-only arm, instant head, 95% CI upper < 0 and points <= -0.05 (points = % of the big model (1 leak-free epoch) normal-input LogLoss, review-weighted, 722 users of the 732 set, paired user bootstrap 2,000 seed 0).
