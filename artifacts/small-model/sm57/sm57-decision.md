# The instant-only small model (SM57): instant-only vs the matched joint arm

Leak-free scoring (deleted cards: neighbour), normal inputs, w8+s8, review-weighted LogLoss, user bootstrap 2,000 seed 0, 95% CI; points = % of the big model (1 leak-free epoch). Both arms: the leak-free small model (SM72s) epoch 25 + epoch 26 + the release's one-epoch 32k Muon cool-down (same users, order, seeds, KD targets); the instant-only arm trains both with every curve loss term at weight 0. Release: the released small model (cool-down from epoch 25, SM63). The instant-only arm's curve head is untrained (curve rows for completeness only).

| pool | users | comparison | head | model | ref | delta [95% CI] | points [95% CI] | users better / worse |
|---|---|---|---|---|---|---|---|---|
| 732 (rule) | 722 | instant-only - joint (the rule: instant) (untrained curve head) | curve | 0.34911 | 0.30013 | +0.048980 [+0.045394, +0.052806] | +16.383 [+14.946, +18.044] | 10 / 712 |
| 732 (rule) | 722 | instant-only - joint (the rule: instant) | instant | 0.27609 | 0.27650 | -0.000405 [-0.000446, -0.000365] | -0.147 [-0.162, -0.133] | 643 / 78 |
| 732 (rule) | 722 | instant-only - the release (untrained curve head) | curve | 0.34911 | 0.30003 | +0.049082 [+0.045460, +0.052991] | +16.417 [+14.959, +18.091] | 10 / 712 |
| 732 (rule) | 722 | instant-only - the release | instant | 0.27609 | 0.27654 | -0.000446 [-0.000482, -0.000408] | -0.162 [-0.176, -0.148] | 646 / 75 |
| 732 (rule) | 722 | joint - the release (the one-more-epoch confound) | curve | 0.30013 | 0.30003 | +0.000103 [-0.000052, +0.000355] | +0.034 [-0.018, +0.119] | 377 / 344 |
| 732 (rule) | 722 | joint - the release (the one-more-epoch confound) | instant | 0.27650 | 0.27654 | -0.000040 [-0.000055, -0.000026] | -0.015 [-0.020, -0.009] | 442 / 278 |
| 732 (rule) | 722 | instant-only - the big model (1 leak-free epoch) (untrained curve head) | curve | 0.34911 | 0.29897 | +0.050146 [+0.046540, +0.054088] | +16.773 [+15.319, +18.478] | 13 / 709 |
| 732 (rule) | 722 | instant-only - the big model (1 leak-free epoch) | instant | 0.27609 | 0.27576 | +0.000336 [+0.000270, +0.000401] | +0.122 [+0.098, +0.145] | 237 / 484 |
| 732 (rule) | 722 | joint - the big model (1 leak-free epoch) | curve | 0.30013 | 0.29897 | +0.001167 [+0.000650, +0.001765] | +0.390 [+0.217, +0.590] | 232 / 489 |
| 732 (rule) | 722 | joint - the big model (1 leak-free epoch) | instant | 0.27650 | 0.27576 | +0.000741 [+0.000656, +0.000823] | +0.269 [+0.239, +0.298] | 139 / 583 |
| 732 (rule) | 722 | the release - the big model (1 leak-free epoch) | curve | 0.30003 | 0.29897 | +0.001064 [+0.000594, +0.001609] | +0.356 [+0.199, +0.533] | 246 / 476 |
| 732 (rule) | 722 | the release - the big model (1 leak-free epoch) | instant | 0.27654 | 0.27576 | +0.000781 [+0.000698, +0.000862] | +0.283 [+0.254, +0.312] | 125 / 597 |
| 732+long | 725 | instant-only - joint (the rule: instant) (untrained curve head) | curve | 0.35217 | 0.30206 | +0.050116 [+0.046195, +0.054335] | +16.637 [+15.104, +18.365] | 10 / 715 |
| 732+long | 725 | instant-only - joint (the rule: instant) | instant | 0.27694 | 0.27738 | -0.000446 [-0.000520, -0.000384] | -0.161 [-0.190, -0.139] | 646 / 78 |
| 732+long | 725 | instant-only - the release (untrained curve head) | curve | 0.35217 | 0.30199 | +0.050186 [+0.046188, +0.054394] | +16.661 [+15.130, +18.394] | 10 / 715 |
| 732+long | 725 | instant-only - the release | instant | 0.27694 | 0.27742 | -0.000485 [-0.000559, -0.000425] | -0.175 [-0.205, -0.153] | 649 / 75 |
| 732+long | 725 | joint - the release (the one-more-epoch confound) | curve | 0.30206 | 0.30199 | +0.000071 [-0.000095, +0.000316] | +0.023 [-0.032, +0.105] | 380 / 344 |
| 732+long | 725 | joint - the release (the one-more-epoch confound) | instant | 0.27738 | 0.27742 | -0.000039 [-0.000054, -0.000025] | -0.014 [-0.020, -0.009] | 443 / 280 |

Smaller-model bar (<= +0.5% curve / <= +0.4% instant vs the big model (1 leak-free epoch), % of its LogLoss; not the rule):

| model | curve % [95% CI] | instant % [95% CI] | passes |
|---|---|---|---|
| joint | +0.390 [+0.217, +0.590] | +0.269 [+0.239, +0.298] | yes |
| instant-only | +16.773 [+15.319, +18.478] | +0.122 [+0.098, +0.145] | yes (instant only) |
| release | +0.356 [+0.199, +0.533] | +0.283 [+0.254, +0.312] | yes |

Training (wall minutes over all segments; reviews/s, training minutes and peak memory of the last segment; nvidia-smi max includes the desktop's baseline):

| arm / stage | status | segments | wall min | last segment: reviews | training min | reviews/s | peak allocated MiB | peak reserved MiB | nvidia-smi max / baseline MiB |
|---|---|---|---|---|---|---|---|---|---|
| instant-only / epoch 26 | complete | 1 | 111.2 | 581,942,724 | 110.6 | 87,706 | 10,892 | 13,014 | 15710 / 2627 |
| instant-only / cool-down | complete | 1 | 477.1 | 581,942,724 | 476.9 | 20,337 | 7,428 | 12,840 | 15429 / 2305 |
| joint / epoch 26 | complete | 1 | 111.2 | 581,942,724 | 110.8 | 87,562 | 10,892 | 13,014 | 15700 / 2305 |
| joint / cool-down | complete | 2 | 376.0 | 130,783,458 | 101.7 | 21,425 | 7,420 | 12,780 | 15106 / 464 |

Rule: go if instant-only - joint on the instant head has its 95% CI entirely below zero and a gain of at least 0.05 points (minimum added by the user 11-07, before the joint arm was scored); else stop. Go proposes stage 2 (from-scratch comparison at small scale, DESIGN.md), not run.

Verdict: **go** (instant -0.147 [-0.162, -0.133]: CI below zero and at least the 0.05-point minimum).
