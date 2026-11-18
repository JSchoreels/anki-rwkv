# Show-time interval retrain (SM91) step 3: cool-down with show-time teacher targets vs step 1's arm (old targets), both scored on show-time inputs

Leak-free scoring (deleted cards: neighbour), w8+s8, review-weighted LogLoss, user bootstrap 2,000 seed 0, 95% CI; points = % of the big model (1 leak-free epoch) with normal inputs. New: step 1's arm epoch 21 + the quarter-epoch 32k Muon cool-down on show-time inputs with KD targets from the show-time big model on show-time inputs. Reference: step 1's arm (the same epoch 21 + the same cool-down with the old targets: the big model (1 leak-free epoch) on normal inputs).

| pool | users | comparison | head | new | ref | delta [95% CI] | points [95% CI] | users better / worse |
|---|---|---|---|---|---|---|---|---|
| 732 (rule) | 722 | new e2s - step 1's arm e2s (the rule) | curve | 0.30088 | 0.30085 | +0.000034 [-0.000085, +0.000162] | +0.011 [-0.028, +0.055] | 341 / 380 |
| 732 (rule) | 722 | new e2s - step 1's arm e2s (the rule) | instant | 0.27754 | 0.27755 | -0.000011 [-0.000023, -0.000000] | -0.004 [-0.008, -0.000] | 388 / 331 |
| 732 (rule) | 722 | new e2s - the big model (1 leak-free epoch) e2s | curve | 0.30088 | 0.29958 | +0.001305 [+0.000774, +0.001950] | +0.436 [+0.256, +0.652] | 236 / 486 |
| 732 (rule) | 722 | new e2s - the big model (1 leak-free epoch) e2s | instant | 0.27754 | 0.27716 | +0.000381 [+0.000065, +0.000623] | +0.138 [+0.023, +0.232] | 218 / 504 |
| 732 (rule) | 722 | step 1's arm e2s - the big model (1 leak-free epoch) e2s | curve | 0.30085 | 0.29958 | +0.001271 [+0.000743, +0.001918] | +0.425 [+0.249, +0.648] | 242 / 480 |
| 732 (rule) | 722 | step 1's arm e2s - the big model (1 leak-free epoch) e2s | instant | 0.27755 | 0.27716 | +0.000391 [+0.000075, +0.000635] | +0.142 [+0.027, +0.235] | 212 / 510 |
| 732 (rule) | 722 | new e2s - the show-time big model e2s | curve | 0.30088 | 0.29941 | +0.001469 [+0.000938, +0.002045] | +0.491 [+0.313, +0.692] | 181 / 541 |
| 732 (rule) | 722 | new e2s - the show-time big model e2s | instant | 0.27754 | 0.27618 | +0.001354 [+0.001239, +0.001457] | +0.491 [+0.450, +0.534] | 70 / 652 |
| 732 (rule) | 722 | step 1's arm e2s - the show-time big model e2s | curve | 0.30085 | 0.29941 | +0.001435 [+0.000923, +0.002023] | +0.480 [+0.307, +0.683] | 182 / 540 |
| 732 (rule) | 722 | step 1's arm e2s - the show-time big model e2s | instant | 0.27755 | 0.27618 | +0.001365 [+0.001251, +0.001470] | +0.495 [+0.454, +0.537] | 61 / 661 |
| 732 (rule) | 722 | new e2s - the released small model e2s | curve | 0.30088 | 0.30041 | +0.000470 [+0.000217, +0.000699] | +0.157 [+0.072, +0.237] | 230 / 489 |
| 732 (rule) | 722 | new e2s - the released small model e2s | instant | 0.27754 | 0.27772 | -0.000180 [-0.000471, +0.000035] | -0.065 [-0.164, +0.012] | 285 / 437 |
| 732 (rule) | 722 | step 1's arm e2s - the released small model e2s | curve | 0.30085 | 0.30041 | +0.000436 [+0.000195, +0.000684] | +0.146 [+0.066, +0.230] | 236 / 486 |
| 732 (rule) | 722 | step 1's arm e2s - the released small model e2s | instant | 0.27755 | 0.27772 | -0.000170 [-0.000455, +0.000043] | -0.062 [-0.161, +0.016] | 284 / 438 |
| 732 (rule) | 722 | new e2s - the big model (1 leak-free epoch) normal (cost left) | curve | 0.30088 | 0.29897 | +0.001918 [+0.001406, +0.002528] | +0.642 [+0.468, +0.856] | 169 / 552 |
| 732 (rule) | 722 | new e2s - the big model (1 leak-free epoch) normal (cost left) | instant | 0.27754 | 0.27576 | +0.001781 [+0.001634, +0.001920] | +0.646 [+0.594, +0.697] | 45 / 676 |
| 732 (rule) | 722 | step 1's arm e2s - the big model (1 leak-free epoch) normal | curve | 0.30085 | 0.29897 | +0.001884 [+0.001379, +0.002498] | +0.630 [+0.458, +0.841] | 171 / 551 |
| 732 (rule) | 722 | step 1's arm e2s - the big model (1 leak-free epoch) normal | instant | 0.27755 | 0.27576 | +0.001791 [+0.001642, +0.001933] | +0.650 [+0.599, +0.701] | 47 / 675 |
| 732 (rule) | 722 | new normal - step 1's arm normal | curve | 0.30077 | 0.30072 | +0.000049 [-0.000070, +0.000178] | +0.016 [-0.023, +0.060] | 320 / 400 |
| 732 (rule) | 722 | new normal - step 1's arm normal | instant | 0.27743 | 0.27741 | +0.000020 [+0.000008, +0.000032] | +0.007 [+0.003, +0.011] | 334 / 388 |
| 732 (rule) | 722 | new e2s - new normal (its own show-time cost) | curve | 0.30088 | 0.30077 | +0.000115 [+0.000092, +0.000142] | +0.038 [+0.031, +0.047] | 217 / 488 |
| 732 (rule) | 722 | new e2s - new normal (its own show-time cost) | instant | 0.27754 | 0.27743 | +0.000107 [+0.000079, +0.000144] | +0.039 [+0.029, +0.052] | 192 / 504 |
| 732+long | 725 | new e2s - step 1's arm e2s (the rule) | curve | 0.30298 | 0.30312 | -0.000144 [-0.000550, +0.000126] | -0.048 [-0.183, +0.042] | 344 / 380 |
| 732+long | 725 | new e2s - step 1's arm e2s (the rule) | instant | 0.27844 | 0.27844 | -0.000008 [-0.000021, +0.000007] | -0.003 [-0.008, +0.002] | 389 / 333 |

Smaller-model bar (<= +0.5% curve / <= +0.4% instant, % of the reference's own LogLoss; not the rule):

| model | reference | curve % [95% CI] | instant % [95% CI] | passes |
|---|---|---|---|---|
| new | (a) normal inputs vs the big model (1 leak-free epoch) normal | +0.603 [+0.429, +0.820] | +0.607 [+0.562, +0.654] | no |
| new | (b) e2s-query vs the big model (1 leak-free epoch) e2s-query | +0.436 [+0.255, +0.651] | +0.137 [+0.023, +0.231] | yes |
| new | (c) e2s-query vs the show-time big model e2s-query | +0.491 [+0.313, +0.691] | +0.490 [+0.449, +0.533] | no |
| step 1's arm | (a) normal inputs vs the big model (1 leak-free epoch) normal | +0.587 [+0.415, +0.798] | +0.600 [+0.556, +0.646] | no |
| step 1's arm | (b) e2s-query vs the big model (1 leak-free epoch) e2s-query | +0.424 [+0.248, +0.647] | +0.141 [+0.026, +0.234] | yes |
| step 1's arm | (c) e2s-query vs the show-time big model e2s-query | +0.479 [+0.307, +0.682] | +0.494 [+0.453, +0.536] | no |

Rule: adopt the show-time targets if the new model beats step 1's arm on e2s-query by >= 0.05 points on curve or instant with the 95% CI below zero, and no head is worse by >= 0.05 points with the CI above zero; else no gain.

Verdict: **no gain** (no head gains >= 0.05 points with the CI below zero (curve +0.011 [-0.028, +0.055], instant -0.004 [-0.008, -0.000])).
