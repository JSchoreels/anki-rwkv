# Show-time retrain of the release (SM92): the new model vs the released small model, both scored on show-time inputs

Leak-free scoring (deleted cards: neighbour), w8+s8, review-weighted LogLoss, user bootstrap 2,000 seed 0, 95% CI; points = % of the big model (1 leak-free epoch) with normal inputs. New: the leak-free small model (SM72s) epoch 25 + epoch 26 on show-time inputs + the release's one-epoch 32k Muon cool-down on show-time inputs (same users, order, seeds, KD targets). Release: the released small model (cool-down from epoch 25). Confound: the new model has one more epoch than the release (the normal-vs-normal row shows its size).

| pool | users | comparison | head | model | ref | delta [95% CI] | points [95% CI] | users better / worse |
|---|---|---|---|---|---|---|---|---|
| 732 (rule) | 722 | new e2s - the release e2s (the rule) | curve | 0.30036 | 0.30041 | -0.000058 [-0.000271, +0.000255] | -0.019 [-0.092, +0.085] | 447 / 273 |
| 732 (rule) | 722 | new e2s - the release e2s (the rule) | instant | 0.27688 | 0.27772 | -0.000842 [-0.001136, -0.000630] | -0.305 [-0.405, -0.232] | 571 / 151 |
| 732 (rule) | 722 | new normal - the release normal (the extra-epoch confound) | curve | 0.30024 | 0.30003 | +0.000210 [+0.000006, +0.000527] | +0.070 [+0.002, +0.177] | 290 / 431 |
| 732 (rule) | 722 | new normal - the release normal (the extra-epoch confound) | instant | 0.27676 | 0.27654 | +0.000218 [+0.000170, +0.000272] | +0.079 [+0.063, +0.098] | 230 / 491 |
| 732 (rule) | 722 | new e2s - new normal (its own show-time cost) | curve | 0.30036 | 0.30024 | +0.000116 [+0.000093, +0.000145] | +0.039 [+0.031, +0.048] | 224 / 482 |
| 732 (rule) | 722 | new e2s - new normal (its own show-time cost) | instant | 0.27688 | 0.27676 | +0.000119 [+0.000088, +0.000155] | +0.043 [+0.033, +0.056] | 204 / 497 |
| 732 (rule) | 722 | the release e2s - the release normal (its show-time cost) | curve | 0.30041 | 0.30003 | +0.000384 [+0.000310, +0.000484] | +0.128 [+0.106, +0.160] | 149 / 559 |
| 732 (rule) | 722 | the release e2s - the release normal (its show-time cost) | instant | 0.27772 | 0.27654 | +0.001180 [+0.000912, +0.001531] | +0.428 [+0.337, +0.546] | 102 / 609 |
| 732 (rule) | 722 | new e2s - the big model (1 leak-free epoch) e2s | curve | 0.30036 | 0.29958 | +0.000776 [+0.000214, +0.001377] | +0.260 [+0.072, +0.464] | 308 / 414 |
| 732 (rule) | 722 | new e2s - the big model (1 leak-free epoch) e2s | instant | 0.27688 | 0.27716 | -0.000281 [-0.000600, -0.000041] | -0.102 [-0.216, -0.016] | 341 / 381 |
| 732 (rule) | 722 | the release e2s - the big model (1 leak-free epoch) e2s | curve | 0.30041 | 0.29958 | +0.000834 [+0.000354, +0.001378] | +0.279 [+0.118, +0.462] | 297 / 425 |
| 732 (rule) | 722 | the release e2s - the big model (1 leak-free epoch) e2s | instant | 0.27772 | 0.27716 | +0.000561 [+0.000473, +0.000646] | +0.203 [+0.172, +0.236] | 208 / 514 |
| 732 (rule) | 722 | new normal - the big model (1 leak-free epoch) normal | curve | 0.30024 | 0.29897 | +0.001274 [+0.000741, +0.001887] | +0.426 [+0.247, +0.631] | 211 / 511 |
| 732 (rule) | 722 | new normal - the big model (1 leak-free epoch) normal | instant | 0.27676 | 0.27576 | +0.000999 [+0.000892, +0.001106] | +0.362 [+0.327, +0.397] | 102 / 620 |
| 732 (rule) | 722 | the release normal - the big model (1 leak-free epoch) normal | curve | 0.30003 | 0.29897 | +0.001064 [+0.000594, +0.001609] | +0.356 [+0.199, +0.533] | 246 / 476 |
| 732 (rule) | 722 | the release normal - the big model (1 leak-free epoch) normal | instant | 0.27654 | 0.27576 | +0.000781 [+0.000698, +0.000862] | +0.283 [+0.254, +0.312] | 125 / 597 |
| 732 (rule) | 722 | new e2s - the show-time big model e2s | curve | 0.30036 | 0.29941 | +0.000941 [+0.000448, +0.001459] | +0.315 [+0.149, +0.491] | 250 / 472 |
| 732 (rule) | 722 | new e2s - the show-time big model e2s | instant | 0.27688 | 0.27618 | +0.000693 [+0.000609, +0.000772] | +0.251 [+0.222, +0.280] | 141 / 579 |
| 732 (rule) | 722 | the release e2s - the show-time big model e2s | curve | 0.30041 | 0.29941 | +0.000999 [+0.000554, +0.001464] | +0.334 [+0.184, +0.494] | 233 / 489 |
| 732 (rule) | 722 | the release e2s - the show-time big model e2s | instant | 0.27772 | 0.27618 | +0.001535 [+0.001297, +0.001828] | +0.557 [+0.478, +0.656] | 96 / 626 |
| 732 (rule) | 722 | new e2s - the big model (1 leak-free epoch) normal (cost left) | curve | 0.30036 | 0.29897 | +0.001390 [+0.000854, +0.002014] | +0.465 [+0.283, +0.671] | 200 / 522 |
| 732 (rule) | 722 | new e2s - the big model (1 leak-free epoch) normal (cost left) | instant | 0.27688 | 0.27576 | +0.001119 [+0.000994, +0.001243] | +0.406 [+0.364, +0.447] | 89 / 633 |
| 732 (rule) | 722 | the release e2s - the big model (1 leak-free epoch) normal | curve | 0.30041 | 0.29897 | +0.001448 [+0.000976, +0.001981] | +0.484 [+0.326, +0.665] | 205 / 517 |
| 732 (rule) | 722 | the release e2s - the big model (1 leak-free epoch) normal | instant | 0.27772 | 0.27576 | +0.001961 [+0.001659, +0.002315] | +0.711 [+0.615, +0.833] | 70 / 652 |
| 732+long | 725 | new e2s - the release e2s (the rule) | curve | 0.30232 | 0.30236 | -0.000042 [-0.000252, +0.000267] | -0.014 [-0.084, +0.087] | 449 / 274 |
| 732+long | 725 | new e2s - the release e2s (the rule) | instant | 0.27776 | 0.27858 | -0.000826 [-0.001105, -0.000622] | -0.299 [-0.397, -0.228] | 573 / 152 |
| 732+long | 725 | new normal - the release normal (the extra-epoch confound) | curve | 0.30221 | 0.30199 | +0.000220 [+0.000017, +0.000526] | +0.073 [+0.006, +0.175] | 292 / 432 |
| 732+long | 725 | new normal - the release normal (the extra-epoch confound) | instant | 0.27764 | 0.27742 | +0.000218 [+0.000171, +0.000270] | +0.079 [+0.063, +0.096] | 230 / 493 |

Smaller-model bar (<= +0.5% curve / <= +0.4% instant, % of the reference's own LogLoss; not the rule):

| model | reference | curve % [95% CI] | instant % [95% CI] | passes |
|---|---|---|---|---|
| new | (a) normal inputs vs the big model (1 leak-free epoch) normal | +0.426 [+0.247, +0.631] | +0.362 [+0.327, +0.397] | yes |
| new | (b) e2s-query vs the big model (1 leak-free epoch) e2s-query | +0.259 [+0.072, +0.463] | -0.101 [-0.214, -0.015] | yes |
| new | (c) e2s-query vs the show-time big model e2s-query | +0.314 [+0.148, +0.491] | +0.251 [+0.222, +0.280] | yes |
| the release | (a) normal inputs vs the big model (1 leak-free epoch) normal | +0.356 [+0.199, +0.533] | +0.283 [+0.254, +0.312] | yes |
| the release | (b) e2s-query vs the big model (1 leak-free epoch) e2s-query | +0.278 [+0.117, +0.461] | +0.202 [+0.171, +0.235] | yes |
| the release | (c) e2s-query vs the show-time big model e2s-query | +0.334 [+0.184, +0.494] | +0.556 [+0.477, +0.655] | no |

Rule: ship candidate if the new model beats the release on e2s-query by >= 0.05 points on curve or instant with the 95% CI below zero, and no head is worse by >= 0.05 points with the CI above zero; else no ship.

Verdict: **ship candidate** (gain >= 0.05 points with the CI below zero: instant -0.305 [-0.405, -0.232]).
