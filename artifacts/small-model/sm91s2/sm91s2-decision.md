# Show-time interval retrain (SM91) step 2: the show-time big model vs the big model (1 leak-free epoch), both scored on show-time inputs

Leak-free scoring (deleted cards: neighbour), w8+s8, `--inputs e2s-query`, review-weighted LogLoss, user bootstrap 2,000 seed 0, 95% CI; points = % of the big model (1 leak-free epoch) with normal inputs. New: the big model (1 leak-free epoch) + one added epoch with the leak-free retrain (SM72) recipe (users 5000-10000, 3.5e-4 -> 0), trained on show-time intervals. Reference: the big model (1 leak-free epoch), rows from the duration-leak scoring (SM31).

| pool | users | comparison | head | new | ref | delta [95% CI] | points [95% CI] | users better / worse |
|---|---|---|---|---|---|---|---|---|
| 732 (rule) | 722 | new e2s - reference e2s (the rule) | curve | 0.29941 | 0.29958 | -0.000164 [-0.000352, +0.000040] | -0.055 [-0.117, +0.013] | 457 / 264 |
| 732 (rule) | 722 | new e2s - reference e2s (the rule) | instant | 0.27618 | 0.27716 | -0.000974 [-0.001280, -0.000741] | -0.353 [-0.459, -0.276] | 584 / 137 |
| 732 (rule) | 722 | reference e2s - reference normal (SM31's show-time cost) | curve | 0.29958 | 0.29897 | +0.000614 [+0.000486, +0.000783] | +0.205 [+0.165, +0.258] | 126 / 586 |
| 732 (rule) | 722 | reference e2s - reference normal (SM31's show-time cost) | instant | 0.27716 | 0.27576 | +0.001400 [+0.001108, +0.001774] | +0.508 [+0.411, +0.633] | 93 / 619 |
| 732 (rule) | 722 | new e2s - reference normal (cost left) | curve | 0.29941 | 0.29897 | +0.000449 [+0.000283, +0.000653] | +0.150 [+0.095, +0.218] | 237 / 485 |
| 732 (rule) | 722 | new e2s - reference normal (cost left) | instant | 0.27618 | 0.27576 | +0.000426 [+0.000349, +0.000514] | +0.155 [+0.129, +0.185] | 168 / 551 |
| 732 (rule) | 722 | new normal - reference normal (the added epoch on normal inputs) | curve | 0.29932 | 0.29897 | +0.000355 [+0.000193, +0.000548] | +0.119 [+0.065, +0.183] | 257 / 465 |
| 732 (rule) | 722 | new normal - reference normal (the added epoch on normal inputs) | instant | 0.27603 | 0.27576 | +0.000270 [+0.000223, +0.000322] | +0.098 [+0.082, +0.116] | 207 / 512 |
| 732 (rule) | 722 | new e2s - new normal (its own show-time cost) | curve | 0.29941 | 0.29932 | +0.000095 [+0.000070, +0.000126] | +0.032 [+0.024, +0.041] | 204 / 497 |
| 732 (rule) | 722 | new e2s - new normal (its own show-time cost) | instant | 0.27618 | 0.27603 | +0.000156 [+0.000119, +0.000202] | +0.057 [+0.043, +0.073] | 174 / 534 |
| 732+long | 725 | new e2s - reference e2s (the rule) | curve | 0.30155 | 0.30184 | -0.000286 [-0.000600, -0.000019] | -0.095 [-0.199, -0.006] | 460 / 264 |
| 732+long | 725 | new e2s - reference e2s (the rule) | instant | 0.27701 | 0.27796 | -0.000950 [-0.001242, -0.000723] | -0.343 [-0.446, -0.266] | 586 / 138 |
| 732+long | 725 | new normal - reference normal (the added epoch on normal inputs) | curve | 0.30146 | 0.30123 | +0.000230 [-0.000078, +0.000496] | +0.076 [-0.026, +0.165] | 259 / 466 |
| 732+long | 725 | new normal - reference normal (the added epoch on normal inputs) | instant | 0.27686 | 0.27658 | +0.000280 [+0.000228, +0.000334] | +0.101 [+0.084, +0.120] | 208 / 514 |

Win-back (not the rule): curve: gain +0.055 points = 26.8% of SM31's big-model show-time cost (+0.205); instant: gain +0.353 points = 69.5% of SM31's big-model show-time cost (+0.508).

Rule: go to step 3 if the new model beats the big model (1 leak-free epoch) by >= 0.10 points on curve or instant with the 95% CI below zero; else stop.

Verdict: **go to step 3** (gain >= 0.10 points with the CI below zero: instant -0.353 points [-0.459, -0.276]).
