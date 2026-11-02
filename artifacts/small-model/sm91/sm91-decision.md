# Show-time interval retrain (SM91) step 1: show-time-trained arm vs normal-trained reference, both scored on show-time inputs

Leak-free scoring (deleted cards: neighbour), w8+s8, `--inputs e2s-query`, review-weighted LogLoss, user bootstrap 2,000 seed 0, 95% CI; points = % of the big model (1 leak-free epoch) with normal inputs. Arm: the leak-free small model (SM72s) epoch 20 + 1 constant-LR epoch + quarter-epoch 32k Muon cool-down, trained on show-time intervals. Reference: SM72s epoch 21 + the same cool-down, trained on normal intervals (the tiny architecture additions (SM77) reference). Same start, users, order, seeds, KD targets.

| pool | users | comparison | head | new | ref | delta [95% CI] | points [95% CI] | users better / worse |
|---|---|---|---|---|---|---|---|---|
| 732 (rule) | 722 | arm e2s - reference e2s (the rule) | curve | 0.30085 | 0.30111 | -0.000256 [-0.000376, -0.000160] | -0.086 [-0.126, -0.054] | 452 / 267 |
| 732 (rule) | 722 | arm e2s - reference e2s (the rule) | instant | 0.27755 | 0.27831 | -0.000762 [-0.001024, -0.000564] | -0.277 [-0.367, -0.209] | 552 / 166 |
| 732 (rule) | 722 | reference e2s - reference normal (its show-time cost) | curve | 0.30111 | 0.30073 | +0.000377 [+0.000308, +0.000463] | +0.126 [+0.105, +0.153] | 150 / 559 |
| 732 (rule) | 722 | reference e2s - reference normal (its show-time cost) | instant | 0.27831 | 0.27722 | +0.001094 [+0.000840, +0.001422] | +0.397 [+0.311, +0.507] | 95 / 609 |
| 732 (rule) | 722 | arm e2s - reference normal (cost left) | curve | 0.30085 | 0.30073 | +0.000121 [+0.000001, +0.000218] | +0.040 [+0.000, +0.073] | 231 / 490 |
| 732 (rule) | 722 | arm e2s - reference normal (cost left) | instant | 0.27755 | 0.27722 | +0.000332 [+0.000266, +0.000414] | +0.120 [+0.098, +0.149] | 176 / 543 |

Win-back (not the rule): curve: gain +0.086 points = 68.3% of the reference's own show-time cost (+0.126) and 67.2% of SM31's released-model cost (+0.128); instant: gain +0.277 points = 69.8% of the reference's own show-time cost (+0.397) and 64.7% of SM31's released-model cost (+0.428).

Rule: go to step 2 if the arm beats the reference by >= 0.10 points on curve or instant with the 95% CI below zero; else stop.

Verdict: **go to step 2** (gain >= 0.10 points with the CI below zero: instant -0.277 points [-0.367, -0.209]).
