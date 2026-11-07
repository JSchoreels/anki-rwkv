# The instant model for RWKV-Instant (SM93): the instant-only small model (SM57) vs the release (SM63), 1,500 validation users

Verdict: **go** (instant -0.164 [-0.179, -0.152]: CI below zero and at least 0.05).

Leak-free scoring (deleted cards: neighbour), normal inputs, w8+s8, review-weighted LogLoss, paired user bootstrap 2,000, seed 0, 95% CI. Unit: % of the release's LogLoss on the same users (the big model (1 leak-free epoch) trained on the ids >= 5000; on the 722 users 1% of the release's LogLoss = 1.003 points of the big model's). Bar: instant CI below 0 and gain >= 0.05.

Row check (6 users of the 732 set re-scored in this tree vs the earlier rows): {"sm57-inst-dec1ep32k/w8+s8": {"users": 6, "max_abs_logloss_diff": 0.0}, "s-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon/w8+s8": {"users": 6, "max_abs_logloss_diff": 0.0}}

| users | comparison | head | candidate | release | % [95% CI] | users better / worse |
|---|---|---|---|---|---|---|
| 1487 (all) | candidate - release (untrained curve head, never served) | curve | 0.35133 | 0.30297 | +15.964 [+14.936, +17.057] | 19 / 1468 |
| 1487 (all) | candidate - release | instant | 0.27908 | 0.27954 | -0.164 [-0.179, -0.152] | 1349 / 137 |
| 726 (ids < 5000 (the 732 set)) | candidate - release (untrained curve head, never served) | curve | 0.35443 | 0.30385 | +16.648 [+15.107, +18.389] | 10 / 716 |
| 726 (ids < 5000 (the 732 set)) | candidate - release | instant | 0.27886 | 0.27935 | -0.175 [-0.204, -0.153] | 650 / 75 |
| 761 (ids >= 5000 (new here)) | candidate - release (untrained curve head, never served) | curve | 0.34875 | 0.30223 | +15.390 [+14.117, +16.808] | 9 / 752 |
| 761 (ids >= 5000 (new here)) | candidate - release | instant | 0.27926 | 0.27969 | -0.156 [-0.169, -0.143] | 699 / 62 |

Reference (722-user rule pool, points of the big model (1 leak-free epoch)): instant -0.162 [-0.176, -0.148], 646 / 75 users (the SM57 result).
