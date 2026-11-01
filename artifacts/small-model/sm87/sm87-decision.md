# Frozen-model grade side head (SM87): decision

A side head (MLP 494 -> 128 -> 4, zero-init output) adds a correction to the released instant (grade) logits of a frozen copy of the released small model (the leak-free small model (SM72s) 25 epochs + the 1-epoch 32k Muon cool-down); the trunk, the curve and the released instant head are untouched. Inputs at each instant row: the trunk's residual stream after the input encoder and after each of the 4 layer rounds (5 x 64, layer-normalised), the instant head's hidden layer (128), the released instant prediction (5), and 41 explicit causal grade features (the card's last 3 grades and answer times, review count, lapses, previous interval; elapsed days and seconds as the released model sees them; the user's pass rate over the last 50 / 500 / all reviews, grade mix over the last 500, running median time per grade). Ablation in the same pass: without the model states (released prediction + explicit features, MLP 46 -> 128 -> 4). Loss: the instant binary LogLoss. Training: the 1,991 users of the SM82 quarter-epoch cool-down (none in the 732 set), each history's first 400,000 reviews, 116,570,913 instant rows, one pass, 14,230 AdamW steps of 8,192 rows (lr 0.002, cosine), side heads only; 46.7 min (last run segment). Training-set instant LogLoss: side head 0.285671, ablation 0.285671, released 0.285705.

Scoring: the scorer's own path (w8+s8, leak-free inputs) on 722 users of the 732 set; one forward per user gives the released rows and, from the same features, the corrected and the ablation rows. Released rows equal to the recorded released scores (every grade field): True.

| model | curve LogLoss | vs the big model (1 leak-free epoch) | instant LogLoss | vs the big model (1 leak-free epoch) |
|---|---|---|---|---|
| released small model | 0.30003 | +0.356% | 0.27654 | +0.283% |
| + grade side head (SM87) | 0.30003 | +0.356% | 0.27652 | +0.278% |
| + ablation without the model states | 0.30003 | +0.356% | 0.27653 | +0.279% |
| big model (1 leak-free epoch) | 0.29897 | | 0.27576 | |

| comparison (instant) | points [95% CI] | LogLoss delta | users better |
|---|---|---|---|
| side head vs released | -0.005 [-0.007, -0.004] | -0.000015 | 465 / 722 |
| ablation vs released | -0.005 [-0.007, -0.003] | -0.000013 | 451 / 722 |
| side head vs ablation: what the model states add | -0.001 [-0.003, +0.001] | -0.000002 | 363 / 722 |

Points = LogLoss change as % of the big model's (1 leak-free epoch) LogLoss, review-weighted; negative = better.

Curve: unchanged (True): corrected and ablation rows' curve fields = the released rows' on all 722 users; the correction applied inside the model (apply-check, 5 users) keeps the curve outputs bitwise: True.

Size: side head 63,876 values (ablation 6,532); the explicit features need per card the last 3 grades and times, review and lapse counts and the previous interval, and per user windowed pass and grade counts plus a running median time per grade at serving time.

Verdict: **stop** (instant gain +0.005 points, CI [+0.004, +0.007]; go >= 0.15 with CI clear of zero, stop < 0.05).
