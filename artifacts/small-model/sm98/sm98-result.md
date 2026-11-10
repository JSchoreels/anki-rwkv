# The show-time instant-only model, stage 2 (SM98): result (2026-11-10)

Launcher finished 10:49 Tue 2026-11-10 with "done (verdict no go)". The rule `sm98_rule.py` ran in the launcher; see
`sm98-decision.json` / `sm98-decision.md`. Units: points = % of the big model (1 leak-free epoch) normal-input instant
LogLoss 0.275757 (732 set, 722 users, review-weighted, paired user bootstrap 2,000, seed 0, 95% CI), unless stated.
Chain: 4 more constant-LR instant-only epochs (32-35) on show-time inputs from the show-time instant-only small model
(SM96)'s epoch 31, then the release's one-epoch 32k Muon cool-down on show-time inputs from epoch 35 (curve loss weight 0).

## Verdict: no go

Primary: the show-time instant-only model, stage 2 (SM98) minus the show-time instant-only small model (SM96), instant
head, both on show-time inputs: **-0.035 points [-0.041, -0.029]**, 489 users better / 232 worse (new 0.276191 vs
0.276287). The CI is entirely below 0, but the gain is smaller than the 0.05-point minimum, so the rule says no go. The
curve head is untrained (weight 0) and is not part of the verdict.

## Secondary (732 set, instant head, points)

| new minus | inputs | new | ref | points [95% CI] | better / worse |
|---|---|---|---|---|---|
| the show-time retrain of the release (SM92) | show-time | 0.276191 | 0.276876 | -0.248 [-0.270, -0.228] | 658 / 64 |
| the instant-only small model (SM57) stage 4 | show-time | 0.276191 | 0.277236 | -0.379 [-0.465, -0.309] | 600 / 122 |
| the release (SM63) | show-time | 0.276191 | 0.277718 | -0.554 [-0.657, -0.474] | 672 / 50 |
| the show-time instant-only small model (SM96) | normal | 0.27606 | 0.276134 | -0.027 [-0.033, -0.021] | 457 / 265 |
| the show-time retrain of the release (SM92) | normal | 0.27606 | 0.276757 | -0.253 [-0.274, -0.232] | 659 / 62 |
| the instant-only small model (SM57) stage 4 | normal | 0.27606 | 0.275897 | +0.059 [+0.040, +0.080] | 290 / 431 |
| the new model's show-time cost (show-time minus normal inputs) | both | 0.276191 | 0.27606 | +0.048 [+0.035, +0.061] | 211 / 496 |

## 1,500 validation users outside the 732 set, show-time inputs (1,487 users with a release reference), % of the release's own LogLoss

| comparison | new | ref | % [95% CI] | better / worse |
|---|---|---|---|---|
| new minus the release (SM63) | 0.279162 | 0.2806 | -0.512 [-0.564, -0.469] | 1399 / 88 |
| new minus the show-time instant-only small model (SM96) | 0.279162 | 0.279249 | -0.031 [-0.037, -0.024] | 1010 / 475 |
| new minus the instant-only small model (SM57) stage 4 | 0.279162 | 0.280122 | -0.343 [-0.386, -0.304] | 1267 / 219 |
| new minus the release, ids < 5000 (the 732 set) (726 users) | 0.278971 | 0.280514 | -0.550 [-0.652, -0.476] | 676 / 50 |
| new minus the release, ids >= 5000 (761 users) | 0.279321 | 0.280671 | -0.481 [-0.535, -0.433] | 723 / 38 |

Row check on the 6 check users: max abs LogLoss difference 0.0. Only 767 of the 771 pool users were scored on show-time
inputs: the last 4 did not fit even at the 0.92 GPU memory cap (no progress at cap 0.92, 767 of 771). 1,487 users have a
release reference.

## Speed

| step | reviews | wall min (segments) | training min (log) | reviews/s (log) | reviews/s (wall) | nvidia-smi max |
|---|---|---|---|---|---|---|
| epoch 32 | 581,942,724 | 111.5 | 110.9 | 87,421 | 86,987 | 14,072 MiB |
| epoch 33 | 581,942,724 | 112.0 | 111.3 | 87,134 | 86,599 | 14,069 MiB |
| epoch 34 | 581,942,724 | 112.0 | 111.3 | 87,161 | 86,599 | 14,202 MiB |
| epoch 35, segment 1 (stopped 19:57 for the break, 58.3% of reviews done) | 339,476,344 | 129.1 | n/a | n/a | 43,816 | 14,524 MiB |
| epoch 35, segment 2 (resumed 01:00, remaining users) | 242,466,380 | 51.4 | 50.8 | 79,527 | 78,622 | 15,727 MiB |
| epoch 35 total | 581,942,724 | 180.5 | n/a | n/a | 53,734 | 15,727 MiB |
| 1-epoch 32k Muon cool-down | 581,942,724 | 479.6 | 479.4 | 20,232 | 20,223 | 13,265 MiB |

Epoch 35's first segment ran slowly (another process held about 7 GB of the GPU, baseline 7,022 MiB), so only its second
segment has a `complete:` line. Overall nvidia-smi max: 15,727 MiB (epoch 35 segment 2; the cool-down peaked at 13,265).

10-epoch ETA at the log speeds (8k tier = 581.94M reviews per epoch, 2k tier = 147.6M); constant-LR epochs use epoch 32's
87,421 reviews/s (epochs 33-34 are within 0.3%):

| phase | 2k tier | 8k tier |
|---|---|---|
| 10 constant-LR epochs at 87,421 reviews/s | 4.7 h | 18.5 h |
| 10 cool-downs at 20,232 reviews/s | 20.3 h | 79.9 h |

Total GPU time from the launch (12:12 Mon 2026-11-09) to the end (10:49 Tue 2026-11-10): 22.6 h wall minus the 5 h
20:00-01:00 break = about 17.6 GPU hours (1,057 min).
