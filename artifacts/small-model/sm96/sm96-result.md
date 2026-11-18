# The show-time instant-only small model (SM96): result (2026-11-09)

Launcher finished 11:54 Mon 2026-11-09 with "done (verdict go)". The rule `sm96_rule.py` ran in the launcher; see
`sm96-decision.json` / `sm96-decision.md`. Units: points = % of the big model (1 leak-free epoch) normal-input instant
LogLoss 0.275757 (732 set, 722 users, review-weighted, paired user bootstrap 2,000, seed 0, 95% CI), unless stated.

## Verdict: go

Primary: the show-time instant-only small model (SM96) minus the show-time retrain of the release (SM92), instant head,
both on show-time inputs: **-0.214 points [-0.233, -0.195]**, 647 users better / 75 worse (new 0.276287 vs 0.276876).
The CI is entirely below 0 and the gain is at least 0.05 points, so the rule says go. The curve head is untrained (weight
0) and is not part of the verdict.

## Secondary (732 set, instant head, points)

| new minus | inputs | new | ref | points [95% CI] | better / worse |
|---|---|---|---|---|---|
| the instant-only small model (SM57) stage 4 | show-time | 0.276287 | 0.277236 | -0.344 [-0.429, -0.275] | 584 / 137 |
| the release (SM63) | show-time | 0.276287 | 0.277718 | -0.519 [-0.620, -0.442] | 664 / 58 |
| the instant-only small model (SM57) stage 4 | normal | 0.276134 | 0.275897 | +0.086 [+0.070, +0.104] | 213 / 506 |
| the show-time retrain of the release (SM92) | normal | 0.276134 | 0.276757 | -0.226 [-0.246, -0.207] | 646 / 75 |
| the new model's show-time cost (show-time minus normal inputs) | both | 0.276287 | 0.276134 | +0.055 [+0.042, +0.072] | 199 / 502 |

## 1,500 validation users outside the 732 set, show-time inputs (1,487 users scored), % of the release's own LogLoss

| comparison | new | ref | % [95% CI] | better / worse |
|---|---|---|---|---|
| new minus the release (SM63) | 0.279249 | 0.2806 | -0.481 [-0.534, -0.438] | 1383 / 104 |
| new minus the instant-only small model (SM57) stage 4 | 0.279249 | 0.280122 | -0.312 [-0.354, -0.274] | 1226 / 257 |
| new minus the release, ids < 5000 (the 732 set) (726 users) | 0.279058 | 0.280514 | -0.519 [-0.622, -0.444] | 668 / 58 |
| new minus the release, ids >= 5000 (761 users) | 0.279408 | 0.280671 | -0.450 [-0.503, -0.404] | 715 / 46 |

Row check on the 6 check users: max abs LogLoss difference 0.0. 767 of 771 pool users were scored (the rest did not fit
at cap 0.92); 1,487 users have a release reference.

## Speed

| step | reviews | wall min (segments) | training min (log) | reviews/s (log) | reviews/s (wall) | nvidia-smi max |
|---|---|---|---|---|---|---|
| epoch 31 (constant LR, show-time inputs) | 581,942,724 | 124.6 | 124.4 | 77,996 | 77,845 | 15,440 MiB |
| 1-epoch 32k Muon cool-down | 581,942,724 | 483.2 | 482.9 | 20,084 | 20,072 | 15,544 MiB |

10-epoch ETA at the log speeds (8k tier = 581.94M reviews per epoch, 2k tier = 147.6M):

| phase | 2k tier | 8k tier |
|---|---|---|
| 10 constant-LR epochs at 77,996 reviews/s | 5.3 h | 20.7 h |
| 10 cool-downs at 20,084 reviews/s | 20.4 h | 80.5 h |
