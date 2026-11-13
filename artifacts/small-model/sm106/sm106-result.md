# The Tiny two-head model (SM106): result

Verdict: **go** (11-13, 01:32:45). The Tiny two-head model (SM106) passes rule A on both heads against the big model (1 leak-free epoch), on show-time inputs (722 users of the 732 set; % = % of the big model (1 leak-free epoch)'s show-time LogLoss, curve 0.29958 / instant 0.277157; paired user bootstrap 2,000, seed 0, 95% CI). The full decision is in `sm106-decision.md`.

## A, the pruning bar: the Tiny two-head model (SM106) minus the big model (1 leak-free epoch)

| head | new | big | % [95% CI] | bar (point estimate) | pass |
|---|---|---|---|---|---|
| curve | 0.300676 | 0.29958 | +0.366% [+0.183, +0.574] | <= +0.5% | pass |
| instant | 0.277369 | 0.277157 | +0.076% [-0.033, +0.163] | <= +0.4% | pass |

The earlier prediction in this file (written after the early stop) was about +0.52% on curve. The cool-down recovered more than that prediction assumed: the final curve gap is +0.366%.

## B, report only (show-time inputs, % of the big model (1 leak-free epoch))

| new minus | head | new | ref | % [95% CI] | better / worse |
|---|---|---|---|---|---|
| the show-time retrain (SM92) | curve | 0.300676 | 0.300356 | +0.107% [+0.074, +0.137] | 213 / 509 |
| the show-time retrain (SM92) | instant | 0.277369 | 0.276876 | +0.178% [+0.158, +0.199] | 122 / 600 |
| the Tiny Instant-Only model (SM101) | instant | 0.277369 | 0.27672 | +0.234% [+0.215, +0.254] | 90 / 631 |
| the release (SM63) | curve | 0.300676 | 0.300414 | +0.088% [+0.001, +0.210] | 311 / 411 |
| the release (SM63) | instant | 0.277369 | 0.277718 | -0.126% [-0.223, -0.052] | 353 / 368 |

## 1,500 validation users, show-time inputs (1487 users), % of the release's own LogLoss

| comparison | head | new | ref | % [95% CI] | better / worse |
|---|---|---|---|---|---|
| new minus the release (SM63) | curve | 0.303517 | 0.303329 | +0.062% [+0.015, +0.118] | 651 / 836 |
| new minus the release (SM63) | instant | 0.280366 | 0.2806 | -0.083% [-0.132, -0.041] | 730 / 755 |
| new minus the Tiny Instant-Only model (SM101) | instant | 0.280366 | 0.279693 | +0.241% [+0.223, +0.261] | 151 / 1333 |

Row check: 6 users, max absolute LogLoss difference 0.0.

## Training and GPU cost

All four trainings used the 8k tier (581,942,724 reviews).

| step | wall minutes | reviews/s | nvidia-smi max (MiB) |
|---|---|---|---|
| epoch 27 | 106.8 | 91,020 | 13,136 |
| epoch 28 | 107.3 (106.7 training) | 90,865 | 14,744 |
| epoch 29 | 105.8 (105.4 training) | 92,039 | 13,444 |
| cool-down (32k Muon) | 463.6 (463.2 training) | 20,941 | 14,650 |

- nvidia-smi max over the whole run: 14,744 MiB (epoch 28; baselines 1,424 to 2,151 MiB). Peak allocated 9,295 MiB in the epochs and 6,612 MiB in the cool-down.
- Total GPU minutes: about 841 (trainings 783.5 from `gpu/segments.jsonl`, scoring and smokes 57 from `gpu/tickets.jsonl`).

## 10-epoch ETA

| tier | reviews per epoch | constant-LR speed (91,020 reviews/s) | cool-down speed (20,941 reviews/s) |
|---|---|---|---|
| 8k | 581.94M | 106.6 min per epoch, about 17.8 h for 10 epochs | 463.2 min per epoch, about 77.2 h for 10 epochs |
| 2k | 147.6M | 27.0 min per epoch, about 4.5 h for 10 epochs | 117.5 min per epoch, about 19.6 h for 10 epochs |

The 2k tier was not measured here; the 8k speeds are assumed for it.

## Early check and history (the run was first stopped, then finished)

The early check after epoch 27 stopped the run on 11-12 at 09:59:48 (10:00 Thursday): on show-time inputs, 722 users, the Tiny two-head model (SM106) at epoch 27 minus epoch 26 of the show-time retrain (SM92) was +0.259% on curve against a +0.25% stop bar. The user then chose to finish anyway, and the run was relaunched at 13:17 on 11-12 with `OVERRIDE_EARLY_STOP=1` (commit 303be480). Rule A was unchanged. The run continued with epochs 28 and 29 and the cool-down, ending 11-13 at 01:32:45.

Early check numbers (stop bar: point estimate above +0.25% on either head):

| head | Tiny two-head model (SM106) epoch 27 | show-time retrain (SM92) epoch 26 | % [95% CI] | better / worse |
|---|---|---|---|---|
| curve | 0.303854 | 0.303079 | +0.259% [+0.131, +0.399] | 301 / 421 |
| instant | 0.279473 | 0.278939 | +0.193% [+0.151, +0.234] | 172 / 550 |

The early-stop write-up predicted about +0.52% on curve (the show-time retrain (SM92) sits at +0.259% vs the big model (1 leak-free epoch), plus the cut cost seen at epoch 27, about +0.26%), which would have failed the +0.5% bar. Epochs 28 and 29 and the cool-down brought the gap down to +0.366%, so the early check was too pessimistic.
