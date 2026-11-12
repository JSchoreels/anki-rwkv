# The Tiny two-head model (SM106): result

Verdict: **early stop** (11-12, 09:59:48). The pre-registered early check after epoch 27 stopped the run; the prepared cool-down never ran.

## Early check (show-time inputs, 722 users)

Epoch 27 of the Tiny two-head model (SM106) minus epoch 26 of the show-time retrain (SM92) (the cut's source), in % of the big model (1 leak-free epoch)'s show-time LogLoss. Stop bar: point estimate above +0.25% on either head.

| head | Tiny two-head model (SM106) | show-time retrain (SM92) | % [95% CI] | better / worse |
|---|---|---|---|---|
| curve | 0.303854 | 0.303079 | +0.259% [+0.131, +0.399] | 301 / 421 |
| instant | 0.279473 | 0.278939 | +0.193% [+0.151, +0.234] | 172 / 550 |

The curve head is above the +0.25% bar, so the run stopped.

## Training and GPU cost

- Epoch 27 (8k tier, 581,942,724 reviews): 106.8 min wall (106.6 min training), 91,020 reviews/s.
- nvidia-smi max: 13,136 MiB (baseline 1,424 MiB); peak allocated 9,290 MiB, reserved 12,924 MiB.
- Total GPU minutes: about 132 (epoch 27 training 106.8, scoring the show-time retrain (SM92) epoch 26 reference 13, scoring epoch 27 12, trainer smoke under 1).

## 10-epoch ETA at the measured 91,020 reviews/s

- 8k tier (581.94M reviews per epoch): 106.6 min per epoch, about 17.8 h for 10 epochs.
- 2k tier (147.6M reviews per epoch): about 27.0 min per epoch, about 4.5 h for 10 epochs (the 8k speed assumed; the 2k tier was not measured here).

## Reading

The show-time retrain (SM92) itself sits at +0.259% on curve vs the big model (1 leak-free epoch) (0.300356 vs 0.29958 on show-time). Adding the cut cost seen here (about +0.26%) would put the Tiny two-head model (SM106) near +0.52% on curve, just above the +0.5% bar, so the final model would most likely have failed rule A anyway. Instant would have been fine (about +0.09%).
