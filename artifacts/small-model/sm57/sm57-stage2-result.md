# The instant-only small model (SM57) stage 2: result

Unit: points = % of the big model (1 leak-free epoch) LogLoss, paired user bootstrap 2,000, seed 0, 95% CI, 722 users of the 732 set, leak-free, normal inputs, w8+s8.

**Verdict: GO.** At the final epoch (5), instant-only - joint on the instant head is -0.773 points [-0.861, -0.685] (CI below zero, at least 0.05 points).

## Trajectory

| epoch | instant-only - joint, instant (points) | users better / worse | instant-only - big model, instant | joint - big model, instant | joint - big model, curve |
|---|---|---|---|---|---|
| 1 | -0.461 [-0.621, -0.329] | 479 / 243 | +6.029 [+5.677, +6.397] | +6.490 [+6.079, +6.914] | +6.737 [+6.141, +7.417] |
| 3 | -0.516 [-0.594, -0.445] | 557 / 165 | +3.266 [+3.078, +3.454] | +3.782 [+3.555, +4.019] | +4.309 [+3.888, +4.786] |
| 5 | -0.773 [-0.861, -0.685] | 661 / 60 | +2.541 [+2.382, +2.699] | +3.315 [+3.104, +3.521] | +4.529 [+4.133, +4.963] |

## Readings

- Epoch 1: instant-only already leads from the start (-0.461 points, 479 users better vs 243 worse, CI below zero); the gap widens through epoch 5.
- Same fresh init in both arms confirmed (initial_parameter_sha256 equal), which also equals the leak-free small model (SM72s) epoch 1 init.
- F = 5 (plan: projected 388.5 min <= 420 min cap after epoch 1: pair 60.1 min, scoring 14.5 min per checkpoint). The time plan did not cut any epoch.

## Speed (training only; wall minutes include ~0.5 min set-up)

| arm / epoch | training min | reviews/s | peak allocated / reserved MiB | nvidia-smi max - baseline MiB |
|---|---|---|---|---|
| instant-only e01 | 29.8 | 82,038 | 10,350 / 12,746 | 12,372 |
| joint e01 | 29.4 | 83,234 | 10,350 / 12,746 | 12,183 |
| instant-only e02 | 29.0 | 84,209 | 10,354 / 12,726 | 11,772 |
| joint e02 | 28.8 | 84,907 | 10,354 / 12,726 | 11,758 |
| instant-only e03 | 29.3 | 83,303 | 10,354 / 12,768 | 12,838 |
| joint e03 | 29.6 | 82,671 | 10,354 / 12,768 | 12,838 |
| instant-only e04 | 29.7 | 82,227 | 10,336 / 12,966 | 13,386 |
| joint e04 | 29.8 | 81,905 | 10,336 / 12,966 | 13,343 |
| instant-only e05 | 29.5 | 82,985 | 10,356 / 12,460 | 12,787 |
| joint e05 | 29.4 | 83,141 | 10,357 / 12,460 | 12,751 |

Mean reviews/s: instant-only 82,952, joint 83,172 (no speed difference; the curve loss weight 0 does not save compute).

## 10-epoch ETA (training only, plus ~1-2 min set-up per epoch)

- 2k tier: 10 x 146.65M / 83,000 = ~17,670 s = ~4.9 h (instant-only 294.7 min, joint 293.9 min).
- 8k tier: 10 x 581.94M / 83,000 = ~70,100 s = ~19.5 h (instant-only 19.49 h, joint 19.44 h).
