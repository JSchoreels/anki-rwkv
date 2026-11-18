# The instant-only small model (SM57) stage 3 (curve-only, the mirror arm): result

Unit: points = % of the big model (1 leak-free epoch) LogLoss, paired user bootstrap 2,000, seed 0, 95% CI, 722 users of the 732 set, leak-free, normal inputs, w8+s8. Curve-only = stage 2's joint arm with the instant loss weight at 0; the joint and instant-only arms are stage 2's.

**Verdict: STOP (curve-only).** At epoch 5, curve-only - joint on the curve head is +0.873 points [+0.655, +1.104]: the CI is above zero, so the curve is better learned jointly.

## Trajectory

| epoch | curve-only - joint, curve (points) | users better / worse | curve-only - joint, instant | curve-only - big model, curve | joint - big model, curve |
|---|---|---|---|---|---|
| 1 | +0.135 [-0.063, +0.323] | 287 / 435 | +43.670 [+39.749, +47.869] | +6.872 [+6.329, +7.465] | +6.737 [+6.141, +7.417] |
| 3 | +1.017 [+0.838, +1.208] | 159 / 563 | +46.378 [+42.331, +50.657] | +5.326 [+4.883, +5.832] | +4.309 [+3.888, +4.786] |
| 5 | +0.873 [+0.655, +1.104] | 219 / 503 | +46.846 [+42.812, +51.140] | +5.402 [+4.888, +5.973] | +4.529 [+4.133, +4.963] |

Curve-only's instant head is untrained (+50.160 [+46.003, +54.582] vs the big model), as expected.

## Two dedicated models vs the joint model

| epoch | instant-only - joint, instant | curve-only - joint, curve | instant-only - big, instant | curve-only - big, curve | joint - big, instant / curve |
|---|---|---|---|---|---|
| 1 | -0.461 [-0.621, -0.329] | +0.135 [-0.063, +0.323] | +6.029 [+5.677, +6.397] | +6.872 [+6.329, +7.465] | +6.490 [+6.079, +6.914] / +6.737 [+6.141, +7.417] |
| 3 | -0.516 [-0.594, -0.445] | +1.017 [+0.838, +1.208] | +3.266 [+3.078, +3.454] | +5.326 [+4.883, +5.832] | +3.782 [+3.555, +4.019] / +4.309 [+3.888, +4.786] |
| 5 | -0.773 [-0.861, -0.685] | +0.873 [+0.655, +1.104] | +2.541 [+2.382, +2.699] | +5.402 [+4.888, +5.973] | +3.315 [+3.104, +3.521] / +4.529 [+4.133, +4.963] |

Reading: the split is one-sided. The instant head is better trained alone (go at epoch 5), but the curve head is better learned jointly (stop). Both rules are not go, so two dedicated models do not beat the shared one on both heads. The shared model stays for the curve, and a dedicated instant model serves instant. Same fresh init confirmed (initial_parameter_sha256 equal to stage 2's joint arm).

## Speed (training only; wall minutes include ~0.5 min set-up)

| epoch | curve-only training min | curve-only reviews/s | joint (stage 2) training min | joint reviews/s | peak allocated / reserved MiB (both) |
|---|---|---|---|---|---|
| 1 | 29.6 | 82,522 | 29.4 | 83,234 | 10,350 / 12,746 |
| 2 | 29.9 | 81,706 | 28.8 | 84,907 | 10,354 / 12,726 |
| 3 | 29.5 | 82,829 | 29.6 | 82,671 | 10,354 / 12,768 |
| 4 | 27.7 | 88,083 | 29.8 | 81,905 | 10,336 / 12,966 |
| 5 | 29.5 | 82,847 | 29.4 | 83,141 | 10,357 / 12,460 |

nvidia-smi max - baseline: curve-only e01-e04 about 12.1-13.6 GB (e05's baseline of 11,421 MiB was polluted by another job; max 14,786 MiB). Mean reviews/s: curve-only 83,597, joint 83,172 (no speed difference).

## 10-epoch ETA (training only, plus ~1-2 min set-up per epoch)

- 2k tier: 10 x 146.65M / 83,597 = ~17,540 s = ~4.9 h.
- 8k tier: 10 x 581.94M / 83,597 = ~69,610 s = ~19.3 h.
