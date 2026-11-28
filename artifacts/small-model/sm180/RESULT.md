# 4-bit weights (SM180): result

Finished Sat 11-28 03:02 (`SM180 GPU DONE`). Full tables: `sm180-ptq.md`, `sm180-qat.md`.

## Question and rule

Can the int8 weights of the six release models go to int4 (one float32 scale per group of 32 or 16 input columns) at a negligible cost? The rule is that int4 against the same model at int8 (same state format: F6 for Big and Small, T24 for Tiny) must have a 95% CI upper bound of at most +0.15% on every head, on the 723-user list of the 732 set. If neither group size passes for Small or Tiny, one straight-through QAT fine-tune is trained per failing size and scored against a matched int8 control, with the same bar.

## Post-training quantization (PTQ)

Cost vs the same model at int8 weights, point and 95% CI.

| model | head | w4g32 | w4g16 |
|---|---|---|---|
| Big two-head (SM91s2) | curve | +0.092% [+0.051, +0.140] free | +0.057% [+0.022, +0.097] free |
| Big two-head (SM91s2) | instant | +0.243% [+0.222, +0.265] | +0.156% [+0.133, +0.187] |
| Big Instant-Only (SM99) | instant | +0.456% [+0.409, +0.506] | +0.242% [+0.203, +0.300] |
| Small two-head (SM118) | curve | +0.201% [+0.144, +0.265] | +0.157% [+0.113, +0.206] |
| Small two-head (SM118) | instant | +1.009% [+0.948, +1.073] | +1.050% [+0.952, +1.157] |
| Small Instant-Only (SM113) | instant | +0.562% [+0.518, +0.615] | +0.797% [+0.726, +0.876] |
| six-layer Tiny two-head (SM130) | curve | +0.260% [+0.218, +0.304] | +0.227% [+0.183, +0.268] |
| six-layer Tiny two-head (SM130) | instant | +0.839% [+0.788, +0.890] | +0.254% [+0.235, +0.274] |
| six-layer Tiny Instant-Only (SM130) | instant | +0.635% [+0.594, +0.682] | +0.450% [+0.401, +0.510] |

Verdict: PTQ fails for all three sizes at both group sizes. The tightest cell per size is the instant head (Big at w4g16: upper bound +0.187%, just over the bar; Small two-head (SM118) at w4g32: +1.073%; six-layer Tiny two-head (SM130) at w4g16: +0.274%). The instant head is the sensitive one, as the earlier compression screen suggested. Only the curve head of Big two-head (SM91s2) is free.

## Quantization-aware training (QAT), group 32

Targets: Small two-head (SM118) and six-layer Tiny two-head (SM130). Arm at w4g32 vs its control at int8 (the rule):

| model | head | arm w4g32 vs control int8 | verdict |
|---|---|---|---|
| Small two-head (SM118) | curve | +0.083% [+0.038, +0.131] | free |
| Small two-head (SM118) | instant | +0.103% [+0.094, +0.114] | free |
| six-layer Tiny two-head (SM130) | curve | +0.085% [+0.055, +0.114] | free |
| six-layer Tiny two-head (SM130) | instant | +0.190% [+0.177, +0.204] | not free |

Report-only lines (curve / instant):
- Arm vs control, both at w4g32 (what QAT buys over PTQ): Small -0.094% / -0.616%; Tiny -0.203% / -0.372%.
- Control PTQ cost (control w4g32 vs control int8): Small +0.177% / +0.724%; Tiny +0.289% / +0.565%.
- Arm vs control, both at int8: Small +0.095% / +1.377%; Tiny +0.113% / +0.267%.
- Arm w4g32 vs the original release model at int8: Small +0.107% / +0.108%; Tiny +0.082% / +0.202%.

Verdict: QAT free for Small two-head (SM118); QAT not free for six-layer Tiny two-head (SM130), because its instant head has an upper bound of +0.204%, over +0.15%. The Small arm's int8 instant head is itself worse than its control by +1.377%, so a QAT-trained Small model should only be shipped at int4, never rescored at int8.

Training: Tiny arm 17.9 min at 62,767 reviews/s (67,310,673 reviews); Small arm 39.5 min at 28,326 reviews/s and Small control 38.9 min at 28,765 reviews/s (67,109,790 reviews each). The Tiny control was the reused SM171 one. GPU scoring tickets: Tiny QAT 40 min, Small QAT 107 min (three passes); PTQ Tiny 57, Small 93, Big 117 min.

## Weight bytes

| model | int8 | w4g32 (saved) | w4g16 (saved) |
|---|---|---|---|
| Big (both models) | 2,995,488 | 1,891,088 (36.9%) | 2,216,976 (26.0%) |
| Small (both models) | 410,459 | 272,347 (33.6%) | 310,459 (24.4%) |
| six-layer Tiny (both models) | 226,191 | 149,287 (34.0%) | 170,807 (24.5%) |

## Meaning

Plain int4 weights are not free for any size. After a short QAT, Small two-head (SM118) passes at w4g32 and saves 138,112 bytes (33.6%) of weights. Six-layer Tiny two-head (SM130) misses by a small margin (upper bound +0.204% against +0.15%) on its instant head. Not tested here: the Small Instant-Only (SM113) model, Big QAT, a longer or different QAT for Tiny, and float16 scales.
