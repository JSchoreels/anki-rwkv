# Trained smaller state for Tiny (SM137): pilot (v16k8, Instant-Only epoch 42)

Decision: **go**. Pre-registered: go if the lane epoch costs at most +0.30% instant vs the matched control (the six-layer Tiny with longer recovery (SM135)'s epoch 42), w8+s8, show-time inputs, 722 users; above that a v16k8 pilot is followed by a v16k12 pilot, any other lane shape stops; stop above +5.00% or on a non-finite user.

| comparison | instant % [95% CI] | LogLoss new / reference | better / worse users |
|---|---|---|---|
| lane epoch 42 (w8+s8) vs SM135 epoch 42 (w8+s8) | +0.065% [+0.054, +0.077] | 0.280106 / 0.279924 | 207 / 514 |
| lane epoch 42 (w8+s8hx8) vs SM135 epoch 42 (w8+s8) | +0.066% [+0.055, +0.078] | 0.280109 / 0.279924 | 202 / 517 |
| zero-shot lane cut of SM130 epoch 41 vs SM130 epoch 41 (w8+s8) | +0.389% [+0.364, +0.412] | 0.280782 / 0.279695 | 63 / 659 |
