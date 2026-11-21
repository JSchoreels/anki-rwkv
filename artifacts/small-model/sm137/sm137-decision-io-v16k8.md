# Trained smaller state for Tiny (SM137): decision (io, v16k8)

Verdict: **no go**. Rule: every trained head's 95% CI upper bound of 100 x (student at w8+s8hx8 / the longer Small Instant-Only model (SM113) at w8+s8 - 1) <= +1.00%, every rule user finite, long-card check pass (False).

| head | vs Small, release format % [95% CI] | LogLoss student / Small |
|---|---|---|
| instant | +0.576% [+0.534, +0.619] | 0.277522 / 0.275933 |
