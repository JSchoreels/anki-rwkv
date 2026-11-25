# Sharpness-aware cool-down (SM172): result

Verdict: **NO GO** (decided; 723 rule users of the 732 set; paired user bootstrap, 2,000 draws, seed 0, 95% CI). Model: six-layer Tiny two-head (SM130). Full tables: `sm172-result.md` and `sm172-result.json`. Diagnostics: `gpu/arms-summary.json`.

## Question and rules

Does a short cool-down with sharpness-aware minimization (SAM) find a flatter minimum, so that compressing the card and note states to rank 2 (Q2) costs at least 0.05% less against the same model's own int8 states (REF) than after the identical cool-down without SAM? The control is the cool-down of the state-energy penalty screen (SM171), reused without retraining or rescoring. Two radii were tried: rho 0.2 and rho 0.05.

The arm passes only if, on at least one head (curve or instant), its Q2 cost minus the control's Q2 cost is at most -0.05% with the CI upper bound below 0, and on both heads the REF guard (the arm's int8 against the original model's int8, minus the control's) has a CI upper bound of at most +0.05%. NO GO if no arm passes.

## Verdict per arm and head

| arm | head | Q2 cost change vs control | REF guard (int8 change vs the release model, minus the control's) |
|---|---|---|---|
| rho 0.2 | curve | -0.004% [-0.007, -0.000] | +0.169% [+0.127, +0.211] (fails) |
| rho 0.2 | instant | -0.006% [-0.008, -0.003] | +0.046% [+0.038, +0.054] (fails) |
| rho 0.05 | curve | -0.000% [-0.002, +0.001] | +0.021% [+0.006, +0.037] (ok) |
| rho 0.05 | instant | -0.003% [-0.004, -0.001] | +0.001% [-0.003, +0.005] (ok) |

No arm comes near the -0.05% bar: the best Q2 gain is 0.006%. The rho 0.2 arm also makes int8 itself worse (+0.161% curve, +0.055% instant against the original model).

## Cost against each model's own int8 (curve / instant)

| model | T24 | Q2 |
|---|---|---|
| rho 0.2 | +0.043% / +0.074% | +0.080% [+0.067, +0.092] / +0.130% [+0.118, +0.142] |
| rho 0.05 | +0.045% / +0.076% | +0.084% [+0.069, +0.097] / +0.133% [+0.121, +0.145] |
| control (SM171) | +0.044% / +0.078% | +0.084% [+0.069, +0.098] / +0.135% [+0.123, +0.148] |
| original model (SM148 rows) | +0.040% / +0.074% | +0.080% / +0.132% |

T24 cost change vs the control: rho 0.2 -0.001% / -0.004%, rho 0.05 +0.001% / -0.002%. Q2 stays 0.04% to 0.06% above T24 on the curve head for the control (SAM does not close that gap; rho 0.05 Q2 vs control T24 is +0.060% curve, +0.056% instant).

Each arm against the original model at int8 (curve / instant): rho 0.2 +0.161% / +0.055%; rho 0.05 +0.013% / +0.009%; control -0.008% / +0.008%.

## Instant Q2 cost by answer count

The change vs the control is at most about -0.03% in every bucket and is significantly negative only for 3 to 4, 16 to 31 and 32 to 127 answers (for example 32 to 127: rho 0.2 -0.019% [-0.030, -0.007], rho 0.05 -0.014% [-0.019, -0.008]). The 128+ bucket is too wide to read (CI crosses 0).

## SAM diagnostics and dropped share

- Mean clipped first-pass gradient norm 0.060 in both arms. Loss rise L(w + e) - L(w): 0.0365 (rho 0.2) and 0.0050 (rho 0.05), against the first-order value rho times the gradient norm of 0.0120 and 0.0031, so the loss is clearly curved along the gradient. Second-pass gradient norm 0.389 (rho 0.2) and 0.155 (rho 0.05).
- bfloat16 survival: e itself survives (part along e 1.00 to 1.01 at every radius); rounding noise is 1.7 x rho at rho 0.05 and 0.75 x rho at rho 0.2.
- Dropped share (SM171 diagnostic, mean of the first 10 / last 10 records), stream 0 / stream 2: rho 0.2 0.057 / 0.070 -> 0.062 / 0.076; rho 0.05 0.057 / 0.069 -> 0.062 / 0.074; control 0.057 / 0.068 -> 0.062 / 0.073. SAM barely changes the states' energy profile.

## Training and GPU time

Both arms trained 67,310,673 reviews in 2,570 updates: rho 0.2 at 31,117 reviews/s (36.1 min), rho 0.05 at 30,783 reviews/s (36.4 min); the control ran 55,723 reviews/s (20.1 min). Peak 4,565 MiB allocated, 8,648 MiB reserved. GPU minutes: smoke 0.3, two trainings 72.5, integrity 0.2, scoring 47.1 (pass 1) + 0.9 (retry pass at cap 0.92), total about 121 min. Integrity: the original model's int8 on the 3 smoke users matches the SM148 rows to 0 (max difference 0.00e+00).

## Conclusion

SAM does not make the compressed states cheaper at either radius, and the larger radius hurts int8. T24 stays the chosen format and SAM is closed for state compression.
