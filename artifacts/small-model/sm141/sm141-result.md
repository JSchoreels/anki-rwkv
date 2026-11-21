# Tiered card state by answer count (SM141): result

Status: **decided**. Rules (design.md): R1 every user finite; R2 dataset-wide cost vs int8 cards and notes (w8+s8hx8), benchmark LogLoss, 95% CI upper bound <= +0.05% per trained head; R3 reviews of cards with 128+ earlier answers on the long-card users (CPU) <= +0.15%; R4 reviews of young cards (1-M earlier answers) <= +0.1% (int3 / int2 schemes). Go: per size the smallest scheme that passes for both models.

## Go per size

| size | passing schemes | pick | state bytes vs int8 (732 set) |
|---|---|---|---|
| Big | none | REF | +0.0% |
| Small | none | REF | +0.0% |
| Tiny | none | REF | +0.0% |

## Per model and scheme (% vs int8 cards and notes)

| model | scheme | head | R2 dataset-wide | R3 long cards 128+ (CPU, label) | R4 young | R1 | pass |
|---|---|---|---|---|---|---|---|
| the show-time big model (SM91s2) | A | curve | +0.015% [+0.009, +0.021] | +0.177% [-0.025, +0.404] (AN) | - | True | False |
| the show-time big model (SM91s2) | A | instant | +0.016% [+0.012, +0.021] | +0.256% [-0.119, +0.515] (AN) | - | True | False |
| the show-time big model (SM91s2) | AN | curve | +0.033% [+0.020, +0.046] | +0.177% [-0.025, +0.404] (AN) | - | True | False |
| the show-time big model (SM91s2) | AN | instant | +0.043% [+0.035, +0.052] | +0.256% [-0.119, +0.515] (AN) | - | True | False |
| the show-time big model (SM91s2) | B4 | curve | +0.041% [+0.025, +0.060] | +0.155% [-0.071, +0.395] (B8) | +0.021% [+0.011, +0.032] | True | False |
| the show-time big model (SM91s2) | B4 | instant | +0.054% [+0.046, +0.063] | +0.252% [-0.139, +0.494] (B8) | +0.039% [+0.034, +0.044] | True | False |
| the show-time big model (SM91s2) | B8 | curve | +0.051% [+0.034, +0.073] | +0.155% [-0.071, +0.395] (B8) | +0.026% [+0.012, +0.039] | True | False |
| the show-time big model (SM91s2) | B8 | instant | +0.068% [+0.059, +0.078] | +0.252% [-0.139, +0.494] (B8) | +0.050% [+0.045, +0.056] | True | False |
| the show-time big model (SM91s2) | D4 | curve | +0.090% [+0.064, +0.114] | +0.170% [-0.023, +0.340] (D4) | +0.077% [+0.052, +0.105] | True | False |
| the show-time big model (SM91s2) | D4 | instant | +0.152% [+0.140, +0.166] | +0.245% [-0.144, +0.465] (D4) | +0.165% [+0.149, +0.182] | True | False |
| the show-time instant-only big model (SM99) | A | instant | +0.017% [+0.014, +0.021] | +0.295% [-0.155, +0.647] (AN) | - | True | False |
| the show-time instant-only big model (SM99) | AN | instant | +0.053% [+0.043, +0.066] | +0.295% [-0.155, +0.647] (AN) | - | True | False |
| the show-time instant-only big model (SM99) | B4 | instant | +0.065% [+0.055, +0.078] | +0.269% [-0.180, +0.621] (B8) | +0.041% [+0.036, +0.046] | True | False |
| the show-time instant-only big model (SM99) | B8 | instant | +0.079% [+0.069, +0.093] | +0.269% [-0.180, +0.621] (B8) | +0.054% [+0.049, +0.061] | True | False |
| the show-time instant-only big model (SM99) | D4 | instant | +0.164% [+0.150, +0.180] | +0.323% [-0.147, +0.665] (D4) | +0.174% [+0.158, +0.193] | True | False |
| the show-time two-head cool-down for Small (SM118) | A | curve | +0.015% [+0.011, +0.020] | +0.475% [+0.061, +2.047] (AN) | - | True | False |
| the show-time two-head cool-down for Small (SM118) | A | instant | +0.020% [+0.015, +0.024] | +0.226% [-0.051, +0.475] (AN) | - | True | False |
| the show-time two-head cool-down for Small (SM118) | AN | curve | +0.021% [+0.015, +0.027] | +0.475% [+0.061, +2.047] (AN) | - | True | False |
| the show-time two-head cool-down for Small (SM118) | AN | instant | +0.034% [+0.027, +0.041] | +0.226% [-0.051, +0.475] (AN) | - | True | False |
| the show-time two-head cool-down for Small (SM118) | B4 | curve | +0.032% [+0.025, +0.040] | +0.387% [+0.016, +1.468] (B8) | +0.015% [+0.008, +0.023] | True | False |
| the show-time two-head cool-down for Small (SM118) | B4 | instant | +0.045% [+0.038, +0.052] | +0.240% [-0.059, +0.529] (B8) | +0.024% [+0.021, +0.027] | True | False |
| the show-time two-head cool-down for Small (SM118) | B8 | curve | +0.041% [+0.034, +0.050] | +0.387% [+0.016, +1.468] (B8) | +0.024% [+0.018, +0.031] | True | False |
| the show-time two-head cool-down for Small (SM118) | B8 | instant | +0.059% [+0.052, +0.066] | +0.240% [-0.059, +0.529] (B8) | +0.039% [+0.036, +0.043] | True | False |
| the show-time two-head cool-down for Small (SM118) | D4 | curve | +0.099% [+0.084, +0.117] | +0.048% [-0.670, +0.233] (D4) | +0.099% [+0.076, +0.127] | True | False |
| the show-time two-head cool-down for Small (SM118) | D4 | instant | +0.147% [+0.137, +0.157] | +0.211% [-0.186, +0.410] (D4) | +0.164% [+0.152, +0.178] | True | False |
| the longer Small Instant-Only model (SM113) | A | instant | +0.024% [+0.019, +0.029] | +0.142% [-0.052, +0.321] (AN) | - | True | False |
| the longer Small Instant-Only model (SM113) | AN | instant | +0.036% [+0.030, +0.043] | +0.142% [-0.052, +0.321] (AN) | - | True | False |
| the longer Small Instant-Only model (SM113) | B4 | instant | +0.055% [+0.048, +0.062] | +0.199% [-0.015, +0.426] (B8) | +0.035% [+0.031, +0.041] | True | False |
| the longer Small Instant-Only model (SM113) | B8 | instant | +0.072% [+0.065, +0.079] | +0.199% [-0.015, +0.426] (B8) | +0.052% [+0.047, +0.058] | True | False |
| the longer Small Instant-Only model (SM113) | D4 | instant | +0.212% [+0.201, +0.225] | +0.196% [-0.061, +0.348] (D4) | +0.247% [+0.231, +0.264] | True | False |
| the six-layer Tiny two-head model (SM130) | A | curve | +0.016% [+0.012, +0.020] | +0.297% [+0.182, +0.705] (AN) | - | True | False |
| the six-layer Tiny two-head model (SM130) | A | instant | +0.021% [+0.017, +0.026] | +0.265% [+0.077, +0.435] (AN) | - | True | False |
| the six-layer Tiny two-head model (SM130) | AN | curve | +0.036% [+0.028, +0.044] | +0.297% [+0.182, +0.705] (AN) | - | True | False |
| the six-layer Tiny two-head model (SM130) | AN | instant | +0.060% [+0.051, +0.071] | +0.265% [+0.077, +0.435] (AN) | - | True | False |
| the six-layer Tiny two-head model (SM130) | B4 | curve | +0.049% [+0.041, +0.059] | +0.305% [+0.174, +0.823] (B8) | +0.030% [+0.022, +0.038] | True | False |
| the six-layer Tiny two-head model (SM130) | B4 | instant | +0.082% [+0.072, +0.093] | +0.291% [+0.178, +0.502] (B8) | +0.057% [+0.051, +0.064] | True | False |
| the six-layer Tiny two-head model (SM130) | B8 | curve | +0.062% [+0.052, +0.072] | +0.305% [+0.174, +0.823] (B8) | +0.041% [+0.034, +0.049] | True | False |
| the six-layer Tiny two-head model (SM130) | B8 | instant | +0.103% [+0.092, +0.114] | +0.291% [+0.178, +0.502] (B8) | +0.079% [+0.073, +0.086] | True | False |
| the six-layer Tiny two-head model (SM130) | D4 | curve | +0.162% [+0.140, +0.184] | +0.266% [+0.146, +0.558] (D4) | +0.169% [+0.143, +0.195] | True | False |
| the six-layer Tiny two-head model (SM130) | D4 | instant | +0.261% [+0.245, +0.278] | +0.265% [-0.048, +0.406] (D4) | +0.303% [+0.281, +0.326] | True | False |
| the six-layer Tiny Instant-Only model (SM130) | A | instant | +0.038% [+0.031, +0.045] | +0.283% [+0.045, +0.383] (AN) | - | True | False |
| the six-layer Tiny Instant-Only model (SM130) | AN | instant | +0.086% [+0.073, +0.100] | +0.283% [+0.045, +0.383] (AN) | - | True | False |
| the six-layer Tiny Instant-Only model (SM130) | B4 | instant | +0.115% [+0.103, +0.129] | +0.227% [-0.047, +0.357] (B8) | +0.070% [+0.064, +0.077] | True | False |
| the six-layer Tiny Instant-Only model (SM130) | B8 | instant | +0.150% [+0.137, +0.164] | +0.227% [-0.047, +0.357] (B8) | +0.103% [+0.096, +0.111] | True | False |
| the six-layer Tiny Instant-Only model (SM130) | D4 | instant | +0.367% [+0.348, +0.387] | +0.268% [+0.029, +0.419] (D4) | +0.410% [+0.386, +0.435] | True | False |
