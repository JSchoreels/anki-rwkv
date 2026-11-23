# Low-rank int8 card and note states, post-training (SM148): result tables

Status: **pending**. Rules (design.md, pre-registered 11-23): R1 every user finite; R2 dataset-wide cost vs int8 cards and notes (w8+s8hx8), benchmark LogLoss on the 722 rule users of the 732 set, 95% CI upper bound <= +0.05% per head; R2T (six-layer Tiny models) the label vs the matching Small model at int8, upper bound <= +1.00%; R3 reviews of cards with 128+ earlier answers on the 50 long-card users (CPU): point <= +0.5% and upper bound <= +1.0% per head; G at most 1% of the long-card users' cards with a review |dR| > 0.10. Go per size: the label with the smallest state bytes passing for both models (Tiny under R2T; the strict go under R2 is listed too).

## Go per size

| size | rule | passing | go pick | bytes vs int8 | strict (R2) passing | strict pick |
|---|---|---|---|---|---|---|
| Big | R2 | Q4 | Q4 | -74.4% | Q4 | Q4 |
| Small | R2 | Q4 | Q4 | -52.3% | Q4 | Q4 |
| Tiny | Tiny rule (R2T) | T24, Q4 | T24 | -64.7% | Q4 | Q4 |

## Rules per model, label and head

| model | label | head | R2 vs int8 | R2T vs Small int8 | R3 128+ (long-card users) | G cards > 0.10 | pass |
|---|---|---|---|---|---|---|---|
| Big two-head (SM91s2) | Q2 | curve | +0.080% [+0.061, +0.101] | - | +0.137% [-0.067, +0.428] | 0.1554% | no |
| Big two-head (SM91s2) | Q2 | instant | +0.109% [+0.096, +0.122] | - | +0.324% [-0.069, +0.740] | 0.2894% | no |
| Big two-head (SM91s2) | T24 | curve | +0.034% [+0.024, +0.043] | - | -0.007% [-0.140, +0.167] | 0.0071% | no |
| Big two-head (SM91s2) | T24 | instant | +0.054% [+0.049, +0.059] | - | +0.011% [-0.133, +0.219] | 0.0243% | no |
| Big two-head (SM91s2) | Q4 | curve | +0.014% [+0.008, +0.020] | - | +0.014% [-0.082, +0.167] | 0.0000% | yes |
| Big two-head (SM91s2) | Q4 | instant | +0.018% [+0.015, +0.021] | - | -0.001% [-0.123, +0.170] | 0.0012% | yes |
| Big Instant-Only (SM99) | Q2 | instant | +0.118% [+0.107, +0.132] | - | +0.363% [-0.021, +0.741] | 0.3535% | no |
| Big Instant-Only (SM99) | T24 | instant | +0.059% [+0.054, +0.065] | - | -0.011% [-0.152, +0.168] | 0.0302% | no |
| Big Instant-Only (SM99) | Q4 | instant | +0.019% [+0.016, +0.022] | - | +0.020% [-0.154, +0.232] | 0.0036% | yes |
| Small two-head (SM118) | Q2 | curve | +0.047% [+0.038, +0.056] | - | -0.172% [-2.221, +0.309] | 0.0165% | no |
| Small two-head (SM118) | Q2 | instant | +0.079% [+0.072, +0.087] | - | +0.292% [+0.242, +0.516] | 0.0388% | no |
| Small two-head (SM118) | T24 | curve | +0.029% [+0.023, +0.035] | - | -0.156% [-1.243, +0.091] | 0.0035% | no |
| Small two-head (SM118) | T24 | instant | +0.050% [+0.045, +0.055] | - | +0.036% [-0.071, +0.102] | 0.0129% | no |
| Small two-head (SM118) | Q4 | curve | +0.005% [+0.003, +0.007] | - | -0.133% [-1.073, +0.095] | 0.0000% | yes |
| Small two-head (SM118) | Q4 | instant | +0.009% [+0.007, +0.011] | - | +0.060% [+0.023, +0.142] | 0.0012% | yes |
| Small Instant-Only (SM113) | Q2 | instant | +0.089% [+0.080, +0.098] | - | +0.310% [+0.194, +0.775] | 0.0835% | no |
| Small Instant-Only (SM113) | T24 | instant | +0.056% [+0.051, +0.062] | - | +0.037% [-0.003, +0.176] | 0.0215% | no |
| Small Instant-Only (SM113) | Q4 | instant | +0.009% [+0.007, +0.011] | - | +0.050% [+0.011, +0.213] | 0.0004% | yes |
| six-layer Tiny two-head (SM130) | Q2 | curve | +0.081% [+0.066, +0.095] | +0.548% [+0.475, +0.629] | +0.548% [+0.043, +1.724] | 0.1285% | no |
| six-layer Tiny two-head (SM130) | Q2 | instant | +0.131% [+0.119, +0.144] | +0.656% [+0.611, +0.703] | +0.304% [+0.052, +0.463] | 0.2829% | no |
| six-layer Tiny two-head (SM130) | T24 | curve | +0.040% [+0.031, +0.049] | +0.508% [+0.436, +0.588] | +0.002% [-0.085, +0.371] | 0.0278% | yes (strict: no) |
| six-layer Tiny two-head (SM130) | T24 | instant | +0.074% [+0.067, +0.082] | +0.598% [+0.555, +0.643] | -0.044% [-0.080, +0.073] | 0.0580% | yes (strict: no) |
| six-layer Tiny two-head (SM130) | Q4 | curve | +0.007% [+0.003, +0.011] | +0.474% [+0.404, +0.554] | -0.014% [-0.074, +0.165] | 0.0004% | yes |
| six-layer Tiny two-head (SM130) | Q4 | instant | +0.015% [+0.012, +0.018] | +0.539% [+0.498, +0.582] | -0.020% [-0.067, +0.119] | 0.0020% | yes |
| six-layer Tiny Instant-Only (SM130) | Q2 | instant | +0.159% [+0.145, +0.174] | +0.724% [+0.675, +0.774] | +0.273% [+0.180, +0.713] | 0.3158% | yes (strict: no) |
| six-layer Tiny Instant-Only (SM130) | T24 | instant | +0.089% [+0.081, +0.099] | +0.655% [+0.610, +0.701] | +0.009% [-0.017, +0.109] | 0.0776% | yes (strict: no) |
| six-layer Tiny Instant-Only (SM130) | Q4 | instant | +0.016% [+0.014, +0.019] | +0.581% [+0.539, +0.625] | +0.029% [-0.004, +0.186] | 0.0000% | yes |

## By answer count before the review (GPU, 732 set, % vs int8, read heads)

| model | label | head | 1-2 | 3-4 | 5-8 | 9-15 | 16-31 | 32-127 | 128- |
|---|---|---|---|---|---|---|---|---|---|
| Big two-head (SM91s2) | Q2 | curve | +0.050% | +0.042% | +0.085% | +0.089% | +0.118% | +0.149% | +0.165% |
| Big two-head (SM91s2) | Q2 | instant | +0.045% | +0.078% | +0.118% | +0.144% | +0.179% | +0.213% | +0.288% |
| Big two-head (SM91s2) | T24 | curve | +0.018% | +0.015% | +0.041% | +0.047% | +0.050% | +0.024% | +0.024% |
| Big two-head (SM91s2) | T24 | instant | +0.019% | +0.038% | +0.073% | +0.099% | +0.062% | +0.029% | +0.026% |
| Big two-head (SM91s2) | Q4 | curve | +0.011% | +0.008% | +0.015% | +0.014% | +0.017% | +0.027% | +0.017% |
| Big two-head (SM91s2) | Q4 | instant | +0.008% | +0.012% | +0.020% | +0.026% | +0.034% | +0.030% | +0.028% |
| Big Instant-Only (SM99) | Q2 | instant | +0.045% | +0.085% | +0.128% | +0.154% | +0.198% | +0.222% | +0.328% |
| Big Instant-Only (SM99) | T24 | instant | +0.020% | +0.042% | +0.081% | +0.106% | +0.066% | +0.030% | +0.013% |
| Big Instant-Only (SM99) | Q4 | instant | +0.008% | +0.012% | +0.019% | +0.025% | +0.036% | +0.034% | +0.038% |
| Small two-head (SM118) | Q2 | curve | +0.016% | +0.021% | +0.045% | +0.064% | +0.084% | +0.084% | +0.062% |
| Small two-head (SM118) | Q2 | instant | +0.026% | +0.050% | +0.084% | +0.108% | +0.134% | +0.139% | +0.180% |
| Small two-head (SM118) | T24 | curve | +0.004% | +0.017% | +0.038% | +0.051% | +0.045% | +0.014% | -0.116% |
| Small two-head (SM118) | T24 | instant | +0.012% | +0.034% | +0.066% | +0.088% | +0.064% | +0.040% | +0.024% |
| Small two-head (SM118) | Q4 | curve | +0.002% | +0.003% | +0.004% | +0.006% | +0.012% | +0.013% | -0.086% |
| Small two-head (SM118) | Q4 | instant | +0.003% | +0.007% | +0.008% | +0.011% | +0.018% | +0.021% | +0.036% |
| Small Instant-Only (SM113) | Q2 | instant | +0.026% | +0.052% | +0.097% | +0.123% | +0.140% | +0.173% | +0.286% |
| Small Instant-Only (SM113) | T24 | instant | +0.013% | +0.037% | +0.076% | +0.103% | +0.073% | +0.034% | +0.035% |
| Small Instant-Only (SM113) | Q4 | instant | +0.004% | +0.006% | +0.007% | +0.015% | +0.014% | +0.020% | +0.035% |
| six-layer Tiny two-head (SM130) | Q2 | curve | +0.023% | +0.039% | +0.071% | +0.108% | +0.153% | +0.218% | +0.340% |
| six-layer Tiny two-head (SM130) | Q2 | instant | +0.049% | +0.093% | +0.151% | +0.164% | +0.196% | +0.248% | +0.258% |
| six-layer Tiny two-head (SM130) | T24 | curve | +0.009% | +0.022% | +0.046% | +0.079% | +0.076% | +0.024% | -0.005% |
| six-layer Tiny two-head (SM130) | T24 | instant | +0.021% | +0.056% | +0.110% | +0.118% | +0.083% | +0.044% | +0.014% |
| six-layer Tiny two-head (SM130) | Q4 | curve | -0.002% | +0.007% | +0.007% | +0.013% | +0.017% | +0.017% | +0.011% |
| six-layer Tiny two-head (SM130) | Q4 | instant | +0.006% | +0.012% | +0.015% | +0.019% | +0.025% | +0.026% | +0.022% |
| six-layer Tiny Instant-Only (SM130) | Q2 | instant | +0.061% | +0.147% | +0.183% | +0.200% | +0.230% | +0.253% | +0.249% |
| six-layer Tiny Instant-Only (SM130) | T24 | instant | +0.026% | +0.092% | +0.127% | +0.155% | +0.085% | +0.026% | +0.021% |
| six-layer Tiny Instant-Only (SM130) | Q4 | instant | +0.010% | +0.014% | +0.017% | +0.024% | +0.021% | +0.025% | +0.046% |

## Per-review and per-card |dR| vs int8 (long-card users, CPU)

| model | label | head | reviews P50 / P99 / max | reviews > 0.10 | cards > 0.05 | cards > 0.10 |
|---|---|---|---|---|---|---|
| Big two-head (SM91s2) | Q4 | curve | 0.0011 / 0.0167 / 0.0890 | 0 (0.0000%) | 0.0385% | 0.0000% |
| Big two-head (SM91s2) | Q4 | instant | 0.0011 / 0.0193 / 0.3035 | 2 (0.0001%) | 0.1228% | 0.0012% |
| Big two-head (SM91s2) | T24 | curve | 0.0023 / 0.0269 / 0.1479 | 12 (0.0007%) | 0.5533% | 0.0071% |
| Big two-head (SM91s2) | T24 | instant | 0.0024 / 0.0314 / 0.3879 | 43 (0.0024%) | 1.1405% | 0.0243% |
| Big two-head (SM91s2) | Q2 | curve | 0.0032 / 0.0426 / 0.2110 | 314 (0.0178%) | 3.1551% | 0.1554% |
| Big two-head (SM91s2) | Q2 | instant | 0.0033 / 0.0481 / 0.4161 | 681 (0.0385%) | 4.7944% | 0.2894% |
| Big Instant-Only (SM99) | Q4 | instant | 0.0012 / 0.0199 / 0.2941 | 6 (0.0003%) | 0.1720% | 0.0036% |
| Big Instant-Only (SM99) | T24 | instant | 0.0025 / 0.0325 / 0.3947 | 53 (0.0030%) | 1.3996% | 0.0302% |
| Big Instant-Only (SM99) | Q2 | instant | 0.0035 / 0.0490 / 0.4221 | 823 (0.0466%) | 5.1318% | 0.3535% |
| Small two-head (SM118) | Q4 | curve | 0.0007 / 0.0099 / 0.0632 | 0 (0.0000%) | 0.0008% | 0.0000% |
| Small two-head (SM118) | Q4 | instant | 0.0008 / 0.0121 / 0.1261 | 3 (0.0001%) | 0.0074% | 0.0012% |
| Small two-head (SM118) | T24 | curve | 0.0017 / 0.0213 / 0.2516 | 10 (0.0003%) | 0.1912% | 0.0035% |
| Small two-head (SM118) | T24 | instant | 0.0019 / 0.0265 / 0.3676 | 36 (0.0011%) | 0.6884% | 0.0129% |
| Small two-head (SM118) | Q2 | curve | 0.0023 / 0.0284 / 0.2769 | 48 (0.0015%) | 0.6331% | 0.0165% |
| Small two-head (SM118) | Q2 | instant | 0.0025 / 0.0342 / 0.3647 | 109 (0.0034%) | 1.7485% | 0.0388% |
| Small Instant-Only (SM113) | Q4 | instant | 0.0008 / 0.0133 / 0.1903 | 1 (0.0000%) | 0.0161% | 0.0004% |
| Small Instant-Only (SM113) | T24 | instant | 0.0019 / 0.0285 / 0.3508 | 57 (0.0018%) | 1.0547% | 0.0215% |
| Small Instant-Only (SM113) | Q2 | instant | 0.0026 / 0.0377 / 0.3510 | 248 (0.0077%) | 2.5748% | 0.0835% |
| six-layer Tiny two-head (SM130) | Q4 | curve | 0.0009 / 0.0133 / 0.1656 | 1 (0.0000%) | 0.0153% | 0.0004% |
| six-layer Tiny two-head (SM130) | Q4 | instant | 0.0010 / 0.0162 / 0.1126 | 5 (0.0002%) | 0.0592% | 0.0020% |
| six-layer Tiny two-head (SM130) | T24 | curve | 0.0021 / 0.0270 / 0.7181 | 76 (0.0024%) | 0.6590% | 0.0278% |
| six-layer Tiny two-head (SM130) | T24 | instant | 0.0023 / 0.0329 / 0.3967 | 157 (0.0049%) | 1.7638% | 0.0580% |
| six-layer Tiny two-head (SM130) | Q2 | curve | 0.0030 / 0.0396 / 0.7211 | 399 (0.0123%) | 2.6653% | 0.1285% |
| six-layer Tiny two-head (SM130) | Q2 | instant | 0.0031 / 0.0457 / 0.4169 | 893 (0.0276%) | 4.8111% | 0.2829% |
| six-layer Tiny Instant-Only (SM130) | Q4 | instant | 0.0011 / 0.0173 / 0.0979 | 0 (0.0000%) | 0.0697% | 0.0000% |
| six-layer Tiny Instant-Only (SM130) | T24 | instant | 0.0025 / 0.0347 / 0.3625 | 219 (0.0068%) | 2.3186% | 0.0776% |
| six-layer Tiny Instant-Only (SM130) | Q2 | instant | 0.0034 / 0.0474 / 0.3616 | 1065 (0.0330%) | 5.4548% | 0.3158% |

## Speed (report only): review evaluations per second, model time

| model | where | REF | Q4 | T24 | Q2 |
|---|---|---|---|---|---|
| Big two-head (SM91s2) | gpu_732 | 76714 | 28850 (0.376x) | 29225 (0.381x) | 29769 (0.388x) |
| Big two-head (SM91s2) | cpu_long | 913 | 832 (0.911x) | 846 (0.927x) | 866 (0.949x) |
| Big Instant-Only (SM99) | gpu_732 | 77646 | 29192 (0.376x) | 29355 (0.378x) | 29777 (0.383x) |
| Big Instant-Only (SM99) | cpu_long | 924 | 844 (0.913x) | 843 (0.912x) | 863 (0.934x) |
| Small two-head (SM118) | gpu_732 | 124598 | 56438 (0.453x) | 56937 (0.457x) | 57783 (0.464x) |
| Small two-head (SM118) | cpu_long | 1529 | 1404 (0.918x) | 1415 (0.925x) | 1451 (0.949x) |
| Small Instant-Only (SM113) | gpu_732 | 126143 | 57045 (0.452x) | 57352 (0.455x) | 58007 (0.46x) |
| Small Instant-Only (SM113) | cpu_long | 1540 | 1406 (0.913x) | 1419 (0.921x) | 1451 (0.942x) |
| six-layer Tiny two-head (SM130) | gpu_732 | 295619 | 83189 (0.281x) | 83964 (0.284x) | 85607 (0.29x) |
| six-layer Tiny two-head (SM130) | cpu_long | 3690 | 3205 (0.869x) | 3218 (0.872x) | 3369 (0.913x) |
| six-layer Tiny Instant-Only (SM130) | gpu_732 | 300392 | 84612 (0.282x) | 84917 (0.283x) | 86248 (0.287x) |
| six-layer Tiny Instant-Only (SM130) | cpu_long | 3725 | 3191 (0.857x) | 3253 (0.873x) | 3376 (0.906x) |
