# Rank-1 int8 states (SM150): result tables

Status: **decided**. Rules (design.md, pre-registered 11-23): the cost is the 95% CI upper bound per head of the LogLoss change on the 722 rule users of the 732 set. Big: vs its own int8 (w8+s8hx8) <= +0.15%. Small / Tiny: TOTAL cost vs the matching Big model at int8 (two-head vs Big two-head (SM91s2), Instant-Only vs Big Instant-Only (SM99), the model's own heads) <= +0.5% (Small) / +1.0% (Tiny); every user finite; at least 715 users. Go per size: the passing label with the smallest state bytes for both models among rank 1 (R1), rank 1 then 2 from 16 answers (R12), rank 2 (Q2), rank 4 (Q4) (int8 factors). Long-card (128+ answers) and per-card |dR| are warnings only (long-card runs: Small and Tiny only).

## Go per size

| size | rule | passing | go pick | bytes vs int8 | labels with a warning |
|---|---|---|---|---|---|
| Big | vs own int8, upper <= 0.15% | Q2, Q4 | Q2 | -82.8% | none |
| Small | vs Big at int8, upper <= 0.5% | Q2, Q4 | Q2 | -68.2% | R1: sm118, sm113; R12: sm118, sm113; Q2: sm118, sm113; Q4: sm118, sm113 |
| Tiny | vs Big at int8, upper <= 1.0% | Q2, Q4 | Q2 | -66.7% | R1: sm130th, sm130io; R12: sm130th, sm130io; Q2: sm130th, sm130io; Q4: sm130th, sm130io |

## Cost per model, label and head

| model | label | head | vs own int8 | vs Big int8 | long-card 128+ (warning) | cards > 0.10 (warning) | pass |
|---|---|---|---|---|---|---|---|
| Big two-head (SM91s2) | R1 | curve | +0.403% [+0.327, +0.507] | - | not run | - | no |
| Big two-head (SM91s2) | R1 | instant | +0.501% [+0.465, +0.543] | - | not run | - | no |
| Big two-head (SM91s2) | R12 | curve | +0.262% [+0.211, +0.331] | - | not run | - | no |
| Big two-head (SM91s2) | R12 | instant | +0.344% [+0.320, +0.368] | - | not run | - | no |
| Big two-head (SM91s2) | Q2 | curve | +0.080% [+0.061, +0.101] | - | not run | - | yes |
| Big two-head (SM91s2) | Q2 | instant | +0.109% [+0.096, +0.122] | - | not run | - | yes |
| Big two-head (SM91s2) | Q4 | curve | +0.014% [+0.008, +0.020] | - | not run | - | yes |
| Big two-head (SM91s2) | Q4 | instant | +0.018% [+0.015, +0.021] | - | not run | - | yes |
| Big Instant-Only (SM99) | R1 | instant | +0.550% [+0.510, +0.592] | - | not run | - | no |
| Big Instant-Only (SM99) | R12 | instant | +0.381% [+0.355, +0.408] | - | not run | - | no |
| Big Instant-Only (SM99) | Q2 | instant | +0.118% [+0.107, +0.132] | - | not run | - | yes |
| Big Instant-Only (SM99) | Q4 | instant | +0.019% [+0.016, +0.022] | - | not run | - | yes |
| Small two-head (SM118) | R1 | curve | +0.290% [+0.258, +0.321] | +0.499% [+0.334, +0.666] | -0.423% [-5.767, +0.719] | 1.7156% | no |
| Small two-head (SM118) | R1 | instant | +0.462% [+0.430, +0.500] | +0.608% [+0.565, +0.659] | +0.729% [+0.459, +1.325] | 3.7843% | no |
| Small two-head (SM118) | R12 | curve | +0.207% [+0.186, +0.229] | +0.419% [+0.256, +0.581] | -0.225% [-2.759, +0.384] | 0.6335% | no |
| Small two-head (SM118) | R12 | instant | +0.341% [+0.319, +0.364] | +0.490% [+0.457, +0.525] | +0.291% [+0.109, +0.527] | 1.9664% | no |
| Small two-head (SM118) | Q2 | curve | +0.047% [+0.038, +0.056] | +0.258% [+0.098, +0.422] | -0.172% [-2.221, +0.309] | - | yes |
| Small two-head (SM118) | Q2 | instant | +0.079% [+0.072, +0.087] | +0.229% [+0.202, +0.257] | +0.292% [+0.242, +0.516] | - | yes |
| Small two-head (SM118) | Q4 | curve | +0.005% [+0.003, +0.007] | +0.218% [+0.061, +0.379] | -0.133% [-1.073, +0.095] | - | yes |
| Small two-head (SM118) | Q4 | instant | +0.009% [+0.007, +0.011] | +0.160% [+0.135, +0.186] | +0.060% [+0.023, +0.142] | - | yes |
| Small Instant-Only (SM113) | R1 | instant | +0.539% [+0.499, +0.581] | +0.542% [+0.500, +0.589] | +1.235% [+0.935, +2.201] | 4.3645% | no |
| Small Instant-Only (SM113) | R12 | instant | +0.401% [+0.374, +0.433] | +0.411% [+0.379, +0.446] | +0.338% [+0.203, +0.838] | 2.4824% | yes |
| Small Instant-Only (SM113) | Q2 | instant | +0.089% [+0.080, +0.098] | +0.101% [+0.075, +0.127] | +0.310% [+0.194, +0.775] | - | yes |
| Small Instant-Only (SM113) | Q4 | instant | +0.009% [+0.007, +0.011] | +0.025% [-0.001, +0.050] | +0.050% [+0.011, +0.213] | - | yes |
| six-layer Tiny two-head (SM130) | R1 | curve | +0.492% [+0.428, +0.559] | +1.162% [+0.972, +1.355] | +1.193% [+0.870, +1.955] | 4.5776% | no |
| six-layer Tiny two-head (SM130) | R1 | instant | +0.696% [+0.639, +0.760] | +1.361% [+1.276, +1.454] | +0.812% [+0.566, +1.912] | 6.7571% | no |
| six-layer Tiny two-head (SM130) | R12 | curve | +0.311% [+0.273, +0.345] | +0.989% [+0.807, +1.175] | +0.581% [+0.067, +1.887] | 1.7372% | no |
| six-layer Tiny two-head (SM130) | R12 | instant | +0.449% [+0.417, +0.482] | +1.118% [+1.051, +1.188] | +0.351% [+0.056, +0.576] | 3.3043% | no |
| six-layer Tiny two-head (SM130) | Q2 | curve | +0.081% [+0.066, +0.095] | +0.759% [+0.584, +0.939] | +0.548% [+0.043, +1.724] | - | yes |
| six-layer Tiny two-head (SM130) | Q2 | instant | +0.131% [+0.119, +0.144] | +0.802% [+0.745, +0.861] | +0.304% [+0.052, +0.463] | - | yes |
| six-layer Tiny two-head (SM130) | Q4 | curve | +0.007% [+0.003, +0.011] | +0.688% [+0.513, +0.866] | -0.014% [-0.074, +0.165] | - | yes |
| six-layer Tiny two-head (SM130) | Q4 | instant | +0.015% [+0.012, +0.018] | +0.687% [+0.635, +0.742] | -0.020% [-0.067, +0.119] | - | yes |
| six-layer Tiny Instant-Only (SM130) | R1 | instant | +0.791% [+0.734, +0.851] | +1.358% [+1.284, +1.434] | +2.049% [+1.534, +2.928] | 7.4338% | no |
| six-layer Tiny Instant-Only (SM130) | R12 | instant | +0.527% [+0.494, +0.564] | +1.104% [+1.042, +1.168] | +0.264% [+0.157, +0.788] | 4.2372% | no |
| six-layer Tiny Instant-Only (SM130) | Q2 | instant | +0.159% [+0.145, +0.174] | +0.735% [+0.684, +0.785] | +0.273% [+0.180, +0.713] | - | yes |
| six-layer Tiny Instant-Only (SM130) | Q4 | instant | +0.016% [+0.014, +0.019] | +0.595% [+0.549, +0.642] | +0.029% [-0.004, +0.186] | - | yes |

## By answer count before the review (GPU, 732 set, % vs own int8)

| model | label | head | 1-2 | 3-4 | 5-8 | 9-15 | 16-31 | 32-127 | 128- |
|---|---|---|---|---|---|---|---|---|---|
| Big two-head (SM91s2) | R1 | curve | +0.125% | +0.237% | +0.426% | +0.577% | +0.641% | +0.706% | +0.739% |
| Big two-head (SM91s2) | R1 | instant | +0.212% | +0.392% | +0.561% | +0.677% | +0.733% | +0.840% | +0.972% |
| Big two-head (SM91s2) | R12 | curve | +0.073% | +0.171% | +0.312% | +0.466% | +0.332% | +0.201% | +0.195% |
| Big two-head (SM91s2) | R12 | instant | +0.129% | +0.283% | +0.451% | +0.611% | +0.336% | +0.241% | +0.286% |
| Big two-head (SM91s2) | Q2 | curve | +0.050% | +0.042% | +0.085% | +0.089% | +0.118% | +0.149% | +0.165% |
| Big two-head (SM91s2) | Q2 | instant | +0.045% | +0.078% | +0.118% | +0.144% | +0.179% | +0.213% | +0.288% |
| Big two-head (SM91s2) | Q4 | curve | +0.011% | +0.008% | +0.015% | +0.014% | +0.017% | +0.027% | +0.017% |
| Big two-head (SM91s2) | Q4 | instant | +0.008% | +0.012% | +0.020% | +0.026% | +0.034% | +0.030% | +0.028% |
| Big Instant-Only (SM99) | R1 | instant | +0.212% | +0.403% | +0.615% | +0.757% | +0.834% | +0.929% | +1.348% |
| Big Instant-Only (SM99) | R12 | instant | +0.127% | +0.294% | +0.500% | +0.693% | +0.388% | +0.262% | +0.327% |
| Big Instant-Only (SM99) | Q2 | instant | +0.045% | +0.085% | +0.128% | +0.154% | +0.198% | +0.222% | +0.328% |
| Big Instant-Only (SM99) | Q4 | instant | +0.008% | +0.012% | +0.019% | +0.025% | +0.036% | +0.034% | +0.038% |
| Small two-head (SM118) | R1 | curve | +0.067% | +0.149% | +0.277% | +0.401% | +0.542% | +0.607% | +0.329% |
| Small two-head (SM118) | R1 | instant | +0.128% | +0.277% | +0.498% | +0.639% | +0.769% | +0.858% | +0.818% |
| Small two-head (SM118) | R12 | curve | +0.052% | +0.138% | +0.247% | +0.345% | +0.307% | +0.182% | -0.017% |
| Small two-head (SM118) | R12 | instant | +0.098% | +0.243% | +0.441% | +0.595% | +0.424% | +0.270% | +0.165% |
| Small two-head (SM118) | Q2 | curve | +0.016% | +0.021% | +0.045% | +0.064% | +0.084% | +0.084% | +0.062% |
| Small two-head (SM118) | Q2 | instant | +0.026% | +0.050% | +0.084% | +0.108% | +0.134% | +0.139% | +0.180% |
| Small two-head (SM118) | Q4 | curve | +0.002% | +0.003% | +0.004% | +0.006% | +0.012% | +0.013% | -0.086% |
| Small two-head (SM118) | Q4 | instant | +0.003% | +0.007% | +0.008% | +0.011% | +0.018% | +0.021% | +0.036% |
| Small Instant-Only (SM113) | R1 | instant | +0.149% | +0.398% | +0.567% | +0.664% | +0.862% | +1.046% | +1.268% |
| Small Instant-Only (SM113) | R12 | instant | +0.117% | +0.350% | +0.500% | +0.647% | +0.473% | +0.326% | +0.350% |
| Small Instant-Only (SM113) | Q2 | instant | +0.026% | +0.052% | +0.097% | +0.123% | +0.140% | +0.173% | +0.286% |
| Small Instant-Only (SM113) | Q4 | instant | +0.004% | +0.006% | +0.007% | +0.015% | +0.014% | +0.020% | +0.035% |
| six-layer Tiny two-head (SM130) | R1 | curve | +0.119% | +0.227% | +0.465% | +0.660% | +0.877% | +1.250% | +1.374% |
| six-layer Tiny two-head (SM130) | R1 | instant | +0.228% | +0.518% | +0.770% | +0.860% | +1.057% | +1.332% | +1.216% |
| six-layer Tiny two-head (SM130) | R12 | curve | +0.086% | +0.167% | +0.379% | +0.536% | +0.457% | +0.293% | +0.342% |
| six-layer Tiny two-head (SM130) | R12 | instant | +0.156% | +0.386% | +0.608% | +0.675% | +0.500% | +0.346% | +0.283% |
| six-layer Tiny two-head (SM130) | Q2 | curve | +0.023% | +0.039% | +0.071% | +0.108% | +0.153% | +0.218% | +0.340% |
| six-layer Tiny two-head (SM130) | Q2 | instant | +0.049% | +0.093% | +0.151% | +0.164% | +0.196% | +0.248% | +0.258% |
| six-layer Tiny two-head (SM130) | Q4 | curve | -0.002% | +0.007% | +0.007% | +0.013% | +0.017% | +0.017% | +0.011% |
| six-layer Tiny two-head (SM130) | Q4 | instant | +0.006% | +0.012% | +0.015% | +0.019% | +0.025% | +0.026% | +0.022% |
| six-layer Tiny Instant-Only (SM130) | R1 | instant | +0.259% | +0.561% | +0.749% | +0.972% | +1.326% | +1.723% | +2.310% |
| six-layer Tiny Instant-Only (SM130) | R12 | instant | +0.201% | +0.468% | +0.636% | +0.884% | +0.623% | +0.340% | +0.235% |
| six-layer Tiny Instant-Only (SM130) | Q2 | instant | +0.061% | +0.147% | +0.183% | +0.200% | +0.230% | +0.253% | +0.249% |
| six-layer Tiny Instant-Only (SM130) | Q4 | instant | +0.010% | +0.014% | +0.017% | +0.024% | +0.021% | +0.025% | +0.046% |

## Per-review and per-card |dR| vs int8 (Small and Tiny long-card users, CPU)

| model | label | head | reviews P50 / P99 / max | reviews > 0.10 | cards > 0.05 | cards > 0.10 |
|---|---|---|---|---|---|---|
| Small two-head (SM118) | R12 | curve | 0.0049 / 0.0561 / 0.5930 | 2028 (0.0627%) | 9.6685% | 0.6335% |
| Small two-head (SM118) | R12 | instant | 0.0052 / 0.0689 / 0.5274 | 6389 (0.1977%) | 16.2701% | 1.9664% |
| Small two-head (SM118) | R1 | curve | 0.0064 / 0.0715 / 0.7942 | 7219 (0.2234%) | 13.8579% | 1.7156% |
| Small two-head (SM118) | R1 | instant | 0.0067 / 0.0864 / 0.5272 | 17681 (0.5471%) | 19.7856% | 3.7843% |
| Small Instant-Only (SM113) | R12 | instant | 0.0057 / 0.0731 / 0.5502 | 8463 (0.2619%) | 18.8222% | 2.4824% |
| Small Instant-Only (SM113) | R1 | instant | 0.0073 / 0.0898 / 0.4884 | 20463 (0.6331%) | 22.1540% | 4.3645% |
| six-layer Tiny two-head (SM130) | R12 | curve | 0.0059 / 0.0675 / 0.7246 | 5575 (0.1725%) | 15.3537% | 1.7372% |
| six-layer Tiny two-head (SM130) | R12 | instant | 0.0062 / 0.0779 / 0.4541 | 11283 (0.3491%) | 21.4648% | 3.3043% |
| six-layer Tiny two-head (SM130) | R1 | curve | 0.0078 / 0.0927 / 0.7943 | 23972 (0.7417%) | 20.6076% | 4.5776% |
| six-layer Tiny two-head (SM130) | R1 | instant | 0.0083 / 0.1039 / 0.4602 | 37212 (1.1514%) | 26.4409% | 6.7571% |
| six-layer Tiny Instant-Only (SM130) | R12 | instant | 0.0068 / 0.0846 / 0.5724 | 16639 (0.5148%) | 22.8600% | 4.2372% |
| six-layer Tiny Instant-Only (SM130) | R1 | instant | 0.0091 / 0.1113 / 0.6179 | 48336 (1.4956%) | 27.0591% | 7.4338% |

## Speed (report only): review evaluations per second, model time

| model | where | REF | R1 | R12 | Q2 | Q4 |
|---|---|---|---|---|---|---|
| Big two-head (SM91s2) | gpu_732 | 76714 | 30824 (0.402x) | 30392 (0.396x) | 29769 (0.388x) | 28850 (0.376x) |
| Big two-head (SM91s2) | cpu_long | 913 | None | None | 866 (0.949x) | 832 (0.911x) |
| Big Instant-Only (SM99) | gpu_732 | 77646 | 30908 (0.398x) | 30727 (0.396x) | 29777 (0.383x) | 29192 (0.376x) |
| Big Instant-Only (SM99) | cpu_long | 924 | None | None | 863 (0.934x) | 844 (0.913x) |
| Small two-head (SM118) | gpu_732 | 124598 | 57718 (0.463x) | 56718 (0.455x) | 57783 (0.464x) | 56438 (0.453x) |
| Small two-head (SM118) | cpu_long | 1529 | 1466 (0.959x) | 1422 (0.93x) | 1451 (0.949x) | 1404 (0.918x) |
| Small Instant-Only (SM113) | gpu_732 | 126143 | 58229 (0.462x) | 57779 (0.458x) | 58007 (0.46x) | 57045 (0.452x) |
| Small Instant-Only (SM113) | cpu_long | 1540 | 1465 (0.951x) | 1436 (0.932x) | 1451 (0.942x) | 1406 (0.913x) |
| six-layer Tiny two-head (SM130) | gpu_732 | 295619 | 84726 (0.287x) | 83121 (0.281x) | 85607 (0.29x) | 83189 (0.281x) |
| six-layer Tiny two-head (SM130) | cpu_long | 3690 | 3425 (0.928x) | 3283 (0.89x) | 3369 (0.913x) | 3205 (0.869x) |
| six-layer Tiny Instant-Only (SM130) | gpu_732 | 300392 | 85449 (0.284x) | 84730 (0.282x) | 86248 (0.287x) | 84612 (0.282x) |
| six-layer Tiny Instant-Only (SM130) | cpu_long | 3725 | 3427 (0.92x) | 3326 (0.893x) | 3376 (0.906x) | 3191 (0.857x) |
