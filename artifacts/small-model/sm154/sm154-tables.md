### Big two-head (SM91s2)

Budget: at most +0.15% vs its own int8 (95% CI upper bound, every head). Cost columns: point estimate and upper bound in % of LogLoss, first against its own int8 rows.

| format | state bytes vs int8 | curve vs own int8 | instant vs own int8 | passes budget | frontier status |
|---|---|---|---|---|---|
| int8 rows (REF) | +0.00% | +0.000 / +0.000 | +0.000 / +0.000 | yes | frontier |
| int4 cards below 16 answers (A16) | -25.54% | +0.006 / +0.011 | +0.005 / +0.006 | yes | curve: inside hull; instant: frontier |
| int4 cards below 32 answers (A32) | -27.22% | +0.011 / +0.017 | +0.011 / +0.013 | yes | dominated |
| int4 cards below 128 answers (A) | -27.74% | +0.014 / +0.020 | +0.015 / +0.019 | yes | dominated |
| compact int4 cards (C4) | -27.75% | +0.014 / +0.021 | +0.015 / +0.019 | yes | dominated |
| card-only int4 cards (CO4) | -32.85% | +0.131 / +0.162 | +0.161 / +0.184 | no | dominated |
| int4 cards and notes below 16 answers (AN16) | -40.25% | +0.008 / +0.014 | +0.012 / +0.013 | yes | inside hull |
| int3 at 1-2 answers, int4 below 16 (Y2N) | -42.75% | +0.012 / +0.021 | +0.016 / +0.018 | yes | dominated |
| int4 cards and notes below 32 answers (AN32) | -43.21% | +0.017 / +0.025 | +0.022 / +0.026 | yes | dominated |
| int3 cards at 1-4 answers, int4 below 16 (Y4C) | -43.48% | +0.008 / +0.017 | +0.015 / +0.016 | yes | inside hull |
| int4 cards and notes below 128 answers (AN) | -44.20% | +0.030 / +0.043 | +0.039 / +0.046 | yes | dominated |
| int3 at 1-4 answers, int4 below 16 (Y4N) | -45.34% | +0.013 / +0.024 | +0.023 / +0.025 | yes | dominated |
| int3 at 1-4 answers, int4 below 128 (B4) | -49.28% | +0.039 / +0.058 | +0.050 / +0.057 | yes | dominated |
| int3 at 1-8 answers, int4 below 128 (B8) | -52.60% | +0.048 / +0.071 | +0.064 / +0.071 | yes | dominated |
| int2 at 1-4 answers, int4 below 128 (D4) | -54.37% | +0.088 / +0.113 | +0.149 / +0.162 | no | dominated |
| rank 4 (Q4) | -74.41% | +0.014 / +0.020 | +0.018 / +0.021 | yes | frontier |
| rank 2 then 4 from 16 answers (T24) [the user's choice] | -81.72% | +0.034 / +0.043 | +0.054 / +0.059 | yes | frontier |
| rank 2 (Q2) | -82.78% | +0.080 / +0.101 | +0.109 / +0.122 | yes | inside hull |
| rank 1 then 2 from 16 answers (R12) | -87.56% | +0.262 / +0.331 | +0.344 / +0.368 | no | frontier |
| rank 1 (R1) | -88.08% | +0.403 / +0.507 | +0.501 / +0.543 | no | frontier |

Frontier on the curve head (CI upper bound), from int8 toward fewer bytes: int8 rows (REF) to rank 4 (Q4) saves 74.41% of the int8 bytes for +0.020% of cost, which is 0.0003% per saved %; rank 4 (Q4) to rank 2 then 4 from 16 answers (T24) [the user's choice] saves 7.31% of the int8 bytes for +0.023% of cost, which is 0.0031% per saved %; rank 2 then 4 from 16 answers (T24) [the user's choice] to rank 1 then 2 from 16 answers (R12) saves 5.84% of the int8 bytes for +0.288% of cost, which is 0.0493% per saved %; rank 1 then 2 from 16 answers (R12) to rank 1 (R1) saves 0.52% of the int8 bytes for +0.176% of cost, which is 0.3385% per saved %.
The budget line is crossed on the curve frontier at about -83.9% bytes vs int8.

Frontier on the instant head (CI upper bound), from int8 toward fewer bytes: int8 rows (REF) to int4 cards below 16 answers (A16) saves 25.54% of the int8 bytes for +0.006% of cost, which is 0.0002% per saved %; int4 cards below 16 answers (A16) to rank 4 (Q4) saves 48.87% of the int8 bytes for +0.015% of cost, which is 0.0003% per saved %; rank 4 (Q4) to rank 2 then 4 from 16 answers (T24) [the user's choice] saves 7.31% of the int8 bytes for +0.038% of cost, which is 0.0052% per saved %; rank 2 then 4 from 16 answers (T24) [the user's choice] to rank 1 then 2 from 16 answers (R12) saves 5.84% of the int8 bytes for +0.309% of cost, which is 0.0529% per saved %; rank 1 then 2 from 16 answers (R12) to rank 1 (R1) saves 0.52% of the int8 bytes for +0.175% of cost, which is 0.3365% per saved %.
The budget line is crossed on the instant frontier at about -83.4% bytes vs int8.

### Big Instant-Only (SM99)

Budget: at most +0.15% vs its own int8 (95% CI upper bound, every head). Cost columns: point estimate and upper bound in % of LogLoss, first against its own int8 rows.

| format | state bytes vs int8 | instant vs own int8 | passes budget | frontier status |
|---|---|---|---|---|
| int8 rows (REF) | +0.00% | +0.000 / +0.000 | yes | frontier |
| int4 cards below 16 answers (A16) | -25.54% | +0.006 / +0.007 | yes | frontier |
| int4 cards below 32 answers (A32) | -27.22% | +0.012 / +0.014 | yes | inside hull |
| int4 cards below 128 answers (A) | -27.74% | +0.016 / +0.020 | yes | dominated |
| compact int4 cards (C4) | -27.75% | +0.016 / +0.020 | yes | dominated |
| card-only int4 cards (CO4) | -32.85% | +0.166 / +0.187 | no | dominated |
| int4 cards and notes below 16 answers (AN16) | -40.25% | +0.013 / +0.015 | yes | inside hull |
| int3 at 1-2 answers, int4 below 16 (Y2N) | -42.75% | +0.018 / +0.020 | yes | dominated |
| int4 cards and notes below 32 answers (AN32) | -43.21% | +0.025 / +0.029 | yes | dominated |
| int3 cards at 1-4 answers, int4 below 16 (Y4C) | -43.48% | +0.016 / +0.018 | yes | inside hull |
| int4 cards and notes below 128 answers (AN) | -44.20% | +0.046 / +0.054 | yes | dominated |
| int3 at 1-4 answers, int4 below 16 (Y4N) | -45.34% | +0.025 / +0.028 | yes | dominated |
| int3 at 1-4 answers, int4 below 128 (B4) | -49.28% | +0.059 / +0.068 | yes | dominated |
| int3 at 1-8 answers, int4 below 128 (B8) | -52.60% | +0.072 / +0.081 | yes | dominated |
| int2 at 1-4 answers, int4 below 128 (D4) | -54.37% | +0.159 / +0.173 | no | dominated |
| rank 4 (Q4) | -74.41% | +0.019 / +0.022 | yes | frontier |
| rank 2 then 4 from 16 answers (T24) [the user's choice] | -81.72% | +0.059 / +0.065 | yes | frontier |
| rank 2 (Q2) | -82.78% | +0.118 / +0.132 | yes | inside hull |
| rank 1 then 2 from 16 answers (R12) | -87.56% | +0.381 / +0.408 | no | frontier |
| rank 1 (R1) | -88.08% | +0.550 / +0.592 | no | frontier |

Frontier on the instant head (CI upper bound), from int8 toward fewer bytes: int8 rows (REF) to int4 cards below 16 answers (A16) saves 25.54% of the int8 bytes for +0.007% of cost, which is 0.0003% per saved %; int4 cards below 16 answers (A16) to rank 4 (Q4) saves 48.87% of the int8 bytes for +0.015% of cost, which is 0.0003% per saved %; rank 4 (Q4) to rank 2 then 4 from 16 answers (T24) [the user's choice] saves 7.31% of the int8 bytes for +0.043% of cost, which is 0.0059% per saved %; rank 2 then 4 from 16 answers (T24) [the user's choice] to rank 1 then 2 from 16 answers (R12) saves 5.84% of the int8 bytes for +0.343% of cost, which is 0.0587% per saved %; rank 1 then 2 from 16 answers (R12) to rank 1 (R1) saves 0.52% of the int8 bytes for +0.184% of cost, which is 0.3538% per saved %.
The budget line is crossed on the instant frontier at about -83.2% bytes vs int8.

### Small two-head (SM118)

Budget: at most +0.5% total vs Big at int8 (95% CI upper bound, every head). Cost columns: point estimate and upper bound in % of LogLoss, first against its own int8 rows, then against Big two-head (SM91s2) at int8.

| format | state bytes vs int8 | curve vs own int8 | curve vs Big at int8 | instant vs own int8 | instant vs Big at int8 | passes budget | frontier status |
|---|---|---|---|---|---|---|---|
| int8 rows (REF) | +0.00% | +0.000 / +0.000 | +0.213 / +0.374 | +0.000 / +0.000 | +0.151 / +0.176 | yes | frontier |
| int4 cards below 16 answers (A16) | -25.36% | +0.006 / +0.008 | +0.219 / +0.380 | +0.008 / +0.009 | +0.159 / +0.184 | yes | curve: dominated; instant: inside hull |
| int4 cards below 32 answers (A32) | -27.04% | +0.010 / +0.013 | +0.223 / +0.384 | +0.014 / +0.016 | +0.165 / +0.190 | yes | dominated |
| int4 cards below 128 answers (A) | -27.55% | +0.015 / +0.019 | +0.228 / +0.389 | +0.019 / +0.023 | +0.170 / +0.196 | yes | dominated |
| compact int4 cards (C4) | -27.56% | +0.013 / +0.017 | +0.226 / +0.388 | +0.019 / +0.023 | +0.170 / +0.196 | yes | dominated |
| int4 cards and notes below 16 answers (AN16) | -36.32% | +0.007 / +0.008 | +0.220 / +0.381 | +0.011 / +0.013 | +0.162 / +0.187 | yes | dominated |
| card-only int4 cards (CO4) | -36.60% | +0.171 / +0.201 | +0.385 / +0.559 | +0.211 / +0.232 | +0.362 / +0.399 | no | dominated |
| int3 at 1-2 answers, int4 below 16 (Y2N) | -38.57% | +0.012 / +0.016 | +0.225 / +0.387 | +0.016 / +0.019 | +0.168 / +0.193 | yes | dominated |
| int4 cards and notes below 32 answers (AN32) | -38.94% | +0.013 / +0.016 | +0.226 / +0.387 | +0.020 / +0.023 | +0.171 / +0.197 | yes | dominated |
| int3 cards at 1-4 answers, int4 below 16 (Y4C) | -39.53% | +0.017 / +0.021 | +0.230 / +0.392 | +0.018 / +0.021 | +0.170 / +0.195 | yes | dominated |
| int4 cards and notes below 128 answers (AN) | -39.81% | +0.019 / +0.026 | +0.233 / +0.394 | +0.031 / +0.037 | +0.182 / +0.209 | yes | dominated |
| int3 at 1-4 answers, int4 below 16 (Y4N) | -40.91% | +0.017 / +0.022 | +0.231 / +0.393 | +0.022 / +0.024 | +0.173 / +0.198 | yes | dominated |
| int3 at 1-4 answers, int4 below 128 (B4) | -44.40% | +0.031 / +0.039 | +0.244 / +0.405 | +0.042 / +0.048 | +0.193 / +0.220 | yes | dominated |
| int3 at 1-8 answers, int4 below 128 (B8) | -47.40% | +0.039 / +0.048 | +0.253 / +0.416 | +0.056 / +0.062 | +0.207 / +0.233 | yes | dominated |
| int2 at 1-4 answers, int4 below 128 (D4) | -48.99% | +0.100 / +0.118 | +0.313 / +0.477 | +0.146 / +0.156 | +0.298 / +0.325 | yes | dominated |
| rank 4 (Q4) | -52.30% | +0.005 / +0.007 | +0.218 / +0.379 | +0.009 / +0.011 | +0.160 / +0.186 | yes | frontier |
| rank 2 then 4 from 16 answers (T24) [the user's choice] | -66.26% | +0.028 / +0.035 | +0.241 / +0.405 | +0.049 / +0.054 | +0.201 / +0.227 | yes | frontier |
| rank 2 (Q2) | -68.24% | +0.045 / +0.055 | +0.258 / +0.422 | +0.078 / +0.085 | +0.229 / +0.257 | yes | frontier |
| rank 1 then 2 from 16 answers (R12) | -77.37% | +0.206 / +0.227 | +0.419 / +0.581 | +0.339 / +0.362 | +0.490 / +0.525 | no | frontier |
| rank 1 (R1) | -78.36% | +0.285 / +0.319 | +0.499 / +0.666 | +0.456 / +0.495 | +0.608 / +0.659 | no | frontier |

Frontier on the curve head (CI upper bound), from int8 toward fewer bytes: int8 rows (REF) to rank 4 (Q4) saves 52.30% of the int8 bytes for +0.005% of cost, which is 0.0001% per saved %; rank 4 (Q4) to rank 2 then 4 from 16 answers (T24) [the user's choice] saves 13.96% of the int8 bytes for +0.026% of cost, which is 0.0019% per saved %; rank 2 then 4 from 16 answers (T24) [the user's choice] to rank 2 (Q2) saves 1.98% of the int8 bytes for +0.017% of cost, which is 0.0086% per saved %; rank 2 (Q2) to rank 1 then 2 from 16 answers (R12) saves 9.13% of the int8 bytes for +0.159% of cost, which is 0.0174% per saved %; rank 1 then 2 from 16 answers (R12) to rank 1 (R1) saves 0.99% of the int8 bytes for +0.085% of cost, which is 0.0859% per saved %.
The budget line is crossed on the curve frontier at about -72.7% bytes vs int8.

Frontier on the instant head (CI upper bound), from int8 toward fewer bytes: int8 rows (REF) to rank 4 (Q4) saves 52.30% of the int8 bytes for +0.010% of cost, which is 0.0002% per saved %; rank 4 (Q4) to rank 2 then 4 from 16 answers (T24) [the user's choice] saves 13.96% of the int8 bytes for +0.041% of cost, which is 0.0029% per saved %; rank 2 then 4 from 16 answers (T24) [the user's choice] to rank 2 (Q2) saves 1.98% of the int8 bytes for +0.030% of cost, which is 0.0152% per saved %; rank 2 (Q2) to rank 1 then 2 from 16 answers (R12) saves 9.13% of the int8 bytes for +0.268% of cost, which is 0.0294% per saved %; rank 1 then 2 from 16 answers (R12) to rank 1 (R1) saves 0.99% of the int8 bytes for +0.134% of cost, which is 0.1354% per saved %.
The budget line is crossed on the instant frontier at about -76.5% bytes vs int8.

### Small Instant-Only (SM113)

Budget: at most +0.5% total vs Big at int8 (95% CI upper bound, every head). Cost columns: point estimate and upper bound in % of LogLoss, first against its own int8 rows, then against Big Instant-Only (SM99) at int8.

| format | state bytes vs int8 | instant vs own int8 | instant vs Big at int8 | passes budget | frontier status |
|---|---|---|---|---|---|
| int8 rows (REF) | +0.00% | +0.000 / +0.000 | +0.016 / +0.042 | yes | frontier |
| int4 cards below 16 answers (A16) | -25.36% | +0.011 / +0.013 | +0.027 / +0.052 | yes | dominated |
| int4 cards below 32 answers (A32) | -27.04% | +0.017 / +0.019 | +0.033 / +0.059 | yes | dominated |
| int4 cards below 128 answers (A) | -27.55% | +0.022 / +0.026 | +0.038 / +0.064 | yes | dominated |
| compact int4 cards (C4) | -27.56% | +0.022 / +0.026 | +0.038 / +0.064 | yes | dominated |
| int4 cards and notes below 16 answers (AN16) | -36.32% | +0.014 / +0.016 | +0.030 / +0.056 | yes | dominated |
| card-only int4 cards (CO4) | -36.60% | +0.290 / +0.315 | +0.306 / +0.341 | yes | dominated |
| int3 at 1-2 answers, int4 below 16 (Y2N) | -38.57% | +0.023 / +0.026 | +0.039 / +0.064 | yes | dominated |
| int4 cards and notes below 32 answers (AN32) | -38.94% | +0.022 / +0.025 | +0.038 / +0.064 | yes | dominated |
| int3 cards at 1-4 answers, int4 below 16 (Y4C) | -39.53% | +0.027 / +0.029 | +0.043 / +0.068 | yes | dominated |
| int4 cards and notes below 128 answers (AN) | -39.81% | +0.033 / +0.038 | +0.049 / +0.076 | yes | dominated |
| int3 at 1-4 answers, int4 below 16 (Y4N) | -40.91% | +0.033 / +0.036 | +0.049 / +0.073 | yes | dominated |
| int3 at 1-4 answers, int4 below 128 (B4) | -44.40% | +0.052 / +0.058 | +0.068 / +0.094 | yes | dominated |
| int3 at 1-8 answers, int4 below 128 (B8) | -47.40% | +0.069 / +0.075 | +0.085 / +0.110 | yes | dominated |
| int2 at 1-4 answers, int4 below 128 (D4) | -48.99% | +0.212 / +0.226 | +0.228 / +0.253 | yes | dominated |
| rank 4 (Q4) | -52.30% | +0.009 / +0.011 | +0.025 / +0.050 | yes | frontier |
| rank 2 then 4 from 16 answers (T24) [the user's choice] | -66.26% | +0.055 / +0.061 | +0.071 / +0.096 | yes | frontier |
| rank 2 (Q2) | -68.24% | +0.085 / +0.094 | +0.101 / +0.127 | yes | frontier |
| rank 1 then 2 from 16 answers (R12) | -77.37% | +0.395 / +0.429 | +0.411 / +0.446 | yes | frontier |
| rank 1 (R1) | -78.36% | +0.526 / +0.570 | +0.542 / +0.589 | no | frontier |

Frontier on the instant head (CI upper bound), from int8 toward fewer bytes: int8 rows (REF) to rank 4 (Q4) saves 52.30% of the int8 bytes for +0.008% of cost, which is 0.0002% per saved %; rank 4 (Q4) to rank 2 then 4 from 16 answers (T24) [the user's choice] saves 13.96% of the int8 bytes for +0.046% of cost, which is 0.0033% per saved %; rank 2 then 4 from 16 answers (T24) [the user's choice] to rank 2 (Q2) saves 1.98% of the int8 bytes for +0.031% of cost, which is 0.0157% per saved %; rank 2 (Q2) to rank 1 then 2 from 16 answers (R12) saves 9.13% of the int8 bytes for +0.319% of cost, which is 0.0349% per saved %; rank 1 then 2 from 16 answers (R12) to rank 1 (R1) saves 0.99% of the int8 bytes for +0.143% of cost, which is 0.1444% per saved %.
The budget line is crossed on the instant frontier at about -77.7% bytes vs int8.

### six-layer Tiny two-head (SM130)

Budget: at most +1.0% total vs Big at int8 (95% CI upper bound, every head). Cost columns: point estimate and upper bound in % of LogLoss, first against its own int8 rows, then against Big two-head (SM91s2) at int8.

| format | state bytes vs int8 | curve vs own int8 | curve vs Big at int8 | instant vs own int8 | instant vs Big at int8 | passes budget | frontier status |
|---|---|---|---|---|---|---|---|
| int8 rows (REF) | +0.00% | +0.000 / +0.000 | +0.681 / +0.861 | +0.000 / +0.000 | +0.672 / +0.727 | yes | frontier |
| int4 cards below 16 answers (A16) | -18.96% | +0.008 / +0.009 | +0.689 / +0.870 | +0.010 / +0.012 | +0.683 / +0.738 | yes | curve: dominated; instant: inside hull |
| int4 cards below 32 answers (A32) | -20.21% | +0.011 / +0.014 | +0.692 / +0.874 | +0.016 / +0.018 | +0.688 / +0.744 | yes | dominated |
| int4 cards below 128 answers (A) | -20.59% | +0.015 / +0.018 | +0.696 / +0.878 | +0.020 / +0.023 | +0.692 / +0.747 | yes | dominated |
| int4 cards and notes below 16 answers (AN16) | -35.34% | +0.012 / +0.014 | +0.693 / +0.874 | +0.020 / +0.022 | +0.692 / +0.748 | yes | dominated |
| int3 at 1-2 answers, int4 below 16 (Y2N) | -37.53% | +0.021 / +0.027 | +0.702 / +0.883 | +0.031 / +0.034 | +0.703 / +0.759 | yes | dominated |
| int3 cards at 1-4 answers, int4 below 16 (Y4C) | -37.74% | +0.020 / +0.023 | +0.701 / +0.882 | +0.029 / +0.032 | +0.701 / +0.757 | yes | dominated |
| int4 cards and notes below 32 answers (AN32) | -38.01% | +0.019 / +0.022 | +0.700 / +0.881 | +0.032 / +0.036 | +0.704 / +0.761 | yes | dominated |
| int4 cards and notes below 128 answers (AN) | -38.91% | +0.032 / +0.039 | +0.713 / +0.895 | +0.054 / +0.061 | +0.726 / +0.782 | yes | dominated |
| int3 at 1-4 answers, int4 below 16 (Y4N) | -39.80% | +0.026 / +0.033 | +0.707 / +0.888 | +0.041 / +0.046 | +0.714 / +0.770 | yes | dominated |
| int3 at 1-8 answers, int4 below 16 (I3b) | -42.71% | +0.038 / +0.045 | +0.719 / +0.901 | +0.062 / +0.067 | +0.734 / +0.792 | yes | dominated |
| int3 at 1-4 answers, int4 below 128 (B4) | -43.38% | +0.046 / +0.055 | +0.727 / +0.910 | +0.076 / +0.084 | +0.749 / +0.807 | yes | dominated |
| int2 at 1-4 answers, int4 below 16 (I2a) | -44.27% | +0.140 / +0.163 | +0.822 / +1.001 | +0.224 / +0.240 | +0.897 / +0.959 | no | dominated |
| int3 at 1-8 answers, int4 below 128 (B8) | -46.28% | +0.059 / +0.069 | +0.740 / +0.923 | +0.097 / +0.105 | +0.770 / +0.828 | yes | dominated |
| int2 at 1-4 answers, int4 below 128 (D4) | -47.84% | +0.160 / +0.183 | +0.842 / +1.021 | +0.258 / +0.275 | +0.931 / +0.994 | no | dominated |
| int2 at 1-4 answers, int3 below 16 (I2b) | -48.64% | +0.176 / +0.202 | +0.858 / +1.038 | +0.282 / +0.300 | +0.956 / +1.019 | no | dominated |
| rank 4 (Q4) | -51.06% | +0.007 / +0.011 | +0.688 / +0.866 | +0.015 / +0.018 | +0.687 / +0.742 | yes | frontier |
| rank 2 then 4 from 16 answers (T24) [the user's choice] | -64.65% | +0.040 / +0.049 | +0.722 / +0.899 | +0.074 / +0.082 | +0.746 / +0.802 | yes | frontier |
| rank 2 (Q2) | -66.68% | +0.078 / +0.092 | +0.759 / +0.939 | +0.129 / +0.142 | +0.802 / +0.861 | yes | frontier |
| rank 1 then 2 from 16 answers (R12) | -75.57% | +0.306 / +0.344 | +0.989 / +1.175 | +0.443 / +0.478 | +1.118 / +1.188 | no | frontier |
| rank 1 (R1) | -76.58% | +0.478 / +0.548 | +1.162 / +1.355 | +0.684 / +0.751 | +1.361 / +1.454 | no | frontier |

Frontier on the curve head (CI upper bound), from int8 toward fewer bytes: int8 rows (REF) to rank 4 (Q4) saves 51.06% of the int8 bytes for +0.005% of cost, which is 0.0001% per saved %; rank 4 (Q4) to rank 2 then 4 from 16 answers (T24) [the user's choice] saves 13.59% of the int8 bytes for +0.033% of cost, which is 0.0024% per saved %; rank 2 then 4 from 16 answers (T24) [the user's choice] to rank 2 (Q2) saves 2.03% of the int8 bytes for +0.040% of cost, which is 0.0197% per saved %; rank 2 (Q2) to rank 1 then 2 from 16 answers (R12) saves 8.89% of the int8 bytes for +0.236% of cost, which is 0.0265% per saved %; rank 1 then 2 from 16 answers (R12) to rank 1 (R1) saves 1.01% of the int8 bytes for +0.180% of cost, which is 0.1782% per saved %.
The budget line is crossed on the curve frontier at about -69.0% bytes vs int8.

Frontier on the instant head (CI upper bound), from int8 toward fewer bytes: int8 rows (REF) to rank 4 (Q4) saves 51.06% of the int8 bytes for +0.015% of cost, which is 0.0003% per saved %; rank 4 (Q4) to rank 2 then 4 from 16 answers (T24) [the user's choice] saves 13.59% of the int8 bytes for +0.060% of cost, which is 0.0044% per saved %; rank 2 then 4 from 16 answers (T24) [the user's choice] to rank 2 (Q2) saves 2.03% of the int8 bytes for +0.059% of cost, which is 0.0291% per saved %; rank 2 (Q2) to rank 1 then 2 from 16 answers (R12) saves 8.89% of the int8 bytes for +0.327% of cost, which is 0.0368% per saved %; rank 1 then 2 from 16 answers (R12) to rank 1 (R1) saves 1.01% of the int8 bytes for +0.266% of cost, which is 0.2634% per saved %.
The budget line is crossed on the instant frontier at about -70.5% bytes vs int8.

### six-layer Tiny Instant-Only (SM130)

Budget: at most +1.0% total vs Big at int8 (95% CI upper bound, every head). Cost columns: point estimate and upper bound in % of LogLoss, first against its own int8 rows, then against Big Instant-Only (SM99) at int8.

| format | state bytes vs int8 | instant vs own int8 | instant vs Big at int8 | passes budget | frontier status |
|---|---|---|---|---|---|
| int8 rows (REF) | +0.00% | +0.000 / +0.000 | +0.579 / +0.625 | yes | frontier |
| int4 cards below 16 answers (A16) | -18.96% | +0.018 / +0.020 | +0.597 / +0.643 | yes | dominated |
| int4 cards below 32 answers (A32) | -20.21% | +0.026 / +0.030 | +0.606 / +0.652 | yes | dominated |
| int4 cards below 128 answers (A) | -20.59% | +0.035 / +0.040 | +0.614 / +0.660 | yes | dominated |
| int4 cards and notes below 16 answers (AN16) | -35.34% | +0.030 / +0.033 | +0.609 / +0.656 | yes | dominated |
| int3 at 1-2 answers, int4 below 16 (Y2N) | -37.53% | +0.043 / +0.047 | +0.622 / +0.670 | yes | dominated |
| int3 cards at 1-4 answers, int4 below 16 (Y4C) | -37.74% | +0.042 / +0.045 | +0.621 / +0.668 | yes | dominated |
| int4 cards and notes below 32 answers (AN32) | -38.01% | +0.047 / +0.052 | +0.627 / +0.674 | yes | dominated |
| int4 cards and notes below 128 answers (AN) | -38.91% | +0.078 / +0.089 | +0.657 / +0.704 | yes | dominated |
| int3 at 1-4 answers, int4 below 16 (Y4N) | -39.80% | +0.058 / +0.063 | +0.638 / +0.686 | yes | dominated |
| int3 at 1-8 answers, int4 below 16 (I3b) | -42.71% | +0.092 / +0.099 | +0.672 / +0.720 | yes | dominated |
| int3 at 1-4 answers, int4 below 128 (B4) | -43.38% | +0.108 / +0.118 | +0.687 / +0.735 | yes | dominated |
| int2 at 1-4 answers, int4 below 16 (I2a) | -44.27% | +0.313 / +0.334 | +0.894 / +0.949 | yes | dominated |
| int3 at 1-8 answers, int4 below 128 (B8) | -46.28% | +0.144 / +0.155 | +0.723 / +0.772 | yes | dominated |
| int2 at 1-4 answers, int4 below 128 (D4) | -47.84% | +0.364 / +0.384 | +0.945 / +1.001 | no | dominated |
| int2 at 1-4 answers, int3 below 16 (I2b) | -48.64% | +0.402 / +0.425 | +0.984 / +1.042 | no | dominated |
| rank 4 (Q4) | -51.06% | +0.016 / +0.019 | +0.595 / +0.642 | yes | frontier |
| rank 2 then 4 from 16 answers (T24) [the user's choice] | -64.65% | +0.088 / +0.098 | +0.668 / +0.717 | yes | frontier |
| rank 2 (Q2) | -66.68% | +0.155 / +0.170 | +0.735 / +0.785 | yes | frontier |
| rank 1 then 2 from 16 answers (R12) | -75.57% | +0.522 / +0.561 | +1.104 / +1.168 | no | frontier |
| rank 1 (R1) | -76.58% | +0.775 / +0.837 | +1.358 / +1.434 | no | frontier |

Frontier on the instant head (CI upper bound), from int8 toward fewer bytes: int8 rows (REF) to rank 4 (Q4) saves 51.06% of the int8 bytes for +0.017% of cost, which is 0.0003% per saved %; rank 4 (Q4) to rank 2 then 4 from 16 answers (T24) [the user's choice] saves 13.59% of the int8 bytes for +0.075% of cost, which is 0.0055% per saved %; rank 2 then 4 from 16 answers (T24) [the user's choice] to rank 2 (Q2) saves 2.03% of the int8 bytes for +0.068% of cost, which is 0.0335% per saved %; rank 2 (Q2) to rank 1 then 2 from 16 answers (R12) saves 8.89% of the int8 bytes for +0.383% of cost, which is 0.0431% per saved %; rank 1 then 2 from 16 answers (R12) to rank 1 (R1) saves 1.01% of the int8 bytes for +0.266% of cost, which is 0.2634% per saved %.
The budget line is crossed on the instant frontier at about -71.7% bytes vs int8.
