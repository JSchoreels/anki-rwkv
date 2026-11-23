# State-format rate-distortion table (SM154): result

Step 0 of idea 3 in `artifacts/ideas-1123/OPPORTUNITIES.md` (rate-distortion allocation of rank and factor precision). CPU only, from existing result rows, no scoring. Rebuild with `python sm154_table.py` (needs numpy; it reads the committed rows under `C:/Users/jonat/workspace/wt/sm1xx`). Output: `sm154-tables.json` and `sm154-tables.md` (the same tables as below).

## What was tabulated

For each of the six release models (Big two-head (SM91s2), Big Instant-Only (SM99), Small two-head (SM118), Small Instant-Only (SM113), six-layer Tiny two-head (SM130), six-layer Tiny Instant-Only (SM130)) every state format that was scored on the 732 set with show-time inputs, from these experiments: compact state for Small and Big (SM131), tiered card state by answer count (SM141), earlier int8 switch (SM144, it has rows: four labels), int3 young tier on the early switch (SM145), aggressive state tiers for Tiny (SM146), low-rank int8 card and note states (SM148) and rank-1 int8 states (SM150). That is 20 formats for each Big and Small model and 21 for each six-layer Tiny model, including the int8 reference (`w8+s8hx8`).

The cost of a format is the change in LogLoss in % of the reference LogLoss, review-weighted, with a paired user bootstrap (2,000 draws, seed 0, 95% CI), per head, as in every 732-set read-out. Two references are used. The first is the model's own int8 rows. The second, for Small and Tiny only, is the matching Big model at int8: Big two-head (SM91s2) for the two-head models and Big Instant-Only (SM99) for the Instant-Only models. The budgets judge the second one: Big at most +0.15% vs its own int8, Small at most +0.5% and Tiny at most +1.0% total vs Big at int8, on the 95% CI upper bound of every head. State bytes are the bytes of all answered cards and notes of the 732 set, in % of the int8 total (negative means smaller); they come from the experiments' own byte files, and the compact formats were computed with the same entity counts and the same layout.

Every row set was rescored from the committed per-user rows, not copied from the reports. All formats share one user set: the 718 users that have a finite row in every row set (the two Big models were scored on 718 of 720 users in the low-rank test (SM148) because the five largest users ran out of GPU memory, so the other models were cut to the same 718). The check block in `sm154-tables.json` compares 33 published numbers of the 11-23 verdict of the low-rank test (SM148) (cost against Big at int8 for the reference, rank 4 (Q4), rank 2 then 4 (T24) and rank 2 (Q2)); the upper bounds agree to within 0.004%. The same label scored again by a later experiment (for example the int8 rows) agrees to within 1e-6 in LogLoss per user.

## Rows that are left out (not comparable, not in any hull)

- The compact state for Tiny (SM129) rows: they were scored on earlier one-layer Tiny checkpoints, not on the six-layer Tiny models (SM130), and there are no int8 rows of those checkpoints on the same pool. The six-layer Tiny models get the same compact int4 cards from the tiered scheme (A) of the tiered card state by answer count (SM141).
- The fork default (`w8+s4c8n`, int4 cards with float32 row scales) of the compact state for Small and Big (SM131) and of the compact state for Tiny (SM129): its int8 base is the older layout with float32 scales and larger shift vectors, so its bytes are not on the same scale as the release int8 rows, and it overflows for Big user 3876.
- Formats that only have bytes and were never scored: Y8N, Y8C and Y2C of the int3 young tier on the early switch (SM145), and three report-only labels of the aggressive state tiers for Tiny (SM146).
- Smaller shift vectors and factors on T24 states (SM152) was still running; there are no rows.
- Not part of the cost measure: speed (the low-rank labels ran at 0.28 to 0.46 times int8 speed on the GPU, 0.86 to 0.96 on the CPU) and the long-card warnings (cards with 128 or more answers) of the rank-1 int8 states (SM150) and the low-rank test (SM148). The hull uses the dataset-wide 732-set cost only.

Reading the frontier columns: "frontier" means the format is a vertex of the lower convex hull of bytes against the upper-bound cost for that head; "dominated" means another format has at most the same bytes and at most the same cost; "inside hull" means no format dominates it but a mix of two frontier formats would beat it. The cost per saved % of bytes is the added cost (in % of LogLoss) divided by the % of the int8 bytes saved between two neighbouring frontier formats.

## Tables per model

Each cell is point / upper bound of the 95% CI, in %. Rows run from the most bytes (int8) to the fewest. T24 is rank 2 below 16 answers and rank 4 from 16 answers (`w8+lr2.16.4q8x8`), the format the user chose.

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

## What this suggests

The frontier is the low-rank family for every model. The tiered integer formats (int4 and int3 by answer count) reach only -25% to -54% of the int8 bytes, and all of them are dominated or inside the hull once a low-rank format is on the table: rank 4 (Q4) saves -51% to -74% of the bytes and costs the same order as the best tiered formats (at most +0.021% against its own int8 for Big, +0.011% for Small, +0.019% for Tiny). The only tiered format that stays on a frontier is the int4-below-16-answers cards (A16), for the two Big models, and only because its cost is almost zero (+0.011% upper bound) for -25.5% of the bytes. The card-only int4 format (CO4) is dominated everywhere, and the int2 and int3 young tiers cost more than the bytes they add.

The frontier bends in two places. The first is just after rank 4 (Q4): from int8 to rank 4 (Q4) each saved % of bytes costs 0.0001% to 0.0003% of LogLoss (essentially nothing), and from rank 4 (Q4) to rank 2 then 4 (T24) it costs 0.002% to 0.006% per saved %, a factor of about 10 to 20 more. The second is just after rank 2 (Q2): the next step, to rank 1 then 2 (R12), costs 0.017% to 0.06% per saved %, and rank 1 (R1) 0.09% to 0.35%. The user's choice, rank 2 then 4 (T24), is on the frontier for every model and head, at the first bend. For Big it is the last vertex before the steep part (rank 2 (Q2) falls inside the Big hull: it saves only 1.1% more of the bytes than T24 for +0.06% more cost); for Small and Tiny it is the vertex between rank 4 (Q4) and rank 2 (Q2).

Against the budgets, rank 2 then 4 (T24) leaves these margins on the upper bound: Big two-head (SM91s2) +0.043% curve and +0.059% instant against +0.15%, Big Instant-Only (SM99) +0.065%; Small two-head (SM118) +0.405% curve and +0.227% instant against +0.5%, Small Instant-Only (SM113) +0.096%; six-layer Tiny two-head (SM130) +0.899% curve and +0.802% instant against +1.0%, six-layer Tiny Instant-Only (SM130) +0.717%. The budget line is crossed on the frontier at about -83% to -84% (Big two-head (SM91s2) instant head at -83.4%, Big Instant-Only (SM99) at -83.2%), at -72.7% on the curve head of Small two-head (SM118) and -77.7% for Small Instant-Only (SM113), and at -69.0% on the curve head of six-layer Tiny two-head (SM130) and -71.7% for six-layer Tiny Instant-Only (SM130). The room left under the budgets, in bytes beyond T24, is therefore about 1.5% to 1.7% of the int8 bytes for Big, 6.5% to 11.5% for Small and 4.3% to 7.0% for Tiny, at a cost of 0.009% to 0.06% per saved %.

Where allocation per head could buy most. The size gap eats most of the Small and Tiny budgets before any state format is applied: int8 rows of Small two-head (SM118) already sit at +0.374% curve, six-layer Tiny two-head (SM130) at +0.861% curve and six-layer Tiny Instant-Only (SM130) at +0.625%. The room for the format is then only +0.13% (Small two-head (SM118) curve), +0.14% (six-layer Tiny two-head (SM130) curve) and +0.27% to +0.46% for the other heads. The curve head of the two-head models is the binding head for Small and Tiny, and the instant head is the binding head for Big (against its own int8, the instant head costs 30% to 90% more than the curve head for the same format). A format only moves along the frontier. A per-WKV-head allocation can beat the straight line between two frontier vertices only if the cost is concentrated in a few heads, which is what the low-rank structure probe (SM147) found according to the ideas file (Big card layer 2 head 1 keeps 84% of the energy at rank 2 while other layers keep 96% to 98.7%). The segment to aim at is rank 4 (Q4) to rank 2 (Q2): both ends are on the frontier for Small and Tiny, the chord costs 0.002% to 0.03% per saved %, and giving rank 4 only to the weakest heads could land below that chord (rank-2 bytes at a cost closer to rank 4). Below rank 2 a per-head search is not worth it, because rank 1 then 2 (R12) already costs several times more per saved % than the T24 to Q2 step. Big is where a gain would show first, because its margin is thinnest (rank 2 (Q2) on Big Instant-Only (SM99) is +0.132% against +0.15%).

Limits: the hull uses the upper bound of the dataset-wide cost on 718 users and ignores speed and the long-card warnings; the five largest users are missing for every model; and the cost per saved % between close vertices is noisy, because both ends carry a bootstrap error of about 0.01%.
