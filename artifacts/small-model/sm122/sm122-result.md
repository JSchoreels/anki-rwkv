# Retrieval-effort probe (SM122): results

Inspired by Pyc & Rawson (2009) (the retrieval effort hypothesis: harder but successful retrievals produce better later memory). Design and pre-registered rules: `design.md`. Answer time = front + back, from the question shown to the grade pressed, capped at the preset's maximum answer seconds; in the models it is feature 8 on the history rows.

**Summary.** The paper (Pyc & Rawson 2009) says that harder (slower) successful recalls lead to better later memory. (1) What the models learned: doubling the recorded answer time of a success (answer time = front + back, from the question shown to the grade press) lowers the predicted recall at the next review, the opposite of the paper. For the show-time retrain (SM92) the instant R falls by 1.60% [-1.78, -1.44] and the curve R at 1 w by 1.66% and at 30 d by 2.10%; for the Tiny two-head model (SM106) 1.57% (curve 1 w 1.71%, 30 d 2.16%); for the show-time big model (SM91s2) 1.85% (curve 1 w 1.92%, 30 d 2.25%). Halving the time raises R by about the same amount. The three models agree. The models read a long answer time as a sign of a weaker memory, which is what the data mostly show before any later effect. (2) Real outcomes: beyond what the model already predicts, later recall barely depends on the duration of the previous success. The within-user fastest-to-slowest-fifth difference in residual is +0.23% [+0.03, +0.44] for the show-time retrain (SM92) and +0.19% [+0.00, +0.39] for the Tiny two-head model (SM106) on the instant head (slightly positive, the direction of the paper, but tiny), and +0.32% [-0.07, +0.70] for the show-time big model (SM91s2). The cross-fitted LogLoss gain of per-fifth offsets is +0.001% for Small and Tiny and negative for Big, so no model passes the rule (CI excluding 0 and at least 0.05%). (3) One more same-day success: the models raise the instant R at the next review by +0.63% (SM92), +0.50% (SM106) and +0.93% (SM91s2); the real later-day residual of two or more successes against one success is +0.03%, +0.07% and +0.35% with CIs spanning 0, so real data show no measurable effect. (4) Implications: the duration effect inside the models is real-sized (about 1.6-1.9% of R per doubling), so the simulator, which feeds sampled answer times back as inputs, must draw them from a realistic time model; the real outcomes do not support the models having a hidden retrieval-effort benefit to exploit in scheduling.

Users: a pre-registered random subsample of the 732-user standard readout set (`users.json`), first 20,000 reviews of each; targets at least 100 reviews apart; user-block bootstrap (1000 Poisson replicates). An effect counts only with the 95% CI excluding 0 and a size of at least 0.05%.

## Small two-head: the show-time retrain (SM92)

200 users, 1,219,157 later-day pairs (previous review Good or Easy, timed), 29,014 counterfactual targets.

### Question 1: counterfactual duration of the last success (change of predicted R, % of R)

| arm | mean log2 change | instant R at the next review | LogLoss change (instant) | curve R at the next review | curve R at 1 d | curve R at 1 w | curve R at 30 d |
|---|---|---|---|---|---|---|---|
| x0.5 | -1.00 | +1.790% [+1.591, +2.007] | +1.114% [+0.793, +1.427] | +1.785% [+1.596, +1.997] | +1.193% [+1.048, +1.349] | +1.948% [+1.737, +2.184] | +2.657% [+2.402, +2.945] |
| x2 | +0.95 | -1.600% [-1.776, -1.436] | +0.777% [+0.534, +1.038] | -1.587% [-1.759, -1.427] | -1.137% [-1.277, -1.006] | -1.664% [-1.852, -1.496] | -2.102% [-2.310, -1.913] |
| card's 10th percentile | -0.68 | +1.335% [+1.176, +1.511] | +2.020% [+1.640, +2.395] | +1.357% [+1.199, +1.530] | +0.896% [+0.776, +1.032] | +1.400% [+1.235, +1.569] | +1.921% [+1.716, +2.126] |
| card's 90th percentile | +0.86 | -1.428% [-1.588, -1.278] | +0.623% [+0.307, +0.915] | -1.399% [-1.558, -1.254] | -0.958% [-1.082, -0.845] | -1.469% [-1.637, -1.315] | -1.969% [-2.162, -1.782] |
| user's 10th percentile | -1.47 | +2.500% [+2.211, +2.795] | +2.268% [+1.753, +2.770] | +2.541% [+2.261, +2.830] | +1.692% [+1.466, +1.914] | +2.816% [+2.509, +3.163] | +3.888% [+3.518, +4.295] |
| user's 90th percentile | +1.49 | -1.981% [-2.190, -1.786] | +1.348% [+1.013, +1.686] | -1.963% [-2.174, -1.772] | -1.404% [-1.571, -1.249] | -2.079% [-2.301, -1.879] | -2.648% [-2.906, -2.416] |

By stratum (x2 arm; instant R at the next review / curve R at 1 w):

| stratum | n | instant | curve 1 w |
|---|---|---|---|
| all | 29,014 | -1.600% [-1.776, -1.436] | -1.664% [-1.852, -1.496] |
| Good | 24,460 | -1.599% [-1.785, -1.432] | -1.699% [-1.902, -1.517] |
| Easy | 4,554 | -1.606% [-1.992, -1.277] | -1.471% [-1.795, -1.179] |
| t is a same-day step | 13,523 | -2.178% [-2.424, -1.948] | -2.651% [-2.931, -2.393] |
| t is a later-day review | 15,491 | -1.162% [-1.314, -1.027] | -0.993% [-1.135, -0.872] |
| next after 1 d | 7,027 | -1.864% [-2.102, -1.619] | -2.550% [-2.834, -2.263] |
| next after 2-7 d | 10,408 | -1.596% [-1.794, -1.413] | -1.870% [-2.084, -1.662] |
| next after 8-30 d | 7,325 | -1.341% [-1.527, -1.163] | -1.111% [-1.286, -0.952] |
| next after > 30 d | 4,254 | -1.633% [-1.901, -1.395] | -0.978% [-1.155, -0.817] |

Spillover (x2 arm) on the 1,190,143 non-target pairs: +0.0325% [+0.0285, +0.0366].

### Question 2: does real later recall depend on the previous success's duration beyond the model?

**within user (duration vs the user's median for that grade)**, 1,219,157 rows. Quintile 1 = fastest.

| head | Q1 resid | Q2 | Q3 | Q4 | Q5 resid | Q5 - Q1 | slope per doubling | LogLoss gain of per-quintile offsets (user CI; user-day CI) |
|---|---|---|---|---|---|---|---|---|
| instant | -0.06% | +0.11% | +0.09% | -0.02% | +0.17% | +0.23% [+0.03, +0.44] | +0.05% [-0.02, +0.12] | +0.001% [-0.002, +0.003]; [-0.001, +0.003] |
| curve | +0.01% | +0.20% | +0.12% | -0.08% | +0.21% | +0.20% [-0.26, +0.73] | +0.06% [-0.12, +0.28] | +0.000% [-0.004, +0.004]; [-0.003, +0.003] |

Actual recall vs predicted by quintile (instant): Q1 90.88% vs 90.94%; Q2 89.70% vs 89.60%; Q3 87.63% vs 87.55%; Q4 85.66% vs 85.68%; Q5 84.69% vs 84.55%

**within card (duration vs the card's earlier median, at least 2 earlier)**, 774,482 rows. Quintile 1 = fastest.

| head | Q1 resid | Q2 | Q3 | Q4 | Q5 resid | Q5 - Q1 | slope per doubling | LogLoss gain of per-quintile offsets (user CI; user-day CI) |
|---|---|---|---|---|---|---|---|---|
| instant | -0.16% | +0.12% | +0.01% | -0.04% | -0.06% | +0.11% [-0.15, +0.39] | +0.04% [-0.04, +0.14] | -0.001% [-0.004, +0.003]; [-0.004, +0.003] |
| curve | -0.30% | +0.13% | -0.06% | -0.06% | -0.14% | +0.16% [-0.39, +0.89] | +0.11% [-0.14, +0.47] | -0.002% [-0.008, +0.004]; [-0.008, +0.004] |

Actual recall vs predicted by quintile (instant): Q1 89.43% vs 89.57%; Q2 89.46% vs 89.35%; Q3 88.73% vs 88.73%; Q4 87.69% vs 87.73%; Q5 86.97% vs 87.02%

### Question 3: one more same-day success

Model (counterfactual insertion, 600 s after the target): instant R at the next review +0.628% [+0.402, +0.843] (go); curve R at the next review +0.830% [+0.532, +1.103] (go); curve 1 d +0.743% [+0.506, +0.958] (go); curve 1 w +0.367% [+0.086, +0.640] (go); curve 30 d +1.408% [+0.958, +1.853] (go).

Real outcomes (residual = actual - predicted recall at the next later-day review, % of predicted; days without an Again):

| group | n | recall | instant pred | instant resid | curve resid |
|---|---|---|---|---|---|
| later day, 1 success | 772,966 | 91.67% | 91.65% | +0.02% [-0.05, +0.09] | -0.04% |
| later day, 2 successes | 14,599 | 90.00% | 89.72% | +0.31% [-0.22, +0.83] | +0.68% |
| later day, >= 3 successes | 2,891 | 78.55% | 79.65% | -1.37% [-3.49, +0.77] | -2.14% |
| first day, 2 successes | 167,225 | 92.46% | 92.41% | +0.06% [-0.09, +0.22] | +0.07% |
| first day, >= 3 successes | 25,472 | 81.72% | 81.43% | +0.36% [-0.16, +0.83] | -0.71% |

Instant: later day, 2 or more successes minus 1 success: +0.03% [-0.53, +0.50]; LogLoss gain of group offsets -0.000% [-0.003, +0.003].

Curve: later day, 2 or more successes minus 1 success: +0.28% [-0.69, +1.09]; LogLoss gain of group offsets +0.004% [-0.004, +0.013].

CPU: 16881 s for 200 users (84 s per user, all arms), peak 1.71 GB; errors: 0.

## Tiny two-head: the Tiny two-head model (SM106)

200 users, 1,219,157 later-day pairs (previous review Good or Easy, timed), 29,014 counterfactual targets.

### Question 1: counterfactual duration of the last success (change of predicted R, % of R)

| arm | mean log2 change | instant R at the next review | LogLoss change (instant) | curve R at the next review | curve R at 1 d | curve R at 1 w | curve R at 30 d |
|---|---|---|---|---|---|---|---|
| x0.5 | -1.00 | +1.736% [+1.542, +1.944] | +1.011% [+0.693, +1.316] | +1.828% [+1.638, +2.043] | +1.218% [+1.067, +1.376] | +1.999% [+1.779, +2.245] | +2.701% [+2.435, +3.004] |
| x2 | +0.95 | -1.567% [-1.738, -1.400] | +0.773% [+0.531, +1.022] | -1.623% [-1.804, -1.457] | -1.155% [-1.299, -1.019] | -1.708% [-1.908, -1.531] | -2.161% [-2.379, -1.958] |
| card's 10th percentile | -0.68 | +1.298% [+1.146, +1.464] | +1.932% [+1.573, +2.305] | +1.383% [+1.224, +1.559] | +0.908% [+0.785, +1.039] | +1.439% [+1.266, +1.618] | +1.967% [+1.754, +2.185] |
| card's 90th percentile | +0.86 | -1.409% [-1.567, -1.265] | +0.605% [+0.296, +0.900] | -1.441% [-1.605, -1.290] | -0.985% [-1.110, -0.871] | -1.521% [-1.695, -1.363] | -2.027% [-2.225, -1.834] |
| user's 10th percentile | -1.47 | +2.481% [+2.200, +2.777] | +2.246% [+1.692, +2.814] | +2.643% [+2.350, +2.943] | +1.738% [+1.513, +1.963] | +2.947% [+2.607, +3.298] | +4.040% [+3.633, +4.469] |
| user's 90th percentile | +1.49 | -1.986% [-2.194, -1.786] | +1.373% [+1.038, +1.695] | -2.023% [-2.248, -1.826] | -1.446% [-1.619, -1.284] | -2.148% [-2.396, -1.944] | -2.712% [-2.981, -2.477] |

By stratum (x2 arm; instant R at the next review / curve R at 1 w):

| stratum | n | instant | curve 1 w |
|---|---|---|---|
| all | 29,014 | -1.567% [-1.738, -1.400] | -1.708% [-1.908, -1.531] |
| Good | 24,460 | -1.567% [-1.746, -1.403] | -1.741% [-1.949, -1.554] |
| Easy | 4,554 | -1.570% [-1.923, -1.255] | -1.533% [-1.867, -1.221] |
| t is a same-day step | 13,523 | -2.138% [-2.373, -1.912] | -2.710% [-3.002, -2.436] |
| t is a later-day review | 15,491 | -1.134% [-1.283, -0.998] | -1.028% [-1.173, -0.899] |
| next after 1 d | 7,027 | -1.812% [-2.053, -1.570] | -2.588% [-2.868, -2.313] |
| next after 2-7 d | 10,408 | -1.568% [-1.758, -1.388] | -1.921% [-2.143, -1.706] |
| next after 8-30 d | 7,325 | -1.321% [-1.500, -1.146] | -1.156% [-1.346, -0.987] |
| next after > 30 d | 4,254 | -1.594% [-1.867, -1.351] | -1.018% [-1.198, -0.854] |

Spillover (x2 arm) on the 1,190,143 non-target pairs: +0.0328% [+0.0288, +0.0372].

### Question 2: does real later recall depend on the previous success's duration beyond the model?

**within user (duration vs the user's median for that grade)**, 1,219,157 rows. Quintile 1 = fastest.

| head | Q1 resid | Q2 | Q3 | Q4 | Q5 resid | Q5 - Q1 | slope per doubling | LogLoss gain of per-quintile offsets (user CI; user-day CI) |
|---|---|---|---|---|---|---|---|---|
| instant | -0.02% | +0.17% | +0.13% | +0.01% | +0.18% | +0.19% [+0.00, +0.39] | +0.04% [-0.03, +0.11] | +0.001% [-0.001, +0.003]; [-0.001, +0.003] |
| curve | +0.02% | +0.22% | +0.16% | -0.04% | +0.27% | +0.25% [-0.22, +0.78] | +0.08% [-0.10, +0.30] | +0.000% [-0.004, +0.004]; [-0.003, +0.004] |

Actual recall vs predicted by quintile (instant): Q1 90.88% vs 90.90%; Q2 89.70% vs 89.55%; Q3 87.63% vs 87.52%; Q4 85.66% vs 85.65%; Q5 84.69% vs 84.54%

**within card (duration vs the card's earlier median, at least 2 earlier)**, 774,482 rows. Quintile 1 = fastest.

| head | Q1 resid | Q2 | Q3 | Q4 | Q5 resid | Q5 - Q1 | slope per doubling | LogLoss gain of per-quintile offsets (user CI; user-day CI) |
|---|---|---|---|---|---|---|---|---|
| instant | -0.06% | +0.17% | +0.02% | -0.03% | -0.05% | +0.01% [-0.26, +0.29] | +0.01% [-0.07, +0.11] | -0.001% [-0.005, +0.002]; [-0.004, +0.002] |
| curve | -0.29% | +0.15% | -0.04% | -0.03% | -0.09% | +0.20% [-0.35, +0.93] | +0.13% [-0.13, +0.49] | -0.002% [-0.009, +0.005]; [-0.008, +0.004] |

Actual recall vs predicted by quintile (instant): Q1 89.43% vs 89.48%; Q2 89.46% vs 89.31%; Q3 88.73% vs 88.71%; Q4 87.69% vs 87.72%; Q5 86.97% vs 87.01%

### Question 3: one more same-day success

Model (counterfactual insertion, 600 s after the target): instant R at the next review +0.504% [+0.319, +0.692] (go); curve R at the next review +0.789% [+0.507, +1.066] (go); curve 1 d +0.886% [+0.655, +1.122] (go); curve 1 w +0.270% [-0.021, +0.541]; curve 30 d +0.929% [+0.506, +1.376] (go).

Real outcomes (residual = actual - predicted recall at the next later-day review, % of predicted; days without an Again):

| group | n | recall | instant pred | instant resid | curve resid |
|---|---|---|---|---|---|
| later day, 1 success | 772,966 | 91.67% | 91.62% | +0.06% [-0.02, +0.13] | +0.01% |
| later day, 2 successes | 14,599 | 90.00% | 89.67% | +0.37% [-0.18, +0.90] | +0.72% |
| later day, >= 3 successes | 2,891 | 78.55% | 79.54% | -1.24% [-3.31, +0.83] | -2.53% |
| first day, 2 successes | 167,225 | 92.46% | 92.39% | +0.08% [-0.06, +0.22] | +0.16% |
| first day, >= 3 successes | 25,472 | 81.72% | 81.36% | +0.44% [-0.05, +0.91] | -0.63% |

Instant: later day, 2 or more successes minus 1 success: +0.07% [-0.48, +0.53]; LogLoss gain of group offsets -0.001% [-0.003, +0.003].

Curve: later day, 2 or more successes minus 1 success: +0.21% [-0.85, +1.05]; LogLoss gain of group offsets +0.005% [-0.004, +0.014].

CPU: 16704 s for 200 users (83 s per user, all arms), peak 1.72 GB; errors: 0.

## Big two-head: the show-time big model (SM91s2)

60 users, 338,211 later-day pairs (previous review Good or Easy, timed), 8,323 counterfactual targets.

### Question 1: counterfactual duration of the last success (change of predicted R, % of R)

| arm | mean log2 change | instant R at the next review | LogLoss change (instant) | curve R at the next review | curve R at 1 d | curve R at 1 w | curve R at 30 d |
|---|---|---|---|---|---|---|---|
| x0.5 | -1.00 | +2.066% [+1.647, +2.553] | +1.335% [+0.710, +2.049] | +2.067% [+1.677, +2.505] | +1.388% [+1.104, +1.713] | +2.215% [+1.784, +2.704] | +2.825% [+2.317, +3.365] |
| x2 | +0.96 | -1.853% [-2.185, -1.546] | +0.659% [+0.248, +1.093] | -1.835% [-2.140, -1.530] | -1.320% [-1.569, -1.089] | -1.919% [-2.279, -1.583] | -2.249% [-2.622, -1.885] |

By stratum (x2 arm; instant R at the next review / curve R at 1 w):

| stratum | n | instant | curve 1 w |
|---|---|---|---|
| all | 8,323 | -1.853% [-2.185, -1.546] | -1.919% [-2.279, -1.583] |
| Good | 6,954 | -1.866% [-2.221, -1.556] | -1.969% [-2.368, -1.611] |
| Easy | 1,369 | -1.783% [-2.276, -1.339] | -1.664% [-2.095, -1.257] |
| t is a same-day step | 4,140 | -2.425% [-2.915, -1.973] | -2.901% [-3.462, -2.371] |
| t is a later-day review | 4,183 | -1.365% [-1.617, -1.136] | -1.165% [-1.383, -0.969] |
| next after 1 d | 1,993 | -1.911% [-2.322, -1.538] | -2.631% [-3.217, -2.105] |
| next after 2-7 d | 2,800 | -1.969% [-2.366, -1.592] | -2.304% [-2.728, -1.907] |
| next after 8-30 d | 2,044 | -1.609% [-2.019, -1.274] | -1.408% [-1.797, -1.107] |
| next after > 30 d | 1,486 | -1.897% [-2.365, -1.480] | -1.173% [-1.482, -0.902] |

Spillover (x2 arm) on the 329,888 non-target pairs: +0.0369% [+0.0305, +0.0457].

### Question 2: does real later recall depend on the previous success's duration beyond the model?

**within user (duration vs the user's median for that grade)**, 338,211 rows. Quintile 1 = fastest.

| head | Q1 resid | Q2 | Q3 | Q4 | Q5 resid | Q5 - Q1 | slope per doubling | LogLoss gain of per-quintile offsets (user CI; user-day CI) |
|---|---|---|---|---|---|---|---|---|
| instant | -0.16% | +0.17% | -0.01% | -0.00% | +0.16% | +0.32% [-0.07, +0.70] | +0.09% [-0.03, +0.20] | -0.013% [-0.024, -0.003]; [-0.021, -0.005] |
| curve | -0.03% | +0.42% | +0.04% | -0.05% | +0.34% | +0.36% [-0.35, +1.12] | +0.11% [-0.14, +0.36] | -0.019% [-0.037, -0.003]; [-0.033, -0.006] |

Actual recall vs predicted by quintile (instant): Q1 88.38% vs 88.52%; Q2 88.44% vs 88.29%; Q3 86.28% vs 86.29%; Q4 84.23% vs 84.23%; Q5 82.87% vs 82.73%

**within card (duration vs the card's earlier median, at least 2 earlier)**, 215,404 rows. Quintile 1 = fastest.

| head | Q1 resid | Q2 | Q3 | Q4 | Q5 resid | Q5 - Q1 | slope per doubling | LogLoss gain of per-quintile offsets (user CI; user-day CI) |
|---|---|---|---|---|---|---|---|---|
| instant | -0.18% | +0.18% | -0.03% | -0.13% | +0.07% | +0.25% [-0.42, +0.86] | +0.06% [-0.15, +0.25] | -0.007% [-0.020, +0.006]; [-0.015, +0.002] |
| curve | +0.20% | +0.33% | -0.04% | -0.09% | +0.20% | -0.01% [-0.88, +0.76] | -0.01% [-0.31, +0.24] | -0.007% [-0.018, +0.004]; [-0.018, +0.004] |

Actual recall vs predicted by quintile (instant): Q1 88.69% vs 88.85%; Q2 88.30% vs 88.14%; Q3 87.51% vs 87.53%; Q4 86.05% vs 86.16%; Q5 85.58% vs 85.52%

### Question 3: one more same-day success

Model (counterfactual insertion, 600 s after the target): instant R at the next review +0.934% [+0.618, +1.253] (go); curve R at the next review +1.641% [+1.247, +2.070] (go); curve 1 d +1.110% [+0.792, +1.472] (go); curve 1 w +1.262% [+0.872, +1.677] (go); curve 30 d +3.746% [+2.964, +4.712] (go).

Real outcomes (residual = actual - predicted recall at the next later-day review, % of predicted; days without an Again):

| group | n | recall | instant pred | instant resid | curve resid |
|---|---|---|---|---|---|
| later day, 1 success | 210,231 | 90.54% | 90.55% | -0.02% [-0.15, +0.11] | -0.08% |
| later day, 2 successes | 4,476 | 91.87% | 91.74% | +0.14% [-0.64, +0.87] | +1.53% |
| later day, >= 3 successes | 828 | 89.37% | 88.11% | +1.43% [-1.38, +4.37] | +2.39% |
| first day, 2 successes | 41,427 | 90.74% | 90.98% | -0.27% [-0.64, +0.04] | -0.49% |
| first day, >= 3 successes | 6,507 | 79.56% | 79.58% | -0.03% [-1.14, +1.06] | -1.83% |

Instant: later day, 2 or more successes minus 1 success: +0.35% [-0.45, +1.18]; LogLoss gain of group offsets -0.003% [-0.010, +0.005].

Curve: later day, 2 or more successes minus 1 success: +1.73% [-0.55, +3.48]; LogLoss gain of group offsets +0.003% [-0.030, +0.041].

CPU: 2882 s for 60 users (48 s per user, all arms), peak 2.16 GB; errors: 0.

