# Wider simulator search with time-aware scheduling (SM86): the 30-minute limit with desired retention 0.30-0.98, and a per-card time-aware policy

Extension of response-time prediction (SM82) stage 2 (CPU only). Same 150 users of the 732 set (150 with their own FSRS-6 fit), same simulator and cost models (A (actual), A (head), B, C; see stage2-result.md section 3): daily limit 1,800 s charged with each cost model's own costs, due reviews first, then new cards while time is left, 365 days, objective = memorized cards (sum of recall) at day 365. Desired retention 0.30-0.98 by 0.02 (stage 2: 0.60-0.98). Stage 2 reproduced on the overlapping grid: max relative difference 0.00000000 (and 0.00000000 against stage 2's simulator function). Run time 860 s.

## Takeaways

- Unlimited new cards (stage 2's setup): there is still no interior optimum. With desired retention 0.30-0.98, 85-88% of users sit at the 0.30 bound for every cost model (A (actual), A (head), B, C; median optimum 0.30, mean 0.33, nobody at 0.98). With an unlimited deck the sum of recall at day 365 rewards learning new cards over reviewing, so the cost models cannot move a bound optimum: the fixed-cost optimum loses only 0.24% (B) and 0.35% (C) on average (0% for 87% of users; 90th percentile 0.90% / 0.71%). Desired retention 0.90 would memorize about 65% fewer cards than the optimum. This setup says little about real scheduling.
- At most 10 new cards a day: the optimum is interior or near the top and spread out. Median optimum 0.92 (A (actual), A (head), B) and 0.90 (C); quartiles about 0.80-0.98; 26% of users at 0.98 under A, 27% under B and C; nobody at 0.30; 5-9% below 0.60. Time costs that depend on recall (B, C) move the optimum down slightly vs A (head): mean -0.007 (B), -0.011 (C); lower for 27% / 38% of users, higher for 17% / 18%.
- Cost of keeping the fixed-cost optimum when the truth is B or C: 0.70% fewer memorized cards on average for B (90th percentile 2.19%; none for 56% of users) and 1.82% for C (90th percentile 5.49%; none for 44%). Desired retention 0.90 loses 3.7% (B) / 3.6% (C) at the median and 12.6% / 11.9% on average, because some users overshoot their 30-minute limit and lose cards at high retention.
- Time-aware per-card policy (stability gain per predicted second, tuned offset) vs the best single desired retention (both tuned on the evaluated simulation): no gain. Mean -0.5% (time-unaware) / -0.3% (time-aware) under B, -0.3% / -0.2% under C, median 0, better for 34-43% of users. Time-aware beats the same policy with fixed costs by only +0.27% (B) and +0.13% (C) on average, with a wide spread (10th-90th percentile -0.6% to +1.3% for B, -1.0% to +1.4% for C). Against desired retention 0.90 it gains 24-26% on average (mean of per-user ratios, driven by users where 0.90 is a poor choice). The tuned offset is at a bound (-0.20 or 0.56) for 18-20% of users (median -0.04 to -0.06), so the offset range is a limit.
- Limits and extrapolation: this is the FSRS-6 simulator (the users' own fits) with the answer-time head's cost curves, not the RWKV model itself. Desired retention below about 0.6 is outside anything the users scheduled; the cost curve is held flat below its lowest bin (median recall about 0.34) and those bins come from stretched-interval counterfactual queries, so counterfactual intervals are extrapolation. The 10-new-cards cap is a modelling choice, the policy is a heuristic, and 150 users of the 732 set are used.

## 1. Single desired retention, 30-minute limit, unlimited new cards (stage 2's setup)

With an unlimited deck, every second not spent on reviews learns a new card, and with the FSRS-6 power curve (the users' own fits) an unreviewed card keeps much of its recall within a year; so the sum of recall at day 365 rewards breadth.

| costs | optimum, median | mean | quartiles | 10th-90th percentile | users at 0.30 | users at 0.98 | users below 0.60 (outside stage 2's grid) | memorized at day 365 at the optimum (median) |
|---|---|---|---|---|---|---|---|---|
| A (actual) | 0.300 | 0.333 | 0.30-0.30 | 0.30-0.46 | 0.860 | 0.000 | 0.960 | 18926.710 |
| A (head) | 0.300 | 0.331 | 0.30-0.30 | 0.30-0.48 | 0.853 | 0.000 | 0.960 | 18999.901 |
| B | 0.300 | 0.329 | 0.30-0.30 | 0.30-0.34 | 0.873 | 0.000 | 0.960 | 18447.479 |
| C | 0.300 | 0.329 | 0.30-0.30 | 0.30-0.33 | 0.880 | 0.000 | 0.967 | 18380.333 |

Distribution of the optimum (share of users; bins of three grid points):

| costs | 0.30-0.34 | 0.36-0.40 | 0.42-0.46 | 0.48-0.52 | 0.54-0.58 | 0.60-0.64 | 0.66-0.70 | 0.72-0.76 | 0.78-0.82 | 0.84-0.88 | 0.90-0.94 | 0.96-1.00 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A (actual) | 0.873 | 0.013 | 0.013 | 0.027 | 0.033 | 0.013 | 0.013 | 0.007 | 0.007 | 0.000 | 0.000 | 0.000 |
| A (head) | 0.873 | 0.013 | 0.007 | 0.047 | 0.020 | 0.020 | 0.013 | 0.007 | 0.000 | 0.000 | 0.000 | 0.000 |
| B | 0.900 | 0.007 | 0.007 | 0.020 | 0.027 | 0.013 | 0.007 | 0.020 | 0.000 | 0.000 | 0.000 | 0.000 |
| C | 0.900 | 0.007 | 0.007 | 0.027 | 0.027 | 0.007 | 0.000 | 0.020 | 0.007 | 0.000 | 0.000 | 0.000 |

- optimum A (actual) minus A (head): mean +0.002, median +0.00; higher for 7% of users, lower 5%, same 88%.
- optimum B minus A (head): mean -0.002, median +0.00; higher for 9% of users, lower 5%, same 87%.
- optimum C minus A (head): mean -0.002, median +0.00; higher for 7% of users, lower 7%, same 87%.
- if the costs are really B, keeping the fixed-cost (A (head)) optimum memorizes 0.24% fewer cards on average (median 0.00%, 90th percentile 0.90%; no loss for 87% of users); desired retention 0.90 memorizes 65.3% fewer than the optimum on average (median 70.8%).
- if the costs are really C, keeping the fixed-cost (A (head)) optimum memorizes 0.35% fewer cards on average (median 0.00%, 90th percentile 0.71%; no loss for 87% of users); desired retention 0.90 memorizes 65.3% fewer than the optimum on average (median 70.6%).

Memorized cards at day 365 by desired retention (median over users):

| desired retention | A (actual) | A (head) | B | C | cards learned, A (head) | cards learned, C | reviews done, C |
|---|---|---|---|---|---|---|---|
| 0.300 | 18869.493 | 18999.901 | 18447.479 | 18380.333 | 31937.500 | 31755.000 | 0.000 |
| 0.360 | 17901.432 | 17705.813 | 17151.670 | 16790.273 | 29496.000 | 28835.000 | 1011.000 |
| 0.420 | 16514.137 | 16295.661 | 15591.390 | 15591.335 | 26176.000 | 25148.000 | 7478.500 |
| 0.480 | 15093.466 | 15209.546 | 14354.281 | 13666.666 | 22836.500 | 21111.000 | 11536.000 |
| 0.540 | 13172.498 | 12677.231 | 11748.832 | 11450.734 | 18459.000 | 16820.000 | 13996.000 |
| 0.600 | 11272.313 | 10929.921 | 9753.950 | 9237.772 | 14896.000 | 12728.500 | 17255.500 |
| 0.660 | 9729.759 | 9430.630 | 8377.826 | 7699.806 | 11895.500 | 9763.000 | 19273.500 |
| 0.720 | 8410.475 | 8281.289 | 7319.451 | 7005.375 | 9957.500 | 8441.500 | 22056.500 |
| 0.780 | 7571.036 | 7666.135 | 6856.266 | 6480.465 | 8585.500 | 7287.000 | 24914.000 |
| 0.840 | 6676.802 | 6548.585 | 6061.675 | 5718.292 | 7196.500 | 6267.500 | 28600.500 |
| 0.900 | 5277.197 | 5209.704 | 5087.620 | 4910.589 | 5518.000 | 5183.000 | 32099.000 |
| 0.960 | 3197.048 | 3221.667 | 3357.465 | 3319.187 | 3297.000 | 3389.000 | 40465.500 |
| 0.980 | 2082.091 | 1986.994 | 2403.201 | 2581.847 | 2008.500 | 2609.500 | 49786.500 |

## 2. Single desired retention, 30-minute limit, at most 10 new cards a day

Same limit, but at most 10 new cards a day (3,650 in the year, the deck of stage 2's no-limit runs): time left over is not spent, so high retention costs cards only when the reviews exceed 30 minutes.

| costs | optimum, median | mean | quartiles | 10th-90th percentile | users at 0.30 | users at 0.98 | users below 0.60 (outside stage 2's grid) | memorized at day 365 at the optimum (median) |
|---|---|---|---|---|---|---|---|---|
| A (actual) | 0.920 | 0.863 | 0.81-0.98 | 0.64-0.98 | 0.000 | 0.260 | 0.060 | 3480.296 |
| A (head) | 0.920 | 0.864 | 0.82-0.98 | 0.62-0.98 | 0.000 | 0.260 | 0.053 | 3474.794 |
| B | 0.920 | 0.857 | 0.80-0.98 | 0.64-0.98 | 0.000 | 0.273 | 0.060 | 3461.768 |
| C | 0.900 | 0.853 | 0.78-0.98 | 0.64-0.98 | 0.000 | 0.273 | 0.087 | 3466.107 |

Distribution of the optimum (share of users; bins of three grid points):

| costs | 0.30-0.34 | 0.36-0.40 | 0.42-0.46 | 0.48-0.52 | 0.54-0.58 | 0.60-0.64 | 0.66-0.70 | 0.72-0.76 | 0.78-0.82 | 0.84-0.88 | 0.90-0.94 | 0.96-1.00 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A (actual) | 0.000 | 0.020 | 0.000 | 0.013 | 0.027 | 0.053 | 0.040 | 0.060 | 0.067 | 0.153 | 0.207 | 0.360 |
| A (head) | 0.000 | 0.013 | 0.000 | 0.000 | 0.040 | 0.067 | 0.033 | 0.047 | 0.080 | 0.153 | 0.207 | 0.360 |
| B | 0.000 | 0.020 | 0.007 | 0.000 | 0.033 | 0.053 | 0.060 | 0.040 | 0.093 | 0.153 | 0.167 | 0.373 |
| C | 0.000 | 0.020 | 0.007 | 0.007 | 0.053 | 0.040 | 0.053 | 0.060 | 0.087 | 0.120 | 0.173 | 0.380 |

- optimum A (actual) minus A (head): mean -0.001, median +0.00; higher for 18% of users, lower 19%, same 63%.
- optimum B minus A (head): mean -0.007, median +0.00; higher for 17% of users, lower 27%, same 56%.
- optimum C minus A (head): mean -0.011, median +0.00; higher for 18% of users, lower 38%, same 44%.
- if the costs are really B, keeping the fixed-cost (A (head)) optimum memorizes 0.70% fewer cards on average (median 0.00%, 90th percentile 2.19%; no loss for 56% of users); desired retention 0.90 memorizes 12.6% fewer than the optimum on average (median 3.7%).
- if the costs are really C, keeping the fixed-cost (A (head)) optimum memorizes 1.82% fewer cards on average (median 0.29%, 90th percentile 5.49%; no loss for 44% of users); desired retention 0.90 memorizes 11.9% fewer than the optimum on average (median 3.6%).

Memorized cards at day 365 by desired retention (median over users):

| desired retention | A (actual) | A (head) | B | C | cards learned, A (head) | cards learned, C | reviews done, C |
|---|---|---|---|---|---|---|---|
| 0.300 | 2035.265 | 2035.265 | 2035.265 | 2034.856 | 3650.000 | 3650.000 | 0.000 |
| 0.360 | 2100.013 | 2100.013 | 2093.607 | 2090.979 | 3650.000 | 3650.000 | 95.000 |
| 0.420 | 2208.085 | 2206.877 | 2199.447 | 2197.761 | 3650.000 | 3650.000 | 920.500 |
| 0.480 | 2352.521 | 2350.607 | 2335.033 | 2320.414 | 3650.000 | 3650.000 | 2072.000 |
| 0.540 | 2503.803 | 2502.042 | 2498.504 | 2484.082 | 3650.000 | 3650.000 | 3745.500 |
| 0.600 | 2669.034 | 2669.034 | 2669.034 | 2650.459 | 3650.000 | 3650.000 | 5445.000 |
| 0.660 | 2851.229 | 2851.229 | 2849.268 | 2839.741 | 3650.000 | 3650.000 | 7502.000 |
| 0.720 | 3017.801 | 3016.868 | 3016.121 | 3006.907 | 3650.000 | 3650.000 | 9380.000 |
| 0.780 | 3185.936 | 3187.661 | 3182.113 | 3170.083 | 3650.000 | 3650.000 | 12093.000 |
| 0.840 | 3327.583 | 3327.050 | 3322.083 | 3319.333 | 3650.000 | 3650.000 | 14981.000 |
| 0.900 | 3449.200 | 3447.623 | 3442.419 | 3438.115 | 3650.000 | 3642.000 | 18807.500 |
| 0.960 | 2883.703 | 2915.847 | 3001.335 | 3049.072 | 2981.000 | 3113.500 | 29358.500 |
| 0.980 | 1912.829 | 1916.095 | 2263.020 | 2435.599 | 1937.000 | 2461.000 | 37977.500 |

Extrapolation: the head's cost curve is binned over the curve-grid points (stage 2); below the lowest bin's mean recall (median over users 0.339) the cost and grade mix are held flat, and the lowest bins come mostly from the stretched-interval queries (up to 365 days), the least tested part of the counterfactual. Desired retention below about 0.6 is outside anything the users scheduled.

## 3. A time-aware per-card policy vs one desired retention (10 new cards a day, 30-minute limit)

Per card, after every review (and at learning): target recall r*(S, D) = the r (0.30-0.98) with the most expected FSRS-6 stability gain (in ln S) per predicted second (the review plus, for Again, the relearning step), then minus one per-user offset (-0.20 to 0.56 by 0.04) tuned on the evaluated simulation, as the single desired retention is: both are best cases. Time-aware: the policy's predicted costs are the truth's own recall-dependent costs (B, or C with the grade mix over recall). Time-unaware control: the same policy built with fixed per-grade costs (A (head)). Under B and C the predicted time depends on recall only, so cards differ only through their memory state (S, D); the time cost reshapes the target curve as a whole.

Untuned target r* at difficulty 5 (median over users), S = 1 / 10 / 100 days: A (head) costs 0.81 / 0.83 / 0.88; B 0.83 / 0.85 / 0.89; C 0.85 / 0.87 / 0.91.

| truth (charged costs) | policy's predicted costs | memorized at day 365, tuned offset (median) | untuned (offset 0) | offset, median | users at an offset bound | mean target recall | vs the best single desired retention, mean | median | users better | vs the single desired retention chosen with A (head), mean | vs desired retention 0.90, mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B_recall_cost | A (head) (time-unaware) | 3445.328 | 3310.374 | -0.060 | 0.200 | 0.883 | -0.005 | -0.000 | 0.340 | 0.002 | 0.260 |
| B_recall_cost | B (time-aware) | 3430.057 | 3339.358 | -0.040 | 0.193 | 0.893 | -0.003 | 0.000 | 0.380 | 0.004 | 0.263 |
| C_recall_cost_and_mix | A (head) (time-unaware) | 3423.228 | 3306.223 | -0.060 | 0.187 | 0.886 | -0.003 | 0.000 | 0.433 | 0.017 | 0.238 |
| C_recall_cost_and_mix | C (time-aware) | 3432.001 | 3364.103 | -0.040 | 0.180 | 0.890 | -0.002 | 0.000 | 0.367 | 0.018 | 0.239 |

- truth B: the time-aware policy vs the same policy with fixed costs (each with its tuned offset): mean +0.27%, median +0.01% memorized cards (10th-90th percentile -0.60% to +1.33%; better for 55% of users, worse for 35%).
- truth C: the time-aware policy vs the same policy with fixed costs (each with its tuned offset): mean +0.13%, median +0.00% memorized cards (10th-90th percentile -0.98% to +1.44%; better for 49% of users, worse for 41%).

## Limits

- Same simulator as stage 2: the memory is the users' FSRS-6 fits (not the RWKV model), learning is a fixed cost per new card, one relearning step per lapse, 365 days, objective = sum of recall at day 365 (it rewards breadth: more cards at lower recall). The new-card cap of sections 2-3 is a modelling choice, not something the users set.
- Costs: the answer-time head on richer frozen features (SM82c) as a per-user cost curve over recall (stage 2's B and C); below the lowest bin and at low desired retention the cost is extrapolated flat, and those bins come from the stretched-interval counterfactual queries. Desired retention 0.30-0.60 is outside the observed scheduling.
- Policy: one hand-built heuristic (stability gain per predicted second + one tuned offset), not an optimal policy (no dynamic programming over the limit); the offset and the single desired retention are both tuned per user on the evaluated simulation (best cases for both). 150 users of the 732 set.

