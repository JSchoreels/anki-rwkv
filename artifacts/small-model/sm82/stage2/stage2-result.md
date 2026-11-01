# Response-time prediction (SM82) stage 2: answer-time curves, fast-Good patterns, time-aware workload

Model: the frozen released small model (the leak-free small model (SM72s) 25 epochs + the 1-epoch 32k Muon cool-down) + the answer-time head on richer frozen features (SM82c), grade bitwise the released model's. Users: 150 sampled from the 732 set (minus never-fit), 4,141,508 test reviews on a later day than the card's previous one; curve grid on up to 3,000 random test reviews per user (436,110). Query: only the review's instant (query) row changes (the card's interval, cumulative interval, review day encodings and day of week; the head's elapsed-days feature); every earlier review, the session counters and the user's running time statistics stay as observed. Identity point (x1) vs the recorded time rows of the answer-time head on richer frozen features (SM82c): max relative LogLoss gap 0.00000000 over 150 users, same review counts True. Errors: 0.

## Takeaways

- What a scheduler could do with the curves: for the same card, a longer wait makes the answer slower and less likely to be fast. From 0.25x to 4x the real interval, recall falls from 0.885 to 0.768, expected time from 15.0 s to 17.9 s (about -5% to +13% vs the real interval) and the chance of a fast Good (under 3.9 s) from 0.137 to 0.105 (+7% to -18%). Per review, each 0.1 drop in recall adds about 2.1 s to the time of a passed review (user quartiles 1.0 to 5.6 s). The effect is modest (elasticity 0.073): time is a secondary cost next to recall, but it is real and differs a lot between users.
- Extrapolation check: at the intervals the scheduler actually chose, predicted and actual recall agree within about 0.002 in every interval-growth bucket (including growth above 4x: 0.805 vs 0.806), time within about 0.4 s, fast Good within about 0.006. So the model is calibrated where we can test it; it says nothing about stretched intervals that were never observed.
- Fast-Good patterns (descriptive, not causal): the share rises with position in the day (0.098 for the first 10 reviews, 0.236 from the 201st on), with more earlier reviews of the card, and when the interval grew 1.7-4x (0.21-0.22); it is lower for cards with earlier lapses (0.134 with one lapse vs 0.173 with none) and for slow users (0.029 in the slowest third vs 0.254 in the fastest).
- Workload, no time limit: with recall-dependent predicted costs (B) time per memorized card is about 8% higher than with fixed per-grade costs (A (head)) at desired retention 0.70-0.85 (ratio 1.077-1.079), 5% at 0.90 and equal at 0.95 (0.996), because low-recall reviews are slower. The ratio to the 0.90 cost falls less steeply with B for low retention (0.624 vs 0.603 at 0.70) and more steeply for high retention (1.333 vs 1.423 at 0.95), so high retention is relatively cheaper than the fixed-cost view says. Adding the changed grade mix (C) shifts this further the same way.
- Workload, 30-minute daily limit: the best desired retention does not move in the median (0.60 for every cost model, mean 0.615 to 0.624); it moves for 17-20% of users (up) versus 7-11% (down). Keeping the fixed-cost optimum when predicted costs are the truth loses 0.5% of memorized cards on average (0.75% with the grade mix; 90th percentile 1.4% and 2.7%). The robust change is the cost shape, not a new optimum.
- Limits: the counterfactual intervals are extrapolation (only the review's query row changes; session counters, other reviews and running statistics stay as observed), and the absolute-day grid is further from the real interval than the multiplier grid. The simulator uses each user's own FSRS-6 parameters; the RWKV model is not inside it, only its answer-time head turned into a per-user, per-grade cost curve over recall (recall bins mix card difficulty with same-card slowdown). Most optima sit at the 0.60 grid bound, so movement can only be read at the margin. The secondary objective, time to fluency, is not computed. 150 users of the 732 set; fast = under 3.9 s (bins 0-6 end at 3,891 ms), not 5.5 s.

## 1. Answer-time curves

User-averaged over the grid reviews (each user equal weight). Recall = 1 - P(Again) from the instant head; E time = expected recorded time (grade-mixed by the grade head, bin centres clipped at the user's cap); E time | pass = mixed over Hard/Good/Easy; P(fast Good) = P(Good) x P(under 3.9 s | Good). Fast = under 3,891 ms = the head's bins 0-6: the quantity the response-time prediction (SM82) and answer-time head on richer frozen features (SM82c) tables label 'under 5.5 s' (their bin 6 ends at 3.89 s; 5.48 s is the end of bin 7).

| query interval | median days | recall | E time (s) | E time pass (s) | E time Again (s) | P(Good) | P(fast Good) | E time vs x1 | E time pass vs x1 | P(fast Good) vs x1 |
|---|---|---|---|---|---|---|---|---|---|---|
| x0.25 | 2.45333 | 0.885 | 15.026 | 14.604 | 19.160 | 0.569 | 0.137 | -0.050 | -0.044 | 0.067 |
| x0.5 | 4.78333 | 0.871 | 15.308 | 14.837 | 19.306 | 0.555 | 0.134 | -0.033 | -0.029 | 0.043 |
| x0.75 | 7.12333 | 0.860 | 15.590 | 15.079 | 19.497 | 0.543 | 0.131 | -0.015 | -0.013 | 0.020 |
| x1 | 9.53 | 0.851 | 15.824 | 15.280 | 19.668 | 0.534 | 0.128 | 0.000 | 0.000 | 0.000 |
| x1.5 | 14.2833 | 0.830 | 16.420 | 15.800 | 20.097 | 0.512 | 0.121 | 0.038 | 0.034 | -0.054 |
| x2 | 19.06 | 0.816 | 16.762 | 16.092 | 20.355 | 0.499 | 0.118 | 0.059 | 0.053 | -0.084 |
| x3 | 28.59 | 0.789 | 17.400 | 16.636 | 20.831 | 0.474 | 0.110 | 0.100 | 0.089 | -0.139 |
| x4 | 38.12 | 0.768 | 17.883 | 17.042 | 21.196 | 0.455 | 0.105 | 0.130 | 0.115 | -0.179 |
| d1 | 1 | 0.906 | 14.573 | 14.226 | 18.977 | 0.583 | 0.144 | -0.079 | -0.069 | 0.126 |
| d3 | 3 | 0.875 | 15.458 | 14.999 | 19.527 | 0.556 | 0.131 | -0.023 | -0.018 | 0.023 |
| d7 | 7 | 0.839 | 16.222 | 15.640 | 19.979 | 0.524 | 0.122 | 0.025 | 0.024 | -0.047 |
| d14 | 14 | 0.799 | 17.047 | 16.314 | 20.564 | 0.486 | 0.112 | 0.077 | 0.068 | -0.126 |
| d30 | 30 | 0.748 | 18.087 | 17.173 | 21.307 | 0.440 | 0.101 | 0.143 | 0.124 | -0.213 |
| d90 | 90 | 0.679 | 19.556 | 18.408 | 22.374 | 0.383 | 0.087 | 0.236 | 0.205 | -0.319 |
| d365 | 365 | 0.626 | 20.689 | 19.390 | 23.188 | 0.345 | 0.078 | 0.307 | 0.269 | -0.392 |

Per review over the 8 interval multipliers, slope of the review's own curve (per user the median over its reviews, then the median over users): E time | pass rises by 2.070 s per 0.1 drop in recall (user quartiles 1.045, 2.070, 5.615); grade-mixed E time by 2.537 s; elasticity d ln E / d ln t = 0.073 (user quartiles 0.051, 0.073, 0.100).

Time vs predicted recall (every grid point of every review pooled per user, then user-averaged; users with at least 30 points in the bin):

| predicted recall | users | points | E time (s) | E time pass (s) | E time Again (s) | P(under 3.9 s given Good) | P(fast Good) |
|---|---|---|---|---|---|---|---|
| 0.00-0.50 | 144 | 818,187 | 22.855 | 20.839 | 23.873 | 0.135 | 0.023 |
| 0.50-0.60 | 144 | 308,612 | 21.600 | 20.052 | 23.515 | 0.130 | 0.042 |
| 0.60-0.70 | 146 | 411,779 | 20.791 | 19.538 | 23.153 | 0.139 | 0.054 |
| 0.70-0.80 | 147 | 590,273 | 19.621 | 18.689 | 22.480 | 0.151 | 0.070 |
| 0.80-0.85 | 147 | 415,355 | 18.462 | 17.769 | 21.763 | 0.163 | 0.085 |
| 0.85-0.90 | 149 | 563,926 | 17.622 | 17.103 | 21.321 | 0.175 | 0.097 |
| 0.90-0.93 | 150 | 453,098 | 16.578 | 16.206 | 20.627 | 0.191 | 0.112 |
| 0.93-0.96 | 150 | 610,897 | 15.555 | 15.304 | 19.941 | 0.210 | 0.129 |
| 0.96-0.98 | 150 | 593,124 | 14.352 | 14.210 | 19.049 | 0.241 | 0.153 |
| 0.98-1.00 | 149 | 1,776,126 | 11.658 | 11.611 | 16.660 | 0.340 | 0.222 |

User types (thirds of users by their median recorded time, and by their observed recall):

| split | group | users | range | recall x0.5 | recall x1 | recall x4 | E pass x0.5 (s) | E pass x1 (s) | E pass x2 (s) | E pass x4 (s) | P(fast Good) x1 | P(fast Good) x4 | recall at 1 d | recall at 365 d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| median recorded time | low third | 48 | 0.42-5.96 | 0.900 | 0.884 | 0.810 | 7.133 | 7.353 | 7.809 | 8.353 | 0.254 | 0.211 | 0.923 | 0.670 |
| median recorded time | middle third | 52 | 6.00-11.80 | 0.870 | 0.849 | 0.761 | 12.774 | 13.181 | 13.956 | 14.865 | 0.108 | 0.086 | 0.902 | 0.606 |
| median recorded time | high third | 50 | 11.84-45.00 | 0.844 | 0.821 | 0.735 | 24.379 | 25.073 | 26.266 | 27.647 | 0.029 | 0.023 | 0.894 | 0.604 |
| observed recall | low third | 50 | 0.56-0.82 | 0.763 | 0.733 | 0.621 | 17.295 | 17.690 | 18.403 | 19.191 | 0.090 | 0.071 | 0.826 | 0.456 |
| observed recall | middle third | 50 | 0.82-0.91 | 0.887 | 0.866 | 0.772 | 12.854 | 13.260 | 14.035 | 14.927 | 0.123 | 0.096 | 0.916 | 0.603 |
| observed recall | high third | 50 | 0.91-1.00 | 0.963 | 0.954 | 0.911 | 14.361 | 14.890 | 15.839 | 17.007 | 0.171 | 0.148 | 0.976 | 0.819 |

### Calibration on the intervals the scheduler actually chose

All test reviews of the sampled users at their real interval (x1). Interval growth = this interval / the card's previous later-day interval: the high-growth buckets are the observed reviews most like the long-interval extrapolation.

| interval growth | reviews | users | recall predicted | recall actual | time predicted (s) | time actual (s) | time predicted given the true grade (s) | under 3.9 s given the true grade predicted | under 3.9 s actual | fast Good predicted | fast Good actual | Good under 5.5 s actual |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| first later-day interval | 801,342 | 150 | 0.845 | 0.848 | 14.906 | 14.811 | 14.896 | 0.219 | 0.224 | 0.136 | 0.140 | 0.216 |
| < 0.5 | 363,965 | 150 | 0.858 | 0.859 | 13.707 | 13.608 | 13.713 | 0.259 | 0.267 | 0.149 | 0.154 | 0.233 |
| 0.5-0.9 | 162,823 | 150 | 0.850 | 0.851 | 14.829 | 14.582 | 14.823 | 0.248 | 0.257 | 0.124 | 0.130 | 0.194 |
| 0.9-1.1 | 302,841 | 150 | 0.842 | 0.843 | 14.899 | 14.460 | 14.899 | 0.230 | 0.234 | 0.110 | 0.114 | 0.180 |
| 1.1-1.7 | 649,034 | 148 | 0.905 | 0.907 | 14.029 | 13.753 | 14.022 | 0.256 | 0.257 | 0.126 | 0.130 | 0.195 |
| 1.7-2.5 | 674,935 | 150 | 0.895 | 0.896 | 13.243 | 13.205 | 13.238 | 0.283 | 0.288 | 0.208 | 0.214 | 0.304 |
| 2.5-4 | 728,943 | 150 | 0.899 | 0.901 | 11.874 | 11.828 | 11.867 | 0.310 | 0.314 | 0.214 | 0.218 | 0.307 |
| > 4 | 457,625 | 150 | 0.805 | 0.806 | 14.173 | 13.991 | 14.169 | 0.249 | 0.254 | 0.137 | 0.140 | 0.209 |

| interval (days) | reviews | users | recall predicted | recall actual | time predicted (s) | time actual (s) | time predicted given the true grade (s) | under 3.9 s given the true grade predicted | under 3.9 s actual | fast Good predicted | fast Good actual | Good under 5.5 s actual |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 773,724 | 150 | 0.848 | 0.850 | 14.351 | 14.180 | 14.346 | 0.233 | 0.239 | 0.155 | 0.159 | 0.242 |
| 2-3 | 664,894 | 150 | 0.872 | 0.874 | 13.821 | 13.641 | 13.816 | 0.251 | 0.259 | 0.152 | 0.159 | 0.238 |
| 4-7 | 806,998 | 150 | 0.888 | 0.889 | 13.045 | 12.933 | 13.042 | 0.269 | 0.274 | 0.155 | 0.159 | 0.236 |
| 8-21 | 872,955 | 150 | 0.888 | 0.889 | 13.520 | 13.460 | 13.513 | 0.267 | 0.272 | 0.157 | 0.161 | 0.238 |
| 22-60 | 545,481 | 149 | 0.874 | 0.876 | 13.820 | 13.664 | 13.808 | 0.270 | 0.272 | 0.167 | 0.170 | 0.243 |
| 61-180 | 307,729 | 140 | 0.848 | 0.848 | 14.613 | 14.286 | 14.614 | 0.274 | 0.273 | 0.183 | 0.184 | 0.256 |
| 181+ | 169,727 | 128 | 0.788 | 0.786 | 14.065 | 13.906 | 14.062 | 0.281 | 0.283 | 0.164 | 0.163 | 0.234 |

## 2. Fast-Good patterns (descriptive)

Fast Good = rated Good in under 3.9 s (the head's bins 0-6; the Good-under-5.5 s share is shown for the actual data). Pooled over the sampled users' test reviews; user avg = mean over users with at least 50 reviews in the bucket (removes part of the between-user mix). Correlations only: the scheduler chose these intervals.

**card's earlier review days**

| bucket | reviews | recall | fast Good share | fast Good share (user avg) | fast given Good | Good under 5.5 s share | mean time (s) | mean time (s, user avg) | fast Good predicted |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 801,342 | 0.848 | 0.140 | 0.121 | 0.253 | 0.216 | 14.811 | 16.395 | 0.136 |
| 2 | 641,505 | 0.875 | 0.157 | 0.131 | 0.273 | 0.237 | 13.861 | 15.685 | 0.155 |
| 3-4 | 942,632 | 0.889 | 0.181 | 0.138 | 0.303 | 0.264 | 13.244 | 15.287 | 0.178 |
| 5-8 | 888,879 | 0.875 | 0.171 | 0.136 | 0.302 | 0.248 | 13.600 | 15.407 | 0.167 |
| 9-16 | 538,117 | 0.861 | 0.148 | 0.114 | 0.300 | 0.217 | 13.600 | 15.451 | 0.143 |
| 17+ | 329,033 | 0.864 | 0.180 | 0.135 | 0.362 | 0.255 | 11.506 | 13.963 | 0.173 |

**card's earlier lapses**

| bucket | reviews | recall | fast Good share | fast Good share (user avg) | fast given Good | Good under 5.5 s share | mean time (s) | mean time (s, user avg) | fast Good predicted |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 2,533,487 | 0.903 | 0.173 | 0.138 | 0.299 | 0.252 | 13.408 | 15.261 | 0.169 |
| 1 | 680,179 | 0.839 | 0.134 | 0.109 | 0.250 | 0.207 | 14.898 | 16.483 | 0.131 |
| 2 | 333,240 | 0.822 | 0.143 | 0.102 | 0.268 | 0.217 | 14.395 | 16.696 | 0.138 |
| 3-5 | 399,535 | 0.810 | 0.157 | 0.100 | 0.299 | 0.234 | 13.303 | 16.310 | 0.150 |
| 6+ | 195,067 | 0.769 | 0.184 | 0.108 | 0.379 | 0.252 | 11.395 | 14.640 | 0.172 |

**card's first later-day interval (days)**

| bucket | reviews | recall | fast Good share | fast Good share (user avg) | fast given Good | Good under 5.5 s share | mean time (s) | mean time (s, user avg) | fast Good predicted |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2,487,855 | 0.875 | 0.182 | 0.136 | 0.306 | 0.266 | 13.080 | 15.867 | 0.178 |
| 2-3 | 668,766 | 0.877 | 0.151 | 0.130 | 0.277 | 0.225 | 14.183 | 15.676 | 0.147 |
| 4-7 | 548,580 | 0.882 | 0.136 | 0.128 | 0.281 | 0.203 | 13.647 | 15.092 | 0.135 |
| 8-14 | 163,996 | 0.848 | 0.105 | 0.116 | 0.239 | 0.161 | 15.275 | 16.199 | 0.102 |
| 15+ | 272,311 | 0.804 | 0.106 | 0.116 | 0.235 | 0.168 | 16.231 | 16.542 | 0.100 |

**interval growth (this interval / previous)**

| bucket | reviews | recall | fast Good share | fast Good share (user avg) | fast given Good | Good under 5.5 s share | mean time (s) | mean time (s, user avg) | fast Good predicted |
|---|---|---|---|---|---|---|---|---|---|
| < 0.5 | 363,965 | 0.859 | 0.154 | 0.122 | 0.278 | 0.233 | 13.608 | 15.756 | 0.149 |
| 0.5-0.9 | 162,823 | 0.851 | 0.130 | 0.103 | 0.260 | 0.194 | 14.582 | 16.169 | 0.124 |
| 0.9-1.1 | 302,841 | 0.843 | 0.114 | 0.111 | 0.235 | 0.180 | 14.460 | 15.928 | 0.110 |
| 1.1-1.7 | 649,034 | 0.907 | 0.130 | 0.131 | 0.287 | 0.195 | 13.753 | 15.031 | 0.126 |
| 1.7-2.5 | 674,935 | 0.896 | 0.214 | 0.146 | 0.325 | 0.304 | 13.205 | 14.818 | 0.208 |
| 2.5-4 | 728,943 | 0.901 | 0.218 | 0.150 | 0.337 | 0.307 | 11.828 | 14.447 | 0.214 |
| > 4 | 457,625 | 0.806 | 0.140 | 0.126 | 0.281 | 0.209 | 13.991 | 16.100 | 0.137 |

**this interval (days)**

| bucket | reviews | recall | fast Good share | fast Good share (user avg) | fast given Good | Good under 5.5 s share | mean time (s) | mean time (s, user avg) | fast Good predicted |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 773,724 | 0.850 | 0.159 | 0.130 | 0.263 | 0.242 | 14.180 | 16.177 | 0.155 |
| 2-3 | 664,894 | 0.874 | 0.159 | 0.124 | 0.282 | 0.238 | 13.641 | 15.771 | 0.152 |
| 4-7 | 806,998 | 0.889 | 0.159 | 0.128 | 0.295 | 0.236 | 12.933 | 15.416 | 0.155 |
| 8-21 | 872,955 | 0.889 | 0.161 | 0.128 | 0.295 | 0.238 | 13.460 | 15.452 | 0.157 |
| 22-60 | 545,481 | 0.876 | 0.170 | 0.136 | 0.311 | 0.243 | 13.664 | 15.308 | 0.167 |
| 61-180 | 307,729 | 0.848 | 0.184 | 0.146 | 0.327 | 0.256 | 14.286 | 15.007 | 0.183 |
| 181+ | 169,727 | 0.786 | 0.163 | 0.127 | 0.338 | 0.234 | 13.906 | 15.417 | 0.164 |

**position in the day's reviews**

| bucket | reviews | recall | fast Good share | fast Good share (user avg) | fast given Good | Good under 5.5 s share | mean time (s) | mean time (s, user avg) | fast Good predicted |
|---|---|---|---|---|---|---|---|---|---|
| 1-10 | 386,692 | 0.865 | 0.098 | 0.100 | 0.172 | 0.171 | 18.882 | 18.585 | 0.097 |
| 11-50 | 988,433 | 0.877 | 0.117 | 0.117 | 0.208 | 0.191 | 16.092 | 16.635 | 0.115 |
| 51-100 | 742,623 | 0.876 | 0.141 | 0.128 | 0.256 | 0.215 | 14.128 | 15.385 | 0.136 |
| 101-200 | 843,417 | 0.872 | 0.166 | 0.137 | 0.303 | 0.244 | 12.773 | 14.537 | 0.160 |
| 201+ | 1,180,343 | 0.862 | 0.236 | 0.160 | 0.417 | 0.318 | 10.138 | 13.181 | 0.230 |

## 3. Workload: fixed per-grade costs vs recall-dependent predicted costs

Simulator: an FSRS-6 card simulation with fsrs_optimizer's formulas and each user's own FSRS-6 parameters (the srs-benchmark FSRS-6 recency-weighted fit, last split; 150 users, default parameters for 0; median decay 0.151), one same-day relearning step rated Good after each lapse, no fuzz, common random numbers across cost models; 150 users = each sampled user's own cost model. Per user: learning cost = the user's mean recorded first-review time (median 17.8 s), relearning step = the user's mean same-day review time (median 12.4 s), first-rating mix from the user's first reviews, pass-rating mix from the user's test reviews. The model is not run inside the simulator: its answer-time head is turned into a per-user, per-grade cost curve over recall (the cheapest honest version).

- A (actual): fixed per-grade review costs = the user's mean recorded time per grade (median over users Again/Hard/Good/Easy 19.4, 17.5, 12.6, 9.9 s): the classic setup.
- A (head): fixed per-grade costs = the head's mean per grade at the observed intervals (19.9, 18.5, 12.8, 10.3 s).
- B: per-grade cost = the head's expected time for that grade at the card's current recall (per user, binned over every grid point of every review, linear interpolation in recall).
- C: B + the pass-rating mix (Hard/Good/Easy given recall) from the grade head (changes the stability updates).

Head cost curve (median over users):

| recall | Again (s) | Hard (s) | Good (s) | Easy (s) | Hard share of passes | Good share | Easy share |
|---|---|---|---|---|---|---|---|
| 0.600 | 21.775 | 21.793 | 17.157 | 15.308 | 0.301 | 0.552 | 0.018 |
| 0.700 | 21.404 | 21.244 | 16.521 | 14.085 | 0.286 | 0.599 | 0.017 |
| 0.800 | 21.214 | 20.064 | 15.786 | 13.141 | 0.240 | 0.633 | 0.017 |
| 0.900 | 19.545 | 18.422 | 14.516 | 12.366 | 0.180 | 0.651 | 0.021 |
| 0.950 | 18.024 | 17.207 | 13.149 | 10.777 | 0.130 | 0.665 | 0.021 |
| 0.980 | 16.320 | 14.759 | 10.846 | 8.993 | 0.098 | 0.646 | 0.044 |

### Objective (a) with no time limit: time per memorized card at a given recall level

10 new cards a day for 365 days (3,650 cards), every due review done. Time per memorized card = (first reviews + reviews + relearning steps) / the sum of recall over all cards at day 365. A and B share the same review trajectory (costs do not change the schedule without a time limit), so their ratio is purely the cost shape. Median over users, seconds:

| desired retention | memorized share at day 365 | A (actual) | A (head) | B | C | B / A (head) | C / A (head) |
|---|---|---|---|---|---|---|---|
| 0.700 | 0.820 | 69.497 | 70.526 | 78.399 | 81.240 | 1.078 | 1.132 |
| 0.800 | 0.888 | 83.398 | 82.312 | 92.787 | 96.469 | 1.079 | 1.137 |
| 0.850 | 0.920 | 91.701 | 94.347 | 102.685 | 109.167 | 1.077 | 1.125 |
| 0.900 | 0.948 | 115.891 | 113.466 | 120.531 | 120.227 | 1.047 | 1.076 |
| 0.950 | 0.974 | 162.949 | 166.388 | 164.182 | 164.360 | 0.996 | 0.969 |

Cost relative to desired retention 0.90 (median over users of the user's ratio):

| desired retention | A (actual) | A (head) | B | C |
|---|---|---|---|---|
| 0.500 | 0.459 | 0.458 | 0.448 | 0.443 |
| 0.550 | 0.473 | 0.466 | 0.458 | 0.468 |
| 0.600 | 0.517 | 0.518 | 0.504 | 0.535 |
| 0.650 | 0.559 | 0.558 | 0.559 | 0.592 |
| 0.700 | 0.603 | 0.603 | 0.624 | 0.653 |
| 0.750 | 0.667 | 0.664 | 0.686 | 0.714 |
| 0.800 | 0.749 | 0.746 | 0.767 | 0.792 |
| 0.850 | 0.842 | 0.841 | 0.863 | 0.885 |
| 0.900 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.950 | 1.420 | 1.423 | 1.333 | 1.275 |

With no time limit this ratio keeps falling down to desired retention 0.50 for every cost model (users at the 0.50 bound: A (actual) 73%, A (head) 73%, B 74%, C 74%): with the FSRS-6 power curve (the users' own fits), cards left unreviewed keep much of their recall within a year, so this ratio has no interior optimum and the optimum cannot move. The comparison that matters here is the cost shape (B / A above).

### Objective (a) as most recall for a given time: a 30-minute daily limit

Daily limit 1,800 s (fsrs_optimizer's default), charged with each cost model's own costs: due reviews first, the rest wait a day, then new cards while time is left (unlimited deck); 365 days; desired retention 0.60-0.98 by 0.02; the optimum = the most memorized cards (sum of recall) at day 365. Each cost model runs its own simulation.

| costs | optimal desired retention, median | mean | quartiles | memorized at day 365 at the optimum (median) | users with the optimum at a grid bound |
|---|---|---|---|---|---|
| A_fixed_actual | 0.600 | 0.618 | 0.60-0.60 | 11404.040 | 0.800 |
| A_fixed_head | 0.600 | 0.615 | 0.60-0.60 | 11017.212 | 0.787 |
| B_recall_cost | 0.600 | 0.618 | 0.60-0.62 | 9789.736 | 0.727 |
| C_recall_cost_and_mix | 0.600 | 0.624 | 0.60-0.60 | 9527.863 | 0.760 |

- optimum A_fixed_actual minus A (head): mean +0.003, median +0.00; higher for 14% of users, lower 9%, same 77%.
- optimum B_recall_cost minus A (head): mean +0.003, median +0.00; higher for 17% of users, lower 7%, same 76%.
- optimum C_recall_cost_and_mix minus A (head): mean +0.009, median +0.00; higher for 20% of users, lower 11%, same 69%.
- if the costs are really B_recall_cost, keeping the fixed-cost optimum memorizes 0.51% fewer cards on average (median 0.00%, 90th percentile 1.44%).
- if the costs are really C_recall_cost_and_mix, keeping the fixed-cost optimum memorizes 0.75% fewer cards on average (median 0.00%, 90th percentile 2.71%).

Memorized cards at day 365 by desired retention (median over users):

| desired retention | A (actual) | A (head) | B | C | cards learned, A (head) | cards learned, B |
|---|---|---|---|---|---|---|
| 0.600 | 11272.313 | 10929.921 | 9753.950 | 9237.772 | 14896.000 | 13218.000 |
| 0.640 | 10214.177 | 9933.020 | 8872.900 | 8322.937 | 13100.500 | 11273.500 |
| 0.680 | 9268.126 | 9015.588 | 7839.313 | 7431.763 | 11337.000 | 10204.000 |
| 0.720 | 8410.475 | 8281.289 | 7319.451 | 7005.375 | 9957.500 | 8866.000 |
| 0.760 | 7759.272 | 7890.100 | 7070.157 | 6584.861 | 9182.500 | 8126.500 |
| 0.800 | 7522.929 | 7550.977 | 6756.799 | 6262.295 | 8587.500 | 7573.000 |
| 0.840 | 6676.802 | 6548.585 | 6061.675 | 5718.292 | 7196.500 | 6585.000 |
| 0.880 | 5821.465 | 5809.720 | 5350.710 | 5114.190 | 6211.500 | 5793.500 |
| 0.920 | 4551.113 | 4697.733 | 4599.693 | 4404.408 | 4885.500 | 4805.500 |
| 0.960 | 3197.048 | 3221.667 | 3357.465 | 3319.187 | 3297.000 | 3433.000 |
| 0.980 | 2082.091 | 1986.994 | 2403.201 | 2581.847 | 2008.500 | 2430.500 |

Secondary objective (b), time to fluency: not computed (it needs the card-level model inside the simulator; the head's P(fast Good) depends on the card's history, which a recall-only cost curve does not carry).

## Limits

- Counterfactual query: only the review's instant row moves to the new interval. The session counters, the reviews and new cards between this card's reviews, and the user's running time statistics stay as observed, so a query far from the real interval is an extrapolation; the calibration tables check the observed intervals only (by interval length and by interval growth, the closest observed analogue of a stretched interval).
- The absolute-day grid (1-365 days) mixes cards whose real interval is far from the query (a card really seen after 58 days queried at 1 day); the multiplier grid (0.25x-4x) stays closer to what was observed.
- Fast = under 3.9 s (the head's bins 0-6), not 5.5 s: the response-time prediction (SM82) and answer-time head on richer frozen features (SM82c) tables label that same quantity "under 5.5 s", but their bin 6 ends at 3,891 ms (5,477 ms is the end of bin 7). The curve queries stored bins 0-6 only; a 5.5 s model curve needs a re-query that also stores bins 0-7 (about 25 min of GPU). The actual-data columns show both thresholds.
- Workload: the RWKV model does not run inside the simulator (too heavy for today). Its answer-time head becomes a per-user, per-grade cost curve over recall, applied to the simulator's FSRS-6 recall. The recall bins pool different cards, so part of the slope is card difficulty (hard cards are both slower and less recalled), not only the same card getting slower; the within-review slopes in section 1 are the same-card part.
- The memory model in the simulator is FSRS-6 (the users' own fits), not the RWKV model; learning steps are a constant cost per new card and each lapse gets one relearning step. Both objectives push the optimum to the bottom of the desired-retention range for most users, so whether the optimum moves can only be read at the margin; the robust result is the change in the cost shape.
- 150 users of the 732 set (seeded sample); the curve grid uses up to 3,000 random test reviews per user; patterns are descriptive (the scheduler chose the intervals) and pooled tables mix users (the user-averaged columns remove part of that).
- Objective (b), time to fluency, is not computed.

