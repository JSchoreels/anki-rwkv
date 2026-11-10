# SM100b: model comparison without timeout reviews (decision)

Verdict: **unchanged** (|(b) - (a)| = 0.002 points; unchanged needs < 0.05). Pair: the show-time instant-only small model (SM96) minus the show-time retrain of the release (SM92), instant head, show-time inputs (e2s-query), 722 users of the 732 set, review-weighted, paired user bootstrap 2,000 (seed 0), 95% CI; points = % of the big model (1 leak-free epoch) normal-input instant LogLoss 0.275757. Negative = the show-time instant-only small model (SM96) better.

| reviews | review count | users | LogLoss SM96 | LogLoss SM92 | delta | points [95% CI] | users SM96 better / worse |
|---|---|---|---|---|---|---|---|
| (a) all reviews | 30,126,145 | 722 | 0.276287 | 0.276876 | -0.000589 | -0.214 [-0.232, -0.195] | 647 / 75 |
| (b) without timeout reviews | 29,115,146 | 722 | 0.274182 | 0.274766 | -0.000583 | -0.212 [-0.230, -0.193] | 629 / 93 |
| (c) timeout reviews only | 1,010,999 | 551 | 0.3369 | 0.337651 | -0.000751 | -0.272 [-0.344, -0.200] | 365 / 186 |

(a) reproduces the target [-0.214, -0.233, -0.195]: True (report estimator on the rows: -0.214 [-0.233, -0.195]); the dump estimator's (a) is within 0.002 of it: True. Timeout reviews 1,010,999 = 3.36% of 30,126,145.

the show-time instant-only small model (SM96), instant LogLoss, show-time inputs: {"all": 0.276287, "without_timeouts": 0.274182, "timeouts_only": 0.3369}; normal inputs: {"all": 0.276134, "without_timeouts": 0.274046, "timeouts_only": 0.336281}.
the show-time retrain of the release (SM92), show-time inputs: {"all": 0.276876, "without_timeouts": 0.274766, "timeouts_only": 0.337651}.

Reproduction of the show-time retrain of the release (SM92)'s existing e2s rows: {"users_compared": 722, "max_abs_diff_new_rows_vs_old_rows": 0.0, "rows_bit_for_bit": true, "max_abs_diff_dumped_vs_old_rows": 4.983135763192426e-07}.
