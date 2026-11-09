# The show-time error sensitivity test (SM100): result

Model: the show-time instant-only small model (SM96), instant head, 722 rule users of the 732 set. Units: points = % of the big model (1 leak-free epoch) normal-input instant LogLoss 0.275757. Full tables: `sm100-decision.md` / `sm100-decision.json`.

**Verdict: not negligible** (pre-registered: negligible needs the 95% CI of both +10 s and +60 s inside (-0.05, +0.05) points).

| show-time interval shift | instant LogLoss shifted | vs shift 0 (0.276287), points [95% CI] | mean abs dR | share abs dR > 0.01 |
|---|---|---|---|---|
| +10 s | 0.276534 | +0.090 [+0.069, +0.114] | 0.001213 | 0.02711 |
| +60 s | 0.277381 | +0.397 [+0.331, +0.476] | 0.004053 | 0.06703 |
| +300 s | 0.27934 | +1.107 [+0.951, +1.309] | 0.008344 | 0.10741 |

By show-time interval bucket (LogLoss change, % of bucket): < 10 min (14.0% of reviews) +0.559 / +2.473 / +6.864 for +10 / +60 / +300 s; 10 min - 1 day (22.0%) +0.001 / +0.004 / +0.039; >= 1 day (64.0%) +0.000 for all three (mean abs dR about 1e-6 at +10 s, 6e-6 at +60 s, 2.9e-5 at +300 s).

Timeout subset: 553 capped users, 1,010,999 timeout reviews = 3.36% of 30,126,145 benchmark reviews; failure rate 0.2007 on timeout reviews vs 0.1435 on the capped users' other reviews. On timeout reviews instant LogLoss is 0.336281 with normal inputs, 0.3369 with show-time inputs (+0.184% [+0.131, +0.243]), and 0.336298 with +60 s (-0.179% [-0.237, -0.126] vs show-time). The capped users' other reviews: show-time vs normal +0.058% [+0.040, +0.079].

Reproduction: the shift-0 and normal passes reproduce the rows of the show-time instant-only small model (SM96) exactly (max abs instant LogLoss difference 0.0 on 722 users; 0 missing dumps, 0 review_th mismatches).

Cost: the GPU ticket took 151 wall minutes (`gpu/tickets.jsonl`, 5 passes). Scorer speed per 723-user pass (last line): 57,617, 57,817, 57,855, 57,956 and 59,121 review-evals/s (passes in order shift 0, +60, +10, +300, normal).

## Interpretation

1. **What the verdict measures.** The pre-registered verdict is "not negligible". The test shifted EVERY query's interval, so it measures the model's genuine sensitivity to the interval, not the size of any leak. That sensitivity sits entirely in the < 10 min bucket (14% of reviews; learning steps, where seconds really matter). In the >= 1 day bucket (64% of reviews), |dR| is about 1e-6.
2. **The actual leak.** A leak can only exist on reviews whose recorded duration is wrong: timeouts (measured: 3.36% of reviews, failure rate 0.2007 vs 0.1435) and undo-regrades (not visible in the revlog, unmeasured). On timeout reviews, +60 s lowers LogLoss by 0.179% of those reviews' LogLoss; our reconstruction makes their interval too long, in the direction that helps.
   - Scaled to all reviews: 1,010,999 / 30,126,145 = 3.356% of reviews, times 0.179% = 0.006007%, i.e. about 3.36% x 0.18% = 0.006% of total LogLoss per 60 s of reconstruction error, roughly 0.006 points.
   - It grows with the true error size: the +300 s effect on the < 10 min bucket (+6.864%) is about 2.8x the +60 s effect (+2.473%).
   - This is an informal bound, not the pre-registered rule.
3. **Model comparisons are not affected.** Models scored on the same show-time inputs (e.g. the show-time instant-only small model (SM96) vs the show-time retrain of the release (SM92)) get the same reconstruction, so the error largely cancels in those differences.
4. **Anki is not affected.** The Anki fork queries at the real show time, so scheduling has no leak.
