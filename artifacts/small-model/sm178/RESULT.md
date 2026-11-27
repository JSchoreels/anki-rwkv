# Today's models on donor data (SM178): result

Status: DONE 11-27 16:48 ("SM178 DONE"). Report exit code 2: 196 of 218 donors (89.9%) have a finite row for all six models, below the 95% bar. The 22 missing donors are the largest histories: every missing row is a CUDA out-of-memory error (`error` field in `gpu/sm178-rows.jsonl`) that persisted at cap 0.92; the Big models lost 22 donors, Small and the six-layer Tiny 12 (206 of 218). The tables use the 196 donors covered by all six models. Not rerun (rule: at most once, and the failure is memory, not chance). The omitted donors are the longest histories, so donor numbers are likely if anything optimistic for long-history cases. 732 set: 718 users. Benchmark LogLoss, review-weighted, user bootstrap 95% CI, label REF (int8 weights, int8 card and note rows). Donors and the 732 set are different users, so "donors vs 732" is not paired; only the size comparisons are paired.

## Per model and head

| model | head | donors | 732 set | donors vs 732 |
|---|---|---|---|---|
| Big two-head (SM91s2) | curve | 0.3618 [0.3404, 0.3833] | 0.2997 [0.2828, 0.3160] | +20.74% |
| Big two-head (SM91s2) | instant | 0.3520 [0.3305, 0.3734] | 0.2760 [0.2596, 0.2922] | +27.54% |
| Big Instant-Only (SM99) | instant | 0.3517 [0.3303, 0.3731] | 0.2757 [0.2593, 0.2919] | +27.59% |
| Small two-head (SM118) | curve | 0.3620 [0.3406, 0.3834] | 0.3003 [0.2832, 0.3167] | +20.54% |
| Small two-head (SM118) | instant | 0.3522 [0.3308, 0.3738] | 0.2764 [0.2600, 0.2927] | +27.43% |
| Small Instant-Only (SM113) | instant | 0.3516 [0.3303, 0.3732] | 0.2757 [0.2594, 0.2920] | +27.53% |
| six-layer Tiny two-head (SM130) | curve | 0.3631 [0.3417, 0.3846] | 0.3017 [0.2847, 0.3182] | +20.35% |
| six-layer Tiny two-head (SM130) | instant | 0.3535 [0.3320, 0.3751] | 0.2778 [0.2614, 0.2941] | +27.24% |
| six-layer Tiny Instant-Only (SM130) | instant | 0.3529 [0.3314, 0.3744] | 0.2773 [0.2609, 0.2936] | +27.28% |

## By source (donors)

| model | head | campaign (41 users) | LMSherlock (98 users) | Expertium (57 users) |
|---|---|---|---|---|
| Big two-head (SM91s2) | curve | 0.4006 [0.3753, 0.4253] | 0.3572 [0.3229, 0.3900] | 0.3340 [0.2885, 0.3761] |
| Big two-head (SM91s2) | instant | 0.3910 [0.3659, 0.4147] | 0.3468 [0.3131, 0.3797] | 0.3248 [0.2801, 0.3677] |
| Big Instant-Only (SM99) | instant | 0.3907 [0.3655, 0.4145] | 0.3466 [0.3129, 0.3795] | 0.3246 [0.2799, 0.3674] |
| Small two-head (SM118) | curve | 0.4009 [0.3756, 0.4255] | 0.3570 [0.3227, 0.3899] | 0.3345 [0.2893, 0.3765] |
| Small two-head (SM118) | instant | 0.3912 [0.3656, 0.4150] | 0.3470 [0.3133, 0.3800] | 0.3251 [0.2803, 0.3679] |
| Small Instant-Only (SM113) | instant | 0.3904 [0.3648, 0.4143] | 0.3465 [0.3129, 0.3794] | 0.3246 [0.2799, 0.3674] |
| six-layer Tiny two-head (SM130) | curve | 0.4020 [0.3768, 0.4267] | 0.3582 [0.3239, 0.3911] | 0.3357 [0.2910, 0.3776] |
| six-layer Tiny two-head (SM130) | instant | 0.3928 [0.3677, 0.4166] | 0.3483 [0.3144, 0.3814] | 0.3262 [0.2812, 0.3691] |
| six-layer Tiny Instant-Only (SM130) | instant | 0.3919 [0.3666, 0.4157] | 0.3478 [0.3139, 0.3807] | 0.3258 [0.2808, 0.3686] |

## Sizes: Small / Tiny vs the matching Big model (paired, % of Big's LogLoss)

| model | head | donors | 732 set |
|---|---|---|---|
| Small two-head (SM118) | curve | +0.04% [-0.05, +0.14] | +0.21% [+0.06, +0.37] |
| Small two-head (SM118) | instant | +0.06% [+0.01, +0.11] | +0.15% [+0.13, +0.18] |
| Small Instant-Only (SM113) | instant | -0.03% [-0.11, +0.02] | +0.02% [-0.01, +0.04] |
| six-layer Tiny two-head (SM130) | curve | +0.36% [+0.26, +0.49] | +0.68% [+0.51, +0.86] |
| six-layer Tiny two-head (SM130) | instant | +0.44% [+0.39, +0.49] | +0.67% [+0.62, +0.73] |
| six-layer Tiny Instant-Only (SM130) | instant | +0.34% [+0.29, +0.39] | +0.58% [+0.53, +0.62] |

## Benchmark-set shape (Big two-head rows)

| set | users | benchmark reviews | median per user | users 1M+ reviews | deleted-card share | 16+ prior answers | 32+ prior answers |
|---|---|---|---|---|---|---|---|
| set732 | 718 | 28,701,815 | 19,497 | 0 | 20.20% | 16.50% | 5.64% |
| donors | 196 | 16,865,295 | 57,165 | 0 | 14.29% | 26.81% | 10.62% |
| campaign | 41 | 4,321,115 | 90,370 | 0 | 12.62% | 36.91% | 15.82% |
| LMSherlock | 98 | 7,825,525 | 42,030 | 0 | 16.32% | 21.01% | 7.56% |
| Expertium | 57 | 4,718,655 | 63,085 | 0 | 12.45% | 27.17% | 10.95% |

## Reading

- Donors score clearly worse than the 732 set for every model: about +20% on the curve head and +27% on the instant head. The confidence intervals of donors and the 732 set barely overlap in the middle, and the gap is the same for all six models, so it is a data effect, not a model effect.
- The size ranking holds. Big (SM91s2, SM99) is still best, Small (SM118, SM113) is within +0.06% of it (curve +0.04%, interval includes 0), and the six-layer Tiny (SM130) costs +0.36% curve / +0.44% instant (two-head) and +0.34% (Instant-Only) on donors. The margins on donors are smaller than on the 732 set (Small curve +0.21%, Tiny +0.68%), so the smaller models lose nothing extra on donor data; no ranking flips (Small Instant-Only (SM113) is -0.03% vs Big Instant-Only (SM99), interval includes 0, as on the 732 set it is +0.02%).
- By source, campaign donors are hardest (curve about 0.40), LMSherlock about 0.36 and Expertium about 0.33; the size ordering is identical in each source.
- Why donors may score worse (from the shape table; hypotheses, no donor named): longer histories (median 57,165 benchmark reviews per user vs 19,497; 26.8% of reviews have 16+ prior answers vs 16.5%, 10.6% with 32+ vs 5.6%), even though the 22 longest donors are missing; a Japanese-heavy subject mix (inventory aggregate `pages/data/donations.json`, read only: Japanese 16.8M of the reviews, 94 donors with Japanese as main category); different schedulers and habits (more manual reschedules, filtered decks, unknown add-ons). The dataset has no scheduler field, so the last point is a hypothesis. Deleted-card share is lower on donors (14.3% vs 20.2%), so that does not explain it.
