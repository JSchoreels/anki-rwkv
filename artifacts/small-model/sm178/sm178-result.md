# Today's models on donor data (SM178): tables

Donors: 196 of 218 users with a finite REF row for all six models (coverage per model {'sm91s2-big-showtime': 196, 'sm99-big-inst-e2': 196, 'sm118-e26-dec2ep32k-t91s2': 206, 'sm113-inst-e35st-st-dec2ep32k-t99': 206, 'sm130-th-dec2ep32k-t91s2': 206, 'sm130-io-dec2ep32k-t99': 206}); 732 set: 718 users with SM148 REF rows for all six. Benchmark LogLoss, review-weighted, user bootstrap 2,000 draws, 95% CI.

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
