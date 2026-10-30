# SM72 step 3: small model retrained leak-free (w8+s8, review-weighted, user bootstrap 2,000 seed 0, 95% CI)

## 464 set: 458 users common to every series (leak-free scoring)

| model | all curve | all instant | existing curve | existing instant | deleted curve | deleted instant | vs big leak-free curve | instant |
|---|---|---|---|---|---|---|---|---|
| Big model leak-free (SM72 step 1) | 0.29530 | 0.27215 | 0.29170 | 0.26877 | 0.31039 | 0.28625 | - | - |
| Shipped big model (fork RWKV, leak inputs) | 0.29591 | 0.27260 | 0.29202 | 0.26901 | 0.31223 | 0.28765 | +0.21% [-0.03, +0.39] | +0.17% [+0.14, +0.20] |
| Small leak-free, cool-down from epoch 25 (SM72) | 0.29641 | 0.27293 | 0.29275 | 0.26957 | 0.31175 | 0.28695 | +0.38% [+0.18, +0.63] | +0.29% [+0.25, +0.33] |
| Small leak-free, cool-down from epoch 20 (SM72) | 0.29666 | 0.27303 | 0.29301 | 0.26968 | 0.31191 | 0.28705 | +0.46% [+0.25, +0.73] | +0.33% [+0.28, +0.37] |
| Small leak-free, cool-down from epoch 15 (SM72) | 0.29681 | 0.27326 | 0.29318 | 0.26992 | 0.31203 | 0.28720 | +0.51% [+0.29, +0.80] | +0.41% [+0.36, +0.46] |
| Released small, cool-down from epoch 10 (SM56, leak inputs) | 0.29704 | 0.27384 | 0.29328 | 0.27034 | 0.31275 | 0.28848 | +0.59% [+0.39, +0.82] | +0.62% [+0.55, +0.70] |
| Small, cool-down from epoch 15 (SM56, leak inputs) | 0.29687 | 0.27353 | 0.29306 | 0.27002 | 0.31282 | 0.28821 | +0.53% [+0.32, +0.78] | +0.51% [+0.45, +0.57] |

| pair (A - B, all reviews) | curve delta [95% CI] | instant delta [95% CI] | users A better (curve / instant) |
|---|---|---|---|
| Small leak-free, cool-down from epoch 25 (SM72) - Small leak-free, cool-down from epoch 20 (SM72) | -0.000242 [-0.000565, -0.000016] | -0.000104 [-0.000129, -0.000079] | 286 / 282 of 458 |
| Small leak-free, cool-down from epoch 25 (SM72) - Released small, cool-down from epoch 10 (SM56, leak inputs) | -0.000625 [-0.000936, -0.000308] | -0.000911 [-0.001043, -0.000786] | 329 / 402 of 458 |
| Small leak-free, cool-down from epoch 20 (SM72) - Small leak-free, cool-down from epoch 15 (SM72) | -0.000158 [-0.000320, -0.000032] | -0.000225 [-0.000270, -0.000185] | 261 / 358 of 458 |
| Small leak-free, cool-down from epoch 20 (SM72) - Released small, cool-down from epoch 10 (SM56, leak inputs) | -0.000382 [-0.000725, +0.000115] | -0.000807 [-0.000930, -0.000691] | 312 / 396 of 458 |
| Small leak-free, cool-down from epoch 15 (SM72) - Small, cool-down from epoch 15 (SM56, leak inputs) | -0.000060 [-0.000526, +0.000508] | -0.000271 [-0.000364, -0.000188] | 281 / 308 of 458 |
| Small leak-free, cool-down from epoch 15 (SM72) - Released small, cool-down from epoch 10 (SM56, leak inputs) | -0.000225 [-0.000599, +0.000263] | -0.000582 [-0.000689, -0.000479] | 314 / 372 of 458 |

## 732 set: 722 users common to every series (leak-free scoring)

| model | all curve | all instant | existing curve | existing instant | deleted curve | deleted instant | vs big leak-free curve | instant |
|---|---|---|---|---|---|---|---|---|
| Big model leak-free (SM72 step 1) | 0.29897 | 0.27576 | 0.29458 | 0.27170 | 0.31626 | 0.29176 | - | - |
| Shipped big model (fork RWKV, leak inputs) | 0.29966 | 0.27620 | 0.29493 | 0.27191 | 0.31830 | 0.29314 | +0.23% [+0.07, +0.37] | +0.16% [+0.14, +0.18] |
| Small leak-free, cool-down from epoch 25 (SM72) | 0.30003 | 0.27654 | 0.29559 | 0.27250 | 0.31755 | 0.29245 | +0.36% [+0.20, +0.53] | +0.28% [+0.25, +0.31] |
| Small leak-free, cool-down from epoch 20 (SM72) | 0.30019 | 0.27664 | 0.29574 | 0.27261 | 0.31771 | 0.29254 | +0.41% [+0.25, +0.59] | +0.32% [+0.29, +0.35] |
| Small leak-free, cool-down from epoch 15 (SM72) | 0.30038 | 0.27687 | 0.29597 | 0.27285 | 0.31779 | 0.29272 | +0.47% [+0.30, +0.66] | +0.40% [+0.37, +0.44] |
| Released small, cool-down from epoch 10 (SM56, leak inputs) | 0.30066 | 0.27745 | 0.29607 | 0.27326 | 0.31874 | 0.29397 | +0.57% [+0.41, +0.74] | +0.61% [+0.56, +0.66] |
| Small, cool-down from epoch 15 (SM56, leak inputs) | 0.30048 | 0.27714 | 0.29585 | 0.27293 | 0.31870 | 0.29375 | +0.51% [+0.34, +0.69] | +0.50% [+0.46, +0.55] |

| pair (A - B, all reviews) | curve delta [95% CI] | instant delta [95% CI] | users A better (curve / instant) |
|---|---|---|---|
| Small leak-free, cool-down from epoch 25 (SM72) - Small leak-free, cool-down from epoch 20 (SM72) | -0.000158 [-0.000364, +0.000014] | -0.000104 [-0.000123, -0.000082] | 433 / 457 of 722 |
| Small leak-free, cool-down from epoch 25 (SM72) - Released small, cool-down from epoch 10 (SM56, leak inputs) | -0.000630 [-0.000888, -0.000386] | -0.000908 [-0.001002, -0.000818] | 521 / 632 of 722 |
| Small leak-free, cool-down from epoch 20 (SM72) - Small leak-free, cool-down from epoch 15 (SM72) | -0.000196 [-0.000312, -0.000100] | -0.000229 [-0.000263, -0.000199] | 423 / 567 of 722 |
| Small leak-free, cool-down from epoch 20 (SM72) - Released small, cool-down from epoch 10 (SM56, leak inputs) | -0.000472 [-0.000735, -0.000120] | -0.000804 [-0.000894, -0.000721] | 490 / 622 of 722 |
| Small leak-free, cool-down from epoch 15 (SM72) - Small, cool-down from epoch 15 (SM56, leak inputs) | -0.000092 [-0.000417, +0.000290] | -0.000274 [-0.000348, -0.000202] | 437 / 469 of 722 |
| Small leak-free, cool-down from epoch 15 (SM72) - Released small, cool-down from epoch 10 (SM56, leak inputs) | -0.000276 [-0.000547, +0.000062] | -0.000575 [-0.000655, -0.000501] | 488 / 585 of 722 |

## 732+long set: 725 users common to every series (leak-free scoring)

| model | all curve | all instant | existing curve | existing instant | deleted curve | deleted instant | vs big leak-free curve | instant |
|---|---|---|---|---|---|---|---|---|
| Big model leak-free (SM72 step 1) | 0.30123 | 0.27658 | 0.29585 | 0.27134 | 0.32181 | 0.29661 | - | - |
| Shipped big model (fork RWKV, leak inputs) | 0.30142 | 0.27706 | 0.29559 | 0.27159 | 0.32371 | 0.29798 | +0.06% [-0.32, +0.33] | +0.17% [+0.15, +0.20] |
| Small leak-free, cool-down from epoch 25 (SM72) | 0.30199 | 0.27742 | 0.29663 | 0.27224 | 0.32246 | 0.29725 | +0.25% [+0.02, +0.47] | +0.31% [+0.26, +0.38] |
| Small leak-free, cool-down from epoch 20 (SM72) | 0.30204 | 0.27750 | 0.29667 | 0.27232 | 0.32261 | 0.29736 | +0.27% [-0.02, +0.52] | +0.34% [+0.29, +0.39] |
| Small leak-free, cool-down from epoch 15 (SM72) | 0.30231 | 0.27775 | 0.29697 | 0.27258 | 0.32273 | 0.29755 | +0.36% [+0.10, +0.59] | +0.42% [+0.37, +0.49] |
| Released small, cool-down from epoch 10 (SM56, leak inputs) | 0.30272 | 0.27836 | 0.29726 | 0.27302 | 0.32361 | 0.29879 | +0.49% [+0.31, +0.70] | +0.64% [+0.57, +0.73] |
| Small, cool-down from epoch 15 (SM56, leak inputs) | 0.30234 | 0.27802 | 0.29678 | 0.27265 | 0.32365 | 0.29857 | +0.37% [+0.09, +0.64] | +0.52% [+0.47, +0.58] |

| pair (A - B, all reviews) | curve delta [95% CI] | instant delta [95% CI] | users A better (curve / instant) |
|---|---|---|---|
| Small leak-free, cool-down from epoch 25 (SM72) - Small leak-free, cool-down from epoch 20 (SM72) | -0.000054 [-0.000321, +0.000233] | -0.000081 [-0.000124, -0.000019] | 435 / 459 of 725 |
| Small leak-free, cool-down from epoch 25 (SM72) - Released small, cool-down from epoch 10 (SM56, leak inputs) | -0.000732 [-0.001017, -0.000473] | -0.000932 [-0.001036, -0.000836] | 524 / 635 of 725 |
| Small leak-free, cool-down from epoch 20 (SM72) - Small leak-free, cool-down from epoch 15 (SM72) | -0.000268 [-0.000449, -0.000136] | -0.000245 [-0.000290, -0.000203] | 426 / 569 of 725 |
| Small leak-free, cool-down from epoch 20 (SM72) - Released small, cool-down from epoch 10 (SM56, leak inputs) | -0.000677 [-0.001162, -0.000250] | -0.000851 [-0.000964, -0.000748] | 493 / 625 of 725 |
| Small leak-free, cool-down from epoch 15 (SM72) - Small, cool-down from epoch 15 (SM56, leak inputs) | -0.000035 [-0.000362, +0.000340] | -0.000269 [-0.000347, -0.000190] | 438 / 471 of 725 |
| Small leak-free, cool-down from epoch 15 (SM72) - Released small, cool-down from epoch 10 (SM56, leak inputs) | -0.000410 [-0.000797, -0.000042] | -0.000606 [-0.000693, -0.000525] | 491 / 588 of 725 |

## Constant-LR epochs (464 set, 458 fit users, leak-free scoring)

| epoch | curve | instant | gain vs e1 curve | instant | vs big leak-free curve | instant |
|---|---|---|---|---|---|---|
| 1 | 0.30575 | 0.28190 | +0.00% | +0.00% | +3.54% | +3.59% |
| 2 | 0.30391 | 0.27826 | +0.60% | +1.29% | +2.91% | +2.25% |
| 3 | 0.30307 | 0.27849 | +0.88% | +1.21% | +2.63% | +2.33% |
| 4 | 0.30213 | 0.27677 | +1.18% | +1.82% | +2.31% | +1.70% |
| 5 | 0.30041 | 0.27681 | +1.75% | +1.81% | +1.73% | +1.71% |
| 6 | 0.30020 | 0.27651 | +1.82% | +1.91% | +1.66% | +1.60% |
| 7 | 0.30113 | 0.27707 | +1.51% | +1.72% | +1.97% | +1.81% |
| 8 | 0.30073 | 0.27591 | +1.64% | +2.13% | +1.84% | +1.38% |
| 9 | 0.29931 | 0.27580 | +2.11% | +2.16% | +1.36% | +1.34% |
| 10 | 0.29995 | 0.27611 | +1.90% | +2.06% | +1.57% | +1.46% |
| 11 | 0.30055 | 0.27579 | +1.70% | +2.17% | +1.78% | +1.34% |
| 12 | 0.29950 | 0.27597 | +2.04% | +2.10% | +1.42% | +1.41% |
| 13 | 0.30047 | 0.27582 | +1.73% | +2.16% | +1.75% | +1.35% |
| 14 | 0.29912 | 0.27514 | +2.17% | +2.40% | +1.29% | +1.10% |
| 15 | 0.29953 | 0.27540 | +2.04% | +2.31% | +1.43% | +1.19% |
| 16 | 0.29986 | 0.27563 | +1.93% | +2.23% | +1.54% | +1.28% |
| 17 | 0.29987 | 0.27538 | +1.92% | +2.31% | +1.55% | +1.19% |
| 18 | 0.29880 | 0.27545 | +2.27% | +2.29% | +1.18% | +1.21% |
| 19 | 0.30087 | 0.27576 | +1.60% | +2.18% | +1.89% | +1.33% |
| 20 | 0.30042 | 0.27577 | +1.74% | +2.17% | +1.73% | +1.33% |
| 21 | 0.29991 | 0.27569 | +1.91% | +2.20% | +1.56% | +1.30% |
| 22 | 0.29967 | 0.27534 | +1.99% | +2.33% | +1.48% | +1.17% |
| 23 | 0.29962 | 0.27531 | +2.00% | +2.34% | +1.46% | +1.16% |
| 24 | 0.30101 | 0.27586 | +1.55% | +2.15% | +1.93% | +1.36% |
| 25 | 0.30011 | 0.27509 | +1.84% | +2.42% | +1.63% | +1.08% |

## Training

- epoch 1: 581,942,724 reviews, 109 min, 89,326 reviews/s
- epoch 2: 581,942,724 reviews, 110 min, 88,134 reviews/s
- epoch 3: 581,942,724 reviews, 110 min, 88,238 reviews/s
- epoch 4: 581,942,724 reviews, 108 min, 89,976 reviews/s
- epoch 5: 581,942,724 reviews, 111 min, 87,526 reviews/s
- epoch 6: 581,942,724 reviews, 112 min, 86,896 reviews/s
- epoch 7: 581,942,724 reviews, 110 min, 88,488 reviews/s
- epoch 8: 581,942,724 reviews, 110 min, 88,407 reviews/s
- epoch 9: 581,942,724 reviews, 110 min, 88,484 reviews/s
- epoch 10: 581,942,724 reviews, 108 min, 89,517 reviews/s
- epoch 11: 581,942,724 reviews, 110 min, 88,143 reviews/s
- epoch 12: 581,942,724 reviews, 108 min, 90,188 reviews/s
- epoch 13: 581,942,724 reviews, 110 min, 88,128 reviews/s
- epoch 14: 581,942,724 reviews, 110 min, 88,185 reviews/s
- epoch 15: 581,942,724 reviews, 110 min, 88,448 reviews/s
- epoch 16: 581,942,724 reviews, 112 min, 86,925 reviews/s
- epoch 17: 581,942,724 reviews, 111 min, 87,376 reviews/s
- epoch 18: 581,942,724 reviews, 111 min, 87,482 reviews/s
- epoch 19: 581,942,724 reviews, 111 min, 87,371 reviews/s
- epoch 20: 581,942,724 reviews, 111 min, 87,440 reviews/s
- epoch 21: 581,942,724 reviews, 106 min, 91,876 reviews/s
- epoch 22: 581,942,724 reviews, 106 min, 91,467 reviews/s
- epoch 23: 581,942,724 reviews, 104 min, 93,428 reviews/s
- epoch 24: 581,942,724 reviews, 111 min, 87,621 reviews/s
- epoch 25: 581,942,724 reviews, 110 min, 87,944 reviews/s
- cool-down from e25: 581,942,724 reviews, 477 min, 20,314 reviews/s
- cool-down from e20: 423,748,289 reviews, 349 min, 20,231 reviews/s
- cool-down from e15: 519,842,056 reviews, 419 min, 20,677 reviews/s
- cool-down from e10: 69,478,198 reviews, 0 min, 0 reviews/s
