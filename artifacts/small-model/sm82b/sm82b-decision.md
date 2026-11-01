# Frozen-model answer-time head (SM82b): the answer-time head on the frozen released small model

The response-time prediction (SM82) head (one zero-init Linear from the instant head's 128-wide layer to 4 grades x 16 log-time bins, censored cross-entropy of the true grade's bins) trained alone on a frozen copy of the released small model (the leak-free small model (SM72s) 25 epochs + the 1-epoch 32k Muon cool-down): every other weight is the released master's, bit for bit. Training: the 1,991 users of the SM82 quarter-epoch cool-down, each history's first 400,000 reviews through the frozen model on the time scorer's path (fp32, leak-free inputs), 81,285,752 labelled rows, one pass, 9,923 AdamW steps of 8,192 rows (lr 0.002, cosine), head only; 40.9 min (last run segment).

Grade (unchanged): the scorer (w8+s8, leak-free) on 5 users of the 732 set, the released model vs the frozen model with the head, in the same ticket: every grade field of the rows and the per-review dump arrays (p_ahead, p_instant, y, bin, review_th) compared exactly. Recorded released scores (small-lf-neighbour.jsonl) compared too.

| user | reviews | dump rows | rows and dumps bitwise equal | equal to the recorded released scores |
|---|---|---|---|---|
| 4396 | 7,143 | 4,670 | True | True |
| 722 | 15,248 | 8,215 | True | True |
| 1286 | 29,268 | 21,255 | True | True |
| 3455 | 56,209 | 40,545 | True | True |
| 3781 | 146,097 | 110,800 | True | True |

| head | released model, 722 users (LogLoss) | vs the big model (1 leak-free epoch) | change |
|---|---|---|---|
| curve | 0.30003 | +0.36% | none (bitwise) |
| instant | 0.27654 | +0.28% | none (bitwise) |

Time (722 users, 21,585,938 test reviews with a recorded time on a later day than the card's previous review, 20,440,806 under the cap): censored 16-bin time LogLoss given the true grade, and the mean |ln error| of the predicted median (uncensored reviews). Baselines get a residual distribution per grade fit on the users' non-test reviews (same rows as SM82's).

| predictor | time LogLoss | mean abs ln error |
|---|---|---|
| frozen-model head (given the true grade) | 2.3385 | 0.6696 |
| frozen-model head (grade mixed by the grade head) | 2.3467 | 0.6760 |
| jointly trained head, response-time prediction (SM82) (given the true grade) | 1.9300 | 0.4625 |
| jointly trained head, response-time prediction (SM82) (grade mixed) | 1.9547 | 0.4744 |
| baseline: the card's last time | 2.4466 | 0.7222 |
| baseline: the user's running median per grade | 2.3664 | 0.6428 |

| frozen-model head vs | LogLoss delta [95% CI] | relative | abs ln error delta [95% CI] | relative |
|---|---|---|---|---|
| the card's last time | -0.1082 [-0.1444, -0.0713] | -4.4% | -0.0526 [-0.0753, -0.0288] | -7.3% |
| the user's running median per grade | -0.0279 [-0.0622, +0.0060] | -1.2% | +0.0268 [+0.0069, +0.0476] | +4.2% |
| the jointly trained head (SM82): the cost of freezing | +0.4085 [+0.3802, +0.4368] | +21.2% | +0.2071 [+0.1893, +0.2254] | +44.8% |

| grade | reviews | frozen-model head LogLoss | jointly trained head (SM82) | card's last time | user's median per grade | P(under 3.9 s) predicted / actual | SM82 predicted |
|---|---|---|---|---|---|---|---|
| 1 | 2,833,721 | 2.3629 | 2.0394 | 2.5233 | 2.3602 | 0.110 / 0.140 | 0.148 |
| 2 | 4,414,931 | 2.3628 | 1.9167 | 2.4233 | 2.3682 | 0.211 / 0.214 | 0.222 |
| 3 | 12,582,574 | 2.3285 | 1.9181 | 2.4488 | 2.3830 | 0.296 / 0.330 | 0.330 |
| 4 | 1,754,712 | 2.3096 | 1.8714 | 2.3664 | 2.2528 | 0.332 / 0.399 | 0.400 |

Threshold: bins 0-6 (under 3.891 s); earlier versions of this file said 5.5 s.

Size: 331,511 -> 339,767 parameters; scorer w8 weight bytes 410,459 -> 419,163 (+8,704 B); state unchanged.

Verdict: **fail-time** (grade unchanged bitwise: True; head beats both baselines: False; vs the jointly trained head (SM82): LogLoss +0.4085, |ln err| +0.2071).
