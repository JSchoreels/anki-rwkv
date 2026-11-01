# Answer-time head on richer frozen features (SM82c): decision

An answer-time head (MLP 474 -> 128 -> 4 grades x 16 log-time bins, zero-init output, censored cross-entropy of the true grade's bins) trained alone on a frozen copy of the released small model (the leak-free small model (SM72s) 25 epochs + the 1-epoch 32k Muon cool-down): every other weight is the released master's, bit for bit. Inputs: the trunk's residual stream at the instant row after the input encoder and after each of the 4 layer rounds (5 x 64, layer-normalised), the instant head's hidden layer (128), and 26 explicit causal time features (the card's last 3 answer times, the user's running median per grade, the user's previous time and running average, elapsed days, review counts). Ablation in the same pass: the explicit features alone (MLP 26 -> 128 -> 64). Training: the frozen-model answer-time head (SM82b) setup, the 1,991 users of the SM82 quarter-epoch cool-down, each history's first 400,000 reviews, 80,988,171 labelled rows, one pass, 9,887 AdamW steps of 8,192 rows (lr 0.002, cosine), heads only; 45.0 min (last run segment).

Grade (unchanged): the scorer (w8+s8, leak-free) on 5 users of the 732 set, the released model vs the exported model with the head, in the same ticket: every grade field of the rows and the per-review dump arrays compared exactly. Recorded released scores (small-lf-neighbour.jsonl) compared too.

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

Time (722 users, 21,585,938 test reviews with a recorded time on a later day than the card's previous review, 20,440,806 under the cap): censored 16-bin time LogLoss given the true grade, and the mean |ln error| of the predicted median (uncensored reviews). Baselines as in SM82 (same rows).

| predictor | time LogLoss | mean abs ln error | grade mixed: LogLoss | grade mixed: abs ln error |
|---|---|---|---|---|
| richer frozen features head (SM82c) | 1.9568 | 0.4677 | 1.9820 | 0.4800 |
| ablation: explicit features only | 2.0299 | 0.5025 | 2.0484 | 0.5125 |
| frozen-model answer-time head (SM82b) | 2.3385 | 0.6696 | 2.3467 | 0.6760 |
| jointly trained head, response-time prediction (SM82) | 1.9300 | 0.4625 | 1.9547 | 0.4744 |
| baseline: the card's last time | 2.4466 | 0.7222 | | |
| baseline: the user's running median per grade | 2.3664 | 0.6428 | | |

| comparison | LogLoss delta [95% CI] | relative | abs ln error delta [95% CI] | relative |
|---|---|---|---|---|
| head vs the card's last time | -0.4899 [-0.5135, -0.4685] | -20.0% | -0.2544 [-0.2704, -0.2402] | -35.2% |
| head vs the user's running median per grade | -0.4096 [-0.4388, -0.3842] | -17.3% | -0.1751 [-0.1899, -0.1613] | -27.2% |
| ablation vs the card's last time | -0.4167 [-0.4401, -0.3965] | -17.0% | -0.2197 [-0.2344, -0.2065] | -30.4% |
| ablation vs the user's running median per grade | -0.3365 [-0.3645, -0.3122] | -14.2% | -0.1403 [-0.1542, -0.1275] | -21.8% |
| head vs the ablation: what the model features add | -0.0732 [-0.0783, -0.0676] | -3.6% | -0.0348 [-0.0368, -0.0326] | -6.9% |
| head vs the frozen-model answer-time head (SM82b) | -0.3817 [-0.4091, -0.3536] | -16.3% | -0.2019 [-0.2202, -0.1845] | -30.1% |
| head vs the jointly trained head (SM82): the remaining cost of freezing | +0.0268 [+0.0229, +0.0310] | +1.4% | +0.0052 [+0.0043, +0.0062] | +1.1% |

| grade | reviews | head LogLoss | ablation | SM82b | SM82 | card's last time | user's median per grade | P(under 3.9 s) predicted / actual | ablation / SM82b / SM82 predicted |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2,833,721 | 2.0484 | 2.1140 | 2.3629 | 2.0394 | 2.5233 | 2.3602 | 0.146 / 0.140 | 0.146 / 0.110 / 0.148 |
| 2 | 4,414,931 | 1.9332 | 2.0182 | 2.3628 | 1.9167 | 2.4233 | 2.3682 | 0.211 / 0.214 | 0.212 / 0.211 / 0.222 |
| 3 | 12,582,574 | 1.9548 | 2.0268 | 2.3285 | 1.9181 | 2.4488 | 2.3830 | 0.324 / 0.330 | 0.317 / 0.296 / 0.330 |
| 4 | 1,754,712 | 1.8824 | 1.9462 | 2.3096 | 1.8714 | 2.3664 | 2.2528 | 0.389 / 0.399 | 0.388 / 0.332 / 0.400 |

Threshold: bins 0-6 (under 3.891 s); earlier versions of this file said 5.5 s.

Size: 331,511 -> 400,567 parameters (+69,056); scorer w8 weight bytes 410,459 -> 480,859 (+70,400 B); state unchanged. The 26 explicit features need a per-user running median per grade (4 small heaps or a histogram), the card's last 3 times, and two user scalars at serving time.

Verdict: **pass** (grade unchanged bitwise: True; head beats both baselines: True; vs SM82: LogLoss +0.0268, |ln err| +0.0052; vs SM82b: LogLoss -0.3817, |ln err| -0.2019).
