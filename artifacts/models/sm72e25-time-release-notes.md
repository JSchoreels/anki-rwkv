An **answer-time head** for the small RWKV memory model (the leak-free small model, cool-down from epoch 25, SM72): it predicts how long the next answer will take, separately for Again, Hard, Good and Easy, as a distribution over 16 log-time bins from 0.5 s to 60 s. It is the answer-time head on richer frozen features (SM82c): a small MLP (69,056 parameters) trained alone on top of the released model, whose weights stay **bit for bit unchanged**, so every recall prediction of the model is exactly as before. It ships as a separate optional file next to the model (`time_head.f32.bin`); a reader without it, or a model folder without it, behaves exactly as before.

Inputs at the query: the model's own residual stream after its input encoder and after each of its four layer rounds, its instant head's hidden layer, and 26 causal features of the review history (the card's last three answer times, its review count and last grade, the user's running median time per answer button, the previous answer time, a running average, elapsed days and the review count). Predictions are made only for reviews on a later day than the card's previous review.

## Results

732-user held-out set (722 users), 21,585,938 later-day reviews with a recorded time. Lower is better; brackets are 95% intervals (review-weighted paired user bootstrap, 2,000 resamples).

| Predictor | Time LogLoss (16 bins, censored at the user's cap) | Mean abs ln error of the predicted median |
|---|---|---|
| **This head** | **1.9568** | **0.4677** |
| The user's running median per answer button | 2.3664 | 0.6428 |
| The card's last answer time | 2.4466 | 0.7222 |

Against the running median per button: -0.4096 [-0.4388, -0.3842] LogLoss (**-17.3%**) and -0.1751 [-0.1899, -0.1613] abs ln error (**-27.2%**). Against the card's last time: -20.0% / -35.2%. A head trained together with the model reached 1.9300 / 0.4625; the frozen head is 1.4% / 1.1% behind it and in exchange leaves the model untouched. Predicted vs actual share of answers under 3.9 s: Again 0.146 / 0.140, Hard 0.211 / 0.214, Good 0.324 / 0.330, Easy 0.389 / 0.399.

Grade outputs: on five evaluation users every grade output and per-review prediction of the model with the head was bitwise equal to the released model's (Curve / Instant LogLoss on the 732-user set unchanged at 0.30003 / 0.27654).

Training: 1,991 users disjoint from the held-out set, the first 400,000 reviews of each history, 80,988,171 labelled reviews, one pass, 9,887 AdamW steps of 8,192 rows (lr 0.002, cosine), head only.

## Package (`sm72e25-time-package.zip`)

`time_head.f32.bin` / `time_head.int8.bin` (the head, format version 1), `time_head.json` (specification), `ANSWER_TIME.md` (integration manual section), `head.pth`, `reference/answer_time_numpy.py` (NumPy reference of the 26 features, the head and the summaries), `testvectors/` (the model package's vectors for users 4676 and 1745 plus the head's inputs and outputs at every query, and a NumPy checker), `parity.json`, `manifest.json`. The NumPy checker reproduces the PyTorch head within 4.8e-6 (logits) and recomputes the history features exactly. The Anki fork's Rust runtime matches the vectors within 1.0e-6 (f32 weights) / 8.3e-7 (int8 weights) on the bin probabilities of all 6,543 queries.

The own collection was not evaluated for the head.
