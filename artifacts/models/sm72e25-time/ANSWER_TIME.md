# Integration manual: answer-time head for the small model (package sm72e25-time)

Section for the answer-time head on richer frozen features (SM82c), the optional add-on of the leak-free small model, cool-down from epoch 25 (SM72), package `sm72e25` (its `INTEGRATION.md` is the manual of the model itself; nothing there changes).

## 1. What it is

An MLP head that predicts the answer time of the next review, separately for each answer button: Linear 474 -> 128, ReLU, Linear 128 -> 64 = 4 grades (Again, Hard, Good, Easy) x 16 log-time bins. It was trained alone on a frozen copy of the released model (every model weight bit for bit the release's), so the model's grade, curve and instant outputs never change. 69,056 parameters (276,402 bytes f32, 70,586 bytes int8 per-row weights).

## 2. Files

| file | content |
|---|---|
| `time_head.f32.bin` | `ARWKVWEIGHTS1` container (section 4.1 of the model manual): `time_mlps.0.hidden.weight` [128, 474], `time_mlps.0.hidden.bias` [128], `time_mlps.0.out.weight` [64, 128], `time_mlps.0.out.bias` [64], `time_head.format_version` [1] = 1.0 |
| `time_head.int8.bin` | `ARWKVWEIGHTS8PC1` copy (2-D weights int8 per output row, as the model's int8 file) |
| `time_head.json` | the specification below in machine-readable form |
| `head.pth` | the head's PyTorch state_dict (`time_mlps.0.*`) |
| `reference/answer_time_numpy.py` | NumPy reference: explicit features, head, median, P(fast) |
| `testvectors/user_4676.npz`, `user_1745.npz` | the model package's vectors (unchanged arrays) + the time arrays (section 6) |
| `testvectors/check_time_vectors.py` | NumPy checker |

Ship the head file **next to** the model's weights and keep the model's `weights.f32.bin` unchanged. A reader that does not know the head ignores the extra file; a reader that knows it treats a missing file as "no head". Refuse any `time_head.format_version` other than 1.

## 3. Inputs at a query (instant) row: 474 float32

1. **Frozen model features (448)**, from the model's own forward pass at the query row (section 7 of the model manual), with the same state as the instant head:
   - `layer_norm(x0)`: the residual (64) after the input encoder `features2card` (after its final SiLU);
   - `layer_norm(x1..x4)`: the residual after each of the four interleaved layer rounds: round 0 = steps (0,0) (1,0) (2,0) (3,0) (4,0), round 1 = (0,1) (1,1) (3,1) (4,1), round 2 = (1,2) (3,2) (4,2), round 3 = (1,3) (the last one is the final residual before `prehead_norm`). A round ends where the next step has another layer index;
   - `head_p` (128): `relu(head_p.0(prehead_norm(x4)))`, the instant head's hidden layer.
   `layer_norm` here has no affine parameters: `(x - mean) / sqrt(var + 1e-5)` with the biased variance.
2. **Explicit causal features (26)**, from the user's reviews **before** the query (card id, grade 1-4, duration ms) and the query's own card id and elapsed days. ln times are `ln(max(duration_ms, 1) / 1000)` in float64, centred on ln 8 s; only reviews with duration > 0 enter the time statistics; every review counts for the card and user counters.

| index | feature |
|---|---|
| 0-2 | the card's last 3 recorded ln times (most recent first) - ln 8, 0 if missing |
| 3-5 | "missing" flags of 0-2 |
| 6 | ln(1 + the card's earlier review count) / 5 |
| 7-10 | the card's last grade, one-hot (0 for a new card) |
| 11-14 | the user's running median ln time per grade 1-4 - ln 8 (fallback: the median over all grades, then ln 8 s) |
| 15-18 | ln(1 + recorded times per grade) / 10 |
| 19 | the running median over all grades - ln 8 (ln 8 s if none) |
| 20, 21 | the user's previous recorded ln time - ln 8, and its "missing" flag |
| 22 | exponential average of recorded ln times (start ln 8 s, `a = 0.9 a + 0.1 x`) - ln 8 |
| 23 | ln(1 + max(elapsed days, 0)) / 3 of the query |
| 24 | 1 if the query is the card's first review (elapsed days < 0) |
| 25 | ln(1 + the user's review index) / 10 (the number of earlier reviews) |

The running median is exact (two heaps; an even count averages the two middle ln values). The reference pushes a time with a grade outside 1-4 into the all-grade median twice; real histories only have grades 1-4.

## 4. Output and summaries

Softmax per grade over its 16 logits. Bin 0 < 500 ms; bin k (1..14) = [e(k-1), e(k)) with e(k) = 500 x 120^(k/14) ms; bin 15 >= 60 s (Anki's time cap). Summaries:
- median (ln seconds): linear in ln time inside the bin where the cumulative probability crosses 0.5 (bin 0 from ln 0.25 s, bin 15 = 60 s);
- P(fast | grade) = bins 0..6 (under 3.891 s);
- grade-mixed distribution = sum over grades of the instant head's grade probability x the grade's bins.

**Valid only** for queries on a later day than the card's previous review (elapsed days >= 1). First and same-day reviews are not predicted: in training their interval input holds their own answer time.

## 5. Accuracy (722 users of the 732-user set, 21,585,938 later-day reviews with a recorded time)

| predictor | censored time LogLoss | mean abs ln error of the median |
|---|---|---|
| this head | 1.9568 | 0.4677 |
| the card's last time | 2.4466 | 0.7222 |
| the user's running median per grade | 2.3664 | 0.6428 |

vs the running median per grade: -0.4096 [-0.4388, -0.3842] (-17.3%) LogLoss, -0.1751 [-0.1899, -0.1613] (-27.2%) abs ln error (95% review-weighted paired user bootstrap). Training: 1,991 users disjoint from the 732 set, first 400,000 reviews each, one pass, 9,887 AdamW steps of 8,192 rows.

## 6. Test vectors

Per instant row of the model vectors: `time_row` (row index into the model vectors), `time_review_th` (1-based revlog index), `time_explicit` [n, 26], `fp32_time_frozen` [n, 448], `fp32_time_logits` / `int8_time_logits` [n, 64] (PyTorch, the frozen-feature path the head was trained and scored on; int8 = the model and the head with int8 per-row weights), and the user's revlog columns `rev_card_id`, `rev_rating`, `rev_duration`, `rev_elapsed_days` (enough rows for every `time_review_th`). The model's own arrays are the `sm72e25` package's, unchanged (rebuilt inputs checked bit for bit). `python testvectors/check_time_vectors.py <package>` recomputes the explicit features (exact) and the logits from the stored inputs.
