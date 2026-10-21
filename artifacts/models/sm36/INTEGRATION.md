# Integration manual: small RWKV model "w64" (package sm36)

Audience: a developer who has this folder and nothing else. You do not need the training repository.
Everything here is checked against the files in this folder, the NumPy reference in `reference/`, and the fork's Rust
code (read-only) where it is named. Rust is not compiled or run (see section 12).

Contents: 1 What it is · 2 Accuracy · 3 Package contents · 4 File formats · 5 architecture.json ·
6 Input pipeline · 7 Forward pass for one review · 8 State · 9 Outputs and use · 10 Port plan for the fork ·
11 Performance · 12 Known gaps · 13 Test vectors and reference · 14 License

---

## 1. What it is

A recurrent model that predicts, for a flashcard review history, three things:

| Output | Meaning |
|---|---|
| `p_curve(t)` | Probability of recall as a function of elapsed time `t` (seconds) since the last review of a card. Used for scheduling. |
| `p_imm` | "Instant" recall: probability that the next answer is not Again, if the card is asked now. `1 - p_rating[0]`. |
| `p_rating` | Distribution over the 4 answers (Again, Hard, Good, Easy), softmax. |

Facts:

- 331,511 parameters, d_model 64, 4 heads of 16, 5 streams (card, deck, note, preset, user), 13 layer steps per review.
- It replaces the fork's 2.76M-parameter model ("Alex's model", the RWKV in the Anki fork, Rust runtime under `rslib/src/rwkv/`).
- Same input layout as the fork's model: 92 features per review, same normalisation. Only feature 22 differs (zeroed here).
- The architecture is an architecture from outside research (RWKV-7 style time mixer, power-curve mixture head). The training run
  (label `s-w64kd-gate-sc-lanes-dec12-1ep32k-muon`): 1 epoch plus a learning-rate decay over 5,000 users of the `anki-revlogs-10k`
  dataset, knowledge distillation from the fork's 2.76M model (weight alpha 0.9, 0.5 in the decay phase), recurrent state
  carried across training chunks. The decay phase uses Muon (hybrid Muon/AdamW, outside-research recipe) instead of AdamW;
  the architecture and the pre-decay training are the same as the previous release (SM40).
- Every user in the sections below is a different Anki user of that dataset. The model is not tuned per user.

Output heads at a glance (formulas in section 9): the curve head is a mixture of 3 power curves
`R(t) = sum_i softmax(w)_i * (1 + t / S_i)^(-d_i)`; the instant/rating head is a 2-layer MLP with a 4-way softmax.

---

## 2. Accuracy

Metric: LogLoss (lower is better), review-weighted over the held-out users, on the benchmark's test reviews.
"Curve" = `p_curve` at the true elapsed time to the next review (scored on answered rows). "Instant" = `p_imm` on query rows.
Reference = the fork's 2.76M model with int8 weights and int8 card/note state ("fork w8+s8"). "w8+s8" for the small model
means the same compression: int8 per-output-channel weights (`weights.int8.bin`) and int8 card/note state (section 8.4).
The fork's app runs the big model with f32 weights and int8 card/note state (its default); on 430 of the 464 users
the big model's int8 weights change its LogLoss by -0.01% (curve) / +0.01% (instant), so the reference is the same either way.

| Held-out set | Δ LogLoss curve vs fork w8+s8 | Δ LogLoss instant vs fork w8+s8 |
|---|---:|---:|
| 464 users | +0.81 % | +0.68 % |
| 729 users (33.3M reviews) | +0.70 % | +0.66 % |

(Positive = worse than the fork. The model is about 0.7 % worse in LogLoss and 8.3x smaller in parameters.
The 729-user row is a number from the model owner (one long user of the 730 used for the previous release is still being scored);
the 464-user row is recomputed in `manifest.json`, below. The spaced-review slice was not measured for this model. On the same 729 users the
previous release (SM40) is +0.72 % / +0.71 %.)

Detail for the 464 users (22,656,735 test reviews), from `manifest.json` → `evaluation` (small model w8+s8 vs fork w8+s8,
both on the same users):

| Metric | Small model | Fork | Δ % |
|---|---:|---:|---:|
| Curve LogLoss | 0.30006 | 0.29766 | +0.81 |
| Instant LogLoss | 0.27402 | 0.27216 | +0.68 |
| Curve RMSE(bins) | 0.03978 | 0.03867 | +2.86 |
| Instant RMSE(bins) | 0.02133 | 0.02084 | +2.35 |

Small model alone: AUC curve 0.8022, instant 0.8408 (review-weighted).

Caveats on these numbers:

- They come from the training repository's PyTorch scorer, which feeds stored review durations of answered reviews and the
  training pipeline's feature encodings (section 6.6). Deployment-realistic inputs may cost about +0.0002 (curve) and
  +0.0004 (instant) LogLoss on top (owner's estimate; not part of the table).
- Nothing was measured in Rust.

---

## 3. Package contents

| File | Bytes | Purpose |
|---|---:|---|
| `weights.f32.bin` | 1,351,553 | All tensors, float32, container `ARWKVWEIGHTS1` (the fork's own container). |
| `weights.int8.bin` | 436,391 | Same tensors; 2-D `.weight` tensors as int8 with per-row scales; container `ARWKVWEIGHTS8PC1`. |
| `architecture.json` | 16,034 | All shapes, switches, layer schedule, constants. Read this instead of hard-coding. |
| `parity.json` | 3,190 | NumPy reference vs PyTorch, 3 benchmark users, fp32 and int8: max abs error 1.8e-6 (fp32) / 2.1e-6 (int8). PASS at 1e-5. |
| `manifest.json` | about 21 KB | sha256 of every file, parameter count, training record, evaluation block. |
| `master.pth` | 1,489,805 | PyTorch checkpoint (the source of the two `.bin` files). Not needed at runtime. |
| `reference/` | | Standalone NumPy reference (section 13). |
| `testvectors/` | | Golden inputs and outputs for 2 benchmark users, check script (section 13). |

Identity of the weights: the sha256 of `weights.f32.bin` is listed in `manifest.json` → `files`. Use it as the model id
for caches (section 8.5).

---

## 4. File formats

All integers little-endian. All floats IEEE float32 little-endian.

### 4.1 `weights.f32.bin` (magic `ARWKVWEIGHTS1`)

```
bytes 0..12     magic "ARWKVWEIGHTS1" (13 bytes, no terminator)
u32             tensor count (420)
repeat count times:
    u16         name length n
    n bytes     name, UTF-8
    u8          rank r
    r x u32     dims (row-major, outermost first)
    prod(dims) x f32   values (a rank-0 tensor would have 1 value; none occur)
```

No padding, no checksum, no trailing bytes. This is exactly what the fork's `WeightMap::load` (`rslib/src/rwkv/mod.rs:3192`)
parses. Linear weights are `[out, in]` (PyTorch layout); `y = W x + b`.

### 4.2 `weights.int8.bin` (magic `ARWKVWEIGHTS8PC1`)

```
bytes 0..15     magic "ARWKVWEIGHTS8PC1" (16 bytes)
u32             tensor count (420)
repeat count times:
    u16         name length n
    n bytes     name, UTF-8
    u8          kind: 0 = float32 tensor, 1 = int8 per-row tensor
    u8          rank r
    r x u32     dims
    kind 0:  prod(dims) x f32
    kind 1:  dims[0] x f32 scales, then prod(dims) x i8 codes (row-major)
```

- Kind 1 is used for every tensor whose name ends in `.weight` and whose rank is 2 (217 tensors, listed in
  `architecture.json` → `int8_tensors`). Everything else (203 tensors: biases, norms, `rkvdag_lerp`, `bonus`, `lerp_k`, `gru_*`) is kind 0.
- Dequantisation: `w[row, col] = code[row, col] * scale[row]`, in float32.
- Quantisation (for reference): `scale = max(|row|) / 127` (float32; if 0 then 1.0); `code = clip(round_half_even(w / scale), -127, 127)`.
  Code -128 never occurs.
- The 1x1 dummy weights (section 5.3) are also stored as kind 1; ignore them.

The dequantised values equal the scorer's "w8-pc" weights bit for bit (checked by the exporter). Accuracy numbers with "w8" use this file.
The fork's runtime loads f32 only today; options for int8 in section 10, step 2.

### 4.3 Tensor names and shapes

Tensors are stored in PyTorch `state_dict` order; look them up by name, do not rely on the order.
`i` = stream 0..4 (card, deck, note, preset, user), `l` = layer inside the stream. Prefix `P = rwkv_modules.{i}.blocks.{l}`.

Top level:

| Name | Shape | Note |
|---|---|---|
| `features2card.0.weight` / `.bias` | 128x92 / 128 | Linear 92 -> 128, then SiLU |
| `features2card.2.weight` / `.bias` | 128 / 128 | LayerNorm(128), eps 1e-5 |
| `features2card.3.weight` / `.bias` | 64x128 / 64 | Linear 128 -> 64, then SiLU |
| `prehead_norm.weight` / `.bias` | 64 / 64 | LayerNorm(64) |
| `head_w.0.weight` / `.bias` | 64x64 / 64 | Linear, then ReLU |
| `head_w.2.weight` / `.bias` | 64 / 64 | LayerNorm(64) |
| `head_w.4.weight` / `.bias` | 128x64 / 128 | Linear 64 -> 128 (curve trunk) |
| `gru_w_weight`, `gru_s_weight`, `gru_d_weight` | 3x128 each | Curve head: mixture logits, scale S, decay d |
| `gru_w_bias`, `gru_s_bias`, `gru_d_bias` | 3 each | |
| `head_p.0.weight` / `.bias` | 128x64 / 128 | Linear, then ReLU (rating head) |
| `p_linear.weight` / `.bias` | 4x128 / 4 | Linear -> 4 rating logits |
| `head_ahead_logits.0.*`, `ahead_linear.*`, `w_linear.*` | 1x1 / 1 | Unused dummies in this model. Ignore. |

Per time mixer (`P.time_mixer.`), C = 64, H = 4, K = 16:

| Name | Shape | Note |
|---|---|---|
| `layer_norm.weight` / `.bias` | 64 / 64 | LayerNorm eps 1e-5 |
| `rkvdag_lerp` | 8x1x1x64 | Mix weights, order r, k, v, d, a, g, k_scale, v_scale |
| `bonus` | 1x1x4x16 | |
| `W_r`, `W_k`, `W_v`, `W_o` `.weight` | 64x64 | no bias |
| `k_scale_linear`, `v_scale_linear` `.weight` / `.bias` | 4x64 / 4 | per-head scales |
| `v_lora_simple.A.weight`, `.B_and_lamb.weight` / `.bias` | 2x64, 64x2 / 64 | rank 2. Layer 0: 1x1 dummies, unused |
| `a_lora_simple.A.weight`, `.B_and_lamb.weight` / `.bias` | 4x64, 64x4 / 64 | rank 4 |
| `d_lora_mlp.A.weight`, `.B_and_lamb.weight` / `.bias` | 4x64, 64x4 / 64 | rank 4 (decay) |
| `lora_A_g.weight`, `lora_B_g.weight` | 4x64, 64x4 | rank 4 (gate), no bias |
| `out_group_norm.weight` / `.bias` | 64 / 64 | GroupNorm, 4 groups, eps 6.4e-4 |

Per channel mixer (`P.channel_mixer.`), only when the layer is not stripped:
`layer_norm.weight/.bias` (64), `lerp_k` (1x1x64), `W_k.weight` (64x64), `W_v.weight` (64x64).
For a stripped layer these tensors exist as 1-element dummies (`lerp_k` 1x1x1, `layer_norm` 1, `W_k`/`W_v` 1x1): accept and ignore.

---

## 5. architecture.json

| Field | Value here | Meaning |
|---|---|---|
| `format`, `spec_version` | "anki-rwkv small-model export", 1 | Bump of `spec_version` = incompatible change. |
| `architecture` | "w64" | Preset name in the training repo. |
| `d_model` | 64 | Width of the trunk. |
| `card_features` | 92 | Input size. |
| `features_fc_dim`, `head_fc_dim` | 128, 128 | Hidden width of the input MLP and of the heads. |
| `curve_head` | "power_mixture" | The other value, "softmax_mixture", is the fork's 128-curve head (not used). |
| `num_curves` | 3 | Power curves in the mixture. |
| `num_points` | 128 | Only used by the ahead residual (unused here). |
| `ahead_residual` | false | No ahead residual: its logit is the constant 1e-5. |
| `zero_features` | [22] | Input indices set to 0 before the first Linear. |
| `layer_schedule` | "interleaved" | See `layer_steps`. |
| `layer_steps` | 13 pairs `[stream, layer]` | Execution order of one review (section 7). |
| `streams[]` | 5 objects | Per stream: `name`, `n_layers`, `n_heads`, `head_size`, `channel_mixer_dim`, `decay_lora`, `a_lora`, `v0_mix_amt_lora`, `gate_lora`, `channel_mixer_stripped_layers`, `value_lora_stripped_layer0`. |
| `constants` | see below | Numeric constants of the formulas. |
| `inputs`, `outputs` | text | Short descriptions. |
| `parameters` | 331511 | Count including dummies. |
| `int8_tensors` | 217 names | Tensors stored as int8 in `weights.int8.bin`. |

`streams` (order = module index `i` in tensor names = order in `inputs.streams_order`):

| i | name | layers | heads x size | channel mixer dim | layers without channel mixer |
|---|---|---:|---|---:|---|
| 0 | card_id | 2 | 4 x 16 | 64 | 1 |
| 1 | deck_id | 4 | 4 x 16 | 64 | 1, 2 |
| 2 | note_id | 1 | 4 x 16 | 64 | none |
| 3 | preset_id | 3 | 4 x 16 | 64 | 0, 1, 2 (all) |
| 4 | user_id | 3 | 4 x 16 | 64 | 0, 1, 2 (all) |

`value_lora_stripped_layer0` is true for every stream: layer 0 has no value LoRA (it does not need one, see section 7.2).

`layer_steps` = `[0,0] [1,0] [2,0] [3,0] [4,0] [0,1] [1,1] [3,1] [4,1] [1,2] [3,2] [4,2] [1,3]`
(card L0, deck L0, note L0, preset L0, user L0, card L1, deck L1, preset L1, user L1, deck L2, preset L2, user L2, deck L3).

`constants`: `layer_norm_eps` 1e-5, `group_norm_eps` 6.4e-4, `normalize_eps` 1e-12, `softplus_threshold` 20, `prob_clamp` 1e-5,
`power_clamp` 25, `power_s_eps` 1e-7, `min_elapsed_seconds` 1. (The fields `s_point_spread`, `s_max`, `s_min`, `curve_base`,
`point_spread`, `max_e`, `point_min` belong to the softmax head and the ahead residual; unused here.)

---

## 6. Input pipeline: 92 features from the revlog

### 6.1 The five streams and what identifies each entity

Every review row belongs to exactly one entity per stream. The state of an entity is kept separately (section 8).

| Stream | Entity key | Source |
|---|---|---|
| card_id | card id | revlog `cid` |
| deck_id | deck id of the card | cards table. Missing -> constant 314159265358979323 (shared by all such cards), flag feature 14 = 1 |
| note_id | note id of the card | cards table. Missing -> 314159265358979323 + card id (unique per card), flag feature 13 = 1 |
| preset_id | deck-options preset id of the deck | decks table. Missing -> 314159265358979323, flag feature 15 = 1 |
| user_id | one constant (the whole collection) | |

IDs are only keys. They do not enter the model as numbers; the model sees a random code per entity (features 24..63).

### 6.2 Revlog fields used

Per review, in revlog order (review index `th` = 1, 2, 3, ... over the whole history; `day_offset` is non-decreasing):

| Field | Unit | Meaning |
|---|---|---|
| `card_id`, `note_id`, `deck_id`, `preset_id` | id | as above |
| `rating` | 1..4 | Again, Hard, Good, Easy |
| `duration` | ms | answer time of the review (Anki revlog `time`) |
| `elapsed_seconds` | s | seconds since the previous review of the same card; -1 on the card's first review |
| `elapsed_days` | days | whole days since the previous review of the same card; -1 on the first review |
| `day_offset` | days | day index of the review, minus the day index of the user's first review (so the first review has 0) |
| `state` | int | card state; only feature 22, which is zeroed. Not needed. |

Derived per user, over the whole history: `day_offset_first` = `day_offset` of the card's first review.
Review-type (revlog `type`) information is not used (feature 22 would carry it and is zeroed).

### 6.3 Two kinds of rows: update and query

For each review there are two rows with different roles. The test vectors contain both, in the order the model sees them.

| | Update row (`skip` = 0, `is_query` = 0) | Query row (`skip` = 1, `is_query` = 1) |
|---|---|---|
| Meaning | A review that happened. | A question: "the card is asked now". |
| Rating (9..12) and duration (8) | set | all zero |
| Elapsed, counters, day features | of this review | of the moment of asking (same formulas) |
| State commit | yes (all streams, all layers) | no: the heads are computed from the stored states and the new states are thrown away |
| Used for | `p_curve(t)` of this card (next-review curve), 4 what-if curves for scheduling | `p_imm`, `p_rating` ("instant") |

In the training data each query row sits just before the update row of the same review (same `th`; query first). A card's first review
has no query row. The query row's shift state (time shift) comes from the last non-skipped row of the entity, so queries never disturb the stream.

### 6.4 All 92 inputs at a glance

One row per feature; a sin/cos pair or a multi-dim code is one row. The `Dims` column sums to 92. Exact formulas follow in
6.5 (indices 0..23) and 6.6 (indices 24..91). "log-z" = a log transform, then minus a mean and divided by a standard
deviation (the constants in 6.5). Every input reaches the model as a float.

| # | Feature | Dims | Indices | Type before scaling | What it is | Encoding |
|---:|---|---:|---|---|---|---|
| 1 | Elapsed days | 1 | 0 | int (days) | Days since **this card's** previous review (the interval) | `ln(1+1e-5+x)` log-z; log term 0 on the card's first review |
| 2 | Elapsed days, cumulative | 1 | 1 | int (days) | Running sum of the card's intervals (the card's calendar age) | log-z |
| 3 | Elapsed seconds | 1 | 2 | int (s) | Seconds since this card's previous review | log-z |
| 4 | Elapsed-seconds sub-day phase | 2 | 3, 4 | float x2 | Where the interval falls within 24 h (tells a 1.0-day gap from a 1.5-day gap) | sin, cos of `2 pi (x mod 86400) / 86400` |
| 5 | Elapsed seconds, cumulative | 1 | 5 | int (s) | Running sum of the card's elapsed seconds | log-z |
| 6 | Cumulative-seconds sub-day phase | 2 | 6, 7 | float x2 | 24 h phase of the cumulative clock | sin, cos |
| 7 | Review duration | 1 | 8 | int (ms) | Answer time of this review | `ln(10+x)` log-z; **0 on a query row** (the answer does not exist yet) |
| 8 | Rating | 4 | 9..12 | one-hot | Again / Hard / Good / Easy | 0/1; **all 0 on a query row** |
| 9 | Missing-ID flags | 3 | 13, 14, 15 | bool x3 | The card's note / deck / preset id is missing (deleted card or deck) | 0/1 each |
| 10 | Days since any review | 1 | 16 | int (days) | Days since the user's previous review of **any** card | `ln(ln(e+x))` |
| 11 | Day of week (pseudo) | 1 | 17 | int (0..6) | Position in a 7-day cycle counted from the user's first day (the true weekday is unknown in the data) | `((day_offset mod 7) - 3) / 3` |
| 12 | New cards since the card's last review | 1 | 18 | int | New cards the user started between this card's previous review and now | `ln(3+x)` log-z |
| 13 | Reviews since the card's last review | 1 | 19 | int | Reviews of other cards in that same window | `ln(3+x)` log-z |
| 14 | New cards today | 1 | 20 | int | New cards started so far on this day | `ln(3+x)` log-z |
| 15 | Reviews today | 1 | 21 | int | Reviews done so far on this day | `ln(3+x)` log-z |
| 16 | Card state (**ignored**) | 1 | 22 | int (0..3) | Anki card state (new / learning / review / relearning) | `state - 2`; **zeroed inside the model** (`zero_features`), so you may send 0 |
| 17 | Query flag | 1 | 23 | bool | 1 on the query row of a review (6.3) | 0/1 |
| 18 | Card code | 12 | 24..35 | code | Identity of this card | random per-entity code, values in {-1.5, -0.5, 0.5, 1.5} (6.6, 6.8) |
| 19 | Note code | 12 | 36..47 | code | Identity of the note: sibling cards share it | same |
| 20 | Deck code | 8 | 48..55 | code | Identity of the deck | same |
| 21 | Preset code | 8 | 56..63 | code | Identity of the deck-options preset | same |
| 22 | 3-day cycle | 4 | 64..67 | float x4 | Two clocks with a 3-day period: today's day, and the day of this card's first review (a fixed property of the card) | sin, cos x {review day, first-review day} |
| 23 | 7-day cycle | 4 | 68..71 | float x4 | Same two clocks, 7-day period | same |
| 24 | 30-day cycle | 4 | 72..75 | float x4 | Same, 30 days | same |
| 25 | 100-day cycle | 4 | 76..79 | float x4 | Same, 100 days | same |
| 26 | 365-day cycle | 4 | 80..83 | float x4 | Same, 365 days | same |
| 27 | 3650-day cycle | 4 | 84..87 | float x4 | Same, 3,650 days | same |
| 28 | 36500-day cycle | 4 | 88..91 | float x4 | Same, 36,500 days | same |
| | **Total** | **92** | | | | |

The user stream gets no code: one sequence is always one user (one collection).

### 6.5 The 24 deterministic features (indices 0..23)

`ln` = natural log. "First" = the card's first review (`elapsed = -1`).

| # | Name | Formula |
|---:|---|---|
| 0 | scaled_elapsed_days | `(L(elapsed_days) - 1.51) / 1.62` |
| 1 | scaled_elapsed_days_cumulative | `(L(cum_days) - 2.14) / 2.25`, `cum_days` = running sum of the card's `elapsed_days` including this row |
| 2 | scaled_elapsed_seconds | `(L(elapsed_seconds) - 9.96) / 5.21` |
| 3, 4 | elapsed_seconds_sin / cos | `sin / cos(2 pi (elapsed_seconds mod 86400) / 86400)` (mathematical mod; -1 gives 86399) |
| 5 | scaled_elapsed_seconds_cumulative | `(L(cum_seconds) - 10.86) / 5.80`, running sum of `elapsed_seconds` |
| 6, 7 | elapsed_seconds_cumulative_sin / cos | same as 3, 4 on `cum_seconds` |
| 8 | scaled_duration | `(ln(10 + duration_ms) - 8.9) / 1.07`. For a query row the value is 0 (not the scaled value of 0). |
| 9..12 | rating one-hot | 1 at index `8 + rating` (rating 1 -> 9). All 0 on a query row. |
| 13 | note_id_is_nan | 1 if the card has no note id |
| 14 | deck_id_is_nan | 1 if no deck id |
| 15 | preset_id_is_nan | 1 if no preset id |
| 16 | day_offset_diff | `ln(ln(e + d))`, `d` = `day_offset` minus the `day_offset` of the previous review in the user's whole history (0 for the first) |
| 17 | day_of_week | `((day_offset mod 7) - 3) / 3` |
| 18 | diff_new_cards | `(ln(3 + x) - 2.945) / 2.011`, `x` = number of new cards first-reviewed since this card's previous review (0 on a first review) |
| 19 | diff_reviews | `(ln(3 + x) - 4.64) / 2.59`, `x` = max(0, `th` - `th` of the card's previous review - 1), i.e. reviews of other cards in between |
| 20 | cum_new_cards_today | `(ln(3 + x) - 2.55) / 1.41`, `x` = first-reviews so far on this `day_offset`, including this row if it is one |
| 21 | cum_reviews_today | `(ln(3 + x) - 4.59) / 1.30`, `x` = reviews earlier on this `day_offset` (0-based) |
| 22 | scaled_state | `state - 2`; **forced to 0 by the model** (`zero_features`), so you can skip computing it |
| 23 | is_query | 1 on a query row, else 0 |

where `L(x) = 0 if x == -1 else ln(1 + 1e-5 + x)`. Note that the test `x == -1` also applies to the running sums (the first
review contributes -1 to them; a sum that equals exactly -1 is treated like a first review). Keep this quirk: the fork's
`features_for` (`rslib/src/rwkv/mod.rs:2659`) does the same, and the model was trained with it.

Constants as written above are the normalisation statistics (means and standard deviations) from `rwkv/data_processing.py` `STATISTICS`.
The fork has the same values (`ELAPSED_DAYS_MEAN`, ... at `mod.rs:388-405`).

### 6.6 Entity codes and calendar encodings (indices 24..91)

| Indices | Count | Content |
|---|---:|---|
| 24..35 | 12 | card code |
| 36..47 | 12 | note code |
| 48..55 | 8 | deck code |
| 56..63 | 8 | preset code |
| 64..91 | 28 | calendar: for each period in 3, 7, 30, 100, 365, 3650, 36500 (this order), four values: `sin(f * ((b + day_offset) mod p))`, `cos(...)`, `sin(f * ((b + day_offset_first) mod p))`, `cos(...)`, with `f = 2 pi / p` |

Entity codes: each entity gets, once, a vector of independent random integers in {0,1,2,3} minus 1.5, i.e. values in {-1.5, -0.5, 0.5, 1.5}.
The same entity always gets the same code. The user stream has no code. The codes are only a way to tell entities apart: any random
assignment that is fixed per entity is valid, but it must never change for an entity once its state exists.

Calendar encodings: `b` is a baseline. Training draws a random integer `b` in [0, p) per period; the fork's runtime uses `b = 0`
(`append_day_offset_encoding`, `mod.rs:3071`). Use `b = 0`.

What the test vectors contain: the features exactly as the scorer built them (random codes from a seeded generator and random
baselines), so reproducing columns 24..91 from the raw columns is **not** possible and not needed; columns 0..23 are reproducible
from the `raw_*` arrays (section 13). The fork already generates codes in first-seen order from a fixed Mersenne-Twister state
(`TorchIdRng`, `mod.rs:2993`) and `b = 0`; it differs from the scorer's codes. The accuracy in section 2 was not re-measured with
the fork's code assignment (known gap).

### 6.7 What changes compared with the big model (the fork's model)

Nothing in the feature builder, except that input 22 is ignored. The fork's `features_for` (`mod.rs:2659`) and `CARD_FEATURES = 92`
(`mod.rs:58`) can stay as they are.

### 6.8 Notes on the inputs

- **Query rows (6.3).** Every answer-derived input is zeroed on a query row: rating (9..12) and duration (8). Timing,
  counters, flags, codes and cycles are kept, computed for the moment of asking. So `p_imm` and `p_rating` see only what is
  known before the card is answered.
- **Entity codes are not learned embeddings.** Training re-draws them at random for every batch, so the model can only use
  them to match ("same code as an earlier row" = same card / note / deck / preset). Any fixed random code per entity works
  at deployment; it must not change once the entity has state. Identity enters the model twice: through these codes and
  through the per-entity recurrent state of the five streams (6.1, section 8).
- **Cycle baselines.** In training each period's phase gets a random offset per batch (augmentation: the model cannot
  learn absolute positions in a cycle, only relative ones). At deployment use offset 0, as the fork does (6.6).
- **Labels are not inputs.** Training targets (`label_y`, `label_rating`, the next interval) come from the card's next
  review. A deployment never builds them; the test vectors carry them only to score the outputs.
- **Where the 92 inputs are computed:**

| Path | Computes the 92 inputs? |
|---|---|
| Training and scoring (`rwkv/data_processing.py`, `rwkv/prepare_batch.py`) | yes (source of truth for the formulas in 6.5 and 6.6) |
| NumPy reference in this package (`reference/`) | no: it reads them from the test vectors; indices 0..23 can be rebuilt from the `raw_*` arrays (section 13) |
| Anki fork (`features_for`, `rslib/src/rwkv/mod.rs:2659`) | yes, the same 92 for the big model; codes from its own seeded generator and cycle offset 0 (6.6) |

---

## 7. Forward pass for ONE review (recurrent form)

Executable spec: `reference/small_model_recurrent.py` (`RecurrentModel.step`). Training-side equivalent:
the training code's one-review step (`VariantSrsRWKVRnn.step`). Notation: `lerp(a, b, w) = a + w (b - a)`, `sigmoid`, `SiLU(x) = x sigmoid(x)`.
All math is float32.

### 7.1 Trunk

```
f = features (92); f[22] = 0
x = SiLU( features2card.0 (f) )                  # 92 -> 128
x = LayerNorm( x )  (features2card.2)
x = SiLU( features2card.3 (x) )                  # 128 -> 64
v0[0..4] = unset                                 # one value-residual vector per stream
for (i, l) in layer_steps:                       # 13 steps, in the order of section 5
    state = states[i][entity_i][l]               # may not exist yet
    x, v0[i], new_time_state = time_mixer(i, l, x, v0[i], state.time)
    if layer l of stream i has a channel mixer:
        x, new_channel_shift = channel_mixer(i, l, x, state.channel_shift)
    remember (i, entity_i, l) -> (new_time_state, new_channel_shift)
x = LayerNorm( x )  (prehead_norm)               # call it h
heads(h)                                         # section 7.4
if not skip: commit all remembered new states
```

`entity_i` is the key of the stream-`i` entity of this row (card id for stream 0, deck id for 1, note id for 2, preset id for 3, the user for 4).
The activation `x` flows through all 13 steps, whatever the stream. The interleaved order means a stream's layer 1 sees the output of
every stream's layer 0. (A sequential run, all layers of the card stream and then deck, ... gives different numbers, up to 0.056 in p.)

### 7.2 Time mixer (RWKV-7 style), layer `l` of stream `i`

Prefix `P.time_mixer.`; C = 64, H = 4, K = 16. State input: `shift` (64 values) and `S` (H x K x K), or none for a new entity.

```
xl     = LayerNorm(x)                            # layer_norm
xs     = state.shift   if state exists else xl   # "previous token" for this entity
r,k,v,d,a,g,ks,vs = lerp(xl, xs, rkvdag_lerp[j]) for j = 0..7     # each 64
r      = W_r r
k      = W_k k
k_scale= sigmoid( k_scale_linear(ks) )           # 4 values
v_scale= sigmoid( v_scale_linear(vs) )           # 4 values
if l == 0:  v = W_v v ;  v0 = v
else:       v_lerp = sigmoid( v_lora_simple.B_and_lamb( v_lora_simple.A(v) ) )   # 64
            v = lerp( W_v v, v0, v_lerp )        # v0 = the stream's layer-0 v of THIS review
a      = sigmoid( a_lora_simple.B_and_lamb( a_lora_simple.A(a) ) )               # 64
g      = lora_B_g( sigmoid( lora_A_g(g) ) )                                      # 64
dd     = d_lora_mlp.B_and_lamb( tanh( d_lora_mlp.A(d) ) )
dd     = -0.5 - softplus(-dd)                    # softplus(x) = x if x > 20 else ln(1 + e^x)
w      = exp( -exp(dd) )                         # decay in (0,1), 64 values
reshape r,k,v,w,a to [H, K]
kd     = k_scale[h] * normalize(k[h])            # L2 over K, denominator max(norm, 1e-12)   ("k deformed")
vv     = v_scale[h] * normalize(v[h])
kb     = kd * a                                  # key used for the write
# state update, per head, S is [K(value row i), K(key column j)]:
S_new[i][j] = S[i][j] * w[j]  -  (sum_j' S[i][j'] kd[j']) * a[j] * kd[j]  +  vv[i] * kb[j]
out    = S_new @ r                               # per head: out[i] = sum_j S_new[i][j] r[j]   -> 64
out    = GroupNorm(4 groups, eps 6.4e-4)(out)    # out_group_norm
bonus  = ( sum_K (r * bonus_param * kb) ) * vv   # per head scalar times vv   -> 64
x      = x + W_o( g * (out + bonus) )
new_time_state = ( shift = xl,  S = S_new )
```

Layer 0 never uses `v_lora_simple` (its weights are 1x1 dummies).

### 7.3 Channel mixer

Present unless the layer is in `channel_mixer_stripped_layers`. Prefix `P.channel_mixer.`; state input: `shift` (64) or none.

```
xl = LayerNorm(x)
xs = shift if it exists else xl
k  = W_k( lerp(xl, xs, lerp_k) )          # 64 -> 64
x  = x + W_v( relu(k)^2 )
new_channel_shift = xl
```

### 7.4 Heads

```
h  = prehead_norm(x)
# curve head (only parameters are produced here; p_curve(t) is evaluated later)
u  = head_w.4( LayerNorm( relu( head_w.0(h) ) ) )     # head_w.2 is that LayerNorm; 128 values
w_logits = gru_w_weight @ u + gru_w_bias              # 3
s_raw    = gru_s_weight @ u + gru_s_bias              # 3
d_raw    = gru_d_weight @ u + gru_d_bias              # 3
# rating head
p_logits = p_linear( relu( head_p.0(h) ) )            # 4
```

Keep (w_logits, s_raw, d_raw) per card: that is the card's current forgetting curve (9 floats).

### 7.5 What the skip flag does to the rest of the pass

A query row runs all of 7.1 to 7.4 with the stored states, but commits nothing. Its time shift therefore always comes from the
last update row of the entity. A new entity (no state) uses its own `xl` as shift and a zero `S`.

---

## 8. State: what is stored, size, persistence

### 8.1 Per (stream, layer, entity)

| Item | Size (floats) | Present |
|---|---:|---|
| time shift `xl` | 64 | always |
| WKV matrix `S` | 4 x 16 x 16 = 1,024 | always |
| channel shift | 64 | only if the layer has a channel mixer |

Per entity (all layers of its stream), float32:

| Stream | Layers | Floats | Bytes (f32) |
|---|---:|---:|---:|
| card | 2 (channel mixer in layer 0 only) | 2 x 1,088 + 64 = 2,240 | 8,960 |
| deck | 4 (mixer in layers 0, 3) | 4 x 1,088 + 128 = 4,480 | 17,920 |
| note | 1 (mixer) | 1,152 | 4,608 |
| preset | 3 (no mixers) | 3,264 | 13,056 |
| user | 3 (no mixers) | 3,264 | 13,056 |

Per card (card state + note state, the part that grows with the collection): 13,568 bytes in f32. For comparison the
fork's 2.76M model: 52,224 + 34,816 = 87,040 bytes per card plus note (3 and 2 layers of 4,352 floats). Deck, preset, user
are few entities; their size does not matter.

Also stored per card: the current curve (9 floats), and the feature builder's own state (section 8.3).

### 8.2 What is updated when

- After an update row (skip = 0): replace the states of the 5 entities, for all layers of their streams, with the new states.
- After a query row, or a what-if row: nothing.
- A card that is reviewed in a new session starts from its stored state; there is no decay or reset between sessions. Elapsed time
  enters only through the features (0..7).

### 8.3 State outside the network

The feature builder carries state too (unchanged from the fork's `FeatureState`, `mod.rs:2604`): first `day_offset` of the user, the
previous review's `day_offset`, the set of seen cards, per-card previous `th` and new-card count, today's counters, per-card first
`day_offset` and running elapsed sums, the review counter, the entity codes and the code generator. It must be persisted together with
the network states.

### 8.4 int8 state ("s8")

The accuracy numbers use int8 state for the **card and note streams only** (stream indices 0 and 2); deck, preset and user stay float32.
For those two streams, after each committed update (the step's own output uses the unrounded `S_new`):

- `S` is stored per row of K = 16 values: `scale = max(|row|) / 127` (if 0 then 1), codes `round_half_even(v / scale)` clipped to ±127,
  stored as 4 bytes f32 scale + 16 int8 codes = 20 bytes per row, 64 rows per layer = 1,280 bytes.
- `shift` and `channel shift` are stored as float16 (128 bytes each).
- This is the fork's format (`rslib/src/rwkv/precision.rs`: `encode_matrix`, `write_shift`, `from_model` sets it for modules 0 and 2).

Size with this encoding (fork layout: 19 byte header; per layer 4 + 128 shift + 4 + 1,280 matrix + 4 + 128 channel shift if any):
card 2,987 bytes, note 1,567 bytes, together 4,554 bytes per card (fork 2.76M model: 15,415 + 10,283 = 25,698 bytes).
`reference/small_model_recurrent.py` has `state_bits=8` for this; it is not covered by the golden vectors (section 12).

### 8.5 Caching and invalidation

Persisted states are only valid for the exact model that produced them. Key the cache by:

1. the sha256 of `weights.f32.bin` (from `manifest.json` → `files`; the int8 file gives different states, so key by which file you load as well),
2. the state precision (f32 or int8),
3. the sha256 of `architecture.json`,
4. a feature-builder version of your own.

Any change in one of them invalidates every stored state; rebuild by replaying the revlog from the first review (section 10, step 8).
In the fork, bump `STATE_CACHE_STORE_SCHEMA_VERSION` (`mod.rs:65`, now 4) or store the key in the cache header, so that states of the
2.76M model are never read by this model (their byte layout differs and a wrong read would not always fail loudly).

---

## 9. Outputs and how to use them

### 9.1 Formulas

Given the three heads of a row:

```
p_rating = softmax(p_logits)                              # again, hard, good, easy
p_imm    = clip(1 - p_rating[0], 1e-5, 1 - 1e-5)
# curve at elapsed seconds t (t is clamped to >= 1):
mix_w    = softmax(w_logits)                              # 3
S_i      = exp(clip(s_raw_i, -25, 25))
d_i      = exp(clip(d_raw_i, -25, 25))
R(t)     = sum_i mix_w_i * exp( -d_i * ln(1 + t / (1e-7 + S_i)) )       # = sum_i mix_w_i * (1 + t/S_i)^(-d_i)
raw      = 1e-5 + (1 - 2e-5) * R(t)
p_curve  = sigmoid( ln(raw / (1 - raw)) + 1e-5 )
```

The last line is the full formula of the training code with a zero ahead residual: the residual logit is the constant 1e-5, so
`p_curve` differs from `raw` by about 2.5e-6. The fork's `predict_curve` (`mod.rs:5432`) uses `raw` directly and ignores the ahead
logits; that is accurate enough for this model (difference below 3e-6) and you may keep that shape.

Properties: `R(t)` is non-increasing in `t` (every term is), starts near 1 at `t = 1`, and has no lower floor except 1e-5. This is what lets a
scheduler solve for the interval with bisection or the fork's grid search.

### 9.2 Which row gives which output

| You want | Run | Read |
|---|---|---|
| Forgetting curve of a card after it was answered | the update row of that review (commit state) | `w_logits, s_raw, d_raw` -> `p_curve(t)` for any `t` |
| Interval for target retention `r*` | same | smallest `t` with `p_curve(t) <= r*` |
| Intervals for each possible answer | 4 what-if update rows (rating 1..4, no commit) from the current states | 4 curves -> 4 intervals |
| Retrievability of a card now | the card's stored curve at the elapsed time now (fork: `curve_retrievability`) or a query row | `p_curve(elapsed)` or `p_imm` |
| Probability of each answer when asked now | a query row | `p_rating` |

The scorer scores `p_curve` at the actual elapsed time on update rows (against "not Again") and `p_imm` on query rows. In production
`p_imm` is the model's own answer-probability for a due card; `p_curve(t)` is the quantity to schedule on.

### 9.3 Scheduling in the fork (unchanged)

The fork already does this with its own curve: `answer_heads` runs 4 simulated answers (`simulated_answer_input`), `interval_for_curve` and
`unrounded_intervals_for_answer_curves` search the day grid (`crossings_on_grid`, sub-day points `SUB_DAY_SEARCH_POINTS`), with the
grade-order correction `pava_non_decreasing`. Only `ReviewCurve` and `predict_curve` change (section 10, step 5); the interval code just
calls `predict_curve(curve, seconds)`.

---

## 10. Port plan for the fork's Rust runtime

The fork is the Anki RWKV fork checkout (`<workspace>/anki`). File: `rslib/src/rwkv/mod.rs` unless noted. Line numbers are
from the checkout read for this manual; search by the names if they drift.

Strategy: keep the 2.76M path working (it is the default when no `architecture.json` sits next to the weights file) and make the
shape-dependent constants run-time values. The first target is correctness against `testvectors/` (1e-5), then speed.

1. **Make the shapes data-driven.** Constants at `mod.rs:57-62` (`D_MODEL = 128`, `HEADS = 4`, `HEAD_SIZE`, `HEAD_DIM = 4*D_MODEL`,
   `NUM_CURVES = 128`) and `78-79` (`MODULE_LAYERS`, `CHANNEL_MIXER_DIMS`) become a `ModelSpec` read from `architecture.json`
   (`d_model` 64, per-stream `n_layers`, `n_heads`, `head_size`, `channel_mixer_dim`; `features_fc_dim`/`head_fc_dim` 128 instead of
   `HEAD_DIM`). Fixed arrays `[f32; D_MODEL]` / `[f32; HEAD_SIZE]` in the scratch structs (`ModuleQueryScratch` 5458,
   `LayerQueryScratch` 5622, `TimeMixerQueryScratch` 5761, `ReviewRetrievabilityScratch` 3446, `ChannelMixerQueryScratch` 6337, and
   the batch versions) become `Vec`s, or one const-generic instantiation per shipped architecture (d128/h32 and d64/h16 are enough).
   `MAX_LORA_RANK = 16`, `MAX_CHANNEL_MIXER_DIM = 256` (`mod.rs:296-298`) are already large enough.
2. **Weight loading.** `WeightMap::load` (`mod.rs:3192`) accepts only magic `ARWKVWEIGHTS1`. Our f32 file loads as is. For
   `ARWKVWEIGHTS8PC1` add a branch that reads the kind byte (section 4.2) and either dequantises at load (smallest change, shrinks the
   file, not RAM) or keeps int8 codes plus row scales resident. `packed_weights.rs` is test-only int4 today.
3. **Model construction.** `SrsModel::from_weights` (`mod.rs:3490`), `RwkvModule::load` (5486), `RwkvLayer::load` (5648):
   - `TimeMixer::load` (5822): LoRA ranks are hard-coded (`8`, `16`, `16`, gate `16`). Take them from the tensor shapes or `architecture.json`
     (v 2, a 4, d 4, gate 4). Layer 0: accept and ignore the 1x1 `v_lora_simple` dummies.
   - `ChannelMixer::load` (6366): make it `Option`; for layers in `channel_mixer_stripped_layers` do not load it (tensors are dummies).
     `RwkvLayer::run` (5655) then returns `channel_shift: None` for those layers; `LayerState.channel_shift` is already an `Option`.
   - Heads: drop `w_linear`, `head_ahead_0`, `ahead_linear` (1x1 dummies here). Add `gru_{w,s,d}_{weight,bias}` ([3, 128] and [3]) and use
     `head_w_4` with output 128.
4. **Interleaved layer schedule and per-stream `v0`.** `review_features` (3703) calls `modules[0..4].run` one after another, and
   `RwkvModule::run` (5497) creates `v0` per module call. Replace by a loop over `layer_steps` (from `architecture.json`): for `(i, l)` take
   module `i`'s layer `l` (`layers[l].run(&x, &v0[i], state_i.layers.get(l))`), keep one `v0[i]` for each stream across the loop, and collect
   each module's new `LayerState`s in layer order into its `ModuleState`. The same change is needed in every other place that walks the modules:
   - the query path `review_retrievability_features` (3778) and `run_query` (5529) / `RwkvLayer::run_query_into` (5678);
   - the batched query path `review_retrievability_query_batch` (3612, module loop at 3637) and `RwkvModule::run_query_batch` (5558), `RwkvLayer::run_query_batch` (5703);
   - the bulk replay `bulk.rs`: `run_module` loops over modules at `bulk.rs:140`, and `run_state_only_module_wavefront` (`bulk.rs:243`)
     pipelines the five modules as five stages. With the schedule the pipeline has 13 stages (one per `layer_steps` entry). Layer-at-a-time bulk
     processing still works (it already works one layer at a time over all rows); only the order of layers and the `v0` bookkeeping change.
   - `ModuleState.layers` (5610) keeps one entry per layer of its own stream, as today.
5. **Curve head.**
   - `ReviewCurve` (3823) holds `ahead_logits` and `weights` (2 x 128 floats). Replace by `{w_logits: [f32;3], s_raw: [f32;3], d_raw: [f32;3]}` (or
     precomputed `mix_w`, `S`, `d`). Update `curve_head` (3670), its cache (de)serialisation (3836-3850) and the `ReviewCurve` users
     (`cache_state_len`; tests at 10868-11065).
   - `predict_curve` (5432): evaluate section 9.1. The caller `current_curve_retrievability` (5420) and `interval_for_curve` (5194) need no change.
   - `button_probabilities_head` (3690) and `retrievability_head` stay (`head_p.0` 128 wide).
6. **Input switches.** Zero input 22 before `features_0` (either in `SrsModel::feature_mlp` 3659 and the bulk `feature_mlp_rows`, or in
   `features_for` 2659 where the value is produced). No other feature change (section 6.7).
7. **State store.**
   - Per-entity state sizes change (section 8.1): wherever `D_MODEL`, `HEADS`, `HEAD_SIZE` size a state blob (`TimeState`, `LayerState`, the
     cache writers around 4160-4230 and 5071-5100) use the spec.
   - int8 state (`precision.rs`): `decode` rejects a layer count outside `2..=4` (`precision.rs`, the `(2..=4).contains(&count)` check),
     but the note stream has 1 layer here: allow 1..=4. `read_shift`, `decode`, `encode_matrix` use `D_MODEL`/`HEADS`/`HEAD_SIZE`; replace by spec values.
     `from_model` (`mod.rs:829`) keeps `modules[0]` and `modules[2]` (card, note): the same streams as the scorer's s8.
   - Bump `STATE_CACHE_STORE_SCHEMA_VERSION` (`mod.rs:65`) and/or add the model sha256 to the cache key (section 8.5).
8. **Replay.** After a model switch all states are rebuilt by replaying the collection's revlog (the fork's warm-up path, `bulk.rs`). Expect
   replay time to be dominated by the 13 sequential layer steps per review times 5 entity lookups; the matrices are 4x smaller than the fork's.
9. **Python side.** `qt/aqt/rwkv_scheduler.py`: the model path comes from `ANKI_RWKV_MODEL_PATH` (line 4014) or the embedded
   `RWKV_trained_on_5000_10000.bin` (line 213, 4084). Ship `architecture.json` next to the new `.bin` and pass the folder (or the sha256) down.
   `qt/aqt/rwkv_inference/architecture.py` and the other files in that folder describe the fixed 2.76M architecture; if any of them is used at run
   time for this model, parameterise it the same way.
10. **Acceptance.**
    - Rust replay of the two `testvectors/user_*.npz` histories (convert the arrays to a flat binary or JSON for the Rust test): per-row `p_imm`,
      `p_rating`, `p_curve` at the label time and the 6 sweep times must match to 1e-5 for f32 weights and for int8 weights (dequantised at load).
    - Unit test for the feature builder against the `raw_*` columns of the vectors for features 0..23 (columns 24..91 are random, section 6.6).
    - Then the fork's collection replay against the 2.76M int8 baseline (LogLoss on the user's own collection; expected about +0.7 %).
    - Then the same with int8 state; the expected effect of int8 state is on the order of 1e-3 in probability (section 12).

---

## 11. Performance notes

| | w64 (this model) | fork 2.76M |
|---|---:|---:|
| Parameters | 331,511 | 2,762,884 |
| `weights.f32.bin` | 1,351,553 B | 11,082,637 B |
| `weights.int8.bin` | 436,391 B | n/a |
| Multiply-adds per review (step) | 0.341 M | 2.91 M |
| State per card + note, f32 | 13,568 B | 87,040 B |
| State per card + note, int8 (fork layout) | 4,554 B | 25,698 B |
| Layer steps per review | 13 | 16 (3+4+2+3+4) |

CPU time per review (training repo measurements, 2026-10-16, Ryzen 7 9850X3D, 1 thread, batch 1, a w64 model of the same widths and depths; speed does not
depend on the weight values, but that model may not have had the stripped mixers and interleaved schedule): NumPy step 0.509 ms against 0.939 ms for a NumPy step of the fork
model and 3.95 ms for the eager PyTorch fork (batch 256: 0.061 ms per review for w64). These are not Rust numbers; with 8.5x fewer
multiply-adds a hand-written Rust step should be faster than the fork's, but small matrices make dispatch and memory traffic matter
more than arithmetic. The reference in `reference/` is written for clarity, not speed (about 5 ms per row including 7 curve evaluations).

Memory per 100,000 cards: about 1.36 GB f32 or 0.46 GB int8 for card + note state, against 8.7 GB and 2.6 GB for the fork's model.

### 11.1 Storage and RAM, every precision computed the same way

Example collection: 13,973 cards and 13,198 notes (224k reviews), the collection the fork measured on 2026-10-14 (Apple M1 Max).
"Big model" = the fork's 2,762,884-parameter RWKV; "small model" = this package.

- **Card/note state** = bytes per card x 13,973 + bytes per note x 13,198, with the fork's state format
  (`rslib/src/rwkv/precision.rs`: 19-byte header; per layer an f16 shift, the matrix as rows of a 4-byte scale plus 8- or 4-bit
  codes, an optional f16 channel shift; f32 = 18-byte header plus 12 bytes and the floats per layer). This formula reproduces the
  fork's measured bytes exactly at f32, int8 and int4.
- **Storage** (the state cache file) = weights + card/note state + 20.4 MB of other state (feature builder, decks, presets, user;
  the same in all three fork measurements).
- **RAM** = 129 MB fixed + weights + card/note state. The 129 MB is the mean non-state memory of the fork's four measurements
  (101 to 144 MB). For the big model this lands within 28 MB of the measured values; the small-model rows are estimates.

| Model | Weights | Card/note state | Bytes per card + its note (serialized) | Weights | Card/note state | Storage | RAM, estimated | RAM, measured |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Big | f32 | f32 | 87,136 | 11.1 MB | 1,190 MB | 1,222 MB | 1,330 MB | 1,302 MB |
| Big | f32 | int8 (fork default) | 25,698 | 11.1 MB | 351 MB | 383 MB | 491 MB | 506 MB |
| Big | f32 | int4 | 15,458 | 11.1 MB | 211 MB | 243 MB | 351 MB | 353 MB |
| Big | int4 | int8 | 25,698 | 1.9 MB | 351 MB | 373 MB | 482 MB | 491 MB |
| Small | f32 | f32 | 13,640 | 1.35 MB | 187 MB | 209 MB | 317 MB | |
| Small | f32 | int8 | 4,554 | 1.35 MB | 62 MB | 84 MB | 192 MB | |
| Small | int8 | int8 (as scored) | 4,554 | 0.44 MB | 62 MB | 83 MB | 191 MB | |
| Small | f32 | int4 | 3,018 | 1.35 MB | 41 MB | 63 MB | 171 MB | |
| Small | f32 | compressed, ~400 bits (not released) | ~99 | 1.35 MB | 1.3 MB | 23 MB | 131 MB | |

Accuracy at these settings: the small model is scored with int8 weights and int8 state only. Its int4 state is not scored. The
compressed state (a few codebook indices per card and per note, ~400 bits each) costs about +0.4% curve / +0.8% instant on 34
test users. On the 14k-card collection, int4 state makes the big model slightly worse (LogLoss 0.45615 vs 0.45583 at int8).

### 11.2 Which phones can run it

What limits phones is memory, not speed. Both models keep a state for every card and every note; it grows with the collection and
has to fit in RAM. At int8 state: big model 25.7 KB per card (with its own note), small model 4.6 KB, compressed state (not
released) ~0.1 KB.

Largest collection that fits, by phone year. Assumptions: the model may use about 1/4 of the phone's RAM, plus ~30 MB fixed for its
weights and other state.

| Year | Budget phone (RAM) | Flagship examples | Max cards, big model | Max cards, small model | Max cards, small + compressed (not released) | Small model full rebuild (224k reviews) |
|---|---|---|---:|---:|---:|---:|
| 2014 | Moto G 2nd gen (1 GB) | Galaxy S5 (2 GB), iPhone 6 (1 GB) | ~9k | ~48k | 2M+ | ~1-2 min |
| 2015 | Moto G 3rd gen (1-2 GB) | Galaxy S6 (3 GB), iPhone 6s (2 GB) | ~9-18k | ~48-100k | 2M+ | ~40-80 s |
| 2016 | Galaxy J5 2016 (2 GB) | Galaxy S7 (4 GB), iPhone 7 (2 GB) | ~18k | ~100k | 4M+ | ~40-80 s |
| 2017 | Moto G5 (2 GB) | Galaxy S8 (4 GB), iPhone 8 (2 GB) | ~18k | ~100k | 4M+ | ~30-60 s |
| 2018 | Redmi 6A (2 GB) | Galaxy S9 (4 GB), iPhone XR (3 GB) | ~18k | ~100k | 4M+ | ~30-60 s |
| 2019 | Galaxy A10 (2 GB) | Galaxy S10 (8 GB), iPhone 11 (4 GB) | ~18k | ~100k | 4M+ | ~20-40 s |
| 2020 | Galaxy A21s (3 GB) | Galaxy S20 (8 GB), iPhone SE 2 (3 GB) | ~28k | ~160k | 7M+ | ~15-30 s |
| 2021 | Galaxy A12 (3-4 GB) | iPhone 13 (4 GB), Pixel 6 (8 GB) | ~28-38k | ~160-210k | 7M+ | ~15-30 s |
| 2022+ | Galaxy A13 / A14 (4 GB) | iPhone 14 (6 GB), Galaxy S22 (8 GB) | ~38k | ~210k | 9M+ | ~10-20 s |

A full rebuild recomputes every card's state from the whole review history; it is only needed after a reset. Daily use is fast on
every phone listed: about 2-10 ms per answer and about a second to score a 4,096-card queue (the fork's M1 Max timings scaled by
per-core phone speed, 2-25x slower, and by 8.5x fewer multiply-adds for the small model).

In short: the small model runs on any 2 GB phone from about 2016 with collections up to ~100k cards. The big model needs about
3-4 GB (2020+ budget phones) and stays under ~40k cards even on 4 GB. With compressed state, collection size would no longer
matter on any phone.

Caveats: estimates, not phone measurements. The state sizes are exact (the fork's format, checked against the 14k-card
collection); the RAM budget, the per-core phone speeds and the phone specs (typical configurations) are approximations.
Collections where several cards share a note need less, so the maximums are conservative. No phone build of either model exists
yet: the fork's Rust runtime, which AnkiDroid and AnkiMobile could share through Anki's Rust backend, needs the port in section 10,
then one run on an actual 2 GB phone (rebuild time, peak RAM).

---

## 12. Known gaps ("Verified vs not")

Verified:

- NumPy parallel reference vs PyTorch on this trained checkpoint: users 4676, 1745, 516, fp32 and int8 weights, max |dp| 2.1e-6 over curve, instant and rating
  outputs (`parity.json`; passes at 1e-5).
- The recurrent reference in this folder vs the parallel reference (which is itself within 3.7e-6 of PyTorch), on 2 users, 14,403 rows, fp32 and int8 weights: max difference below 2e-6
  (`testvectors/check_testvectors.py`, tolerance 1e-5).
- The interleaved schedule matters: on the same weights the sequential schedule differs by up to 0.056 in p, so the check tells them apart.
- The int8 file dequantises bit-exactly to the scorer's int8 weights.

Not verified:

- **Rust**: nothing was compiled or run in the fork. Section 10 is a plan read from the code, not tested.
- **int8 state (s8)** in NumPy: the reference implements it (`state_bits=8`) following `precision.rs` and the scorer, but there are no golden vectors for it.
  On the first 2,500 rows of user 4676 it differs from fp32 state by up to 2.5e-3 in probability, which fits the accuracy cost seen in the scorer.
  Small known difference: the scorer also rounds the shift to float16 on a new entity's first review; the reference rounds only stored states.
- **Fork feature builder with this model**: the accuracy in section 2 uses the scorer's entity codes and calendar baselines. The fork assigns codes with its own
  fixed-state generator and uses baseline 0. The effect on accuracy was not measured for this model.
- **Duration leak caveat**: scores use the stored `duration` of answered reviews. A runtime that supplies a different duration (capped, estimated, or missing for
  imports) may lose about +0.0002 (curve) / +0.0004 (instant) LogLoss.
- Accuracy numbers beyond the 464-user block of section 2 come from the model owner and are not recomputed in this package.
- The test vectors come from 2 benchmark users only (3,716 and 4,144 reviews). Long histories (hundreds of thousands of reviews) are not covered by golden values;
  float32 drift over such histories is untested.
- Performance of a Rust implementation is unmeasured.

---

## 13. Test vectors and reference

`reference/` (needs only numpy):

| File | Purpose |
|---|---|
| `weights_io.py` | `read_weights(path)` for both containers, `load_package(dir, "fp32" or "int8")` |
| `small_model_recurrent.py` | One-review-at-a-time model (section 7). `RecurrentModel.step`, `.p_curve`, `.instant`. Optional `state_bits=8`. |
| `small_model_numpy.py` | Copy of the repository's parallel (batch) reference used by `parity.json`. Code unchanged, docstring edited. |
| `run_reference.py` | Loads the package, runs the reference over a test-vector file, prints outputs and the difference to the stored values. |

```
python reference/run_reference.py testvectors/user_4676.npz --rows 20
python reference/run_reference.py testvectors/user_4676.npz --weights int8 --form parallel
```

`testvectors/` (`user_4676.npz`, `user_1745.npz`, benchmark users of `anki-revlogs-10k`; per-user benchmark data may be shared):

| Key | Shape | Content |
|---|---|---|
| `features` | N x 92 float32 | exact model input per row (N = reviews + query rows; 6,554 and 7,849) |
| `id_card_id`, `id_deck_id`, `id_note_id`, `id_preset_id`, `id_user_id` | N int64 | entity key per stream |
| `skip`, `is_query`, `has_label` | N uint8 | row role (section 6.3); `has_label` = a curve/instant label exists |
| `label_elapsed_seconds` | N float32 | elapsed seconds to the next review of the card = the `t` at which `p_curve` is scored (0 on query rows: then `t` is clamped to 1) |
| `label_rating`, `review_th` | N | rating of the labelled review; review index |
| `raw_elapsed_seconds`, `raw_elapsed_days`, `raw_day_offset`, `raw_day_offset_first`, `raw_rating`, `raw_duration`, `raw_state` | N float64 | raw columns that features 0..23 are computed from (zeroed on query rows as in the model input) |
| `sweep_seconds` | 6 float32 | 60 s, 1 h, 1 day, 7 days, 30 days, 365 days |
| `fp32_p_curve`, `int8_p_curve` | N float32 | expected `p_curve` at `label_elapsed_seconds` |
| `fp32_p_curve_sweep`, `int8_p_curve_sweep` | N x 6 | expected `p_curve` at the sweep times |
| `fp32_p_imm`, `int8_p_imm` | N | expected `p_imm` |
| `fp32_p_rating`, `int8_p_rating` | N x 4 | expected rating softmax |
| `par_*` | | Batch-form arrays for `small_model_numpy.NumpyModel` (gathers, shift selects, skips, labels) |

Rows are in model order (query row before the update row of the same review). Expected values were produced by the parallel reference
and checked against PyTorch to 1e-5 when the files were made (`make_testvectors.py`, provenance only).

Check:

```
python testvectors/check_testvectors.py        # about 3.5 minutes; --rows 500 --no-parallel for a quick run
```

It runs both reference forms for fp32 and int8 weights on both users and exits 0 only if every output (`p_curve` at the label time, at the six sweep times,
`p_imm`, `p_rating`) is within 1e-5. Result when this package was built: PASS, largest difference 1.5e-6.

---

## 14. License

TBD by the owner: check the terms of the training data (anki-revlogs-10k), the teacher weights and the architecture source before redistribution.
