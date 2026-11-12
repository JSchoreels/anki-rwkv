# Integration manual: big RWKV model "fork" (package sm99, the show-time instant-only big model (SM99), Big Instant-Only)

Audience: a developer who has this folder and nothing else. Everything is checked against the files in this folder and
the NumPy reference in `reference/`. This package uses the layout of the small-model packages (sm92, sm98, sm101); the
architecture is the Anki fork's own big model, which the fork already runs natively (`rslib/src/rwkv/`, the big engine).

**Read section 6 first if you build inputs:** leak-free inputs and show-time intervals are required.

---

## 1. What it is

The Anki fork's 2,762,884-parameter RWKV architecture (the "Big" size), trained for RWKV-Instant only. **Instant-only:** its curve head was never trained (curve loss weight 0). Only `p_imm` and `p_rating` may be used, never `p_curve`, intervals or the 90% stability. The test vectors still hold its curve outputs, only so that a port can check the whole forward pass.

- 2,762,884 parameters, d_model 128, 4 heads of 32, 5 streams (card, deck, note, preset, user), 16 layer steps per review.
- Training: The show-time big model (SM91s2) trained two more big-model epochs on show-time inputs with the curve loss off: users 5000-10000, fresh AdamW, warmup 200, learning rate 3.5e-4 to 0 in each epoch, 2 x 8,192 carried lanes, no teacher, shuffle seeds 2 and 3. The final model is epoch 2.
- Every input was built leak-free (`RWKV_DELETED_CARDS=neighbour`). The held-out users are disjoint from the training users.

Outputs: `p_curve(t)` (recall after `t` seconds), `p_imm` (1 - P(Again) if asked now) and `p_rating` (softmax over Again,
Hard, Good, Easy).

**Use.** Use `p_imm` and `p_rating` only. In the Anki fork this is the Big size's RWKV-Instant model (key `sm99`, file `qt/aqt/rwkv_inference/sm99/weights.f32.bin`, byte-identical to `weights.f32.bin` of this folder): beside the show-time big model (SM91s2), which keeps the curve, intervals and the 90% stability, or alone when presets use RWKV-Instant without RWKV-Curve. Never as a full model.

## 2. Accuracy

Pool: 722 users of the 732-user held-out set, leak-free inputs (deleted cards: neighbour), w8+s8, review-weighted
instant LogLoss, paired user bootstrap 2,000 resamples (seed 0), 95% intervals. % = change against the reference's own
LogLoss; negative = this model is better. Show-time inputs unless noted.

| Against | This model | Reference | Change [95% CI] | Users better / worse |
|---|---|---|---|---|
| the show-time big model (SM91s2) | 0.275894 | 0.276183 | -0.105% [-0.118, -0.093] | 578 / 143 |
| the show-time instant-only model, stage 2 (SM98) | 0.275894 | 0.276191 | -0.108% [-0.133, -0.084] | 492 / 230 |
| the show-time retrain of the release (SM92) | 0.275894 | 0.276876 | -0.356% [-0.392, -0.322] | 630 / 91 |
| the release (SM63) | 0.275894 | 0.277718 | -0.662% [-0.765, -0.577] | 662 / 59 |
| the big model (1 leak-free epoch), normal inputs | 0.275748 | 0.275757 | -0.004% [-0.024, +0.016] | 392 / 329 |

On 1,487 validation users (show-time inputs) it is -0.115% [-0.124, -0.107] against the show-time big model (SM91s2).
The curve head is untrained and not scored for use.

## 3. Package contents

| File | Bytes | Purpose |
|---|---:|---|
| `weights.f32.bin` | 11,082,637 | All 504 tensors, float32, container `ARWKVWEIGHTS1` (the fork's own big-model file format). |
| `weights.int8.bin` | 3,027,096 | Same tensors; the 265 2-D `.weight` tensors as int8 with per-row scales; container `ARWKVWEIGHTS8PC1`. |
| `architecture.json` | 18,928 | Shapes, switches, layer schedule, constants (`"architecture": "fork"`). |
| `parity.json` | 2,954 | NumPy reference vs PyTorch on 3 held-out users (normal inputs), fp32 and int8: max 1.2e-06 / 1.5e-06. PASS at 1e-5. |
| `manifest.json` | about 27 KB | sha256 of every file, training record, evaluation, `use` and `inputs` rules. |
| `master.pth` | 11,251,089 | PyTorch state dict, the source of the two `.bin` files. Not needed at runtime. |
| `reference/` | | Standalone NumPy reference: `small_model_numpy.py` (batch form), `small_model_recurrent.py` (one review at a time), `weights_io.py`, `run_reference.py`. |
| `testvectors/` | | Golden inputs and outputs for 2 benchmark users on show-time inputs, `check_testvectors.py` (section 9). |

sha256 of `weights.f32.bin`: `682ba1ff207c51af7de5e2abeb25593362aa706b292cbcc338a1591fad0d935c`. Use it as the model id for state caches.

## 4. File formats

All integers little-endian, all floats IEEE float32 little-endian.

`weights.f32.bin`: magic `ARWKVWEIGHTS1` (13 bytes), u32 tensor count, then per tensor in state_dict order: u16 name
length, UTF-8 name, u8 rank, rank x u32 dims (PyTorch shape, Linear weights `[out, in]`), prod(dims) x f32 values. No
padding, no trailing bytes. It is byte-identical to the fork's file (section 1).

`weights.int8.bin`: magic `ARWKVWEIGHTS8PC1` (16 bytes), u32 count, then per tensor: u16 name length, name, u8 kind,
u8 rank, dims; kind 0 = prod(dims) x f32; kind 1 = dims[0] x f32 scales, then prod(dims) x i8 codes. Kind 1 is every 2-D
tensor named `*.weight` (listed in `architecture.json` -> `int8_tensors`); value = code x scale[row], with
scale = max|row| / 127 (1.0 for an all-zero row) and code = round_half_even(w / scale) clamped to [-127, 127].

## 5. Architecture (`architecture.json`)

Per review: the 92 input features go through `features2card` (Linear 92 -> 512, SiLU, LayerNorm, Linear 512 -> 128, SiLU),
then through the 5 streams in order (card 3 layers, deck 4, note 2, preset 3, user 4: 16 layer steps, sequential). Each
layer is an RWKV-7 time mixer (4 heads of 32, LoRA ranks decay 16 / a 16 / value-mix 8 / gate 16, value mix from layer 1
on; layer 0 sets the value that later layers mix in, its value-mix tensors are unused) and a channel mixer (squared-ReLU, hidden 192 for card and note, 256 for deck, preset and user). Each stream keeps
one state per entity (card id, deck id, note id, preset id, the user); a query row computes the heads but commits no
state. Then `prehead_norm` and the heads:

- rating head: Linear 128 -> 512, ReLU, Linear 512 -> 4, softmax = `p_rating`; `p_imm = clip(1 - p_rating[0], 1e-5, 1 - 1e-5)`.
- curve head: `head_w` (Linear 128 -> 128, ReLU, LayerNorm, Linear 128 -> 512), `w_linear` (512 -> 128), softmax = weights
  `w_j` over 128 fixed curves: `R(t) = sum_j w_j * 0.9 ** (t / s_j)`, `s_j = 0.1 + (exp(x_j) - 1) * e^(22 - 18.5)`, `x_j` =
  128 points evenly from 0 to 18.5, `t = max(t, 1)` seconds; `raw = 1e-5 + (1 - 2e-5) * R(t)`.
- ahead residual: `head_ahead_logits.0` (128 -> 512, ReLU) and `ahead_linear` (512 -> 128) give 128 logits at the time points
  `p_j = 0.5 + (exp(x_j) - 1) * e^(21 - 18.5)`, linearly interpolated at `t`: `res = 1e-5 + (1 - 2e-5) * interp(t)`.
- `p_curve(t) = sigmoid(logit(raw) + res)` (the training code's output, stored in the test vectors).

**The Anki fork schedules with the raw forgetting curve `raw` and ignores the ahead residual** (its big engine, since the
first big model). The fork's own parity tests compare against the raw curve; the test vectors here hold the training code's
`p_curve` with the residual, which is what `check_testvectors.py` checks.

No input feature is zeroed by the architecture (the small models zero feature 22; this model uses it).

## 6. Inputs (required: leak-free, show-time)

The 92 inputs per review are the fork's big-model inputs (`rslib/src/rwkv` `CARD_FEATURES`, the same layout the
small-model packages describe in their INTEGRATION.md section 6), with two rules this model was trained under:

- **Leak-free (`RWKV_DELETED_CARDS=neighbour`):** the missing-ID flags (features 13-15) are always 0; the fork applies
  this for the key (model key `sm99`).
- **Show-time intervals:** the instant query row of review j and the curve query for review j use
  floor(max(elapsed_seconds_j - duration_j / 1000, 0)) seconds (from the card's previous answer to the moment review j is
  shown), its sin / cos, and the cumulative interval = the card's earlier answer-to-answer intervals + that show-time
  interval. Update (answer) rows keep the answer-to-answer interval. This is what a live app knows at a query, and what
  RWKV-Instant gives the model in the fork.

The test vectors store every row's 92 features, entity ids, skip (query) flag and query time, so a port can check its
model without its own feature builder first.

## 7. State per card and note

Per (stream, layer, entity): the WKV matrix S (4 heads x 32 x 32 = 4,096 values), the time-mixer shift (128) and the
channel-mixer shift (128). A card has 3 layers, a note 2. Stored size in the fork's formats (exact byte layouts of
`rslib/src/rwkv/engine.rs` and `precision.rs`):

| State precision | Card | Note | Card + note | Big pair (two models, fork) |
|---|---:|---:|---:|---:|
| int4 cards + int8 notes (fork default) | 9,271 B | 10,283 B | 19,554 B | 39,146 B |
| int8 | 15,415 B | 10,283 B | 25,698 B | 51,434 B |
| f32 | 52,278 B | 34,858 B | 87,136 B | 174,310 B |

The pair column is the fork's paired state: per stream one 19-byte header plus one state of each big model. Deck,
preset and user states are per deck, preset and collection, not per card.

## 8. Outputs and use

Use `p_imm` and `p_rating` only. In the Anki fork this is the Big size's RWKV-Instant model (key `sm99`, file `qt/aqt/rwkv_inference/sm99/weights.f32.bin`, byte-identical to `weights.f32.bin` of this folder): beside the show-time big model (SM91s2), which keeps the curve, intervals and the 90% stability, or alone when presets use RWKV-Instant without RWKV-Curve. Never as a full model.

## 9. Test vectors and reference

`python testvectors/check_testvectors.py` (NumPy only, any folder) runs the NumPy reference in both forms (recurrent, one
review at a time, and parallel) with fp32 and int8 weights over the full show-time histories of 2 benchmark users
(4676: 6,554 rows; 1745: 7,849 rows) and compares `p_imm`, `p_rating`, `p_curve` at the label time and at 6 sweep times
(1 minute to 1 year) with the stored values: it must print `RESULT: PASS` (tolerance 1e-5). Last run:
`RESULT: PASS (tolerance 1e-05)`; 8 checks (2 users x fp32 / int8 x recurrent / parallel), largest difference 9.5e-07.

`testvectors/make_testvectors.py` is provenance only (needs the training repository): it built the vectors on show-time
inputs (`e2s-query`) and checked the NumPy reference against the training code's PyTorch model and against the original
big-model class (max 1.2e-06, tolerance 1e-5). `reference/small_model_recurrent.py` is the small packages' file plus
the big model's curve and ahead heads (section 5).

## 10. License

TBD by the owner: check the terms of the training data (anki-revlogs-10k) and the architecture source before redistribution.
