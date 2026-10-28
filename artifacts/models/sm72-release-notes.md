A small RWKV memory model for Anki, **trained without the deleted-card leak**: **331,511 parameters, 8.3x fewer than the big model** (the 2,762,884-parameter RWKV that the Anki RWKV fork ships), within 0.5% of it on held-out users. It has the same architecture and size as the previous release (SM56), but every input it was trained on was built **leak-free**: the signal that a card is deleted later (three flags plus a shared placeholder deck and preset) is removed, deleted cards take the deck and preset of the nearest existing card, and the flags are always 0. In live Anki a card is never deleted at review time, so that signal is future information.

Recipe: trained from scratch on all 8,000 final-tier users (581.9M reviews per epoch): **15 constant-LR epochs** (AdamW 7e-4, 2,048 x 16 strided lanes, carried state), then a one-epoch 32k Muon cool-down from epoch 15 (LR 7e-4 to 0) over the same users. Knowledge distillation from replay targets of the big model after it was given one extra epoch with the leak removed ("big model (1 leak-free epoch)"; its earlier training saw the leak, so it is not fully leak-free). The held-out users are disjoint from all 8,000 training users.

**Earlier releases were scored with the leak in their inputs, so their published numbers are not directly comparable with the numbers below.** All numbers here score every model on leak-free inputs, the big models included.

## Results

LogLoss on held-out benchmark users (lower is better), weighted by reviews, leak-free inputs for all models. Small model scored as int8 per-channel weights and int8 card/note state (w8+s8); the big models scored the same way. Brackets are 95% intervals (review-weighted user bootstrap, 2,000 resamples).

| Evaluation | Big model, 1 leak-free epoch (curve / instant) | Shipped big model (curve / instant) | Small model (curve / instant) | vs big model (1 leak-free epoch) | vs shipped big model |
|---|---|---|---|---|---|
| 464-user set (458 users) | 0.29530 / 0.27215 | 0.29591 / 0.27260 | 0.29681 / 0.27326 | +0.51% [+0.29, +0.80] / +0.41% [+0.36, +0.46] | +0.30% / +0.24% |
| 732-user set (722 users) | 0.29897 / 0.27576 | 0.29966 / 0.27620 | 0.30038 / 0.27687 | +0.47% [+0.30, +0.66] / +0.40% [+0.37, +0.44] | +0.24% / +0.24% |
| 732+long set (725 users) | 0.30123 / 0.27658 | 0.30142 / 0.27706 | 0.30231 / 0.27775 | +0.36% [+0.10, +0.59] / +0.42% [+0.37, +0.49] | +0.30% / +0.25% |
| The author's own collection | not measured for this model | not measured | not measured | - | - |

The "shipped big model" is the fork's big model as shipped (its training saw the leak), here scored on leak-free inputs. The own-collection row was not measured for this model: no per-review data of that collection is included in this release.

**Against the previous release (SM56) on the same leak-free inputs.** Paired difference (this model minus SM56, 95% interval; negative = this model better):

| Set | Curve | Instant |
|---|---|---|
| 464-user set (458 users) | -0.000225 [-0.000599, +0.000263] | -0.000582 [-0.000689, -0.000479] |
| 732-user set (722 users) | -0.000276 [-0.000547, +0.000062] | -0.000575 [-0.000655, -0.000501] |
| 732+long set (725 users) | -0.000410 [-0.000797, -0.000042] | -0.000606 [-0.000693, -0.000525] |

On the 732-user set SM56 scores 0.30066 / 0.27745 on these inputs, i.e. +0.57% / +0.61% against the big model (1 leak-free epoch), versus +0.47% / +0.40% for this model. This model is better on instant (interval clear of 0 on all three sets) and better on curve in the point estimate (the interval just crosses 0 on the first two sets, and is clear of 0 on the 732+long set). SM56 is the released model that was trained with the leak; it is scored here without it.

- **Curve** is the forgetting-curve head (probability of recall at any future time); **instant** is the next-review head.
- Not measured for this model: the spaced-review slice, int4 card state (it was nearly free on an earlier model, +0.00005 LogLoss), and the own collection.
- Weights: 436 KB in int8 (the scored form) or 1.35 MB in fp32. The exported files reproduce the PyTorch model to 1.3e-6 (parity.json, leak-free inputs).

## Important for anyone porting it

The model must be fed leak-free inputs: **section 6.9 of `INTEGRATION.md`** describes exactly how. In short: the three missing-ID flags (features 13, 14, 15) are always 0; a card missing from the cards table keeps a note of its own, and takes the deck of the existing card with the nearest card id and that deck's preset (placeholder preset if the deck is unknown). A runtime that keeps the old behaviour (flags 1, shared placeholder deck and preset for deleted cards) feeds the model inputs it never saw in training. The Anki RWKV fork does this since commit 587253fbc, where the leak-free small model (SM72) is the default small model (the fork never replays deleted cards' reviews, so only the zero flags apply there).

## Memory and phones

What limits phones is memory, not speed: both models keep a state for every card and note. At the fork's default int8 state the big model needs 25.7 KB per card (with its own note), the small model 4.6 KB. On a 14k-card collection that is ~190 MB of RAM instead of 506 MB (measured for the big model), and an 83 MB state cache instead of 372 MB. Architecture and size are unchanged from the previous release (SM56), so the figures are the same. Details and every precision: section 11 of `INTEGRATION.md`.

## What is inside `sm72-package.zip`

- `INTEGRATION.md`: the developer manual (inputs including the new leak-free section 6.9, features, state handling, weight formats, how to port it, known gaps). It is also attached on its own below.
- `weights.f32.bin`, `weights.int8.bin`, `architecture.json`, `master.pth` (PyTorch checkpoint), `manifest.json` (training recipe, sha256 of every file).
- `reference/`: a NumPy-only reference model (parallel and one-review-at-a-time forms) and `run_reference.py`.
- `testvectors/`: inputs and expected outputs for 2 users, **built with the leak-free inputs**; `python testvectors/check_testvectors.py` must print `RESULT: PASS` (it does, largest difference 1.5e-6).

## Not done yet

- No Rust port: the fork's runtime needs the changes listed in section 12 of the manual (it rejects a 1-layer stream, and the layer order differs), plus the leak-free feature builder of section 6.9.
- 8-bit card/note state works in the NumPy reference but has no reference outputs yet.
- The fork builds the input features slightly differently from our scorer; the effect on accuracy is not measured.
- The own collection, the spaced-review slice and int4 card state are not measured for this model.
- The big model (1 leak-free epoch) had only one epoch without the leak, so the "within 0.5% of the big model" claim compares against a teacher that is itself not fully leak-free; a from-scratch leak-free big model is not part of this release.
- Compressed state (a few hundred bits per card instead of int8) is not part of this release; it currently costs too much accuracy.
- No license is set yet. Check the terms of the training data (anki-revlogs-10k) and the teacher model before redistributing.
