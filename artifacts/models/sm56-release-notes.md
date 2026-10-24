A small RWKV memory model for Anki: **331,511 parameters, 8.3x fewer than the big model** (the 2,762,884-parameter RWKV that the Anki RWKV fork ships), within 0.5% of it on held-out users. It has the same architecture, size and Muon final decay as the previous release (8k model, SM55), but it is **trained from scratch on all 8,000 final-tier users (581.9M reviews) in every epoch**: 10 epochs at 2,048 x 16 strided lanes with carried state and AdamW 7e-4 (constant), then the same one-epoch 32k Muon decay over the same 8,000 users. The previous release trained 5,000 users for 12 epochs and used the 8,000 only in the decay. Knowledge distillation from the big model is unchanged (replay targets). The held-out users are disjoint from all 8,000 training users.

## Results

LogLoss on held-out benchmark users (lower is better), weighted by reviews. Both models scored the same way: int8 per-channel weights and int8 card/note state (w8+s8). (The Anki fork itself runs the big model with fp32 weights and int8 card/note state; int8 weights cost the big model nothing measurable, see below.)

| Evaluation | Big model (curve / instant) | Small model (curve / instant) | Difference |
|---|---|---|---|
| 464-user set (458 users scored by both) | 0.29584 / 0.27223 | 0.29680 / 0.27351 | +0.32% / +0.47% |
| 732-user set (723 users) | 0.30139 / 0.27786 | 0.30242 / 0.27917 | +0.34% / +0.47% |
| The author's own collection (one user, 173,690 test reviews, whole history as one sequence) | 0.4417 / 0.4367 | 0.4415 / 0.4371 | -0.03% / +0.10% |

The 726-user row of the previous release (the users every released model fits) is not available for this model: 3 of those users (650, 1601, 4501) were not scored for it, so the comparison is on the 723 users that were scored. The own-collection row is aggregate only: no per-review data of it is included. On that collection the previous release was -0.05% / +0.08%, so the two are tied there.

- **Curve** is the forgetting-curve head (probability of recall at any future time); **instant** is the next-review head.
- Against the previous release (8k model, SM55) on the same users, this model is better. Paired difference (SM56 minus SM55, with interval): 723 users, curve -0.00045 [-0.00079, -0.00017], instant -0.00017 [-0.00023, -0.00011]; 458 users, curve -0.00061 [-0.00117, -0.00022], instant -0.00020 [-0.00027, -0.00013]. On the 723 users SM55 is +0.49% / +0.53% against the big model, this model +0.34% / +0.47%.
- Smaller state: int4 card state + int8 note state is nearly free. It was measured on the release before last (Muon version, SM36) on the same 723 users: +0.00005 curve / +0.00005 instant (+0.64% to +0.66% / +0.63% to +0.64% vs the big model), for 31% less card state (26,624 to 18,432 bits). It is not measured on this model yet, and the reference code and test vectors in this package cover int8 state only.
- Not measured for this model: the spaced-review slice, and int4 card state (see above).
- Quantization costs the big model nothing measurable (-0.01% / +0.01%, measured on 430 of the 464 users).
- Weights: 436 KB in int8 (the scored form) or 1.35 MB in fp32. The exported files reproduce the PyTorch model to 1.6e-6.

## Memory and phones

What limits phones is memory, not speed: both models keep a state for every card and note. At the fork's default int8 state the
big model needs 25.7 KB per card (with its own note), the small model 4.6 KB. On a 14k-card collection that is ~190 MB of RAM
instead of 506 MB (measured for the big model), and an 83 MB state cache instead of 372 MB.

Largest collection that fits, by phone year (the model may use about 1/4 of the phone's RAM; estimates, not phone measurements):

| Year | Budget phone (RAM) | Max cards, big model | Max cards, small model |
|---|---|---:|---:|
| 2014 | Moto G 2nd gen (1 GB) | ~9k | ~48k |
| 2016-2019 | Galaxy J5 2016, Moto G5, Redmi 6A, Galaxy A10 (2 GB) | ~18k | ~100k |
| 2020-2021 | Galaxy A21s, Galaxy A12 (3-4 GB) | ~28-38k | ~160-210k |
| 2022+ | Galaxy A13 / A14 (4 GB) | ~38k | ~210k |

Daily use (2-10 ms per answer) is fast on all of them. Details, flagship examples and every precision: section 11 of `INTEGRATION.md`.

## What is inside `sm56-package.zip`

- `INTEGRATION.md`: the developer manual (inputs, features, state handling, weight formats, how to port it, known gaps). It is also attached on its own below.
- `weights.f32.bin`, `weights.int8.bin`, `architecture.json`, `master.pth` (PyTorch checkpoint), `manifest.json` (training recipe, scores, sha256 of every file).
- `reference/`: a NumPy-only reference model (parallel and one-review-at-a-time forms) and `run_reference.py`.
- `testvectors/`: inputs and expected outputs for 2 users; `python testvectors/check_testvectors.py` must print `RESULT: PASS`.

## Not done yet

- No Rust port: the fork's runtime needs the changes listed in section 12 of the manual (it rejects a 1-layer stream, and the layer order differs).
- 8-bit card/note state works in the NumPy reference but has no reference outputs yet.
- The fork builds the input features slightly differently from our scorer; the effect on accuracy is not measured.
- Compressed state (a few hundred bits per card instead of int8) is not part of this release; it currently costs too much accuracy.
- No license is set yet. Check the terms of the training data (anki-revlogs-10k) and the teacher model before redistributing.
