A small RWKV memory model for Anki: **331,511 parameters, 8.3x fewer than the big model** (the 2,762,884-parameter RWKV that the Anki RWKV fork ships), within 0.6% of it on held-out users. It is the previous release's model (SM36) with the same architecture and size, and the same Muon final decay, but that one-epoch decay ran over **8,000 training users (581.9M reviews) instead of 5,000 (369.5M)**, with full-history teacher targets for the 3,000 extra users. The held-out users are disjoint from all 8,000 training users.

## Results

LogLoss on held-out benchmark users (lower is better), weighted by reviews. Both models scored the same way: int8 per-channel weights and int8 card/note state (w8+s8). (The Anki fork itself runs the big model with fp32 weights and int8 card/note state; int8 weights cost the big model nothing measurable, see below.)

| Evaluation | Big model (curve / instant) | Small model (curve / instant) | Difference |
|---|---|---|---|
| 464-user set (458 users scored by both) | 0.29584 / 0.27223 | 0.29741 / 0.27371 | +0.53% / +0.54% |
| 732-user set (723 users) | 0.30139 / 0.27786 | 0.30286 / 0.27934 | +0.49% / +0.53% |

7 long users of the 730 the previous releases used are still being scored; these notes will be updated with the 730-user row.

- **Curve** is the forgetting-curve head (probability of recall at any future time); **instant** is the next-review head.
- Against the previous release (SM36) on the same users, this model is better. Paired difference (SM55 minus SM36, with interval): 723 users, curve -0.00047 [-0.00088, -0.00017], instant -0.00025 [-0.00029, -0.00022]; 458 users, curve -0.00063 [-0.00127, -0.00019], instant -0.00027 [-0.00032, -0.00022].
- Not measured for this model: the spaced-review slice and the result on the user's own collection.
- Quantization costs the big model nothing measurable (-0.01% / +0.01%, measured on 430 of the 464 users).
- Weights: 436 KB in int8 (the scored form) or 1.35 MB in fp32. The exported files reproduce the PyTorch model to 2.0e-6.

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

## What is inside `sm55-package.zip`

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
