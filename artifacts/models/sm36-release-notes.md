A small RWKV memory model for Anki: **331,511 parameters, 8.3x fewer than the big model** (the 2,762,884-parameter RWKV that the Anki RWKV fork ships), within 1% of it on held-out users. It is the previous release's model (SM40) with the final learning-rate decay run by Muon (hybrid Muon/AdamW, outside-research recipe) instead of AdamW; same architecture, same size.

## Results

LogLoss on held-out benchmark users (lower is better), weighted by reviews. Both models scored the same way: int8 per-channel weights and int8 card/note state (w8+s8). (The Anki fork itself runs the big model with fp32 weights and int8 card/note state; int8 weights cost the big model nothing measurable, see below.)

| Evaluation | Big model (curve / instant) | Small model (curve / instant) | Difference |
|---|---|---|---|
| 464 users, 22.7M reviews | 0.29766 / 0.27216 | 0.30006 / 0.27402 | +0.81% / +0.68% |
| 730 users, 33.8M reviews | 0.30437 / 0.27943 | 0.30646 / 0.28126 | +0.69% / +0.65% |

- **Curve** is the forgetting-curve head (probability of recall at any future time); **instant** is the next-review head.
- Against the previous release (SM40) on the same 730 users: SM40 is +0.71% / +0.71% worse than the big model, this model +0.69% / +0.65%. Paired difference vs SM40: curve −0.00007 [−0.00029, +0.00019] (a tie), instant −0.00016 [−0.00019, −0.00013] (better).
- Not measured for this model: the spaced-review slice and the result on the user's own collection.
- Quantization costs the big model nothing measurable (−0.01% / +0.01%, measured on 430 of the 464 users).
- Weights: 436 KB in int8 (the scored form) or 1.35 MB in fp32. The exported files reproduce the PyTorch model to 2.1e-6.

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

## What is inside `sm36-package.zip`

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

sha256 of `sm36-package.zip`: `866166e30bf76b8f40bfeaca4bcc0bc542983a6a8663c9d01f8a69cb225105a8`
