A small RWKV memory model for Anki, **trained without the deleted-card leak**: **331,511 parameters, 8.3x fewer than the big model** (the 2,762,884-parameter RWKV that the Anki RWKV fork ships), within about 0.4% of it on held-out users. It has the same architecture and size as the previous release (the leak-free small model, cool-down from epoch 20, SM72); only the weights change, because the constant-LR phase is five epochs longer. Like that release, every input it was trained on was built **leak-free**: the signal that a card is deleted later (three flags plus a shared placeholder deck and preset) is removed, deleted cards take the deck and preset of the nearest existing card, and the flags are always 0. In live Anki a card is never deleted at review time, so that signal is future information.

Recipe: trained from scratch on all 8,000 final-tier users (581.9M reviews per epoch): **25 constant-LR epochs** (AdamW 7e-4, 2,048 x 16 strided lanes, carried state), then a one-epoch 32k Muon cool-down from epoch 25 (LR 7e-4 to 0) over the same users, in one run. Knowledge distillation from replay targets of the big model after it was given one extra epoch with the leak removed ("big model (1 leak-free epoch)"; its earlier training saw the leak, so it is not fully leak-free). The held-out users are disjoint from all 8,000 training users.

**Earlier releases were scored with the leak in their inputs, so their published numbers are not directly comparable with the numbers below.** All numbers here score every model on leak-free inputs, the big models included.

## Results

LogLoss on held-out benchmark users (lower is better), weighted by reviews, leak-free inputs for all models. Small model scored as int8 per-channel weights and int8 card/note state (w8+s8); the big models scored the same way. Brackets are 95% intervals (review-weighted user bootstrap, 2,000 resamples).

| Evaluation | Big model, 1 leak-free epoch (curve / instant) | Shipped big model (curve / instant) | Small model (curve / instant) | vs big model (1 leak-free epoch) | vs shipped big model |
|---|---|---|---|---|---|
| 464-user set (458 users) | 0.29530 / 0.27215 | 0.29591 / 0.27260 | 0.29641 / 0.27293 | +0.38% [+0.18, +0.63] / +0.29% [+0.25, +0.33] | +0.17% / +0.12% |
| 732-user set (722 users) | 0.29897 / 0.27576 | 0.29966 / 0.27620 | 0.30003 / 0.27654 | +0.36% [+0.20, +0.53] / +0.28% [+0.25, +0.31] | +0.12% / +0.12% |
| 732+long set (725 users) | 0.30123 / 0.27658 | 0.30142 / 0.27706 | 0.30199 / 0.27742 | +0.25% [+0.02, +0.47] / +0.31% [+0.26, +0.38] | +0.19% / +0.13% |
| The author's own collection | not measured for this model | not measured | not measured | - | - |

**At the fork's default state precision (int4 cards + int8 notes).** The table above is the int8 state reference. The fork's default since 2026-10-20 is int4 card state with int8 note state. These weights were not scored at that precision. For the previous release (same architecture, 732-user set) int4 cards cost curve +0.000039 [+0.000026, +0.000055] and instant +0.000051 [+0.000039, +0.000065] against its own int8 state; a similar small cost is expected here but is not measured.

The "shipped big model" is the fork's big model as shipped (its training saw the leak), here scored on leak-free inputs. The own-collection row was not measured for this model: no per-review data of that collection is included in this release.

**Against the previous release (the leak-free small model, cool-down from epoch 20, SM72) on the same leak-free inputs.** Paired difference (this model minus the previous release, 95% interval; negative = this model better):

| Set | Curve | Instant |
|---|---|---|
| 464-user set (458 users) | -0.000242 [-0.000565, -0.000016] | -0.000104 [-0.000129, -0.000079] |
| 732-user set (722 users) | -0.000158 [-0.000364, +0.000014] | -0.000104 [-0.000123, -0.000082] |
| 732+long set (725 users) | -0.000054 [-0.000321, +0.000233] | -0.000081 [-0.000124, -0.000019] |

On the 732-user set the previous release scores 0.30019 / 0.27664, i.e. +0.41% / +0.32% against the big model (1 leak-free epoch), versus +0.36% / +0.28% for this model. Instant is better with an interval clear of 0 on all three sets; curve is better in the point estimate on all three sets, with an interval clear of 0 on the 464-user set only (it is the only change: 25 constant-LR epochs instead of 20).

**Against the last release trained with the leak (SM56), scored here without it.** Paired difference (this model minus that release, 95% interval; negative = this model better):

| Set | Curve | Instant |
|---|---|---|
| 464-user set (458 users) | -0.000625 [-0.000936, -0.000308] | -0.000911 [-0.001043, -0.000786] |
| 732-user set (722 users) | -0.000630 [-0.000888, -0.000386] | -0.000908 [-0.001002, -0.000818] |
| 732+long set (725 users) | -0.000732 [-0.001017, -0.000473] | -0.000932 [-0.001036, -0.000836] |

On the 732-user set that release (SM56) scores 0.30066 / 0.27745 on these inputs, i.e. +0.57% / +0.61% against the big model (1 leak-free epoch). This model is better on both heads, with an interval clear of 0 on all three sets.

- **Curve** is the forgetting-curve head (probability of recall at any future time); **instant** is the next-review head.
- Not measured for this model: the spaced-review slice, the own collection, and int4 card state (no set was scored at the fork's default state for these weights).
- **Full int4 state (int4 cards AND int4 notes) diverged** with the previous release (same architecture) on long note histories (held-out user 3876: a 9,539-step note sequence grows to ~1e32 by step 6000 and inf at step 6843; float ~5, int8 ~6.5), so scoring failed with a non-finite loss. It was not re-tested with these weights. The Anki fork removed the full int4 option (fork commit f967796bc); a stored "int4" profile falls back to int4 cards + int8 notes. Anyone porting it must keep the note state at int8 or better.
- Weights: 436 KB in int8 (the scored form) or 1.35 MB in fp32. The exported files reproduce the PyTorch model to 1.3e-6 (parity.json, leak-free inputs).

## Important for anyone porting it

The model must be fed leak-free inputs: **section 6.9 of `INTEGRATION.md`** describes exactly how. In short: the three missing-ID flags (features 13, 14, 15) are always 0; a card missing from the cards table keeps a note of its own, and takes the deck of the existing card with the nearest card id and that deck's preset (placeholder preset if the deck is unknown). A runtime that keeps the old behaviour (flags 1, shared placeholder deck and preset for deleted cards) feeds the model inputs it never saw in training. The Anki RWKV fork does this since commit 587253fbc, and it uses this model as its default small model (model key `sm72e25`; the previous release stays bundled as `sm72e20`; the fork never replays deleted cards' reviews, so only the zero flags apply there).

## Memory and phones

What limits phones is memory, not speed: both models keep a state for every card and note. At int8 state the big model needs 25.7 KB per card (with its own note), the small model 4.6 KB. At the fork's default (int4 cards + int8 notes) the small model needs 3.5 KB per card + note (3,530 B; 4,554 B at int8, 3,018 B at full int4, which diverged with the previous release). On a 14k-card collection that is ~178 MB of RAM and a 70 MB state cache at the fork's default (estimates), ~190 MB and 83 MB at int8, instead of 506 MB (measured for the big model at int8) and 372 MB. Architecture and size are unchanged from the previous releases, so the figures are the same. Details and every precision: section 11 of `INTEGRATION.md`.

## What is inside `sm72e25-package.zip`

- `INTEGRATION.md`: the developer manual (inputs including the leak-free section 6.9, features, state handling, weight formats, how to port it, known gaps). It is also attached on its own below.
- `weights.f32.bin`, `weights.int8.bin`, `architecture.json`, `master.pth` (PyTorch checkpoint), `manifest.json` (training recipe, sha256 of every file).
- `reference/`: a NumPy-only reference model (parallel and one-review-at-a-time forms) and `run_reference.py`.
- `testvectors/`: inputs and expected outputs for 2 users, **built with the leak-free inputs**; `python testvectors/check_testvectors.py` must print `RESULT: PASS` (it does, largest difference 1.1e-6).

## Not done yet

- No Rust port: the fork's runtime needs the changes listed in section 12 of the manual (it rejects a 1-layer stream, and the layer order differs), plus the leak-free feature builder of section 6.9.
- 8-bit card/note state works in the NumPy reference but has no reference outputs yet.
- The fork builds the input features slightly differently from our scorer; the effect on accuracy is not measured.
- The own collection, the spaced-review slice and int4 card state are not measured for this model.
- The big model (1 leak-free epoch) had only one epoch without the leak, so the "within 0.4% of the big model" claim compares against a teacher that is itself not fully leak-free; a from-scratch leak-free big model is not part of this release.
- Compressed state (a few hundred bits per card instead of int8) is not part of this release; it currently costs too much accuracy.
- No license is set yet. Check the terms of the training data (anki-revlogs-10k) and the teacher model before redistributing.
