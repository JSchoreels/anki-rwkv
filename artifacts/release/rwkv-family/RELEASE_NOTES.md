# The RWKV family: Big, Small and Tiny memory models for Anki (DRAFT)

> DRAFT. Cells marked **TODO** wait for a package or a final number. Nothing in this folder creates a GitHub release; the `gh release create` line is at the end of `build_release.sh`, commented out.

One release, three sizes, two kinds of model per size. Every model here was trained and scored on **leak-free inputs** (the signal that a card is deleted later is removed, see section 6.9 of `INTEGRATION.md`) and on **show-time inputs** (see below), which is what a live Anki gives a model.

## Which model should I use?

- **Two-head model** (Big, Small or Tiny): use it if you need **both RWKV-Curve and RWKV-Instant from one model**. One forward pass gives the forgetting curve (probability of recall at any future time, hence intervals and the 90% stability) and the next-review (instant) head.
- **Instant-Only model**: use it if you **only schedule with RWKV-Instant**. It was trained with the curve head switched off, so its curve head is untrained and must never be used (no curve, no intervals, no 90% stability). In exchange it is better on the instant head than the two-head model of the same size, and the Tiny one needs less memory.
- Size: Big is the most accurate, Small is about 8x smaller, Tiny is one card layer lighter than Small. Pick the largest size whose memory and speed you can afford.

## Show-time inputs (important for anyone porting a model)

The interval fed to the model at a query is the time **up to the moment the card is shown**, not up to the moment you answer it (the answer time is unknown when a card is shown). All models here were trained for that convention, and all numbers below score them on it (`e2s-query` inputs). A runtime that builds its features with answer-time intervals gives these models inputs they were never trained on. The rule, with the exact formula, is section 6.10 of `INTEGRATION.md`; the leak-free input rule is section 6.9. Read both before building inputs.

## The family

LogLoss on held-out benchmark users (lower is better), 732-user set (722 users), review-weighted, show-time inputs, leak-free, int8 weights and int8 state (w8+s8). "% vs big" is against the big model (1 leak-free epoch) on the same show-time inputs (curve 0.29958, instant 0.277157); negative is better. "Memory" is the exact stored state of one card plus one note at the fork's default precision (int4 cards + int8 notes). Parameters of Big are the fork's 2,762,884.

| Size | Variant | Parameters | Memory per card + note | Show-time curve LogLoss (% vs big) | Show-time instant LogLoss (% vs big) |
|---|---|---|---|---|---|
| Big | Two-head, the show-time big model (SM91s2) | 2,762,884 | 19,554 B | 0.299415 (-0.055%) *step 2 result, TODO final after packaging* | 0.276183 (-0.351%) *step 2 result, TODO final after packaging* |
| Big | Instant-Only, the show-time instant-only big model (SM99) | 2,762,884 | 19,554 B (TODO confirm) | not used (curve head untrained) | TODO (training until about Thu 07:00) |
| Small | Two-head, the show-time retrain (SM92) | 331,511 | 3,530 B | 0.300356 (+0.259%) | 0.276876 (-0.101%) |
| Small | Instant-Only, the Small Instant-Only model (SM98) | 331,511 | 3,530 B | not used (curve head untrained) | 0.276191 (-0.349%) |
| Tiny | Two-head, the Tiny two-head model (SM106) | TODO | TODO | TODO | TODO |
| Tiny | Instant-Only, the Tiny Instant-Only model (SM101) | 311,786 | 2,622 B | not used (curve head untrained) | 0.27672 (-0.158%) |

Notes on the table:

- Memory per card + note at int8 state (the accuracy-reference format): Small 4,554 B, Tiny Instant-Only 3,134 B. In f32: Small 13,640 B, Tiny 9,276 B. When the fork pairs two models (a curve model plus an instant model) the Small pair costs 7,098 B and the Tiny pair 6,190 B; a Small two-head model alone is the 3,530 B above.
- Small Instant-Only vs Small two-head: the instant head gains about 0.25% (0.276191 vs 0.276876). Tiny Instant-Only is about 0.19% behind Small Instant-Only (it drops one card layer) and still better than the Small two-head model on instant (-0.057% [-0.079, -0.034], 460 users better / 261 worse).
- Cost of the smaller state formats on the instant head (80 users of the 732 set, f32 state as reference): int4 cards + int8 notes cost +0.0095% (Small Instant-Only) and +0.0128% (Tiny Instant-Only), below the 0.05% minimum effect. Full int4 state (int4 notes too) can diverge on long note histories and must not be used; keep notes at int8 or better.
- The Instant-Only models' curve numbers are not meaningful and are not listed.
- 95% intervals, per-set tables, other input kinds and the paired comparisons are in each package's `INTEGRATION.md` (section 2) and `manifest.json`.

## What each download contains

Six zips, one per row of the table, plus `INTEGRATION.md` (the full porting manual) and `SHA256SUMS`. Each zip is a package in the same layout as the earlier single-model releases: `INTEGRATION.md`, `weights.f32.bin`, `weights.int8.bin` (per-channel int8), `architecture.json`, `manifest.json`, `parity.json` (NumPy reference vs PyTorch on leak-free inputs, max abs error about 1e-6), `reference/` (NumPy reference implementation), `testvectors/` with `check_testvectors.py` (run it: it must print `RESULT: PASS`) and, where the package has it (Small Instant-Only, Tiny Instant-Only), `master.pth` (the float32 PyTorch checkpoint).

Weight files: Small 1,351,553 B in f32 / 436,391 B in int8 (also used by the Small two-head model, same architecture), Tiny Instant-Only 1,270,749 B / 409,483 B. Big: TODO.

The Small two-head package comes with an optional separate answer-time head (`sm92-time`) in the repository; whether it ships in this release is TODO (decide with the user).

## Important for anyone porting a model

(Copy the "Important for anyone porting it" section of the earlier release notes, updated: leak-free inputs, show-time intervals, int8 note state minimum, Instant-Only models never use `p_curve`.) TODO

## Earlier releases

Earlier single-model releases were scored on different inputs (answer-time intervals, and some with the deleted-card leak), so their numbers are not directly comparable with the table above.
