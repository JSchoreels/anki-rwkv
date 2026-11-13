# The RWKV family: Big, Small and Tiny memory models for Anki

One release, three sizes, two kinds of model per size. Every model here was trained and scored on **leak-free inputs** (the signal that a card is deleted later is removed, see section 6.9 of `INTEGRATION.md`) and on **show-time inputs** (see below), which is what a live Anki gives a model.

## Which model should I use?

- **Two-head model** (Big, Small or Tiny): use it if you need **both RWKV-Curve and RWKV-Instant from one model**. One forward pass gives the forgetting curve (probability of recall at any future time, hence intervals and the 90% stability) and the next-review (instant) head.
- **Instant-Only model**: use it if you **only schedule with RWKV-Instant**. It was trained with the curve head switched off, so its curve head is untrained and must never be used (no curve, no intervals, no 90% stability). In exchange it is better on the instant head than the two-head model of the same size, and the Tiny one needs less memory.
- Size: Big is the most accurate, Small is about 8x smaller, Tiny is one card layer lighter than Small. Pick the largest size whose memory and speed you can afford.

## Show-time inputs (important for anyone porting a model)

The interval fed to the model at a query is the time **up to the moment the card is shown**, not up to the moment you answer it (the answer time is unknown when a card is shown). All models here were trained for that convention, and all numbers below score them on it (`e2s-query` inputs). A runtime that builds its features with answer-time intervals gives these models inputs they were never trained on. The rule, with the exact formula, is section 6.10 of `INTEGRATION.md`; the leak-free input rule is section 6.9. Read both before building inputs.

## The family

LogLoss on held-out benchmark users (lower is better), 732-user set (722 users), review-weighted, show-time inputs, leak-free, int8 weights and int8 state (w8+s8). "% vs big" is against the big model (1 leak-free epoch) on the same show-time inputs (curve 0.29958, instant 0.277157); negative is better. "Memory" is the exact stored state of one card plus one note at the fork's default precision (int4 cards + int8 notes). Both Big models have the fork's 2,762,884 parameters.

| Size | Variant | Parameters | Memory per card + note | Show-time curve LogLoss (% vs big) | Show-time instant LogLoss (% vs big) |
|---|---|---|---|---|---|
| Big | Two-head, the show-time big model (SM91s2) | 2,762,884 | 19,554 B | 0.299415 (-0.06%) | 0.276183 (-0.35%) |
| Big | Instant-Only, the show-time instant-only big model (SM99) | 2,762,884 | 19,554 B | not used (curve head untrained) | 0.275894 (-0.46%) |
| Small | Two-head, the show-time retrain (SM92) | 331,511 | 3,530 B | 0.300356 (+0.26%) | 0.276876 (-0.10%) |
| Small | Instant-Only, the Small Instant-Only model (SM98) | 331,511 | 3,530 B | not used (curve head untrained) | 0.276191 (-0.35%) |
| Tiny | Two-head, the Tiny two-head model (SM106) | 311,786 | 2,622 B | 0.300676 (+0.37%) | 0.277369 (+0.08%) |
| Tiny | Instant-Only, the Tiny Instant-Only model (SM101) | 311,786 | 2,622 B | not used (curve head untrained) | 0.27672 (-0.16%) |

Notes on the table:

- Memory per card + note at int8 state (the accuracy-reference format): Small 4,554 B, the Tiny models (SM106, SM101) 3,134 B. In f32: Small 13,640 B, Tiny 9,276 B. Big costs 19,554 B per card + note at the fork default (25,698 B at int8, 87,136 B serialized in f32). When the fork pairs a size's two models (its two-head model for the curve plus its Instant-Only model) the Big pair costs 39,146 B, the Small pair 7,098 B and the Tiny pair 5,282 B; one model alone costs the figure in the table.
- The Tiny two-head model (SM106) is the show-time retrain (SM92) with one card layer: +0.11% on the curve and +0.18% on instant against it (95% intervals [+0.07, +0.14] and [+0.16, +0.20], % of the big model (1 leak-free epoch)), for 26% less state per card and note. It passed the pruning bar (curve at most +0.5%, instant at most +0.4% vs the big model (1 leak-free epoch)).
- Instant-Only vs two-head: the instant head gains about 0.25% for Small (the Small Instant-Only model (SM98) 0.276191 vs the show-time retrain (SM92) 0.276876), about 0.10% for Big (the show-time instant-only big model (SM99) 0.275894 vs the show-time big model (SM91s2) 0.276183) and about 0.23% for Tiny (the Tiny Instant-Only model (SM101) 0.27672 vs the Tiny two-head model (SM106) 0.277369). The Tiny Instant-Only model (SM101) is about 0.19% behind the Small Instant-Only model (SM98) (it drops one card layer) and still better than the show-time retrain (SM92) on instant (-0.057% [-0.079, -0.034], 460 users better / 261 worse).
- Cost of the smaller state formats on the instant head (80 users of the 732 set, f32 state as reference): int4 cards + int8 notes cost +0.0095% (the Small Instant-Only model (SM98)) and +0.0128% (the Tiny Instant-Only model (SM101)), below the 0.05% minimum effect. Full int4 state (int4 notes too) can diverge on long note histories and must not be used; keep notes at int8 or better.
- The Instant-Only models' curve numbers are not meaningful and are not listed. The Big show-time numbers are the step-2 scores from the show-time report, as packaged.
- 95% intervals, per-set tables, other input kinds and the paired comparisons are in each package's `INTEGRATION.md` (section 2) and `manifest.json`.

## What each download contains

Six zips, one per row of the table, plus `INTEGRATION.md` (the full porting manual) and `SHA256SUMS`. Each zip is a package in the same layout as the earlier single-model releases: `INTEGRATION.md`, `weights.f32.bin`, `weights.int8.bin` (per-channel int8), `architecture.json`, `manifest.json`, `parity.json` (NumPy reference vs PyTorch on leak-free inputs, max abs error about 1e-6), `reference/` (NumPy reference implementation), `testvectors/` with `check_testvectors.py` (run it: it must print `RESULT: PASS`) and, where the package has it (both Big packages, Small Instant-Only, both Tiny packages), `master.pth` (the float32 PyTorch checkpoint; the Small two-head package (SM92) has its checkpoint hash in the manifest but not the file).

**Test vectors and training data (updated 2026-11-13):** the test vectors in every package (`testvectors/user_900001.npz`, `testvectors/user_900002.npz`) come from two synthetic users: simulated review histories (fixed seed) written in the dataset's on-disk format by `scripts/synthetic_testvector_users.py` in this repository. The packages contain no rows of the training data. The zips were replaced on 2026-11-13 (same tag, same models and weight files; only the test vectors and their documentation changed), and `SHA256SUMS` was regenerated. The models were trained on the anki-revlogs-10k dataset, which is linked rather than shipped: https://huggingface.co/datasets/open-spaced-repetition/anki-revlogs-10k (gated, license "anki-revlogs-10k").

Weight files: Small 1,351,553 B in f32 / 436,391 B in int8 (also used by the Small two-head model, same architecture), the Tiny models (SM106, SM101) 1,270,749 B / 409,483 B, and both Big models (SM91s2, SM99) 11,082,637 B / 3,027,096 B. The Big f32 file has the same layout as the fork's `RWKV_trained_on_5000_10000.bin`. The Big packages carry a shorter `INTEGRATION.md` (sections 1-10) that points to the Small packages' section 6 for the full feature list; the fork schedules with the raw forgetting curve and ignores the ahead residual, while the Big test vectors hold the training code's `p_curve` with the residual.

The optional answer-time head that exists for the show-time retrain (SM92) (`sm92-time`, in the repository) is **not shipped** in this release.

## How the fork uses these models

The Anki fork (fork main 001440b03, 2026-11-13) runs exactly these six models, byte for byte, and a size never borrows another size's model. Pick a size in Deck Options; which of its two models run depends on the schedulers your presets use and on the "use one model" option:

| Presets use | Big | Small (default) | Tiny |
|---|---|---|---|
| RWKV-Curve and RWKV-Instant (default) | the show-time big model (SM91s2) for the curve beside the show-time instant-only big model (SM99) for the instant head | the show-time retrain (SM92) beside the Small Instant-Only model (SM98) | the Tiny two-head model (SM106) beside the Tiny Instant-Only model (SM101) |
| both, with "use one model" | the show-time big model (SM91s2) alone | the show-time retrain (SM92) alone | the Tiny two-head model (SM106) alone |
| RWKV-Instant alone (no preset uses RWKV-Curve) | the show-time instant-only big model (SM99) alone | the Small Instant-Only model (SM98) alone | the Tiny Instant-Only model (SM101) alone |
| RWKV-Curve only | the show-time big model (SM91s2) alone | the show-time retrain (SM92) alone | the Tiny two-head model (SM106) alone |

- The curve always comes from the size's two-head model, so "use one model" only changes the instant side.
- The two-head model also provides the answer-time estimate. The show-time big model (SM91s2) and the show-time retrain (SM92) have one; the Tiny two-head model (SM106) does not, so Tiny shows no RWKV answer-time estimate.
- The default small model of the earlier release (SM63) stays in the fork as an explicit profile choice, but no size picks it any more.

The manifests of the Small Instant-Only model (SM98) and the Tiny Instant-Only model (SM101) say the model "runs beside the release (sm72e25)". That wording is out of date: each Instant-Only model runs beside the two-head model of its own size, as listed above.

## Speed in the fork

Measured on 2026-11-13 on a real collection (13,980 cards, 13,205 notes, 225,132 reviews) on a quiet desktop computer, with the fork's Rust code at its default precision, 5 runs each, medians. "State build" is what you wait for after switching size, or when no stored state matches: a full replay of the review history. The app estimate adds about 1.1 s of Python work measured separately. About 1.5 s of every build is reading the history, the same for every size.

| Size | Presets use | Models | State build (app estimate) | Stored state |
|---|---|---|---|---|
| Big | both schedulers | SM91s2 + SM99 | 23.1 s | 611 MB |
| Big | one model, or one scheduler | SM91s2 or SM99 | 14.2 s | 306 MB |
| Small | both schedulers | SM92 + SM98 | 7.8 s | 115 MB |
| Small | one model, or one scheduler | SM92 or SM98 | 5.9 s | 58 MB |
| Tiny | both schedulers | SM106 + SM101 | not measured yet (expected between 5.6 and 7.4 s) | not measured yet |
| Tiny | one model, or one scheduler | SM106 or SM101 | 5.6 s | 46 MB |

The Small pair was timed with the default small model (SM63) in place of the show-time retrain (SM92); both have the same architecture, so the time is the same.

Loading a stored state takes 1.4–1.5 s (Small, Tiny) or 1.9–2.4 s (Big); scoring every card takes 0.12–0.13 s (Small, Tiny) or 0.49 s (Big). These timings match those seen in the app (Big one model about 12–15 s, Small about 4–6 s).

## Important for anyone porting a model

- Use leak-free inputs: remove the signal that a card is deleted later (section 6.9 of `INTEGRATION.md`).
- Use show-time intervals: the interval runs up to the moment the card is shown, not up to the answer (section 6.10 of `INTEGRATION.md`). Inputs built with answer-time intervals are ones these models were never trained on.
- Keep notes at int8 state or better. Cards may be int4 (the fork default); full int4 state can diverge on long note histories.
- Never use an Instant-Only model (the show-time instant-only big model (SM99), the Small Instant-Only model (SM98), the Tiny Instant-Only model (SM101)) for `p_curve`, intervals or the 90% stability. Only `p_imm` and `p_rating` are valid from them.
- Run `testvectors/check_testvectors.py` in each package; it must print `RESULT: PASS`.

## Earlier releases

Earlier single-model releases were scored on different inputs (answer-time intervals, and some with the deleted-card leak), so their numbers are not directly comparable with the table above.
