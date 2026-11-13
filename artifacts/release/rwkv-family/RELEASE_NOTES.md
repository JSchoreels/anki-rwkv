# The RWKV family: Big, Small and Tiny memory models for Anki (DRAFT)

> DRAFT. All six packages exist (the Tiny two-head model (SM106) since 2026-11-13). Nothing in this folder creates a GitHub release; the `gh release create` line is at the end of `build_release.sh`, commented out.

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

- Memory per card + note at int8 state (the accuracy-reference format): Small 4,554 B, the Tiny models (SM106, SM101) 3,134 B. In f32: Small 13,640 B, Tiny 9,276 B. Big costs 19,554 B per card + note at the fork default (25,698 B at int8, 87,136 B serialized in f32). When the fork pairs two models (a curve model plus an instant model) the Big pair costs 39,146 B, the Small pair 7,098 B and the Tiny pair 6,190 B; a Small two-head model alone is the 3,530 B above.
- The Tiny two-head model (SM106) is the show-time retrain (SM92) with one card layer: +0.11% on the curve and +0.18% on instant against it (95% intervals [+0.07, +0.14] and [+0.16, +0.20], % of the big model (1 leak-free epoch)), for 26% less state per card and note. It passed the pruning bar (curve at most +0.5%, instant at most +0.4% vs the big model (1 leak-free epoch)).
- Instant-Only vs two-head: the instant head gains about 0.25% for Small (the Small Instant-Only model (SM98) 0.276191 vs the show-time retrain (SM92) 0.276876), about 0.10% for Big (the show-time instant-only big model (SM99) 0.275894 vs the show-time big model (SM91s2) 0.276183) and about 0.23% for Tiny (the Tiny Instant-Only model (SM101) 0.27672 vs the Tiny two-head model (SM106) 0.277369). The Tiny Instant-Only model (SM101) is about 0.19% behind the Small Instant-Only model (SM98) (it drops one card layer) and still better than the show-time retrain (SM92) on instant (-0.057% [-0.079, -0.034], 460 users better / 261 worse).
- Cost of the smaller state formats on the instant head (80 users of the 732 set, f32 state as reference): int4 cards + int8 notes cost +0.0095% (the Small Instant-Only model (SM98)) and +0.0128% (the Tiny Instant-Only model (SM101)), below the 0.05% minimum effect. Full int4 state (int4 notes too) can diverge on long note histories and must not be used; keep notes at int8 or better.
- The Instant-Only models' curve numbers are not meaningful and are not listed. The Big show-time numbers are the step-2 scores from the show-time report, as packaged.
- 95% intervals, per-set tables, other input kinds and the paired comparisons are in each package's `INTEGRATION.md` (section 2) and `manifest.json`.

## What each download contains

Six zips, one per row of the table, plus `INTEGRATION.md` (the full porting manual) and `SHA256SUMS`. Each zip is a package in the same layout as the earlier single-model releases: `INTEGRATION.md`, `weights.f32.bin`, `weights.int8.bin` (per-channel int8), `architecture.json`, `manifest.json`, `parity.json` (NumPy reference vs PyTorch on leak-free inputs, max abs error about 1e-6), `reference/` (NumPy reference implementation), `testvectors/` with `check_testvectors.py` (run it: it must print `RESULT: PASS`) and, where the package has it (both Big packages, Small Instant-Only, both Tiny packages), `master.pth` (the float32 PyTorch checkpoint; the Small two-head package (SM92) has its checkpoint hash in the manifest but not the file).

Weight files: Small 1,351,553 B in f32 / 436,391 B in int8 (also used by the Small two-head model, same architecture), the Tiny models (SM106, SM101) 1,270,749 B / 409,483 B, and both Big models (SM91s2, SM99) 11,082,637 B / 3,027,096 B. The Big f32 file has the same layout as the fork's `RWKV_trained_on_5000_10000.bin`. The Big packages carry a shorter `INTEGRATION.md` (sections 1-10) that points to the Small packages' section 6 for the full feature list; the fork schedules with the raw forgetting curve and ignores the ahead residual, while the Big test vectors hold the training code's `p_curve` with the residual.

The optional answer-time head that exists for the show-time retrain (SM92) (`sm92-time`, in the repository) is **not shipped** in this release.

## How the fork uses these models

In the Anki fork (fork main cec8ca927, the Tiny one-model choice and Tiny for RWKV-Instant alone (SM112), 2026-11-13), by the schedulers the profile's presets use:

| Presets use | Big | Small (default) | Tiny |
|---|---|---|---|
| RWKV-Instant and RWKV-Curve (default) | the show-time big model (SM91s2) for the curve beside the show-time instant-only big model (SM99) for the instant head | the default small model (SM63) beside the Small Instant-Only model (SM98) | the default small model (SM63) beside the Tiny Instant-Only model (SM101) |
| both, with "use one model" | the show-time big model (SM91s2) alone | the show-time retrain (SM92) alone | the Tiny two-head model (SM106) alone |
| RWKV-Instant alone (no preset uses RWKV-Curve) | the show-time instant-only big model (SM99) alone | the Tiny Instant-Only model (SM101) alone | the Tiny Instant-Only model (SM101) alone |
| RWKV-Curve only | the show-time big model (SM91s2) alone | the default small model (SM63) alone | the default small model (SM63) alone |

- With both schedulers Tiny keeps the default small model (SM63) for the curve: it is more accurate there than the Tiny two-head model (SM106) (+0.09%).
- RWKV-Instant alone runs the Tiny Instant-Only model (SM101) for Small and Tiny alike (the smallest state and the fastest); Big, an explicit choice, keeps its own Instant-Only model.
- The two-head models of Small and Tiny serve as the one model of their size: the show-time retrain (SM92) for Small, the Tiny two-head model (SM106) for Tiny. The Tiny two-head model (SM106) has no answer-time head, so the fork shows no RWKV answer-time estimate with it.

The manifests of the Small Instant-Only model (SM98) and the Tiny Instant-Only model (SM101) say the model "runs beside the release (sm72e25)". That wording is out of date for this family: the right pairing is the one listed above, with the default small model (SM63) providing the curve.

## Important for anyone porting a model

- Use leak-free inputs: remove the signal that a card is deleted later (section 6.9 of `INTEGRATION.md`).
- Use show-time intervals: the interval runs up to the moment the card is shown, not up to the answer (section 6.10 of `INTEGRATION.md`). Inputs built with answer-time intervals are ones these models were never trained on.
- Keep notes at int8 state or better. Cards may be int4 (the fork default); full int4 state can diverge on long note histories.
- Never use an Instant-Only model (the show-time instant-only big model (SM99), the Small Instant-Only model (SM98), the Tiny Instant-Only model (SM101)) for `p_curve`, intervals or the 90% stability. Only `p_imm` and `p_rating` are valid from them.
- Run `testvectors/check_testvectors.py` in each package; it must print `RESULT: PASS`.

## Earlier releases

Earlier single-model releases were scored on different inputs (answer-time intervals, and some with the deleted-card leak), so their numbers are not directly comparable with the table above.
