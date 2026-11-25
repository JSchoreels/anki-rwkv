# RWKV family 2026-11-28: Big, Small and Tiny, with answer-time heads and smaller states

Tag `rwkv-family-2026-11-28`. Built by `artifacts/small-model/sm166/build_release.py` (the weekend release packages (SM166)) against the Anki fork's main branch at 6cb164606.

This release supersedes the 13 November release for new integrations. The 13 November release (`rwkv-family-2026-11-13`) stays downloadable, unchanged.

## What changed since 13 November

- **New Small and Tiny models.** Small: the show-time two-head cool-down for Small (SM118) and the longer Small Instant-Only model (SM113). Tiny: the six-layer Tiny two-head model (SM130) and the six-layer Tiny Instant-Only model (SM130), a lighter architecture (`w64-c1-d2p1u1`, 185,057 parameters instead of 311,786). Big is unchanged: the show-time big model (SM91s2) and the show-time instant-only big model (SM99), same weight files as on 13 November.
- **The older Tiny weights are gone from this release:** the Tiny two-head model (SM106) and the Tiny Instant-Only model (SM101) are only in the 13 November release.
- **Answer-time heads for all six models.** Each package now has `time_head.f32.bin`, `time_head.int8.bin` and `time_head.json` (format 2). They predict how long the answer will take per button (4 grades x 16 log-time bins), which the Anki fork shows in Card Info.
- **Tiny instant ensemble in the fork (SM170).** When the fork runs both Tiny models (the default), Tiny's instant recall combines them: `sigmoid(0.69 * logit(p_IO) + 0.31 * logit(p_TH))`, where `p_IO` is the six-layer Tiny Instant-Only model (SM130)'s instant recall and `p_TH` the six-layer Tiny two-head model (SM130)'s instant head, each clipped to [1e-7, 1 - 1e-7]. Instant LogLoss is -0.067% against the Instant-Only model alone on the 732 set (the deploy-time ensemble test (SM160)) and -0.062% [-0.066, -0.057] on 1,385 users (the decisive check (SM169)). The weights are unchanged; the two Tiny manuals describe the combination in section A.2.
- **Smaller stored state.** The fork now stores each card and note state as low-rank factors (T24: rank 2 below 16 answers, rank 4 from the 16th), with 6-bit factors for Big and Small and 8-bit factors for Tiny. That is 85% (Big), 71% (Small) and 65% (Tiny) fewer state bytes than int8 rows, within the accuracy budgets below. Section A.3 of each `INTEGRATION.md` describes it.
- **Every package has `WEIGHTS-NOTICE.md`**, and every package's `INTEGRATION.md` now starts with a Part A for this release, followed by the 13 November manual of the same slot as Part B.

## Which model should I use?

Same rule as before. Use a **two-head model** if you need RWKV-Curve (the forgetting curve, intervals, the 90% stability) and RWKV-Instant from one model. Use an **Instant-Only model** if you only schedule with RWKV-Instant: its curve head is untrained and must never be used, but its instant head is better. Pick the largest size you can afford. All models expect **leak-free** and **show-time** inputs (sections 6.9 and 6.10 of `INTEGRATION.md`).

## The family

The table gives LogLoss on held-out benchmark users: the 732-user set (722 users), review-weighted, show-time inputs, leak-free, int8 weights and int8 state (w8+s8). "% vs big" is measured against the big model (1 leak-free epoch) on the same inputs (curve 0.29958, instant 0.277157), and negative is better. The last column is the release-candidate re-check of the final state formats (SM164): the 95% CI upper bound of the total cost at the fork's state format. Big is measured against its own int8 (budget +0.15%), Small against the matching Big model at int8 (budget +0.5%) and Tiny the same way (budget +1.0%). Every cell passes.

| Size | Variant | Model | Parameters | Show-time curve LogLoss (% vs big) | Show-time instant LogLoss (% vs big) | Fork state format | Budget check (CI upper, curve / instant) |
|---|---|---|---|---|---|---|---|
| Big | two-head | the show-time big model (SM91s2) | 2,762,884 | 0.299415 (-0.06%) | 0.276183 (-0.35%) | 6-bit factors (`0x26`) | +0.055% / +0.077% of +0.15% |
| Big | Instant-Only | the show-time instant-only big model (SM99) | 2,762,884 | not used | 0.275894 (-0.46%) | 6-bit factors (`0x26`) | - / +0.084% of +0.15% |
| Small | two-head | the show-time two-head cool-down for Small (SM118) | 331,511 | 0.300016 (+0.15%) | 0.276599 (-0.20%) | 6-bit factors (`0x26`) | +0.412% / +0.243% of +0.5% |
| Small | Instant-Only | the longer Small Instant-Only model (SM113) | 331,511 | not used | 0.275933 (-0.44%) | 6-bit factors (`0x26`) | - / +0.116% of +0.5% |
| Tiny | two-head | the six-layer Tiny two-head model (SM130) | 185,057 | 0.301416 (+0.61%) | 0.278049 (+0.32%) | 8-bit factors (`0x28`) | +0.908% / +0.806% of +1.0% |
| Tiny | Instant-Only | the six-layer Tiny Instant-Only model (SM130) | 185,057 | not used | 0.27749 (+0.12%) | 8-bit factors (`0x28`) | - / +0.720% of +1.0% |

Notes on the table:

- Against the 13 November models of the same slot (722 users, w8+s8): the show-time two-head cool-down for Small (SM118) is -0.113% on the curve and -0.100% on instant vs the show-time retrain (SM92). The longer Small Instant-Only model (SM113) is -0.094% vs the Small Instant-Only model (SM98). The six-layer Tiny models are measured against Small, not against the older Tiny: the two-head model is +0.467% (curve) and +0.524% (instant) vs the show-time two-head cool-down for Small (SM118), and the Instant-Only model is +0.564% vs the longer Small Instant-Only model (SM113). Both are inside the user's Tiny budget of +1% vs Small.
- The fork's state format itself costs +0.034% to +0.089% vs the model's own int8 state on the 732 set (section A.2 of each package).
- **Long cards** (128 or more answers) were checked as a warning, not a rule. Small with 6-bit factors has no flagged cell. Tiny with T24 is at +0.002% (curve) and -0.044% (instant). Big with T24 is within ±0.03%, and Big with 6-bit factors is inside the tolerance too: at most +0.022% vs int8, largest CI upper bound +0.299% (the Big long-card check with 6-bit factors (SM165), 50 users).

## State size

These are card + note bytes for one model in the fork's layout, as means over the answer counts of 723 benchmark users. The int8 rows and T24 bytes come from the fork's `docs/rwkv-lowrank-state-2026-11-23.MD`. The 6-bit bytes are estimates from the smaller shift vectors and factors experiment (SM152) savings.

| Size | int8 rows | T24, 8-bit factors | Fork default | Saving vs int8 | Store of a synthetic 14,000-card collection, both models (int8 rows -> default, SM163) |
|---|---|---|---|---|---|
| Big | 23,158 B | 4,244 B | about 3,570 B (6-bit) | -84.6% | 672.7 MB -> 185.0 MB |
| Small | 3,860 B | 1,309 B | about 1,100 B (6-bit) | -71.4% | 119.3 MB -> 51.9 MB |
| Tiny | 2,630 B | 932 B | 932 B (8-bit) | -64.7% | 80.3 MB -> 44.9 MB |

## Speed in the fork (SM163)

Measured on 2026-11-25 on a quiet desktop with fork main eda153ddb, a release build and synthetic data only. The collection has 14,000 cards and 205,785 answers. Each run used the size's pair and all threads, and each number is the median of 3 passes.

| Size | First-launch state build | vs int8 rows | Whole rebuild | Per review p50, show time / answer |
|---|---|---|---|---|
| Big | 14,346 reviews/s | +38.8% | 14.6 s | 3.0 ms / 0.76 ms |
| Small | 45,156 reviews/s | -3.9% | 4.7 s | 0.64 ms / 0.20 ms |
| Tiny | 80,569 reviews/s | -3.9% | 2.7 s | 0.36 ms / 0.11 ms |

On one thread with long histories (every card past 16 answers), Small trailed int8 rows by 17% and Tiny by 24% in this measurement. **Fork speed change since then:** fork main 6cb164606 batches the low-rank codec (SM168). The stored bytes are identical; the one-thread long-history gap shrinks to 3.2% (Small) and 5.0% (Tiny), and the first-launch rebuild gets 5.4% (Big), 4.0% (Small) and 5.0% (Tiny) faster on the same machine (AVX2 and AVX-512 CPUs; other CPUs keep about the old speed).

## How the fork uses these models

The Anki fork's main branch (6cb164606) runs exactly these six models, byte for byte, and a size never borrows another size's model.

| Presets use | Big | Small (default) | Tiny |
|---|---|---|---|
| RWKV-Curve and RWKV-Instant (default) | the show-time big model (SM91s2) beside the show-time instant-only big model (SM99) | the show-time two-head cool-down for Small (SM118) beside the longer Small Instant-Only model (SM113) | the six-layer Tiny two-head model (SM130) beside the six-layer Tiny Instant-Only model (SM130), instant recall from the Tiny instant ensemble (SM170) |
| both, with "use one model" | the show-time big model (SM91s2) alone | the show-time two-head cool-down for Small (SM118) alone | the six-layer Tiny two-head model (SM130) alone |
| RWKV-Instant alone | the show-time instant-only big model (SM99) alone | the longer Small Instant-Only model (SM113) alone | the six-layer Tiny Instant-Only model (SM130) alone |
| RWKV-Curve only | the show-time big model (SM91s2) alone | the show-time two-head cool-down for Small (SM118) alone | the six-layer Tiny two-head model (SM130) alone |

Each size now shows the RWKV answer-time estimate. A head always runs on the model it was trained on: one model alone uses its own head, and a pair uses its Instant-Only model's head.

The manifest of the longer Small Instant-Only model (SM113) says it runs beside the show-time retrain (SM92). That wording is out of date: it runs beside the show-time two-head cool-down for Small (SM118).

## What each download contains

There are six zips, one per row of the family table, plus `INTEGRATION.md` (the Small two-head package's manual), `WEIGHTS-NOTICE.md` and `SHA256SUMS`. Each zip holds:

- `INTEGRATION.md` (Part A for this release, then the 13 November manual of the same slot as Part B);
- `WEIGHTS-NOTICE.md`;
- the weights: `weights.f32.bin`, `weights.int8.bin` and `master.pth` (the float32 PyTorch checkpoint, in every package);
- the answer-time head: `time_head.f32.bin`, `time_head.int8.bin` and `time_head.json`;
- `architecture.json`, `manifest.json` and `parity.json`;
- `reference/` (the NumPy reference, now with `answer_time_numpy.py`);
- `testvectors/`.

Run both checks in `testvectors/`; each must print `RESULT: PASS`:

```
python testvectors/check_testvectors.py
python testvectors/check_time_vectors.py
```

**Test vectors and training data:** every test vector comes from two synthetic users (900001 and 900002). They are simulated review histories (fixed seed) in the dataset's on-disk format, written by `scripts/synthetic_testvector_users.py`. No row of the training data is in any package. The models were trained on the anki-revlogs-10k dataset, which is linked rather than shipped: https://huggingface.co/datasets/open-spaced-repetition/anki-revlogs-10k (gated, license "anki-revlogs-10k").

**Weights are not MIT.** Read `WEIGHTS-NOTICE.md`. The weights (`weights.*.bin`, `time_head.*.bin`, `master.pth`) are not under the code's MIT License, and this project grants no license of its own to them: they stay under Anki's terms for the training data. Those terms allow memory research and individuals' own research; any other use, including shipping the weights in an application, needs permission from Anki. Redistribution is an open question upstream (srs-benchmark issue #353).
