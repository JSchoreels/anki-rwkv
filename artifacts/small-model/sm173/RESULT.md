# Deleted-card note ids (SM173): result

Verdict: **HARMLESS** under the pre-registered rule (`design.md`, "Rules"); no phase 2. Full tables: `sm173-result.md`.

## What was found

- **Cause.** `rwkv/data_processing.get_rwkv_data` stores the placeholder note id `ID_PLACEHOLDER + card_id` in a float64
  column (spacing 64 at 3.14e17), so deleted cards in one 64-wide block of card ids share a note lane; the int32 cast in
  `create_sample` then turns every placeholder into one value, so in scoring all deleted cards of a user share one note
  code. The fork is not affected: it keeps 64-bit integer ids, so each deleted card gets its own placeholder note.
- **Size.** Deleted-card reviews are 27.0% of the training reviews (93.5% of users), 24.7% of the 732 set (95.0% of
  users) and 25.4% of the 1,500 pool; about 47 deleted cards share each lane (`sm173-measure.json`).

## Effect at inference (732 set, 723 rule users, 617 with deleted-card reviews)

Giving each deleted card its own note ("fix", what the fork does) against the current pipeline ("cur"):

| Model | Scope | Curve | Instant |
|---|---|---:|---:|
| six-layer Tiny two-head (SM130) | overall | +0.000% [-0.029, +0.024] | +0.026% [+0.014, +0.039] |
| six-layer Tiny two-head (SM130) | existing cards only | -0.012% [-0.049, +0.013] | +0.010% [+0.006, +0.015] |
| Small two-head (SM118) | overall | +0.018% [-0.032, +0.052] | +0.057% [+0.042, +0.073] |
| Small two-head (SM118) | existing cards only | -0.010% [-0.064, +0.024] | +0.011% [+0.006, +0.018] |

The fix makes the released models slightly **worse**, not better: they were trained with the collapsed ids and adapted
to them. A reseeded control ("reseed vs cur") stays within ±0.01%, so the differences are not noise.

## What it means

- **For the fork:** the fork only shows predictions for existing cards. On existing cards the train/serve mismatch costs
  about +0.010% (Tiny) and +0.011% (Small) instant LogLoss, far under the 0.05% bar. No fork change is needed now.
- **For the next family:** whether training with correct ids would help is not answered by an inference test.
  If the next family is retrained from scratch anyway, use correct 64-bit placeholder ids from the start and keep the
  fork as it is (it is already correct), so the mismatch disappears for free.
- **Open detail:** 53 to 69 users without deleted cards are not bitwise equal between arms (expected: equal). The
  differences are tiny; likely GPU nondeterminism across tickets. Not investigated further.
