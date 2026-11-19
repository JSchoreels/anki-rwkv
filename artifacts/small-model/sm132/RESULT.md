# Again pressed as Hard (SM132): result

**Verdict: the bet does not hold (curve, instant)** for the show-time two-head cool-down for Small (SM118), 722 users, unparseable row lines 0. The bet (95% CI upper bound <= +2.0% at p = 5, 10 and 20%) holds at 5% and 10% but fails at 20% for both heads: curve +2.738% [+2.451, +3.043], instant +3.362% [+3.059, +3.677]. Run 16:01 to 20:02 Thu 11-19, 240 GPU minutes in 1 ticket, 723 users scored per step.

- Again is 15.72% of all reviews (pre-run count 15.91% over 723 users; median user 14.59%), so p = 20% changes 3.15% of all reviews.
- Grade-only arm at 20% is identical to the Anki-state arm (+0.000% difference). These w64 models zero the card-state input in training and evaluation (`zero_features=(22,)`, dim 22 = scaled_state; `expertium/variants.py`, `expertium/model/variant_model.py`), so the card-state input is switched off in these models and only the recorded grade change matters.
- The big model (SM91s2) is slightly less robust than Small (SM118): +0.38% (curve) and +0.27% (instant) worse at 20%.
- LogLoss against the corrupted labels (what Anki's own metrics would show) goes down, but that is label noise (flipped Agains counted as passed), not a better model.
- Limitation: the timeline stays the real one. The short relearning interval after a hidden lapse remains in the history, which the model may partly read as a lapse; the test measures robustness to the recorded grade on a real timeline, not a replayed scheduler.

Full decision tables follow (from `sm132-decision.md`).


Readout (pre-registered): the bet holds for a head iff the 95% CI upper bound of 100 x (corrupted / clean - 1) is <= +2.0% at every p in {5%, 10%, 20%}; verdict holds iff both heads hold; rule users with finite rows at p = 0 and every p (at least 718 of 722). Show-time inputs, leak-free, the 732 set, published format w8+s8, paired user bootstrap 2,000, seed 0. % = 100 x (corrupted / clean - 1) of the review-weighted LogLoss, targets = the true outcomes.

- **The user's bet does not hold (curve, instant)** for the show-time two-head cool-down for Small (SM118) (722 users): curve does not hold, instant does not hold

Unparseable row lines skipped: 0

| p | share of all reviews changed | curve % [CI] | instant % [CI] | clean LogLoss curve / instant | corrupted LogLoss curve / instant | within the bet |
|---|---|---|---|---|---|---|
| 5% | 0.79% | +0.509% [+0.439, +0.580] | +0.582% [+0.515, +0.648] | 0.300016 / 0.276599 | 0.301544 / 0.278209 | yes |
| 10% | 1.57% | +1.118% [+0.983, +1.261] | +1.308% [+1.173, +1.447] | 0.300016 / 0.276599 | 0.30337 / 0.280217 | yes |
| 20% | 3.15% | +2.738% [+2.451, +3.043] | +3.362% [+3.059, +3.677] | 0.300016 / 0.276599 | 0.308231 / 0.285898 | no |
| 30% | 4.72% | +5.000% [+4.542, +5.487] | +6.348% [+5.831, +6.880] | 0.300016 / 0.276599 | 0.315016 / 0.294158 | (not in the bet) |
| 50% | 7.86% | +12.204% [+11.290, +13.131] | +16.100% [+15.005, +17.261] | 0.300016 / 0.276599 | 0.336631 / 0.321133 | (not in the bet) |

Again share: 15.72% of all reviews, 14.27% of the scored (benchmark) reviews of the rule users.

Grade-only arm at 20% (recorded card states kept; report only): curve +2.738% [+2.451, +3.043], instant +3.362% [+3.059, +3.677] vs clean; vs the Anki-state arm at 20%: curve +0.000% [+0.000, +0.000], instant +0.000% [+0.000, +0.000].

## LogLoss against the corrupted labels (secondary, no rule; dominated by label noise)

What Anki's own metrics would show for a collection with this habit: the same predictions scored against labels where a flipped Again counts as passed, vs the clean model on the true labels.

| p | curve % [CI] | instant % [CI] |
|---|---|---|
| 5% | -1.562% [-1.772, -1.347] | -0.983% [-1.262, -0.698] |
| 10% | -3.504% [-3.882, -3.116] | -2.493% [-2.986, -1.990] |
| 20% | -8.299% [-8.928, -7.649] | -6.641% [-7.458, -5.800] |
| 30% | -14.134% [-14.940, -13.289] | -12.023% [-13.082, -10.954] |
| 50% | -29.040% [-30.027, -28.028] | -26.549% [-27.811, -25.234] |

## the show-time big model (SM91s2) (report only, reference = its published rows, 722 users)

| p | curve % [CI] | instant % [CI] | Big minus Small, curve | Big minus Small, instant |
|---|---|---|---|---|
| 10% | +1.301% [+1.122, +1.495] | +1.459% [+1.324, +1.597] | +0.183% [+0.081, +0.319] | +0.151% [+0.125, +0.175] |
| 20% | +3.120% [+2.757, +3.489] | +3.630% [+3.322, +3.939] | +0.382% [+0.221, +0.621] | +0.268% [+0.226, +0.313] |

p = 0 of this run vs the published rows of the show-time two-head cool-down for Small (SM118): curve +0.000% [+0.000, +0.000], instant +0.000% [+0.000, +0.000] (largest per-user difference 0.0e+00).
