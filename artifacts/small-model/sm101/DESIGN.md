# The smaller instant-only model (SM101): design (2026-11-10, prepared, not launched)

## Question

Can an instant-only model with less state per card still beat what the fork ships for RWKV-Instant?

The card-layer cut of the smaller model track (SM73) removes the card stream's second layer. On the normal two-head
recipe, its early check (the card-stream-1-layer check (SM73 stage 2)) cost +0.16 curve / +0.21 instant points, so that
track stopped. Instant-only training buys instant accuracy back.

This design asks whether the cut, trained instant-only on show-time inputs, does two things. First, it should stay
within the instant bar: +0.4% vs the big model (1 leak-free epoch) on normal inputs. Second, it should still beat the
show-time retrain of the release (SM92) on show-time inputs. The show-time retrain of the release (SM92) is what the
fork ships for RWKV-Instant.

Points = % of the big model (1 leak-free epoch) normal-input instant LogLoss (0.275757, 722 users of the 732 set).

Models named here:

| plain name (ID) | what it is |
|---|---|
| the smaller instant-only model (SM101) | this design: card stream 1 layer, instant-only, show-time inputs |
| the show-time instant-only model, stage 2 (SM98) | the same chain without the cut, 35 epochs + cool-down (no go vs the show-time instant-only small model (SM96), but the best of the chain) |
| the show-time instant-only small model (SM96) | 31 epochs + cool-down; went go vs the show-time retrain of the release (SM92) |
| the show-time retrain of the release (SM92) | what the fork's RWKV-Instant runs today |
| the release (SM63) | what the fork's RWKV-Curve runs |

## Where the state saving counts

| runtime case in the fork | instant model's state today | with the smaller instant-only model (SM101) |
|---|---|---|
| RWKV-Instant alone (full model alone) | 4,554 B per card + note (int8) | **3,134 B (-1,420 B, -31%)**; 2,110 B with int4 state |
| RWKV-Curve and RWKV-Instant both on (paired with the release) | 9,146 B (release 4,554 + instant 4,554 + 2 x 19 B pair headers) | **no saving today** (see below) |

- **The saving counts only where the instant model runs alone.**
  - With both heads on, the fork pairs the instant model with the release (SM63). Every stored state holds one state of
    each model (`rslib/src/rwkv/paired.rs`), and the release (SM63) keeps its 2 card layers.
  - The fork's engine has one architecture per build (`rslib/src/rwkv/architecture.rs`: `MODULE_LAYERS = [2, 4, 1, 3, 3]`
    and a fixed 13-step `LAYER_STEPS`). So today the smaller instant-only model (SM101) cannot be paired with the release
    (SM63) at all.
  - Even after an engine change that runs two architectures, the pair would only drop from 9,146 to 7,726 B (-16%).
- **Shipping it alone also needs that engine change**, because one build serves both cases. A model change also needs a
  card-state rebuild, since stored 2-layer card states are not valid for a 1-layer model. No state-format change is
  needed.
- **Where the bytes come from.** They follow the fork's layout (`ModuleState::compact`), using the formula of the SM73
  stage 2 addendum; `sm101_check.py` K8 reproduces it. The card state goes from 2,987 to 1,567 B; the note stays at
  1,567 B. Weights go from 331,511 to 311,786 parameters and the scorer's w8 weight bytes from 410,459 to 385,485.
  Weights do not grow with the collection.

## Warm start: cut the epoch-35 checkpoint of the show-time instant-only model, stage 2 (SM98)

| option | what | verdict |
|---|---|---|
| the cut model of the card-stream-1-layer check (SM73 stage 2) | the leak-free small model (SM72s) epoch 20, cut, + 1 normal two-head epoch | no: 10+ epochs behind this chain (epochs 22-25 two-head, 26-30 instant-only, 31+ show-time), so ~20 GPU h of epochs before the cool-down |
| cut epoch 31 of the show-time instant-only small model (SM96) | 31 epochs, instant-only, show-time | no: same recipe, but 4 epochs less training than the epoch 35 below |
| **cut epoch 35 of the show-time instant-only model, stage 2 (SM98) (chosen)** | 35 epochs, the last 9 instant-only, the last 5 on show-time inputs; constant LR, with its AdamW state | yes: the most trained checkpoint of this exact recipe; its uncut finish is the direct reference; the cut is the only change |

**The cut.** The tool is `scripts/sm73s2.py cut`, from the card-stream-1-layer check (SM73 stage 2). It removes:
- the 30 tensors `rwkv_modules.0.blocks.1.*`, 19,725 values, which form one residual branch;
- the AdamW state of 25 of them.

Everything else stays exactly as it was:
- the other 390 tensors are copied bit for bit;
- the AdamW state of 329 parameters is copied bit for bit;
- the other 12 layer steps keep their order.

The cut checkpoint is `artifacts/small-model/sm101-cut-e35` (master sha256 716edd0fe143). Its source is the epoch-35
master of the show-time instant-only model, stage 2 (SM98) (sha256 f7577d14ba10).

## What runs

| step | recipe | about |
|---|---|---|
| reference score | the show-time instant-only model, stage 2 (SM98)'s constant-LR epoch 35 (w64), show-time inputs, 723 users (the early check's reference) | 13 min |
| **epoch 36** | from the cut; the epoch recipe of the show-time instant-only model, stage 2 (SM98), with the epoch index advanced (see below); the only change is `architecture` w64-c1 | ~110 min |
| **early check** | epoch 36 scored on show-time inputs; rule below | 13 min |
| epochs 37, 38 | the same step from the previous epoch (options unchanged) | ~2 x 110 min |
| **cool-down** | the release's one-epoch 32k Muon cool-down on show-time inputs (see below); equal to the cool-down of the show-time instant-only model, stage 2 (SM98), except the start (epoch 38) and the architecture | ~430-480 min |
| scoring | show-time and normal inputs on the 732 set (723 + 3 long users), then the 1,500 pool on show-time inputs | ~55 min |

The epoch recipe is: 8k final tier, the run's order rule, 2048 x 16 lanes carry, leak-free replay KD, AdamW 7e-4
constant, `curve_loss_weight` 0 and `show_time_intervals` true.

The cool-down is: fresh Muon, LR 7e-4 -> 0 on a one_minus_sin schedule, KD replay alpha 0.9, curve weight 0.

**Why three recovery epochs.** That is what the SM73 design planned for this cut. In the show-time instant-only model,
stage 2 (SM98), 4 more uncut epochs gained only -0.035 points. So the extra epochs here are for the cut to recover, not
a gain of their own.

## Early check (stop rule, `sm101_rule.py early`, fixed before launch)

The check compares two constant-LR checkpoints:
- the arm: epoch 36 of the smaller instant-only model (SM101) (w64-c1);
- the reference: epoch 35 of the show-time instant-only model, stage 2 (SM98) (w64).

Both are scored on the instant head, on show-time inputs, for 722 users, paired.

- **Stop** if the point estimate is above **+0.25 points**. The launcher then ends after about 2.3 GPU h, with no
  further GPU work.
- **Continue** otherwise.

**Why +0.25.**
- The final rule leaves about 0.20 points for the cut on show-time inputs. The uncut model, the show-time instant-only
  model, stage 2 (SM98), is -0.248 vs the show-time retrain of the release (SM92), and a go needs <= -0.05 with the CI
  below 0.
- The two remaining epochs and the cool-down are not expected to win back more than ~0.05 beyond what epoch 36 shows.
- The two checkpoints are one epoch apart; that epoch's own gain is ~0.01 points.
- Constant-LR read-outs move by about 0.06 instant points from one epoch to the next.
- So this is a coarse filter for a clearly bad cut, not a measurement.

## Final rule (`sm101_rule.py decide`, pre-registered)

All numbers here use 722 users of the 732 set, review-weighted LogLoss, the instant head, a paired user bootstrap
(2,000 resamples, seed 0, 95% CI), and points as the unit.

- **A, the bar.** The smaller instant-only model (SM101) minus the big model (1 leak-free epoch), both on normal inputs.
  Pass if the 95% CI's upper end is <= **+0.40** points, so the whole CI is inside the SM73 instant bar.
- **B, what the fork ships.** The smaller instant-only model (SM101) minus the show-time retrain of the release (SM92),
  both on show-time inputs. Pass if the 95% CI is entirely below 0 AND the gain is >= **0.05** points (points <= -0.05).
- **Go = A and B**; no go otherwise.

Reported, not part of the verdict:
- the smaller instant-only model (SM101) minus the show-time instant-only small model (SM96), on show-time and normal
  inputs;
- it minus the show-time instant-only model, stage 2 (SM98), on show-time and normal inputs;
- it minus the show-time retrain of the release (SM92) on normal inputs;
- it minus the release (SM63) on show-time inputs;
- its show-time cost;
- the 1,500 pool on show-time inputs.

The curve head is untrained, so it is never part of the verdict.

**Room under the rule.** The rule's selftest scores the uncut model, the show-time instant-only model, stage 2 (SM98),
as if it were the new model:
- A: +0.110 [+0.081, +0.138]. A cut that costs up to ~0.26 points still passes A.
- B: -0.248 [-0.270, -0.228]. A cut that costs up to ~0.19 points still passes B.
- **So B is the condition that binds.**

**What to expect.** Suppose the cut costs what it cost on the two-head recipe (+0.21). Then the smaller instant-only
model (SM101) lands at about -0.04 points vs the show-time retrain of the release (SM92), which fails B narrowly. It
lands at about +0.32 vs the big model (1 leak-free epoch), which passes A. So it goes only if instant-only training and
the longer finish win back a few hundredths of a point. That is plausible, but not likely enough to skip the run. The
early check limits the loss to ~2.3 GPU h.

**If the fork switches RWKV-Instant.** The fork might switch from the show-time retrain of the release (SM92) to the
show-time instant-only small model (SM96). The honest comparison then becomes the smaller instant-only model (SM101)
minus the show-time instant-only small model (SM96), which the rule reports. The smaller instant-only model (SM101) will
very likely lose ~0.2 points there. The trade then is 31% less state on profiles that use only RWKV-Instant, for ~0.2
points of instant LogLoss.

## Verdict of the design

**Worth running, with the early stop.**

- **The saving is real but narrow.** It is -1,420 B per card + note (-31%), but only on profiles that use RWKV-Instant
  alone, and it needs a fork engine change before it can be used.
- **The accuracy question is open.** The known cut cost (+0.21 points) sits right at the edge of the binding room
  (~0.19 points vs the show-time retrain of the release (SM92)). The result is not obvious either way.

## GPU budget and timeline (if launched ~12:00 Tue 2026-11-10, queue empty)

| step | ends about |
|---|---|
| reference scoring, smoke, epoch 36, its scoring, early rule | 14:20 Tue (early check: ~2.3 GPU h) |
| epochs 37, 38 | ~18:00 |
| cool-down prepare + checks + smoke; first cool-down segment to the 19:58 snapshot stop | 19:58 |
| break 20:00-01:00 (a quiet.json window replaces a night's default) | |
| cool-down resume (~330-360 min left) | ~06:30-07:00 Wed |
| scoring + rules | **~07:30-08:00 Wed 2026-11-11** |

About **14.5 GPU h** in all (the early check is ~2.3 h of it), and ~20 h wall time with the break.

The speeds assumed:
- **Epochs.** The cut model of the card-stream-1-layer check (SM73 stage 2) ran at 92.8k reviews/s on normal inputs.
  The show-time instant-only model, stage 2 (SM98) ran at 87.4k on show-time inputs.
- **Cool-down.** The cut model's quarter cool-down ran at 22.5k reviews/s. The cool-down of the show-time instant-only
  model, stage 2 (SM98) ran at 20.2k.

## Preparation (see HANDOFF.md)

- **Branch.** `sm101` is cut from `origin/sm98`, not from main, because main lacks this chain's trainer code:
  - there is no `w64-c1` preset;
  - there is no `show_time_intervals`;
  - there is no `scripts/sm77_warm.py`.
- **Code.** The code trees are bitwise the ones the show-time instant-only small model (SM96) and the show-time
  instant-only model, stage 2 (SM98) trained with (c99330a7).
- **CPU proofs.** `check-cpu-sm101.json` is ALL OK.
- **Cut equivalence.** The cut's proofs were re-run on this code (`check-cut-equivalence.json`, ALL OK). They show that
  the trainer and scorer paths are bitwise the full model with the layer skipped.
- **GPU smokes (2 users each).** Both passed:
  - the trainer on epoch 36;
  - the scorer on w64-c1 with show-time inputs.
- **Dry run.** OK.
