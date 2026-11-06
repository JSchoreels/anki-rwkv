# The instant-only small model (SM57): design (2026-11-06)

## Question

Many users mostly use RWKV-Instant. The small model shares one trunk between the curve head and the instant head. Does giving the whole model to the instant task (no curve loss at all) give better instant predictions than sharing it with the curve, or does the curve task help instant? The goal is instant LogLoss, not size (a curve-only model could cover the curve users later).

## Design chosen: stage 1, a matched fine-tune from the release's start

Both arms start from the same checkpoint and see exactly the same data; the only difference is the loss.

| | Instant-only arm | Matched joint arm | Release (read only) |
|---|---|---|---|
| Start | the leak-free small model (SM72s) constant-LR epoch 25 (w64), its AdamW state, update offset 551,950 | the same | the same |
| Epoch 26 | `sm77_warm.py epoch --epoch 26` (the run's own epoch-26 protocol: order `Random(split_seed * 1000 + 26)`, epoch 25's label-window trims, run.json options, constant LR 7e-4, leak-free replay KD alpha 0.9) + `{"curve_loss_weight": 0.0}` | the same, no extra option (bit-identical to what SM72s's own epoch 26 would be) | none |
| Cool-down | the release's recipe `sm77_warm.py cooldown --epoch 25 --fraction 1.0` (tier final, shuffle seed 1, 8,000 users, 32k windows, batch_chunks 1, fresh Muon `expertium-iter53`, LR 7e-4 -> 0 one_minus_sin, KD replay `big-leakfree-1ep-final8k-neighbour` alpha 0.9) from its own epoch 26 + `{"curve_loss_weight": 0.0}` | the same from its own epoch 26, no extra option | `wt/sm72s/.../scale-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon` |
| Inputs | normal (answer-time), the Anki fork's default for the release from epoch 25 (SM63) | normal | normal |

Why this is the matched setup: the release's own machinery (the show-time retrain of the release (SM92): same start, same epoch-26 protocol, same cool-down, same scorer and pairing) with one switch. Same users, order, seeds, windows, KD targets, LR schedule and optimizer in both arms (proved by `sm57_pair.py arms`). The joint arm is also epoch 26 of the "more epochs" run of the release from epoch 25 (SM63) (epochs 26-30); its epoch-26 checkpoint is kept (`artifacts/small-model/sm57-joint-e26/expertium/E89-F65e-F77a/master.pth` + `optimizer.pth`). Both arms have one more constant-LR epoch than the release; the joint arm measures that confound, which is why the rule compares against the joint arm, not the release.

Branch `sm57`, cut from `sm92` at `29727c6b` (main has no small-model training or scoring code; `sm92` has the release's cool-down recipe, the resumable decay (SM69) and the SM92 tools; its source is unchanged since SM92 ran).

## The loss option (`curve_loss_weight`, off by default)

`expertium/model/variant_model.py::_get_loss_variant` sums `0.5 * curve BCE + 0.5 * raw-curve BCE + IMMEDIATE_SCALE * instant CE + 1e-5 * w KL + 1e-4 * ahead-logit magnitude + 1e-3 * ahead-logit smoothness` (+ the grade-order hinge when on). KD enters inside the terms: the curve label becomes `alpha * teacher curve + (1 - alpha) * y`, the rating target `alpha * teacher probabilities + (1 - alpha) * one-hot`. The new attribute `curve_loss_weight` (float, default 1.0) multiplies the five curve constants (`0.5 * w`, `0.5 * w`, `1e-5 * w`, `1e-4 * w`, `1e-3 * w`): at 1.0 every constant is the same double, so the loss and gradients are bit for bit today's; at 0.0 the training loss is exactly the instant term (instant CE with its KD targets). The trainer sets it from the track option `"curve_loss_weight"` (`scripts/gpu_r01_training.py`, recorded in result.json), refusing it with `grade_order`; the key is added to the ordered-prefetch allowlist and to `TEACHER_BLIND` (it never changes a batch or the teacher input). Evaluation and the scorer are unaffected.

With weight 0 the curve-only parameters (`head_w.*`, `gru_*`) get zero gradients; the shared trunk and the instant head get all the gradient. Their AdamW moments (epoch 26) and the weight decay still move the curve-only weights a little, so the instant-only arm's curve is not meaningful; it is reported for completeness only.

## Read-out (normal inputs, leak-free)

Scoring as the release: `small_model_heldout.py --split-existing --deleted-cards neighbour`, w8+s8, the 732 set (723 users cap 0.8 + 3 long users cap 0.92); labels `sm57-inst-dec1ep32k/w8+s8`, `sm57-joint-dec1ep32k/w8+s8`, rows `gpu/sm57-rows.jsonl`. Release rows: the leak-free small model (SM72s) normal rows (`small-lf-neighbour.jsonl`).

`sm57_rule.py` (reuses `sm92_rule.compare`: 722 users of the 732 set, review-weighted LogLoss, points = % of the big model (1 leak-free epoch) normal-input LogLoss, paired user bootstrap 2,000, seed 0, 95% CI) reports instant LogLoss of each arm vs the joint arm and vs the release, the joint arm's curve vs the release, both vs the big model, the smaller-model bar, the 732+long pool, and training minutes, reviews/s and GPU memory per arm and stage.

## Pre-registered rule (fixed 2026-11-06 before any run)

Instant-only arm - joint arm, instant head, 722 users:

- **go** if the 95% CI is entirely below zero. Reported with it: whether the gain is at least 0.05 points (the release-candidate size of SM92's rule). Then **stage 2 is proposed, not run**: a from-scratch comparison at small scale (for example a few epochs at 2,048 windows, same seeds, instant-only vs joint from the same random init), to see whether the curve task helps instant learn from the start.
- **stop** otherwise.

## CPU proofs (`sm57_check.py` -> `check-cpu-sm57.json`)

C1 SM72s epochs 21-25 rebuilt == disk. C2 joint epoch 26 == the recipe; instant-only == the recipe + exactly the flag; the two equal except the flag; wrong pairs refused. C3 loss (eager and TorchScript, with and without KD + sync-free loss, a real user's batch): option untouched == the loss code at `29727c6b` bit for bit (loss, per-review losses, every gradient), weight 1.0 == untouched, weight 0.0 = exactly the instant average with only curve-head parameters newly at zero gradient and loss / gradients linear in the weight. C4 a CPU training slice through the trainer's batch path (run options, state-carry lanes), loss, clip and optimizer (AdamW and fresh Muon), TorchScript: the `wt/sm92` tree == this tree with the option absent == weight 1.0, every update and the final weights bit for bit; weight 0.0 trains, finite, loss == instant average. C5 KD cache digest unchanged; only the 4 option files differ from `29727c6b`. C6 cool-downs prepared by this tree from epoch 25 == the release (joint) / the release + flag (instant-only), and equal to each other except the flag. C7 rule self-test (reproduces SM92's published normal pair).

GPU check (`sm57_gpu_check.py`, first ticket, ~5 min): the trainer's smoke (2 users x 512 reviews) of the joint epoch-26 folder run by the `wt/sm92` tree and by this tree, kept weights compared bit for bit; the instant-only smoke trains with `curve_loss_weight` 0.0 recorded; GPU memory of each.

## Schedule (estimates from SM92: epoch ~114 min, cool-down ~480 min wall, scoring ~15 min per model)

Launcher `launch-sm57.sh` (WMI hidden): GPU check -> instant-only epoch 26 -> joint epoch 26 -> instant-only cool-down -> its scoring -> joint cool-down -> its scoring -> rule. Epochs and cool-downs are snapshot-resumable (resumable decay (SM69)): a snapshot stop <= 2 min before a quiet window (quiet.json, scope all/gpu, plus the daily 20:00-01:00 break), resume after it. Expected: epochs ~02:30-06:30, instant-only cool-down ~06:30-14:30, joint cool-down ~14:45-20:00 then 01:00-~04:00 on 11-07, rule ~04:15 on 11-07.
