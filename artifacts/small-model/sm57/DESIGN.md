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

Branch `sm57`, cut from `sm92` at `22ca8edf` (main has no small-model training or scoring code; `sm92` has the release's cool-down recipe, the resumable decay (SM69) and the SM92 tools; its source is unchanged since SM92 ran).

## The loss option (`curve_loss_weight`, off by default)

`expertium/model/variant_model.py::_get_loss_variant` sums `0.5 * curve BCE + 0.5 * raw-curve BCE + IMMEDIATE_SCALE * instant CE + 1e-5 * w KL + 1e-4 * ahead-logit magnitude + 1e-3 * ahead-logit smoothness` (+ the grade-order hinge when on). KD enters inside the terms: the curve label becomes `alpha * teacher curve + (1 - alpha) * y`, the rating target `alpha * teacher probabilities + (1 - alpha) * one-hot`. The new attribute `curve_loss_weight` (float, default 1.0) multiplies the five curve constants (`0.5 * w`, `0.5 * w`, `1e-5 * w`, `1e-4 * w`, `1e-3 * w`): at 1.0 every constant is the same double, so the loss and gradients are bit for bit today's; at 0.0 the training loss is exactly the instant term (instant CE with its KD targets). The trainer sets it from the track option `"curve_loss_weight"` (`scripts/gpu_r01_training.py`, recorded in result.json), refusing it with `grade_order`; the key is added to the ordered-prefetch allowlist and to `TEACHER_BLIND` (it never changes a batch or the teacher input). Evaluation and the scorer are unaffected.

With weight 0 the curve-only parameters (`head_w.*`, `gru_*`) get zero gradients; the shared trunk and the instant head get all the gradient. Their AdamW moments (epoch 26) and the weight decay still move the curve-only weights a little, so the instant-only arm's curve is not meaningful; it is reported for completeness only.

## Read-out (normal inputs, leak-free)

Scoring as the release: `small_model_heldout.py --split-existing --deleted-cards neighbour`, w8+s8, the 732 set (723 users cap 0.8 + 3 long users cap 0.92); labels `sm57-inst-dec1ep32k/w8+s8`, `sm57-joint-dec1ep32k/w8+s8`, rows `gpu/sm57-rows.jsonl`. Release rows: the leak-free small model (SM72s) normal rows (`small-lf-neighbour.jsonl`).

`sm57_rule.py` (reuses `sm92_rule.compare`: 722 users of the 732 set, review-weighted LogLoss, points = % of the big model (1 leak-free epoch) normal-input LogLoss, paired user bootstrap 2,000, seed 0, 95% CI) reports instant LogLoss of each arm vs the joint arm and vs the release, the joint arm's curve vs the release, both vs the big model, the smaller-model bar, the 732+long pool, and training minutes, reviews/s and GPU memory per arm and stage.

## Pre-registered rule (fixed 2026-11-06 before any run)

Instant-only arm - joint arm, instant head, 722 users:

- **go** if the 95% CI is entirely below zero **and** the gain is at least 0.05 points (the release-candidate size of SM92's rule; this minimum was added by the user on 11-07 ~00:40, before the joint arm was scored). Then **stage 2 is proposed, not run**: a from-scratch comparison at small scale (for example a few epochs at 2,048 windows, same seeds, instant-only vs joint from the same random init), to see whether the curve task helps instant learn from the start.
- **stop** otherwise.

## CPU proofs (`sm57_check.py` -> `check-cpu-sm57.json`)

C1 SM72s epochs 21-25 rebuilt == disk. C2 joint epoch 26 == the recipe; instant-only == the recipe + exactly the flag; the two equal except the flag; wrong pairs refused. C3 loss (eager and TorchScript, with and without KD + sync-free loss, a real user's batch): option untouched == the loss code at `22ca8edf` bit for bit (loss, per-review losses, every gradient), weight 1.0 == untouched, weight 0.0 = exactly the instant average with only curve-head parameters newly at zero gradient and loss / gradients linear in the weight. C4 a CPU training slice through the trainer's batch path (run options, state-carry lanes), loss, clip and optimizer (AdamW and fresh Muon), TorchScript: the `wt/sm92` tree == this tree with the option absent == weight 1.0, every update and the final weights bit for bit; weight 0.0 trains, finite, loss == instant average. C5 KD cache digest unchanged; only the 4 option files differ from `22ca8edf`. C6 cool-downs prepared by this tree from epoch 25 == the release (joint) / the release + flag (instant-only), and equal to each other except the flag. C7 rule self-test (reproduces SM92's published normal pair).

GPU check (`sm57_gpu_check.py`, first ticket, ~5 min): the trainer's smoke (2 users x 512 reviews) of the joint epoch-26 folder run by the `wt/sm92` tree and by this tree, kept weights compared bit for bit; the instant-only smoke trains with `curve_loss_weight` 0.0 recorded; GPU memory of each.

## Schedule (estimates from SM92: epoch ~114 min, cool-down ~480 min wall, scoring ~15 min per model)

Launcher `launch-sm57.sh` (WMI hidden): GPU check -> instant-only epoch 26 -> joint epoch 26 -> instant-only cool-down -> its scoring -> joint cool-down -> its scoring -> rule. Epochs and cool-downs are snapshot-resumable (resumable decay (SM69)): a snapshot stop <= 2 min before a quiet window (quiet.json, scope all/gpu, plus the daily 20:00-01:00 break), resume after it. Expected: epochs ~02:30-06:30, instant-only cool-down ~06:30-14:30, joint cool-down ~14:45-20:00 then 01:00-~04:00 on 11-07, rule ~04:15 on 11-07.

## Stage 2 plan (fixed 2026-11-07 ~01:35, before any stage-2 run)

### Question

From the same random init, does the curve task help instant learn early (then the joint arm leads on instant in the first epochs and the stage-1 gain is a late-training effect), or does instant-only lead from the start too?

### Recipe: the factory's 2k tier from scratch

- **Users**: the accuracy-loop tier `screen` (`anki-rwkv/artifacts/accuracy/tiers.json`): 2,000 training users nested in the 8k final tier, stratified by history length, 146,654,211 reviews per epoch (a quarter of the 8k epoch). Disjoint from the 732 set and its 3 long users; none of them has a label-window trim (the leak-free small model (SM72s) trims file covers all 8k users: only user 5599, not in the tier).
- **Training**: the leak-free small model (SM72s)'s own from-scratch recipe, its `run.json` options unchanged: w64, 2,048-review windows x 16 strided state-carry lanes, leak-free replay KD (alpha 0.9, the big model (1 leak-free epoch) targets `big-leakfree-1ep-final8k-neighbour`, per-review so window- and order-independent), AdamW 7e-4 constant after the 5-update warm-up, `training_seed` 2025 (the fresh init and the per-window seeds), GPU memory cap 0.8, `RWKV_DELETED_CARDS=neighbour`. Protocol fields from SM72s's epoch 1. Epoch N order: the run's rule `random.Random(split_seed * 1000 + N)` over the tier's users. Epoch 1 from the trainer's fresh init (no checkpoint: `CatchUp.fresh_checkpoint(2025)`, the same init as SM72s's epoch 1, hash `47a33c78...`); epoch N > 1 from the arm's own epoch N-1 (weights, AdamW state, update offset). No cool-down: the read-out is on constant-LR checkpoints (the question is the trajectory, not a release).
- **Arms**: joint (no extra option) and instant-only (+ `{"curve_loss_weight": 0.0}`), identical otherwise. Folders `artifacts/small-model/sm57s2-{inst,joint}-eNN`, prepared by `sm57_stage2.py prepare`, run by `scripts/expertium_decay.py run` (snapshot every 400 users, resumable).

### Epochs, time cap and scoring

- Up to **5 epochs per arm**, interleaved (instant-only eN, joint eN, then the scoring of both when N is scored), so a cut run still has matched pairs.
- **Time plan** (`sm57_stage2.py plan`, once, after epoch 1 of both arms is trained and scored): F = the largest of 5, 4, 3, 2, 1 such that GPU minutes spent so far + (F - 1) x the measured epoch-1 pair minutes + 2 x the measured scoring minutes per checkpoint x the scored epochs after 1 stays <= 420 min (~7 h). Estimate (not measured): ~30 min per epoch per arm, ~13 min per scoring (stage 1: 12-13 min for 723 users), so F = 5 and ~6.5 h in all.
- **Scored epochs**: 1, 3 and F (both arms each), the cheapest set that answers "does instant-only lead from the start" (epoch 1), shows the trend (epoch 3) and gives the rule (F). Epochs 2 and 4 are kept on disk and can be scored later on demand.
- **Scoring**: as stage 1: `small_model_heldout.py --split-existing --deleted-cards neighbour`, w8+s8, normal inputs, the 732 set minus 9 never-fit (723 users) at cap 0.8 (one retry pass at 0.92 if a pass makes no progress); the 3 long users are skipped (not in the rule pool). Labels `sm57s2-{inst,joint}-eNN/w8+s8`, rows `gpu/sm57s2-rows.jsonl`.

### Pre-registered rule (`sm57_stage2.py rule`)

Instant-only - joint on the instant head at the final epoch F, 722 users of the 732 set (the stage-1 pool), review-weighted LogLoss, points = % of the big model (1 leak-free epoch) normal-input LogLoss, paired user bootstrap 2,000, seed 0, 95% CI (`sm92_rule.compare`, as stage 1):

- **go** if the 95% CI is entirely below zero **and** the gain is at least 0.05 points: instant-only wins from scratch too.
- **stop** otherwise.

Reported (not the rule): the same pair at every scored epoch (the trajectory, with users better / worse); the epoch-1 reading (instant-only leads / joint leads = the curve task helps instant early / no difference, from the epoch-1 CI); the joint arm's curve and both arms' instant vs the big model (1 leak-free epoch); joint vs the release (SM63) for scale; training minutes, reviews/s and GPU memory per arm and epoch; same fresh init (initial_parameter_sha256 of both epoch-1 results).

### Proofs

CPU (`sm57s2_check.py` -> `check-cpu-sm57-stage2.json`): S1 the tier (2,000 users inside the 8k run, disjoint from the 732 set, no trims, 146,654,211 reviews); S2 the fresh init hash is equal for both arms' options and equal to SM72s epoch 1's; S3 the epoch-1 folders differ only by the flag and folder names (rows = the order rule, joint options == SM72s run.json, both from the fresh init), with refusals (joint vs joint, swapped rows, flag 0.5); S4 epochs 1-5 are distinct permutations; S5 a CPU trainer slice from the fresh init (the trainer's batch path, state carry, loss, clip, fresh AdamW, TorchScript): the `wt/sm92` tree == this tree for joint bit for bit, instant-only sees the same updates and first-update predictions with loss == the instant average; S6 the stage-2 rule code reproduces stage 1's published pair on stage 1's rows; S7 the break logic on synthetic quiet files. Each later epoch is re-checked by `sm57_stage2.py pair` before it runs (own previous epoch as start, update offsets and update counts equal across arms).

GPU (`sm57s2_gpu_check.py`, first ticket): the joint epoch-1 trainer smoke (2 users x 512 reviews) by the `wt/sm92` tree == this tree bitwise; the instant-only smoke trains with weight 0 recorded; all three smokes start from the same init (initial_parameter_sha256 == SM72s epoch 1's).

### Breaks and launcher

`launch-sm57-stage2.sh` (new; `launch-sm57.sh` is finished and unchanged), queue owner `sm57`, WMI hidden. Breaks: quiet.json (scope all/gpu) plus the default nightly 20:00-01:00, except that a night with any quiet.json all/gpu window overlapping its 20:00-01:00 uses quiet.json's window instead of the default (the user moves the night's break there); re-read at every check. A new epoch starts only if 40 min fit; a running epoch gets a snapshot stop <= 2 min before a break and resumes after it. Expected (if the queue stays free): start ~02:00 on 11-07, end ~08:30.

## Stage 3 plan (fixed 2026-11-07 ~08:20, before any stage-3 GPU run)

### Question

Stage 2 showed that dropping the curve task helps the instant head (instant-only - joint at epoch 5: -0.773 points [-0.861, -0.685]). The mirror question: does the instant task also cost the curve? If both specialised arms beat the joint arm, two dedicated models, one per Anki scheduler (the model-per-scheduler idea (SM93)), would beat the shared model on both heads.

### One new arm: curve-only

- **Option**: the existing `instant_loss_weight` (SM26) at 0.0, no code change. `scripts/gpu_r01_training.py` applies it whenever it is not None, so 0.0 is applied; the model uses it only as `IMMEDIATE_SCALE`, the factor of the instant average in the loss (and the per-review loss tensor). The instant KD targets (the big model (1 leak-free epoch)'s rating probabilities, alpha 0.9) live inside that instant average (its soft rating target), so weight 0 removes the instant term, instant KD included; the curve terms (curve and raw-curve BCE with the curve KD targets, plus the curve head's three regularizers) are unchanged. The option is teacher-blind and an accepted batch option of the ordered prefetch, so the batches and teacher targets cannot change.
- **Everything else is stage 2's joint arm**: the same fresh init (training_seed 2025, hash `47a33c78...`), the 2k tier, the leak-free small model (SM72s) recipe and options, epoch order, per-window seeds, replay KD targets, AdamW 7e-4 constant, no cool-down, GPU memory cap 0.8. Epoch N > 1 from the arm's own epoch N-1 (weights, AdamW state, update offset). Folders `artifacts/small-model/sm57s3-curve-eNN`, built by `sm57_stage3.py prepare` from stage 2's own construction of the joint folder (`sm57_stage2.build("joint", N)`) with only the flag, the folder paths and the stage name changed; `sm57_stage3.py pair` re-checks every epoch against stage 2's joint folder before it runs.
- **Comparison**: stage 2's joint arm, not retrained (its checkpoints `sm57s2-joint-eNN` and its rows `gpu/sm57s2-rows.jsonl`, labels `sm57s2-joint-eNN/w8+s8`); the instant-only arm's rows for the two-model report.

### Epochs and scoring

5 epochs, scored at epochs 1, 3 and 5 (as stage 2: `small_model_heldout.py --split-existing --deleted-cards neighbour`, w8+s8, normal inputs, the 732 set minus 9 never-fit (723 users), cap 0.8, one retry pass at 0.92 if a pass makes no progress). Labels `sm57s3-curve-eNN/w8+s8`, rows `gpu/sm57s3-rows.jsonl`. Expected: ~5 x 30 min training + 3 x 14 min scoring ~ 3.3 h; no time plan (it fits before the 20:00 break).

### Pre-registered rule (`sm57_stage3.py rule`)

Curve-only - joint on the **curve** head at epoch 5, 722 users of the 732 set (the stage-1/2 pool), review-weighted LogLoss, points = % of the big model (1 leak-free epoch) normal-input LogLoss, paired user bootstrap 2,000, seed 0, 95% CI (`sm92_rule.compare`, as stages 1-2):

- **go** if the 95% CI is entirely below zero **and** the gain is at least 0.05 points: the instant task costs the curve too.
- **stop** otherwise.

Reported (no rule): the same pair at epochs 1 and 3; curve-only's instant head (expected untrained: the instant-head-only weights (`head_p`, `p_linear`) get zero gradients, so only weight decay, if their group has any, moves them); curve-only and joint vs the big model (1 leak-free epoch) on both heads; the **two-model comparison** per scored epoch: (instant-only instant, curve-only curve) vs joint (both heads), and whether both rules are go at epoch 5; training minutes, reviews/s and GPU memory per epoch next to stage 2's joint arm; same fresh init (initial_parameter_sha256 of curve-only e01 == stage 2 joint e01's).

### Proofs

CPU (`sm57s3_check.py` -> `check-cpu-sm57-stage3.json`): T1 the fresh init hash for the curve-only options == joint's == stage 2's epoch-1 results' == SM72s epoch 1's; T2 the epoch-1 folder differs from stage 2's joint e01 only by the flag, folder names and stage name, with refusals (joint vs joint, stage 2's instant-only folder, swapped rows, flag 0.5); T3 the option's path in the source (applied at 0.0, teacher-blind, batch-blind, only `IMMEDIATE_SCALE` in the loss, the instant KD inside that term); T4 a CPU trainer slice from the fresh init with the run's real replay KD (`sm57s3_probe.py`): the probe without the teacher == stage 2's probe bit for bit; the same batches; the same first forward; curve-only's loss == the curve terms of the loss exactly (and == the curve terms of joint's loss on the same first batch); reversing the teacher's rating targets leaves curve-only's losses, gradients and weights bit for bit unchanged (and changes joint's): the instant KD is gone; flipping the curve targets changes curve-only's loss: the curve KD stays; only the instant-head-only parameters (`head_p`, `p_linear`) lose their gradient, disjoint from the curve-only parameters instant-only zeroes; TorchScript, weights finite; T5 the rule code reproduces stage 2's published epoch-5 pair on stage 2's rows; T6 the break logic.

GPU (`sm57s3_gpu_check.py`, first ticket): the joint epoch-1 smoke by this tree == stage 2's GPU-check joint smoke bitwise (the GPU and tree still reproduce stage 2); the curve-only smoke trains with instant_loss_weight 0.0 recorded and different weights; both smokes from the same init == SM72s epoch 1's.

### Breaks and launcher

`launch-sm57-stage3.sh` (new; the stage 1 and 2 launchers are finished and unchanged), queue owner `sm57`, ticket purposes starting with "the instant-only small model (SM57) stage 3: ", WMI hidden. Breaks as stage 2 (quiet.json all/gpu plus the nightly 20:00-01:00 unless quiet.json moves that night's break; re-read at every check); a new epoch starts only if 40 min fit; a running epoch gets a snapshot stop <= 2 min before a break and resumes after it.
