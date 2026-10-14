# Expertium catch-up: 5,000-user-era options

This repository vendors Expertium's 100-user-era iter36 model (commit `39666c6`, [expertium/upstream.json](../expertium/upstream.json)). His 5,000-user era restarted iteration numbers and moved to a width-80 lineage: iter 31 merged the width-80 trunk with PAVA, a GRU curve head and Muon, iter 32 added distillation from a d=128 teacher, and iter 53 (`iter53_muonlora`, promoted 2026-08-18) is his last promoted champion. After it he fixed a query-row clock leak (2026-09-10) and ran warmup-stable-decay endgames. This page lists what was ported, as opt-in options, from [Expertium/rwkv-anki-autoresearch](https://github.com/Expertium/rwkv-anki-autoresearch) at `a9fcf466b7794f2e109ef851916d759c8ea6b768` (2026-10-12). The upstream files read, with their sha256, are pinned under `catch_up` in [expertium/upstream.json](../expertium/upstream.json).

The default model, data and training are unchanged: every switch is off unless a track (R01 options dict) or a full-run `prepare` sets it, and [scripts/check_expertium_catch_up.py](../scripts/check_expertium_catch_up.py) proves the defaults bitwise.

Which champion: iter 53 is his last promoted champion and the last one on the published 92-input layout this repository trains on (val half: curve 0.297523 / instant 0.265191 on his split, not comparable to ours). His later reference models depend on his rebuilt `-id` dataset (review times, 103-114 inputs). Examples: featB, realcyc, and `w10lf`, the leak-free 10+2-epoch base he ships (561,732 parameters). Those layouts cannot be fed from our data, so their feature changes are not ported; their training findings (decay length, clock leak) are reflected below.

## Recipes for consumers

- Width-80 champion, accepted schedule: track options `{**accepted, "architecture": "w80"}` passed through `compatible_options`, or `run_full_training.py prepare ... --extra-options '{"architecture": "w80"}'`.
- Closest to his iter-53 recipe: `--extra-options '{"architecture": "w80", "optimizer": "expertium-iter53", "distillation": {"teacher": "C:/Users/jonat/workspace/anki/qt/aqt/rwkv_inference/RWKV_trained_on_5000_10000.pth", "alpha": 0.9, "decay_alpha": 0.5}}' --final-lr-schedule '{"decay_start": 0.75, "shape": "one_minus_sin"}'`. Read the contamination warning in item 5 first.
- Default width with only the leak fix and feature drop: `--extra-options '{"end_to_start_intervals": true, "architecture": {"base": "iter36", "zero_features": [22]}}'`.
- Decay an existing run: `scripts/expertium_decay.py auto --source-run <run dir> --epoch <N> --out artifacts/<new> --fraction 0.25`, then read `<out>/summary.md`.
- `run_full_training.py prepare` resolves `data/` and the R01 cohort file relative to the checkout it runs from. A worktree needs both (a junction to the main checkout's `data/` and a copy of `artifacts/retraining/gpu-retrain-r01/cohorts-user-holdout-v1.json`). `expertium_decay.py` and `expertium_speed.py` read them from the main checkout directly.

## Switches

All five items are R01 track options. Each lives in one key; `scripts/gpu_r01_training.run_case` reads them through [scripts/expertium_options.py](../scripts/expertium_options.py) and never imports it when none is set.

| Item | Key | Values | Upstream (a9fcf466) |
|---|---|---|---|
| 1. Width-80 champion | `"architecture"` | `"w80"`, `"w80-sequential"`, `"iter36"`, or `{"base": <preset>, <field>: <value>}` | `scratchpad/track2_a18/architecture_d80_lora4_cnd.py` + iter-53 env: `RWKV_GRU_HEAD=3`, `RWKV_NO_AHEAD_RESIDUAL=1`, `RWKV_STRIP_L0_VLORA=1`, `RWKV_STRIP_CMIX=...`, `RWKV_INTERLEAVE=1`, `RWKV_DROPOUT_SCALE=0.5`, `RWKV_ZERO_FEATURES=22` |
| 2. Warmup-stable-decay | `"lr_schedule"` (track) or `full_run.lr_schedule`; decay-only: [scripts/expertium_decay.py](../scripts/expertium_decay.py) | `{"decay_start": 0.75, "decay_end": 1.0, "shape": "one_minus_sin", "final": 0.0, "peak": 7e-4, "warmup_updates": 5}` | `TRAIN_MODE="D"` decay runs, `RWKV_DECAY_SHAPE` |
| 3. Muon + AdamW (incl. LoRA) | `"optimizer"` | `"expertium-iter53"` or `{"kind": "muon", "groups": "matrices"\|"projections", "include_lora": true, "lr_scale": ..., "adjust_lr_fn": "original"\|"match_rms_adamw", "lora_weight_decay": 0.0, "wd_at_adamw_rate": true, "momentum": 0.95}` | `RWKV_MUON=1`, `RWKV_MUON_LR=0.0025`, `RWKV_MUON_INCLUDE_LORA=1` (iter 53) |
| 4a. Interval leak fix | `"end_to_start_intervals"` | `true` | `RWKV_E2S_PUBLISHED=1`; analogue of `RWKV_CLOCK_AT_PREV_ANSWER` |
| 4b. Feature drop | `"architecture": {"base": ..., "zero_features": [22]}` (on in `w80`) | list of input dims | `RWKV_ZERO_FEATURES=22` (iter 15) |
| 5. Distillation | `"distillation"` | `{"teacher": "<.pth>", "alpha": 0.9, "decay_alpha": 0.5, "teacher_architecture": "baseline", "teacher_dtype": "bfloat16"}` | `RWKV_KD_MIX`, `RWKV_KD_ALPHA` (iters 32, 39, 45) |

Full runs take the same options: `scripts/run_full_training.py prepare ... --extra-options '<json>' --final-lr-schedule '<json>'`. A non-width-32 architecture cannot use the width-32 kernels; `scripts.expertium_options.compatible_options(options)` removes them (`row_tiles_exact`, `native_wkv`, `wkv_tile`, `exact_index`, `skip_padding_sources`, `native_index`, `prefetch_rows`, `native_glue`, `exact_group_norm`, `native_group_norm` from the accepted track) and `run_full_training`/`expertium_decay` apply it automatically. `CatchUp` refuses those keys, and `dynamic_graphs`, `reduced_sync` and `group_adam`, with a variant or distillation.

## 1. Width-80 champion (`"architecture": "w80"`)

Code: [expertium/variants.py](../expertium/variants.py) (presets), [expertium/model/variant_model.py](../expertium/model/variant_model.py) (training model, subclass of the vendored `SrsRWKV`), [expertium/model/variant_rnn.py](../expertium/model/variant_rnn.py) (recurrent inference), [rwkv/model/nvrtc.py](../rwkv/model/nvrtc.py) (`head_source`: the NVRTC recurrence rewritten for head size 16; head size 32 compiles the unchanged text).

- d_model 80 = 5 heads x K 16; LoRA ranks decay 4, a 4, value-mix 2, gate 4; channel-mixer factor 1.0; dropout 0.01 / long 0.025 / layer 0.005 (his 0.02/0.05/0.01 at `RWKV_DROPOUT_SCALE=0.5`).
- Streams card 2, deck 4, note 1, preset 3, user 3 layers. Each stream keeps the champion's depth by name. The order is this repository's batch chain: `rwkv.prepare_batch.prepare` builds gathers in `rwkv.architecture`'s tuple order (card, deck, note, preset, user) for every model. Upstream iter 41 moved note before deck by changing his `prepare()`; his iter 43 measured that reorder as free. The only effect here is the order of note and deck inside each interleaved round.
- GRU curve head with N=3 predicted (w, S, d) power curves, no ahead residual (zero residual logits), upstream initialisation; 1x1 dummies for the dead modules, as upstream counts them.
- Channel mixers removed at user 0-2, preset 0-2, deck 1-2, card 1 (iter-53 `RWKV_STRIP_CMIX`, verbatim); layer 0's value-residual LoRA replaced by upstream's 1x1 dummy.
- Front-loaded round-robin layer schedule (`RWKV_INTERLEAVE=1`): round r runs layer r of every stream deep enough. Gathers are re-anchored to the canonical row order and each layer scatters its rows back. `"w80-sequential"` is the same model without it.
- Input dim 22 (`scaled_state`) zeroed in training and evaluation.
- Parameters: 558,209 = upstream's 558,212 minus PAVA's three thetas (PAVA not ported, below). Verified, with strict parallel-to-recurrent loading.
- Recurrent inference: `expertium.variants.build("w80", rnn=True)`, `load_state_dict(master_state, strict=True)`, then `step(features, states)` per review or `predict_rows(features, ids, skips, label_times)`. States route by stream name (module i = stream i of the batch chain). The interleaved schedule runs per review exactly as in training.
- Not ported (deliberately; listed as untested ideas in [GPU_OPTIMIZATIONS.md](GPU_OPTIMIZATIONS.md) X06): PAVA (`RWKV_PAVA_LAMBDA=0.2`) with its probe rows (`RWKV_PROBE_DENSITY=0.08`) and rectified evaluation, which change the training data stream and the scored metric. `RWKV_STATE_CLAMP_TAU=300`, which only engages on recurrences longer than 32,768 rows. Spread interleave (iter 44), QAT and every later deploy-side lever.

## 2. Warmup-stable-decay (`"lr_schedule"`)

The accepted schedule is a 5-update linear warmup to 7e-4, then constant. With `lr_schedule`, progress is counted in this case's training reviews, which are known before training (update counts depend on packing): the learning rate is constant until `decay_start` of them, then follows `shape` to `final x peak` at `decay_end`. Without the key the accepted expression runs; a schedule with `decay_start` 1.0 equals it bitwise.

Upstream recipe: a warmup-stable run, then a separate decay process started from its final checkpoint, LR from peak to 0. His default shape `1 + cos(pi/2 (1 + x))` is identically `1 - sin(pi x / 2)` (`"one_minus_sin"`, LR mass 0.363); `"cosine"` is the standard half cosine (mass 0.5). His decay length grew from 0.25 of the training budget to 1.0 epoch (iter 34, the era's largest single gain). In his leak-free 10-epoch endgame (2026-09-15), decays of 1 and 2 epochs after 10 warmup-stable epochs were close, and 3 epochs hurt the curve head (+0.000248); he settled on 1. The 0.25-epoch default here is the requested short first test; a 1-epoch decay (about 55 min training plus evaluation on the 8k split) is the next length to try.

Two ways to use it:

- In the last epoch of a full run: `run_full_training.py prepare ... --final-lr-schedule '{"decay_start": 0.75, "shape": "one_minus_sin"}'` decays over its last quarter.
- Decay-only from an existing checkpoint, for example tonight's 10-epoch run (read-only; quick validation = its 300 quick users, compare with epoch 10's scores of the same users):

```bash
.venv/Scripts/python.exe scripts/expertium_decay.py prepare --source-run C:/Users/jonat/workspace/anki-rwkv/artifacts/full-run/run-8k-20261013 --epoch 10 --out artifacts/expertium-catch-up/decay-8k-e10-q25 --fraction 0.25 --shape one_minus_sin --evaluate quick
C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py run --owner expertium-catch-up --purpose "decay-only 0.25 epoch from 8k epoch 10" --claim gpu -- .venv/Scripts/python.exe scripts/expertium_decay.py run --out artifacts/expertium-catch-up/decay-8k-e10-q25
```

It trains 0.25 epoch of reviews (about 145M at 8k users) on a fresh shuffle of the same training users from epoch 10's weights and AdamW state, `update_offset` continuing so no warmup recurs. The label database is opened read-only (`lock=False`).

## 3. Muon for matrices, AdamW for the rest (`"optimizer"`)

[scripts/muon_optimizer.py](../scripts/muon_optimizer.py) is the F82 helper copied unmodified from the main checkout (commit `ef8a2de`), then extended. `"expertium-iter53"` puts every 2-D weight of the four AdamW weight-decay groups (time mixer, channel mixer, heads incl. the GRU head, features2card) on `torch.optim.Muon`, and the LoRA projections in their own Muon group at weight decay 0 (iter 53). Norms, biases, lerp vectors and the `*scale*` projections stay on AdamW. On w80 that is 528,320 parameters on Muon, 27,520 of them LoRA (his iter-53 note: 27,520), and 29,889 on AdamW.

His Muon matches `torch.optim.Muon` up to float. Both use the same Newton-Schulz coefficients and bf16 iteration. His Nesterov momentum is in sum form, torch's in EMA form, which is identical after normalisation. His step scale `sqrt(max(1, rows/cols))` is `adjust_lr_fn="original"`. His Muon peak was 0.0025 with AdamW at 1e-3. Here AdamW stays at 7e-4 and the Muon groups use `lr_scale = 0.0025 / 7e-4`, so the Muon peak is his 0.0025 and follows the same schedule. Weight decay is applied at the AdamW rate (`wd_at_adamw_rate`), as he did. The optimizer state is `{"adamw": ..., "muon": ...}`, so it cannot resume an AdamW checkpoint; a decay-only run from tonight's checkpoint keeps AdamW. The F82 projections-only grouping stays the default of `get_hybrid_optimizer`.

## 4. Clock leak and feature drop

Determination: the specific leak Expertium fixed does not exist here, but the same kind of leak does.

- His `RWKV_CLOCK_AT_PREV_ANSWER` fixes clock columns of his rebuilt `-id` layout (`t_since_any_review`, time of day, tenure, deck age, sibling gap) measured to a reconstructed show time. This repository uses the published layout (`rwkv/data_processing.CARD_FEATURE_COLUMNS`, 24 card columns), which has none of them; upstream itself refuses the flag without `-id` features.
- In the published revlogs `elapsed_seconds` runs answer-to-answer, so a query row's interval columns (`scaled_elapsed_seconds`, its sin/cos and the cumulative interval) contain the queried review's own `duration`. That duration is unknown before the review and predicts its outcome. [scripts/expertium_interval_audit.py](../scripts/expertium_interval_audit.py) on 40 random users ([interval-audit.json](../artifacts/expertium-catch-up/interval-audit.json), 1,663,091 query rows):
  - 31.4% of query rows have intervals under 1 hour. There the median duration is 3.0% of the interval, and 5.9% of all query rows carry a duration of at least 10% of the interval.
  - Among under-1-hour rows, P(fail) is 18.7% in the fastest duration quartile (median 2.0 s) and 24-26% in the others.
  - The duration decile carries 0.0024 nats of outcome information per query row on its own. How much of that the model recovers from the interval is unmeasured.
  - Expertium measured his narrower version at +0.0032 instant / +0.0003 curve log loss.
- `"end_to_start_intervals": true` applies his published-layout correction (`RWKV_E2S_PUBLISHED`): `elapsed_seconds - duration` (milliseconds), clamped at 0, the -1 first-review sentinel kept, on every row. Real rows, query rows and curve label times therefore stay consistent. `elapsed_days` (calendar days) is untouched. It is applied in `get_rwkv_data` through the `EXPERTIUM_END_TO_START` environment variable, which `CatchUp` sets before any data read or worker spawn. Training and evaluation must use the same setting.
- Feature drop: `zero_features: [22]` zeroes `scaled_state` at the model input. Query rows already carry 0 there (`add_queries` rejects state). On w80 it is on; for the default width use `{"base": "iter36", "zero_features": [22]}`.

## 5. Distillation (`"distillation"`)

`get_loss(batch, kd_mix=(teacher_curve, teacher_rating_probs, alpha))` on the variant model mixes the targets as his iter 32: curve target = alpha x teacher recall + (1 - alpha) x label; rating target = alpha x teacher probabilities + (1 - alpha) x one-hot (soft cross-entropy). Evaluation never mixes, and the instant binary statistic keeps the hard labels. His recipe is alpha 0.9 during warmup-stable training (iter 39) and 0.5 during decay (iter 45); `decay_alpha` applies once the LR schedule is decaying. He replayed stored teacher outputs by step; here the teacher runs online on the identical batch (no_grad, eval mode, BF16 by default), so any R01 batch order works. With alpha 0 the loss equals the plain loss (checked).

Teacher: the 2.76M srs-benchmark RWKV, e.g. `C:/Users/jonat/workspace/anki/qt/aqt/rwkv_inference/RWKV_trained_on_5000_10000.pth` (2,762,884 parameters, loads strictly into `rwkv.model.srs_model.SrsRWKV`). It needs the variant model (`"architecture": "iter36"` for the default width). **Its file name says it was trained on users 5000-10000**: distilling it leaks those users' held-out reviews into a student. Evaluate KD students only on users below 5000 or on collections outside the dataset (the user's own collection is clean).

## Verification

| Check | Result | Evidence |
|---|---|---|
| Default model untouched: variant class with all switches off vs vendored `SrsRWKV` | bitwise equal initial weights, training loss and every gradient (CPU, real user batch) | [check-cpu.json](../artifacts/expertium-catch-up/check-cpu.json) |
| Default LR | equal to the accepted expression for 0-39 updates, offsets 0 and 13,789 | same |
| w80 parameters / strict recurrent load | 558,209 / passes (also iter36 192,800, w80-sequential) | same |
| Interleave machinery | stream-major order through the canonical gather/scatter path = sequential forward, max abs 0.0 | same |
| Recurrent vs parallel predictions (weights +N(0, 0.01)) | iter36 9.5e-7, w80 3.3e-6, w80-sequential 5.4e-6; vendored RNN 6.0e-7 | same |
| NVRTC head-size 32 source | unchanged (K=16 is a separate rewrite) | same |
| Muon grouping, LoRA group, step | 528,320 Muon / 27,520 LoRA / 29,889 AdamW on w80; finite step, LR restored | same |
| End-to-start | off: frame identical; on: `elapsed_seconds - duration`, sentinel kept | same |
| Distillation | alpha 0 = plain loss; fork teacher loads strictly | same |
| GPU: NVRTC K=16 vs reference; w80 CUDA vs CPU | pending until the 8k run releases the GPU | [check-gpu.json](../artifacts/expertium-catch-up/check-gpu.json) |
| Speed: w80 vs w32 short R01 runs | pending | [speed-20261014](../artifacts/expertium-catch-up/speed-20261014) |

Commands: `.venv/Scripts/python.exe scripts/check_expertium_catch_up.py` (CPU, about 25 s); `scripts/check_expertium_gpu.py` and `scripts/expertium_speed.py case --out artifacts/expertium-catch-up/speed-20261014 --case <name>` through the queue.
