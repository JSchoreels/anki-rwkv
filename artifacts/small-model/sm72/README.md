# Leak-free retrain (SM72): preparation (CPU/code only, 2026-10-24)

Goal (user): retrain the small model with the deleted-card leak removed from its training inputs (SM71's `neighbour`
transform), score it leak-free (`scripts/small_model_heldout.py --deleted-cards neighbour`), compare with the big model
scored leak-free (SM71's `big-neighbour` rows). This note covers the code, the CPU check and the plan; no GPU job ran.

Branch `sm72` (worktree `C:/Users/jonat/workspace/wt/sm72`), cut from `sm71` (4e2c9413). `sm71` was itself cut from the
`sm56` tip 746d8f1d, so it already holds the full SM56 recipe, `scripts/sm56_epochs.py` and the resumable decay
(`decay-resume` 1a1653bb, merged into sm56 at 545c91d3). Nothing else needed merging.

## Where the training inputs are built (correction to the brief)

The SM56 recipe does **not** read an LMDB. `scripts/run_full_training.py` runs each epoch through
`scripts/gpu_r01_training.run_case` with `prefetch_users`, whose ordered prefetch (`scripts/gpu_r01_prefetch.training_batches`
-> `scripts/gpu_batch1.frame_stream`) calls `rwkv.data_processing.get_rwkv_data(DATA, user)` per user on the fly and cuts
2,048-review chunks from that frame (5 preparation workers). The decay (`scripts/expertium_decay.py`) runs the same
`run_case`; the LMDB it opens is only the read-only label-filter database (benchmark review ids, from the revlogs alone).
The LMDB writer in `rwkv/data_processing.py` (`job` -> `create_sample` -> `writer_queue` -> `save_job`, `LMDB_PATH` from
the toml) is the upstream pipeline (`python -m rwkv.data_processing --config ...`, `train_db`, 205 GB), used by the upstream
trainer `rwkv/train_rwkv.py`, not by SM56. `sample_cache` / `preprocessing_cache` are off in SM56, so no cached samples exist.

So there is nothing to rebuild: the transform has to act on `get_rwkv_data` itself.

## Code (default off = unchanged)

- `rwkv/data_processing.get_rwkv_data`: if the environment variable `RWKV_DELETED_CARDS` is set (`neighbour` or `flags`),
  the finished frame goes through `scripts/sm71_leak.neutralise` (the scorer's transform, same function). Unset: the code
  path is the old one. An environment variable (not a track option) so that every process sees it: the trainer, its
  spawned preparation workers (`prestart_pool`), the replay exporter and the upstream LMDB builder. The launcher must
  `export RWKV_DELETED_CARDS=neighbour`.
- `scripts/sm11_replay_export.py`: the manifest records `"deleted_cards": "<mode>"` when the variable is set (key absent
  otherwise, so existing manifests still match).
- `scripts/expertium_options.py`: `ReplayTargets` refuses targets whose manifest `deleted_cards` differs from the
  training process's `RWKV_DELETED_CARDS` (so a leak-free run cannot read the leaky `fork-fp32-final8k`, and a normal run
  cannot read leak-free targets); `CatchUp.describe()` reports `deleted_cards` in the run record.
- `scripts/sm72_check.py`: the CPU check below.

Scoring: keep `RWKV_DELETED_CARDS` **unset** in scorer processes and use `--deleted-cards neighbour` as SM71 does. (Set,
the transform would run twice; it is idempotent on the inputs, but the scorer's deleted-row statistics and
`--split-existing` read `note_id_is_nan`, which would already be 0.)

## CPU check (`scripts/sm72_check.py`, users 4676, 516, 4789, 1058) -> ALL PASS

`check-inputs-sm72.json` / `.log`. Reference = `get_rwkv_data` of commit `sm71`, loaded from git.

| user | reviews | deleted reviews / cards | off: frame equal, sample identical | on: frame = SM71 neighbour | changed frame columns | sample rows changed (all deleted-card rows) | deck/preset ids changed outside deleted rows | matches SM71 check-inputs.json |
|---|---|---|---|---|---|---|---|---|
| 4676 | 3,716 | 353 / 124 | yes / yes | yes | 3 flags + deck_id, preset_id | 582 | 0 | yes |
| 516 | 4,601 | 1,711 / 490 | yes / yes | yes | same | 2,932 | 0 | yes |
| 4789 | 5,102 | 4,934 / 1,231 | yes / yes | yes | same | 8,637 | 0 | yes |
| 1058 | 5,224 | 2,380 / 607 | yes / yes | yes | same | 4,153 | 0 | yes |

Feature changes only in the 3 `*_is_nan` columns; card, note and user ids identical; the only other sample tensors that
differ are the deck/preset module permutations (the deck lane membership), exactly as in SM71.
Cost: `get_rwkv_data` with the transform takes +0 to +0.03 s per user (161k-review user 4462: 0.17-0.26 s either way),
inside the 5 preparation workers, overlapped with the GPU: no measurable epoch-time change expected.

## Estimates

- "LMDB build": none needed (inputs built on the fly). Eval users: the scorer applies the transform itself.
- Leak-free KD replay targets (GPU, required unless decided otherwise, see open decisions): the teacher replay
  `artifacts/cache/kd-replay/fork-fp32-final8k` (581.9M reviews, 8,000 users, 5.8 GB) was exported from
  `get_rwkv_data` frames, i.e. leaky inputs. Measured rate of its last export (3,000 users, 212.5M reviews in 1.74 h,
  CUDA): 34.0k reviews/s -> about **4.8 h GPU** for all 8,000 users, written to a NEW folder
  `artifacts/cache/kd-replay/fork-fp32-final8k-neighbour` (+5.8 GB disk), plus a CPU `verify --sample 8`.
- Training, same recipe as SM56 (`scale-w64kd-final-sc-lanes`: w64, AdamW peak 7e-4 constant, 2,048 x 16 strided
  lanes, replay KD alpha 0.9 / decay_alpha 0.5, seed 2025, order seed 20261013): **10 epochs x ~118 min = 19.7 h**, plus a
  464-set scoring after each (~12 min, `--deleted-cards neighbour`) = 2 h; then the one-epoch 32k Muon decay from e10
  **~8.5 h**; then 732-pool scorings (~11 min each). Total about **35 h GPU** (4.8 + 19.7 + 2 + 8.5 + 0.4), i.e. two to
  three days with the daily breaks. The CPU label-window check (`scripts/sm56_epochs.py check`) must be rerun against the
  new replay folder (about 10-20 min CPU, 4 workers).
- Big model leak-free scores: already in SM71's runbook (`artifacts/small-model/sm71/gpu/big-neighbour*.jsonl`, 464 set and
  732 pool), so the SM72 comparison needs no extra big-model scoring.

## The big model's training (facts for a leak-free big retrain or fine-tune)

Upstream: `open-spaced-repetition/srs-benchmark` (read-only shallow clone at commit 4ddb9fd9, 2026-10-25; this repo vendors
`rwkv/` from commit 3effecc3, `upstream.json`).

- `rwkv/train_rwkv_config.toml`: `TRAIN_USERS_START = 5000`, `TRAIN_USERS_END = 10000` (5,001 users, 370.3M reviews in our
  dataset revision), validation users 1-100, `PEAK_LR = 7e-4`, `WARMUP_STEPS = 20000`, `MAX_TRAIN_GLOBAL_LEN = 66000`
  (reviews per batch), bfloat16, `LOAD_MODEL_NAME = "RWKV_trained_on_5000_10000"`.
- `rwkv/train_rwkv.py`: AdamW betas (0.9, 0.999), eps 1e-18, weight decay 0.01 (1e-2 encoders, 0 for others), `FINAL_LR = 0`;
  WSD schedule: `TRAIN_MODE = "WS"` = linear warmup (start factor 1e-4) over 20k steps then constant; `"D"` = cosine down
  `1 + cos(pi/2 (1 + t/T))` to 0 over the decay epochs.
- `rwkv/README.md`: "approximately 10 epochs in the warmup + stable phase and 2 epochs in the decay phase". Inputs come from
  the LMDB built by `rwkv/data_processing.py` (`data_processing_config_train.toml`, users 1-10000, `MAX_BATCH_SIZE 65536`),
  i.e. the same `get_rwkv_data` with the same `*_is_nan` flags and placeholder deck/preset: **the big model was trained
  on the leaky inputs**.
- 2,762,884 parameters = this repo's R01 "Default" architecture (`docs/GPU_TRAINING.md` line 20; `docs/COMPARISON.md`);
  the R01 Default trainer ran 28,521 reviews/s at R01 iteration 0002 (`docs/GPU_OPTIMIZATIONS.md` F01 row,
  `artifacts/retraining/gpu-retrain-r01/iteration-0002/summary.md`). Our R01 trainer is not the upstream LMDB trainer (other
  batching), so a retrain with it is a reproduction, not a bit-equal rerun.
- Shipped weights: `C:/Users/jonat/workspace/anki/qt/aqt/rwkv_inference/RWKV_trained_on_5000_10000.pth` (+ `.bin`, float32,
  sha256 8fb7fea0...); no optimizer state is shipped.
- Estimates at 28.5k reviews/s: one epoch over users 5000-10000 = 370.3M / 28.5k = **3.6 h**. (a) Full leak-free retrain,
  10 + 2 epochs: **~43 h GPU** (more if the R01 Default path needs the 16 GB card at a smaller batch). (b) Leak-free
  fine-tune from the shipped checkpoint, one cool-down epoch (cosine/one_minus_sin to 0, fresh AdamW state): **~3.6 h**;
  two epochs as upstream's decay: ~7.2 h. Both need the users 5000-10000 histories with `RWKV_DELETED_CARDS=neighbour`.

## Open decisions

1. **Teacher targets.** The KD teacher is the big model, trained on leaky inputs. Options: (a) re-export its replay targets
   on leak-free inputs (4.8 h GPU; the teacher then predicts without the leak signals, but its weights still learned to use
   them; this is the default plan in HANDOFF-sm72.md); (b) first fine-tune the big model leak-free (b above, ~3.6 h), then
   export targets from it (+4.8 h) — cleanest, and gives the leak-free big model for the comparison too; (c) keep the leaky
   targets (inputs and targets disagree; not recommended, and the trainer now refuses it while `RWKV_DELETED_CARDS` is set).
2. Mode: `neighbour` (user's choice) vs `flags`. Both supported by the same variable.
3. Epoch count: SM56 ran past 10 epochs (adaptive rule, cap 15); the brief fixes 10 + decay. Keep the per-epoch 464-set
   readout so the plateau rule can still be read.
4. The deleted cards' note stays a note of its own in both modes (siblings cannot be restored), so a small residual
   signal remains (a singleton note lane), as in SM71's leak-free scoring.
