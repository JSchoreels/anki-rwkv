# Expertium catch-up: handoff (2026-10-14)

Worktree `C:\Users\jonat\workspace\wt\expertium-catch-up`, branch `expertium-catch-up` (from main `e61e7cd`). Not pushed or merged. Full documentation: [docs/EXPERTIUM_CATCH_UP.md](../../docs/EXPERTIUM_CATCH_UP.md); ledger rows X01–X06 in `docs/GPU_OPTIMIZATIONS.md`.

## State

All switches are opt-in R01 track keys: `architecture` (w80 = iter-53 champion; w64 = s64h; w80/w64-sequential; iter36), `lr_schedule` and `scripts/expertium_decay.py`, `optimizer: expertium-iter53`, `end_to_start_intervals`, and `distillation`. Upstream is pinned to Expertium `1f2e6b0` in `expertium/upstream.json` (`catch_up`).

Verified:

- 28 CPU checks pass (`artifacts/expertium-catch-up/check-cpu.json`, `scripts/check_expertium_catch_up.py`):
  - the default model is bitwise unchanged through the variant class;
  - parameter counts: w80 558,209, w64 331,511; both reproduce Expertium's counts at his 103-input layout;
  - recurrent-vs-parallel parity is at most 5.4e-6;
  - the default LR is bitwise unchanged;
  - Muon grouping puts 27,520 LoRA parameters in their own group (his iter-53 number);
  - end-to-start intervals and KD alpha 0 behave as specified.
- GPU (`speed-20261014/check-gpu.json`): the NVRTC head-size-16 recurrence matches the reference (float32 9e-8, gradients 2e-7).
- GPU bitwise (`speed-20261014/exact-iter36-variant/result.json`): the accepted E89-F65e-F77a track vs the same track with `"architecture": "iter36"` gives 5/5 batches with the loss and every gradient bitwise equal.
- Decay-only, 0.25 epoch from the 8k run's epoch 10, scored on its 300 quick users (`decay-8k-e10-q25/summary.md`):
  - curve LogLoss 0.36128 → 0.35906 (paired −0.00180, SE 0.00027);
  - instant LogLoss 0.34473 → 0.34327 (−0.00131, SE 0.00012);
  - training ran at 176,906 reviews/s.
- Speed: 160 users, 12.4M reviews (`speed-20261014/summary.md`, `summary.json`; rebuild with `scripts/expertium_speed.py report --out artifacts/expertium-catch-up/speed-20261014`):

  | Case | Reviews/s | Peak GB |
  |---|---:|---:|
  | w32, accepted kernels | 178,242 | — |
  | w32 generic | 119,308 | — |
  | w32 + KD | 113,121 | — |
  | w80, 256 chunks | 61,985 | 8.5 |
  | w64 interleaved, 512 chunks | 18,711 | 13.2 |
  | w64-sequential, 512 chunks | 19,067 | 13.0 |

  w80 at 512 chunks runs out of memory on the 16 GB card. The slow w64 runs at 512 chunks are likely WDDM memory paging, not the model.

Pending (outputs land in `artifacts/expertium-catch-up/speed-20261014/<case>/expertium/E89-F65e-F77a/result.json`):

- Ticket 000306 (log `speed-20261014/queue-console-3.log`, `all.log`) was still running w80-sequential-b256, then w32-generic-b256, at handoff. It was started from this session and may have been stopped with it; if `all.log` lacks those lines, rerun `jobs/speed2.sh` through the queue (finished cases are skipped).
- Ticket 000318 (detached; log `speed-20261014/queue-console-4.log`) runs w64-b256 and w80-muon-b256, about 20 min.

None of the five items has a matched quality comparison yet, except the decay screen.

## Merge notes for main

- Files touched in shared code, each a minimal hook:
  - `scripts/gpu_r01_training.py` (CatchUp hooks);
  - `scripts/run_full_training.py` (`--extra-options`, `--final-lr-schedule`);
  - `scripts/gpu_r01_prefetch.py` (option allowlist);
  - `scripts/run_gpu_r01.py` (source watch);
  - `scripts/check_r01_exact_options.py` (`architecture` branch);
  - `rwkv/model/nvrtc.py` (K=16 rewrite; K=32 text unchanged);
  - `rwkv/data_processing.py` (env-gated end-to-start);
  - `expertium/model/rwkv_model.py` (one assert);
  - `docs/GPU_OPTIMIZATIONS.md`, `upstream.json`, `expertium/upstream.json`.
- New files: `expertium/{variants,intervals}.py`, `expertium/model/variant_{model,rnn}.py`, and `scripts/{expertium_options,expertium_decay,expertium_speed,expertium_interval_audit,check_expertium_catch_up,check_expertium_gpu}.py`.
- `scripts/muon_optimizer.py` and `scripts/optimizer_curve.py` collide with the untracked copies in the main checkout. Commit `ef8a2de` holds them unmodified; `c584703` extends `muon_optimizer.py` (the F82 default is unchanged). Remove or compare the untracked main copies before merging.
- upstream hashes: main's `scripts/check.py` already fails on `rwkv/model/rwkv_model.py`, from R01 commits dab8fee/a9ed123. This branch does **not** fix that hash. It updates only `adapted_sha256` for `rwkv/data_processing.py` (root `upstream.json`) and `expertium/model/rwkv_model.py` (`expertium/upstream.json`).
- Worktree jobs need `artifacts/cache/r01-native/<identity>/` (copied from main) and queue wrappers that `cd` into the worktree (`artifacts/expertium-catch-up/jobs/*.sh`). The queue runs commands from main's directory.

## Next steps

1. Quality screens, same split and seeds as the w32 screens:
   - w80 and w64 at 256 chunks, 1 epoch, then with the iter-53 Muon and a final decay;
   - the decayed epoch-10 checkpoint on the full 2,000 users;
   - a 1-epoch decay.
2. Kernel port for wider widths. The R01 width-32 kernels (row tiles, native WKV, exact_index, native glue/group norm) are what make w32 1.5x faster than its generic path. Porting them to d=64/80, K=16, plus scripting the interleave loop, is the main speed lever for w64/w80.
3. The end-to-start A/B (`end_to_start_intervals`).
4. KD only on users below 5000 (the fork teacher was trained on users 5000–10000).
