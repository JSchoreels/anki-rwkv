# HANDOFF SM21 v1: training with carried state (2026-10-18)

Branch `state-carry`, worktree `C:/Users/jonat/workspace/wt/state-carry`. Design: `docs/SMALL_MODEL.md`, "Design 2026-10-16: training with carried state (SM21)" and its status note of 2026-10-18. Nothing has run on the GPU; nothing is enqueued.

## Done (CPU only)

- **(a) Kernels.** `rwkv/model/wkv_state.py`: `wkv_state(r, k, v, w, a, kd, skip, S0) -> (out, S_T)`; CUDA = `RWKV7_WKV_State` (`nvrtc.forward_with_state` + the stock `nvrtc.backward`), CPU = `wkv_with_state` (the prototype's reference, moved here). S0 is detached; S_T is returned detached.
  - Generic NVRTC kernel: `rwkv/model/nvrtc.py` `STATE_REWRITES` / `state_kernel_source` / `state_module` (separate module cache key `"state"`). The stock path (`module`, `forward`, `backward`, `kernel_source`) is unchanged; `_compile_options` is the old inline option list moved into a function.
  - Exact row tiles: `scripts/gpu_row_tiles.py` `STATE_REWRITES` (applied after the K=16 rewrite, only when `state=True`); `row_tiled_backend` also installs `nvrtc.state_module` (rows/tile of the backend; wave-aware and pipelines 7/8 refuse).
  - Backward needs no change: it recomputes from the checkpoints, whose first entry is now S0.
- **(b) Trainer.** Option `"state_carry": true | {"precision": "deploy" | "fp32", "carry": true | false}` (allowlisted in `scripts/gpu_r01_prefetch.py`). Code: `scripts/state_carry_train.py` (module docstring has the full description).
  - Worker side: user-level day features (`user_segment_features`, main process, before slicing), int64 ids (`create_sample_fast(int64_ids=True)`), consistent id encodings and day baselines (`prepare_fast(consistent_seed=seed - offset)`), entity rows on the item (`item.carry`), entity universe on the user's first item (`item.carry_universe`).
  - GPU side: `CarryTable` (int8 S + fp16 scale for card/note, fp16 S for deck/preset/user, fp16 shift vectors, has-flag). v1 = every window reads the table as of the update start, and each entity's latest window writes back after the forward. The table is reset at the user's first update. It uses searchsorted + gather/where only: no host sync, and deterministic.
  - Hand-over: `VariantSrsRWKV._interleaved_streams` (jit.ignore) calls `CARRY_HOOK[0].streams` when the hook owns that model. Under TorchScript the method gets a wrapper, so the hook matches on parameter storage and resolves the layers on the original object. The teacher is never carried. Then `carry_layer` runs the stock time/channel mixers with the run's exact switches, the carried shift at the rows before the first non-skipped row, and the S0 kernel.
  - `scripts/gpu_r01_training.py` has one line before the loss: `catch_up.carry.attach(item, batch, user)`. `scripts/expertium_options.py`: `CatchUp.carry`, the `active()` wrap at the top of `CatchUp.loss`, and `describe()` (with live `stats`). `scripts/run_gpu_r01.py` SourceWatch: `rwkv/model/wkv_state.py` and `scripts/state_carry_train.py` are added, conditional on the source manifest.
  - `"carry": false` = the control arm. It has the same inputs and the same Python layer path, starts from zero state, and never reads or writes the table.
  - Requires: `prefetch_users`, `fast_samples`, `columnar_samples`, `batched_id_encodings`. Refuses: `pack_users`, `preprocessing_cache`, `worker_frames`, `batch_segment_features`, fused/native kernels, graphs, concurrent splits.
- **(c) Checks.**
  - `scripts/check_state_carry_kernels.py --cpu`: ALL OK. All 8 state variants compile with NVRTC for compute_120. The CPU reference started from zeros equals `reference_rwkv7` bitwise, chained halves are bitwise, and no gradient reaches S0. Report: `artifacts/small-model/check-state-carry-kernels-cpu.json`.
  - `scripts/check_state_carry_trainer.py` (users 1340 and 8254, ctx2048 e6 checkpoint, 470 s): ALL OK. Report: `artifacts/small-model/check-state-carry-trainer.json`.
    - A: option absent equals the tree before SM21 (7fada06, `git archive`). All prefetch batches, the loss and every gradient have equal sha256.
    - B: empty table equals the stock forward bitwise (train mode).
    - C: window input rows equal the full-history rows bitwise.
    - D: exact carry (fp32 table, 1 window per update) equals the full-history forward, max |dp| 5e-7 / 6e-7. With the deploy table it is 1.4e-3 / 7.8e-4; with zero start (`carry: false`) 0.65 / 0.64.
    - E: v1 with 2 windows per update equals the prototype `FreshCarry` bitwise.
    - F: user boundary.
    - G: scripted model hand-over.
  - The prototype check `scripts/check_state_carry.py` still passes A/B/C after the move.

## Left

1. **GPU parity ticket** (about 10–15 min). Never run outside the queue. Run it from any shell:

   ```
   C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py run --owner small-model --purpose "SM21 GPU parity: S0/S_T kernels + trainer carry path" --claim gpu --expected-minutes 15 -- "C:\Program Files\Git\usr\bin\bash.exe" C:/Users/jonat/workspace/wt/state-carry/artifacts/small-model/jobs/sm21-gpu-parity.sh
   ```

   - It writes `artifacts/small-model/check-state-carry-kernels-gpu.json` (Z/R/G/C over 2 backends x fp32/bf16 x K16/32 x 8 shapes x 4 skip patterns) and `artifacts/small-model/check-state-carry-gpu.json`:
     - P: empty table vs stock, bf16 train mode, run options with row_tiles_exact 4;
     - X: exact carry on the GPU (fp32) vs full history;
     - T: seconds per 16 x 2,048 update, stock vs carry, plus peak memory.
   - Both scripts print `ALL OK`.
   - If Z (zero-state bitwise vs the stock kernel) fails while R/G/C pass, the kernels are still correct. The cause is ptxas contraction, generic kernel only. Report it, then decide whether P's tolerance matters.
   - Record T in `docs/GPU_OPTIMIZATIONS.md` row S09.
2. **Merge SM11** (branch `sm11`, worktree `wt/sm11`, full-history fork targets as a new `Teacher` mode) into `state-carry`, or merge both into `small-model`. Integration points:
   - `scripts/expertium_options.py`: my edits are small (`CatchUp.__init__` 5 lines after `self.teacher = None`, 3 lines at the top of `CatchUp.loss`, 2 lines in `describe`), so conflicts are local.
   - The SM11 Teacher must key targets by (user, `batch.label_review_th`). Carry batches keep today's rows and labels; only the input encodings change.
   - The window-local teacher cache hash includes `state_carry`, because it is not in `TEACHER_BLIND`. So a carry run with the old online/cached teacher would start a cold cache. That is correct, since the inputs differ, but SM11 targets make the hash irrelevant.
   - The teacher model never sees carried state (`owns` is false; check G).
   - After merging, re-run `scripts/check_state_carry_trainer.py` (CPU, 8 min) and SM11's own check.
3. **Experiment** (per the design; needs SM11 merged and the parity ticket passed). Write a runbook from `artifacts/small-model/runbook-2026-10-16b.sh` `cont`.
   - **The `cont` stage cannot be used as is.** Its NAME `scale-w64kd-gate-cached-ctx2048-cont` already exists. It also has no hook for extra options. Copy it into e.g. `artifacts/small-model/runbook-sm21.sh` with:
     - source `scale-w64kd-gate-cached-ctx32768-cont`, FROM=10 (the best: dec10 0.30096 / 0.27471, +1.11% / +0.94%), TO=12, CHUNK=2048, BATCH_CHUNKS=16;
     - two arms: NAME `scale-w64kd-gate-carry-e10` with extra options `{"state_carry": true}`, and `scale-w64kd-gate-carrycontrol-e10` with `{"state_carry": {"carry": false}}`. Both use the SM11 teacher option. Merge the extras into `exp["options"]`, `exp["option_overrides"]` and `run["options"]` in the copy step, next to `chunk_reviews` / `batch_chunks`;
     - per arm: e11, e12, then the decay with carry (the decay prepare inherits the options), then scoring on the 464 test users at w8+s8.
   - Pre-registered rule (as Q6): carry − control ≤ −0.001 on the curve, with upper95 < 0, and instant ≤ +0.0005.
   - Queue: one ticket per epoch, `--expected-minutes 80` until T is known. The design assumed ≈ 66 min per epoch at 2,048 and a carry overhead of 1.05–1.15x.
   - Memory rules: report reviews/s and CPU; respect the GPU break windows; run a DRYRUN first.
4. Optional, after a positive result: v2 (strided lanes + stale lane-start states). See the design.

## Notes

- Rows and windows: one update = 16 consecutive 2,048-review windows of one user, so v1 only carries across updates. For users with ≤ 32,768 training reviews (one update) carry changes nothing except the consistent inputs.
- Table size measured on CPU: user 8254 (10k reviews) 52 MB fp32 / 16 MB deploy. The design's maximum is ≈ 0.4 GB int8 for the largest user.
- Commit identity: `git -c user.name="Jonathan Schoreels" -c user.email=jonathan.schoreels@gmail.com commit`; push `origin state-carry`.
