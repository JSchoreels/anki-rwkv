# HANDOFF SM21 v2: strided lanes (2026-10-18, branch `sm21v2`)

Branch `sm21v2` (cut from `state-carry` at 5a2455e), worktree `C:/Users/jonat/workspace/wt/sm21v2`. CPU only: nothing run on the GPU, nothing enqueued. The v1 handoff below still describes the shared code; arms A (v1 carry) and B (zero) run from the `state-carry` worktree.

## Done

- **Option** `"state_carry": {"lanes": "strided"}` (with `precision` / `carry` as in v1). `"lanes": "update"` or absent = v1, unchanged bit for bit; option absent = unchanged bit for bit. Code: `scripts/state_carry_train.py` (`strided_layout`, `lane_universe`, laned `CarryTable`, `StateCarry.attach` lane offset; module docstring has the rule), `scripts/gpu_r01_prefetch.py` (v2 job layout).
- **Layout.** A user's N windows (2,048 each, the last one possibly short) form 16 lanes of consecutive windows. Update u holds window u of every lane, in lane order, so a window's lane predecessor ran in update u - 1.
  - Ragged rule: U = ceil(N / 16) (today's update count) and N = 16 (U - 1) + m. The first 16 - m lanes hold U - 1 windows and the last m lanes hold U.
  - So every update has today's window count and review count (same LR schedule). The final short window is the last slot of the last update.
  - `item.offset` = u x 32,768 as today, so seeds, dropout and log offsets are unchanged.
  - Users with N <= 16 have one update, and their v2 batch is the v1 batch.
- **Table.** One slot per (lane, stream, layer, entity), only for the (lane, entity) pairs that occur.
  - Codes = searchsorted into the user's entities, then into the sorted lane keys `lane x E + code`. The lane of slot b is b + a Python offset.
  - No new transfers, no host sync, deterministic. Deploy precision is unchanged (int8 card/note, fp16 others).
  - Reset at the user's first update. Lane starts (update 0), and entities first seen in a lane after its start, begin from zero.
  - Alternative, not implemented: seed lane j from lane j - 1's end states of the previous epoch (option 3).
- **CPU check** `scripts/check_state_carry_lanes.py` -> `artifacts/small-model/check-state-carry-lanes.json`. It uses users 1340 / 8254 at 2 and 3 lanes (they have 3 / 5 windows).
  - A0: option absent = the pre-SM21 tree.
  - A1: v1 (`true` and fp32, 2 windows/update, every update with loss + gradients + describe) = the 5a2455e tree, sha256 equal.
  - L: same updates/offsets/seeds/reviews as v1, layout = strided_layout, slot rows = full-history rows bitwise; at L16, v2 batch = v1 batch.
  - B: first update = stock, bitwise.
  - D: fp32 v2 carry vs full history. Lane-0 rows max |dp| 8.3e-7. With lane starts seeded with exact full-history states, every row is at most 8.3e-7 (deploy 1.4e-3). With zero lane starts, the other lanes are off by up to 0.65 in lane-start windows and 0.17–0.46 in continuation windows.
  - F: user boundary.
  - G: JIT hand-over at 2 lanes.
- **Stale share v2** (`artifacts/small-model/feature-screen/sm21_stale_share_v2.py` -> `sm21-stale-share-v2.txt`, same 295 users as v1). Card reviews by class:

  | Class | v2 | v1 |
  |---|---:|---:|
  | new | 13.1% | 13.1% |
  | in-window | 48.6% | 48.6% |
  | exact carry | **19.2%** | 8.9% |
  | **stale** | **0.0%** | 29.4% |
  | lane cut, zero state | **19.1%** | — |

  - The lane cut splits into 9.4% in lane-start windows and 9.7% in later windows (entities first seen in the lane).
  - Notes: exact 14.5% (v1 6.5%), stale 0 (v1 21.2%), cut 13.2%.
  - User stream: 31.0% of windows (the lane starts of lanes 1–15) enter with a zero state. All other windows continue their lane exactly; in v1, 92% skipped ~15k reviews.
  - 33.7% of reviews sit in update 0 (159 of the 295 users have a single update).
  - Table size vs v1: card slots 2.46x, note slots 2.71x (median user 2.1x, max 9.9x / 16x).
- **GPU check script** (not run): `artifacts/small-model/jobs/sm21v2-gpu-check.sh` -> `scripts/check_state_carry_lanes_gpu.py` -> `artifacts/small-model/check-state-carry-lanes-gpu.json`.
  - P: first update vs stock, bf16.
  - X: zero and seeded exact carry vs full history, fp32 (8254 at 2 lanes, 8745 at 16 lanes).
  - T: stock / v1 / v2 seconds per update.
  - M: v1 / v2 table bytes and peak memory on the largest training user (6810, 3.9M reviews; an OOM is recorded, not fatal).
  - `--cpu-smoke` (CPU, 2 lanes, small users, every code path): ALL OK. Do not run the full-size check on CPU: the CPU WKV reference keeps per-step states, so it reached 17 GB of RAM before it was stopped.
- **Runbook** `artifacts/small-model/runbook-2026-10-18-sm21v2.sh`: experiment `scale-w64kd-gate-sc-lanes-cont`, labels `s-w64kd-gate-sc-lanes-e11/-e12/-dec12`.
  - Same recipe as the v1 runbook: from e10 of `scale-w64kd-gate-cached-ctx32768-cont`, e11–e12 at 2,048 x 16, 0.25-epoch decay, 464-user scoring, SM11 replay targets.
  - Rows go to `artifacts/small-model/heldout-sm21v2.jsonl`. Arms A and B are read from `wt/state-carry/artifacts/small-model/heldout-sm21.jsonl`.
  - Decision: C - B, with the Q6 rule. C - A is informational.
  - Dry run: `artifacts/small-model/runbook-2026-10-18-sm21v2-dryrun.log` ends with `DRYRUN OK`.

## Left

1. **GPU check ticket** (~15 min; never outside the queue):

   ```
   C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py run --owner small-model --purpose "SM21 v2 GPU check: strided-lane carry (P/X/T/M)" --claim gpu --expected-minutes 15 -- "C:\Program Files\Git\usr\bin\bash.exe" C:/Users/jonat/workspace/wt/sm21v2/artifacts/small-model/jobs/sm21v2-gpu-check.sh
   ```

   - Pass: `ALL OK` (P and X). Then commit the json.
   - Record T (v2/stock, v2/v1) and M (largest-user table and peak) in `docs/GPU_OPTIMIZATIONS.md` row S09.
   - If M is near 16 GB, allocate lane slots lazily or move finished lanes to host before the experiment.
2. **Experiment** (after the GPU check passes; can run while or after arm B finishes, one arm, about 218 GPU minutes):

   ```
   DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm21v2/artifacts/small-model/runbook-2026-10-18-sm21v2.sh
   nohup bash C:/Users/jonat/workspace/wt/sm21v2/artifacts/small-model/runbook-2026-10-18-sm21v2.sh > C:/Users/jonat/workspace/wt/sm21v2/artifacts/small-model/runbook-2026-10-18-sm21v2.log 2>&1 & disown
   ```

   - The summary goes to `artifacts/small-model/sm21v2-summary.json/.md`.
   - Rerun with `SUMMARY_ONLY=1` once arm B's dec12 rows exist.
   - Do not edit this worktree while it runs.
3. Expectation and caveat: v2 removes the stale class, but 19% of card reviews (13% of notes) start from zero at lane cuts. In v1 those reviews had a stale but informative state. If C does not beat A, the next step is stale lane-start seeding (option 3).

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

## Experiment runbook (written 2026-10-18, NOT launched)

`artifacts/small-model/runbook-2026-10-18-sm21.sh` (dry run log: `artifacts/small-model/runbook-2026-10-18-sm21-dryrun.log`, ends with `DRYRUN OK`). Arm A (carry, `"state_carry": true`) runs completely first, then arm B (control, `{"carry": false}`); both continue from epoch 10 of `scale-w64kd-gate-cached-ctx32768-cont` at 2,048 x 16 with SM11 replay targets, epochs 11-12, 0.25-epoch decay, 464-user scoring. About 436 GPU minutes in total (218 per arm). Launch only after the SM11 replay export is complete:

```
# 1. export complete? (exit 0 = yes)
C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 C:/Users/jonat/workspace/wt/state-carry/scripts/sm11_replay_export.py status --out C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/fork-fp32-gate5k --run C:/Users/jonat/workspace/wt/sm11/artifacts/accuracy/experiments/scale-w64kd-gate/run/run.json
# 2. dry run (CPU, ~20 s; the export warning is expected until it is complete)
DRYRUN=1 bash C:/Users/jonat/workspace/wt/state-carry/artifacts/small-model/runbook-2026-10-18-sm21.sh
# 3. launch (detached; rerun the same command to resume; ONLY_ARM=carry|zero for one arm)
nohup bash C:/Users/jonat/workspace/wt/state-carry/artifacts/small-model/runbook-2026-10-18-sm21.sh > C:/Users/jonat/workspace/wt/state-carry/artifacts/small-model/runbook-2026-10-18-sm21.log 2>&1 & disown
```

Verify: log lines `[sm21 HH:MM:SS]`, last line `done`; results `artifacts/small-model/sm21-summary.json` / `.md` (decision rule A - B: curve <= -0.001 with upper95 < 0 and instant <= +0.0005), new score rows in `artifacts/small-model/heldout-sm21.jsonl` (commit it afterwards). Do not edit this worktree while it runs. Real run refuses uncommitted source in scripts/ expertium/ rwkv/.
