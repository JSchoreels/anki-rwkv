# Show-time interval retrain (SM91) step 2: design (2026-11-02)

## Question

Step 1 (`wt/sm91/artifacts/small-model/sm91/sm91-decision.md`, GO) showed that the small model trained on show-time intervals wins back about 2/3 of its show-time cost (instant -0.28 points, curve -0.09). The KD teacher there was still the big model trained on the leaky answer-to-answer intervals. Step 2: one added big-model epoch on show-time intervals, for

- a show-time big model the Anki fork can ship (the fork predicts at show time);
- a cleaner teacher for step 3 (new teacher targets + small-model cool-down; not part of this step).

The duration-leak scoring (SM31) measured the big model (1 leak-free epoch) on show-time inputs at **+0.205 curve / +0.508 instant points** (points = % of its own normal-input LogLoss, 722 users of the 732 set). How much of that does one show-time epoch win back?

## What runs

| | New model | Reference (reused, read only) |
|---|---|---|
| Start | the big model (1 leak-free epoch): `wt/sm72/artifacts/small-model/sm72-big-leakfree-1ep/expertium/big-leakfree-1ep/master.pth` (sha256 `ac7f0d8a...`, the start the better teacher (SM65) load check passed) | the same checkpoint, as is |
| Training | one added epoch, the leak-free retrain (SM72) step 1 recipe: `scripts/sm72_big_epoch.py` (users 5000-10000 = the fork's own training users, 5,001 users / 370.3M reviews, shuffle seed 1 = the same order, whole histories, fresh AdamW (upstream groups), warmup 200 updates, peak 3.5e-4 -> 0 one_minus_sin over the epoch, 2 strided lanes x 8,192 carried state through the variant class "fork", budgeted carry table 768 MiB, cap 0.8, training seed 2025, `RWKV_DELETED_CARDS=neighbour`, the same 27 label-window trims); **inputs: show-time intervals** (`"show_time_intervals": true`) | none |
| Output | `artifacts/small-model/sm91s2-big-showtime/expertium/big-showtime/master.pth` (new folder; nothing existing is overwritten) | |
| Scoring | leak-free (`--split-existing --deleted-cards neighbour`), w8+s8, `fork::` spec, the 732 set (723 fit users at cap 0.8 + the 3 long users at 0.92), `--inputs e2s-query` and normal inputs; label `sm91s2-big-showtime/w8+s8[@e2s-query]`, rows `gpu/sm91s2-inputs.jsonl` | e2s-query: SM31's rows (`wt/sm31/.../gpu/sm31-inputs.jsonl`, `big-leakfree-1ep/w8+s8@e2s-query`, 726 users); normal: SM72's rows (`wt/sm72/.../gpu-big/bigleak-neighbour.jsonl`, `big-leakfree-1ep/w8+s8`) |

The scorer flags are SM31's big-model rows' flags exactly (`--pool 732 --users-file <732 fit> --model TAG=fork::CK --only TAG/w8+s8 --inputs V --split-existing --deleted-cards neighbour`, scorer processes without `RWKV_DELETED_CARDS` / `EXPERTIUM_END_TO_START`).

## Same trainer, same transform

The big model's epoch runs through `scripts/expertium_decay.py run` -> `gpu_r01_training.run_case` -> `gpu_r01_prefetch.training_batches` (state-carry path, main-process frames, no packing / preprocessing cache / worker_frames): the trainer the step 1 flag was built in. So the step 1 transform is used as is, no port:

- `scripts/gpu_r01_prefetch.py`: `show_time_frame` (the curve query time = the next review's show-time interval) on the full user frame after the deleted-card transform, before the train split;
- `scripts/gpu_r01_samples.py`: `show_time_query_rows` (the instant query rows' interval, cumulative interval, scaled values, sin / cos at show time).

Update rows keep the full interval. The only code change in this step: `scripts/sm72_big_epoch.py prepare --extra-options JSON` (merged into the prepared options after `carry_options`, recorded in `decay.json` "extra_options"; absent = the previous `decay.json`).

No KD here: the big model trains on the labels only (SM72's recipe has no teacher), so the leaky teacher question of step 1 does not arise.

## Base branch

`sm91s2` is cut from `sm91` (`24b3f154`: the step 1 flag, the SM31 scorer, the SM72 big-epoch trainer up to `7f5cfdc4`) with `sm65` merged (`0e78817a`: `sm72_big_epoch.py --init / --track`, the code the better teacher (SM65) big-model epoch ran on, plus SM72's score rows). The merge was clean; `sm91` did not touch `sm72_big_epoch.py`.

## Pre-registered rule (`sm91s2_rule.py`, fixed before launch)

Users: the 722 users of the 732 set that the big model (1 leak-free epoch) normal rows have. Review-weighted benchmark LogLoss, paired new - reference, both on `e2s-query` inputs, user bootstrap 2,000, seed 0, 95% CI; points = % of the big model (1 leak-free epoch) normal-input LogLoss on the same users.

- **go to step 3** if the new model gains >= **0.10 points** on at least one head with that head's 95% CI entirely below zero;
- **stop** otherwise.

Also reported (not the rule): the gain as a share of SM31's +0.205 / +0.508; the cost left (new e2s - reference normal); the new model on normal inputs vs the reference on normal inputs; the new model's own show-time cost (new e2s - new normal); the 725-user 732+long pool.

**Caveat (the added epoch).** The new model has one more epoch than the reference, so part of a gain could come from the extra epoch itself, not from the show-time inputs. The "new normal - reference normal" row is the check: a show-time-trained model is not expected to improve on answer-to-answer inputs, so a clear gain there would point to an epoch effect. The clean control (the same added epoch on normal inputs, ~9.5 GPU h) is not run; it can be added if the read-out is ambiguous. For reference, the better teacher (SM65) added epoch on 3,969 unseen users moved normal-input LogLoss by -0.17% curve / -0.03% instant.

## CPU proofs (`scripts/sm91s2.py cpu-check` -> `check-cpu-sm91s2.json`), run by the launcher after 01:00

The launcher first applies the label-window pre-flight (`scripts/sm72_label_windows.py --trim`, as SM72 step 1), then:

- **T1** trainer samples, flag on vs off, the big model's options (users 8632, 5984, 7885 of its training users: 2,641 / 12,595 / 20,883 reviews = 1 update, 1 update with 2 lanes, 2 updates): only interval columns 2-7 of instant query rows and the curve query time of update rows move, never longer (`scripts/sm91.py` t1_t2, unchanged).
- **T2** trainer rows (flag on) == the SM31 scorer's e2s-query rows bitwise; flag off == its normal rows; on - off == e2s-query - normal.
- **T3a** flag off == the current big-model trainer: full training batches (SM72's options minus `packed_ipc`, span 2 x 8,192) of this tree and of a git archive of `c4fc3b84` (the better teacher (SM65) launch commit, the last big-model epoch) are bitwise equal in every tensor. **T3m** model side: the fork architecture with the start weights, every update in order with the carried state, train mode, the trainer's seeds: batch digest, loss and every gradient sha256 equal between the two trees (users 8632, 5625 at 1,024-review windows x 2 lanes, 2 updates each, so the CPU process stays well under 6 GB; window size is a layout parameter, the code path is the epoch's). T3b / T3mb (information): with the flag only the feature / label tensors change and the loss changes.
- **T4** config: `decay.json` / `protocol.json` == SM72 step 1's (`decay.json`, `protocol.prepared.json`, `protocol.untrimmed.json`) except the flag, the start weights (sha256 checked against the file and the SM65 load check), track / paths and `snapshot_users`; same users, order, trimmed ranges and trims; a fresh start (no `lr_resume`, nothing trained yet); new output paths; SM72's folder against itself is refused (no flag).
- **T5** the rule's self-test: SM31's big-model e2s-query vs normal rows through `compare()` give +0.205 [+0.165, +0.258] curve / +0.508 [+0.411, +0.633] instant (= `sm31-result.json`), the reference against itself 0 / stop, a synthetic model 0.2 points better on instant only: go.
- **T6** the scorer: the reused SM31 rows came from commit `d8b1f2c5`; this tree's scorer and a git archive of `d8b1f2c5` give identical rows on CPU (users 4676, 2388, the big model (1 leak-free epoch), w8+s8, e2s-query and normal). The scorer files are also unchanged by git diff between `origin/sm31` and this branch.

`STOP: CPU proofs not ALL OK` stops the launcher before any GPU ticket.

## Changes against the brief

- **Snapshots every 50 users** instead of 400 (the only recipe field changed besides the flag, start and paths): a hard kill loses <= 50 users (~7.4M reviews, ~5-6 min); a stop request (`gpu/STOP`, a quiet window now, or <= 2 min before one) snapshots at the next user boundary (~1 min). 100 snapshots x 33 MB = ~3.3 GB of disk.
- **Same user order as SM72 step 1** (shuffle seed 1): the recipe as is; a second epoch with the same order is what "same recipe" gives.
- **T3m at 1,024-review windows** (memory), T3a at the real 8,192 windows (batches only, no model). **T6 added** (the reused rows' scorer).
- **SM91 wait fallback:** besides `done` / `STOP: ended`, the launcher goes ahead if the SM91 step 1 launcher process (`launch-sm91(-tonight).sh`) has been gone for 15 min, or after 6 h, so a dead launcher does not idle the GPU all night.

## Cost

GPU: smoke ~2-10 min, epoch ~560-575 min (SM72 step 1: 11,458 reviews/s; SM65: 11,339), scoring e2s-query ~50 + ~20 min, normal ~30 + ~20 min: ~11.5 h. CPU: label windows ~1-2 min, proofs ~10-20 min.
