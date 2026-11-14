# The big instant-only teacher (SM110): design (2026-11-12)

## Question

The show-time instant-only big model (SM99) came out "ceiling above the Small Instant-Only model (SM98)". Its rule was
pre-registered before launch: this verdict means a teacher experiment is worth running next.
- The show-time instant-only big model (SM99) minus the Small Instant-Only model (SM98), instant head, show-time inputs:
  **-0.108% [-0.133, -0.084]**, on 722 users of the 732 set (0.275894 vs 0.276191).

This run asks one thing. If the Small Instant-Only model (SM98)'s recipe is distilled from the show-time instant-only
big model (SM99), does the small instant-only model close part of that gap?

Unit everywhere: % of the big model (1 leak-free epoch)'s normal-input instant LogLoss, 0.275757 on 722 users of the 732
set. All runs are leak-free (deleted cards: neighbour), scored w8+s8.

Models named here:

| plain name (ID) | what it is |
|---|---|
| the big instant-only teacher (SM110) | this design: the Small Instant-Only model (SM98)'s cool-down rerun with the targets of the show-time instant-only big model (SM99) |
| the Small Instant-Only model (SM98) | 35 epochs + the 32k Muon cool-down on show-time inputs, instant-only; the reference and the recipe |
| the show-time instant-only big model (SM99) | the teacher: the big architecture, 2 instant-only epochs on show-time inputs; final model = epoch 2 |
| the show-time big model (SM91 step 2) | the teacher of the show-time teacher targets (SM91 step 3), which tied |
| the big model (1 leak-free epoch) | the teacher of the Small Instant-Only model (SM98)'s targets (normal inputs); the unit's denominator |

## Start point: the Small Instant-Only model (SM98)'s epoch 35, not its final model

The run starts from the Small Instant-Only model (SM98)'s constant-LR epoch 35 and reruns its one-epoch cool-down. The
only change is the KD replay targets. The epoch 35 weights are `wt/sm98/artifacts/small-model/sm98-inst-e35-st`, read
only, master sha256 f7577d14ba10, update offset 772,730.

Why not a warm start from the Small Instant-Only model (SM98)'s final model:
- **The final model's learning rate is already at 0.** Our chain always ends with one cool-down from a constant-LR
  epoch. A second 7e-4 -> 0 cool-down on top of the first would mix two changes, the new targets and a second
  cool-down epoch, and no run has a reference for "two cool-downs".
- **A short low-LR fine-tune from the final model would be a new recipe.** Its learning rate and length would have no
  tested reference.
- **Epoch 35 gives a clean pair, one difference only.** The cool-down from epoch 35 with the new targets is the
  Small Instant-Only model (SM98)'s own cool-down leaf for leaf, except `distillation.teacher`, `distillation.replay`
  and paths (`sm110_pair.py`). It has the same start, users, order, seeds, schedule, alphas, curve loss weight 0,
  show-time flag and trims. The reference is then the Small Instant-Only model (SM98) itself, read from its committed
  rows and never retrained.
- **The cool-down is where the teacher acts at full exposure.** All 8,000 users and 581.9M reviews get the new
  targets with `decay_alpha` 0.5: each target is 0.5 x teacher + 0.5 x label. The show-time teacher targets (SM91
  step 3) only measured a quarter-epoch cool-down (145.6M reviews), so they gave a lower bound.

Not chosen, and kept as a follow-up: also redo epoch 35 from epoch 34 with the new targets. The constant-LR epochs use
alpha 0.9, so the teacher weighs more there. This costs +1.9 GPU h (112 min at 87k reviews/s). It is worth running only
if this run lands in the grey zone: CI below 0, but less than 0.05%.

## Teacher targets

- **Teacher:** the show-time instant-only big model (SM99), final epoch 2,
  `wt/sm99/artifacts/small-model/sm99-big-inst-e2/expertium/big-inst/master.pth`, read only. Its sha256 is c49e45d69819,
  the same bytes as the release package `wt/big-package/artifacts/models/sm99/master.pth`.
- **What it replays:** for each user, the whole history in one sequence, state carried from the first review, as in the
  scorer and in Anki. Per review it stores the curve logit and the 4 rating probabilities, as float16 (10 B per review).
  Users above 600k reviews run in passes with 300k reviews of burn-in.
  - Inputs: show-time intervals (`--inputs e2s-query`), exactly the scorer's `e2s-query` inputs, which is what the
    student trains on.
  - Deleted cards: neighbour. Teacher precision: float32 weights and float32 state.
  - Exporter: `scripts/sm11_replay_export.py --inputs e2s-query`, unchanged since the show-time interval retrain
    (SM91) step 3.
- **Users:** the same 8,000 final-tier training users as the Small Instant-Only model (SM98)'s targets
  (`wt/sm55/artifacts/small-model/sm55-final8k-run.json`), 581,942,738 reviews.
  - 4,031 of them (ids 5000-10000) are training users of the show-time instant-only big model (SM99). They are also
    training users of the old teacher, the big model (1 leak-free epoch).
  - The other 3,969 (ids below 5000) were seen by neither teacher.
  - So the swap keeps the old targets' structure and changes only the teacher. None of the 732 set or the 1,500
    validation users is in the export.
- **Output:** a NEW folder, `anki-rwkv/artifacts/cache/kd-replay/sm99-big-inst-final8k-neighbour-e2s`, about 5.8 GB.
  The old targets are never touched.
- **GPU time:** the same architecture exported the 8,000 users' targets in 239 min at 40,577 reviews/s (the big model
  (1 leak-free epoch), 10-26). The show-time teacher targets (SM91 step 3) ran at 38,174 reviews/s. So expect 240-255 min
  plus about 10 min of start-up: **about 4.3 GPU h.**
- **GPU memory:** the exporter caps itself at 0.8 (`SM11_GPU_MEMORY_FRACTION` default, at most 13.0 of 16.3 GiB). An
  out-of-memory error halves that user's pass length and retries; this happened for 3 users in the show-time interval
  retrain (SM91) step 3. The launcher logs the ticket's nvidia-smi maximum.
- **What the targets look like (CPU proof C4a, users 4001 and 8081):**
  - NaN pattern identical to the old targets and to the show-time teacher targets (SM91 step 3), so the label windows
    and the one trim stay the same;
  - every value finite;
  - mean absolute change of P(again): 0.009 / 0.018 against the old targets, 0.007 / 0.013 against the show-time
    teacher targets (SM91 step 3).

## Training schedule

One cool-down epoch, equal to the Small Instant-Only model (SM98)'s cool-down except the targets:
- 8,000 users, 581,942,724 reviews after the same single trim (user 5599, 14 reviews);
- the release's order and seeds (`sm77_warm.py cooldown --epoch 25`);
- fresh Muon (`expertium-iter53`), LR 7e-4 -> 0 on a one_minus_sin schedule over the epoch;
- 32k-review windows, 1 lane, strided carry;
- KD replay with alpha 0.9 and decay alpha 0.5 (the cool-down uses 0.5);
- `curve_loss_weight` 0 and `show_time_intervals` true;
- GPU cap 0.8; snapshots every 400 users.

The Small Instant-Only model (SM98)'s cool-down measured:
- 479.6 min at 20,232 reviews/s;
- peak allocated 7.4 GiB, reserved 12.5 GiB, nvidia-smi max 13,265 MiB;
- process-tree RSS 11.1 GB (the main process + 5 preparation workers).

There are no constant-LR epochs. The final model is the end of the cool-down,
label `sm110-inst-e35st-st-dec1ep32k-t99/w8+s8`.

## Pre-registered rules (`sm110_rule.py`, fixed before launch)

All numbers: 722 users of the 732 set, review-weighted instant LogLoss, show-time inputs (`e2s-query`), paired user
bootstrap (2,000 resamples, seed 0, 95% CI), as % of the big model (1 leak-free epoch)'s normal-input instant LogLoss.
References are read from their own committed rows (sha256 pinned), never rescored.

**Early stop: the mid-check (`sm110_rule.py mid`).**
- When the cool-down writes its periodic snapshot after 4,000 of the 8,000 users (284.9M reviews, 49%), the launcher
  requests a snapshot stop. It scores that snapshot on show-time inputs.
- x = SM110's snapshot users-04000 minus the Small Instant-Only model (SM98)'s own snapshot users-04000 (kept in
  `wt/sm98`, scored as the run's first ticket). Both snapshots saw the same users, in the same order, at the same
  learning rate. Only the targets differ.
- **Continue** if the point estimate x is below 0. The cool-down then resumes from its snapshot.
- **Stop** otherwise (x >= 0). The cool-down is not finished, the verdict is "no go (stopped at the mid-check)", and
  this saves about 4.3 GPU h.
- **Why halfway and why 0.**
  - By the halfway point, 85% of the cool-down's learning-rate integral is spent: the one_minus_sin schedule puts the
    LR at 30% of its peak there.
  - So the targets' pull on the weights is mostly done. A final gain of 0.05% that has not even started to show at
    halfway is unlikely.
  - Zero is also the line the show-time instant-only big model (SM99)'s early rule used.

**Final go rule (`sm110_rule.py decide`).**
- **Primary:** the big instant-only teacher (SM110) minus the Small Instant-Only model (SM98).
  - **Go** if its 95% CI lies entirely below 0 AND the gain is at least 0.05% (points <= -0.05).
  - **No go** otherwise.
- **Reported with it (no verdict):**
  - the share of the teacher gap closed: (SM110 - SM98) / (SM99 - SM98) on the same users;
  - minus the show-time instant-only big model (SM99), the show-time instant-only small model (SM96), the show-time
    retrain of the release (SM92) and the release (SM63), all on show-time inputs;
  - minus the Small Instant-Only model (SM98) on normal inputs;
  - normal inputs minus the big model (1 leak-free epoch): the smaller-model bar, CI upper end <= +0.40%;
  - the show-time cost (show-time minus normal inputs);
  - the mid-check pair.
- **The 1,500 validation users,** on show-time inputs, as % of the release (SM63)'s own LogLoss: minus the Small
  Instant-Only model (SM98), minus the show-time instant-only big model (SM99), minus the release (SM63), by id part,
  and the row check.
  - The ids >= 5000 in that pool are training users of the show-time instant-only big model (SM99). So "minus SM99"
    there is not a held-out comparison.
  - For the small models the whole pool stays held out.

**Selftest (`sm110_rule.py selftest`, part of the CPU proofs):**
- the Small Instant-Only model (SM98) against itself gives 0, no go, mid stop;
- the show-time instant-only small model (SM96) minus SM98 gives +0.035% [+0.029, +0.041], no go;
- the show-time instant-only big model (SM99) minus SM98 gives -0.108% [-0.133, -0.084], go, mid continue;
- a synthetic model 0.04% better everywhere gives no go (CI below 0 but under the 0.05% bar);
- the edges of both rules, and an end-to-end `mid` + `decide` on relabelled rows.

## Expected gain: about -0.03%, below the bar more often than not

- **The gap on offer is -0.108% [-0.133, -0.084].** This is held out: the 732 set was never trained on by either
  teacher or by the student.
  - SM99's -0.271% on the 1,500 validation users is not a held-out teacher gap. About half of those 1,487 users (761,
    ids >= 5000) are its own training users.
- **Transfer evidence is thin, and it points low.**
  - The show-time teacher targets (SM91 step 3) swapped in a show-time teacher for a quarter-epoch cool-down and got
    -0.004% instant, a tie.
  - The better teacher (SM65) design estimated that a student gets a third to a half of a teacher's gain. It was
    parked for that reason.
  - A heavier teacher weight (SM59) made things worse.
- **What is new here.** The student already matches the show-time big model (SM91 step 2): -0.003% [-0.026, +0.019].
  So the show-time teacher targets (SM91 step 3) came from a teacher at the student's own level. The show-time
  instant-only big model (SM99) is the first teacher clearly above the student on the student's own inputs:
  -0.105% vs the show-time big model (SM91 step 2).
- **What works against it.**
  - The old targets were computed on normal inputs: the teacher saw the answer duration. They are 0.05% more accurate
    in label LogLoss than the new show-time targets (0.275757 vs 0.275894 on the 732 set).
  - Swapping them for show-time targets that were 0.15% less accurate changed nothing (SM91 step 3). So the student
    is not very sensitive to how accurate its targets are.
- **Estimate.** A 20-40% transfer gives -0.02% to -0.045%; the central guess is -0.03%. The chance of clearing the
  -0.05% go bar is about 1 in 4.
- **Downside if it fails.** The mid-check caps a null result at about 8.9 GPU h (18 GPU h is the cost of the whole
  Small Instant-Only model (SM98) chain). Even a no go would settle the question: the small instant-only model cannot
  absorb a 0.1% teacher gap through the cool-down.

## GPU hours and timeline (launch about 12:00 Thu 2026-11-12, queue empty)

`sm110_block.py` on the real quiet.json: Thu 11-12 evening's break is lifted (a 1-minute window at 00:59-01:00 Fri
replaces it), so the next real break is Fri 20:00.

| step | GPU | ends about |
|---|---|---|
| preconditions (CPU: proofs, pair, rule selftest, sha256 pins) | | 12:01 |
| the Small Instant-Only model (SM98)'s snapshot users-04000 scored on show-time inputs (723 users, ~12 min) | 0.2 h | 12:15 |
| **teacher targets**: 581.9M reviews at 38-40.6k reviews/s | 4.3 h | ~16:30 |
| CPU: export status, verify (8 users recomputed on CPU), all 8,000 files vs the old targets, cool-down prepare + trim + label windows + Muon check + pair with the targets (~12 min) | | ~16:42 |
| cool-down trainer smoke (2 users x 512 reviews) | 0.1 h | ~16:50 |
| **cool-down, first half** (284.9M reviews at ~20.2k reviews/s, ~235 min) | 3.9 h | ~20:45 |
| mid-check: snapshot stop, SM110's snapshot scored (~12 min), `mid` rule | 0.25 h | ~21:00 |
| (if stop: done, verdict no go; about 8.9 GPU h in all) | | (~21:00 Thu) |
| **cool-down, second half** (297.0M reviews, ~245 min + ~5 min restart), held 00:57-01:01 by the 1-minute window | 4.2 h | ~01:20 Fri |
| final scoring: show-time and normal (723 + 3 long users each), the 1,500 pool, `decide` (the Small Instant-Only model (SM98): 47 min) | 0.8 h | **~02:10 Fri 2026-11-13** |

**Total: about 13.9 GPU h** (about 8.9 GPU h if the mid-check stops). About 14 h of wall time with an empty queue. The
Fri 20:00 break is far away, so a late start or a slow export still finishes well before it.

10-epoch ETA at the measured speeds (8k tier = 581.94M reviews per epoch, 2k tier = 147.6M):
- 10 cool-downs at 20,232 reviews/s: 79.9 h (8k tier) / 20.3 h (2k tier);
- 10 exports at ~39k reviews/s: 41.4 h / 10.5 h.

## Risks

1. **The effect is below the bar** (main risk, see above). The mid-check limits the cost.
2. **The new targets are less accurate in label LogLoss than the old ones** (no duration leak). If the student gains
   more from privileged-input targets than from matched ones, the result is a tie or slightly worse. That is still a
   clear answer.
3. **The teacher fits 4,031 of the 8,000 users.** The show-time instant-only big model (SM99) has seen them for 4 big
   model epochs, so its targets there sit closer to the labels and carry less soft information. The old teacher had
   the same overlap.
4. **A false stop at the mid-check.** A gain that only appears in the low-LR tail would be missed. Accepted, see the
   reasoning above.
5. **The untrained curve head.** The curve targets of the show-time instant-only big model (SM99) are meaningless, but
   curve loss weight 0 multiplies them by 0. They must be finite, or 0 x NaN poisons the loss.
   - Checked on CPU (C4a) and on all 8,000 files after the export (`sm110_check.py targets`).
   - The trainer also refuses a labelled row without a target.
6. **Export memory and time.** Long users run in passes, and an out-of-memory error halves the pass. The export is
   resumable per user, and tickets stop starting users before a break.
7. **Interruptions** (a reboot cut the show-time instant-only big model (SM99)'s epoch 2). Every step is resumable:
   the export per user, the cool-down from its last 400-user snapshot (the user in flight is retrained). Detached
   launchers die when the Claude app restarts, so check after restarts.
8. **Disk:** +5.8 GB for the targets (1.7 TB free).
9. **The mid-check reference lives in `wt/sm98`:** the Small Instant-Only model (SM98)'s snapshot users-04000 (C2
   pins it). `wt/sm98` must not be cleaned up while this runs.
10. **RAM:** the cool-down's process tree reached 11.1 GB over 6 processes in the Small Instant-Only model (SM98)'s
    run. The launcher holds a step while less than 3 GB is free. The CPU proofs stay far below 6 GB per process.

## What is built, and the new code

No source file changes: `scripts/ expertium/ rwkv/` == c99330a7 (C1), the trees SM96 / SM98 / SM99 / SM101 trained and
scored with. Everything is existing code:
- `scripts/sm11_replay_export.py --inputs e2s-query` (export, status, verify);
- `scripts/sm77_warm.py cooldown`;
- `scripts/check_replay_label_windows.py`, `scripts/check_muon_fresh_decay.py`;
- `scripts/expertium_decay.py` (smoke / run / stop, snapshots, resume);
- `scripts/small_model_heldout.py`;
- `showtime_report.py` (pinned).

New, all under `artifacts/small-model/sm110/` (small):
- `launch-sm110.sh`: the launcher. It combines the Small Instant-Only model (SM98)'s cool-down launcher with the export
  loop of the show-time interval retrain (SM91) step 3, plus the mid-check: a snapshot stop when users-04000 appears,
  scoring, the rule, then the resume;
- `sm110_pair.py`: the pair check, the show-time interval retrain (SM91) step 3's `sm91s3_pair.py` plus a
  `--untrimmed` mode for the recipe proof;
- `sm110_rule.py`: mid, decide, selftest;
- `sm110_check.py`: the CPU proofs C1-C8, plus `targets` after the export;
- `sm110_block.py`: the break windows, the code of `sm106_block.py` byte for byte;
- `extra-options.json`: the Small Instant-Only model (SM98)'s, with teacher + replay swapped.

CPU proofs (`check-cpu-sm110.json`, ALL OK, 1.3 min):
- C1 code trees;
- C2 sha256 pins and the reference snapshot;
- C3 the recipe == SM98's cool-down except the targets, with 6 mutated copies refused;
- C4a SM99's show-time targets on CPU for 2 users: manifest, NaN pattern, finite, moved;
- C4b this tree reproduces the GPU targets of the show-time teacher targets (SM91 step 3) within one float16 step;
- C5 the scorer reproduces the Small Instant-Only model (SM98)'s committed rows exactly on CPU and scores its snapshot;
- C6 rule selftest;
- C7 breaks on fixed times and on the real quiet.json;
- C8 disk.

Dry run: `gpu/launch-sm110-dryrun.log` ends `DRYRUN OK`.
