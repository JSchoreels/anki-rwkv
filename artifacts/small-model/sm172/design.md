# Sharpness-aware cool-down (SM172): design

Written Wed 2026-11-25, before any GPU run of this screen (only the CPU proofs ran first; their bfloat16 numbers fix the two radii below). Branch `sm172` (from `origin/sm171` at 569e1f7a), worktree `C:/Users/jonat/workspace/wt/sm172`. Run tree: `C:/Users/jonat/workspace/wt/sm172-run`, a new detached worktree at `cea0f931` (`origin/sm148`), the same commit the state-energy penalty screen (SM171) trained and scored in, **not edited**. Track: small model, next family.

## Question

Our card and note states are compressed after training: T24 (rank 2 below 16 answers, then rank 4, the chosen format) or Q2 (rank 2 always, `w8+lr2q8x8`), both against int8 (REF, `w8+s8hx8`). For six-layer Tiny two-head (SM130), after the SM171 control cool-down, Q2 costs +0.084% curve / +0.135% instant against its own REF, and T24 +0.044% / +0.078%.

**Question: does a short cool-down with sharpness-aware minimization (SAM) find flatter minima that lose less when the states are compressed, i.e. shrink the Q2 cost against own int8 by at least 0.05% compared with the identical cool-down without SAM, without making int8 worse?** A cheaper Q2 or T24 would make the next family's state format cheaper.

| plain name (ID) | what it is here |
|---|---|
| six-layer Tiny two-head (SM130) | the model: `w64-c1-d2p1u1`, K = 16, 185,057 parameters, release folder `wt/sm130/artifacts/small-model/sm130-th-dec2ep32k-p2` (`master.pth` sha256 48a5e724...) |
| the state-energy penalty screen (SM171) | its control arm `sm171-tiny-ctl` is this screen's control: same start, data, order, seed, learning rate and code path, minus SAM. Its trained model (in `wt/sm171-run`) and scored rows (`../sm171/gpu/sm171-rows-732.jsonl`) are reused, not retrained or rescored |
| the low-rank int8 states experiment (SM148) | the codec and scorer labels REF / T24 / Q2 and the original model's rows (`artifacts/small-model/sm148/gpu/sm148-rows-732.jsonl` in the run tree) |

## SAM (plain form)

Per update, with g the update's gradient at the weights w (after the trainer's clip, which only rescales g):

- e = rho g / ||g|| (global 2-norm over every parameter with a gradient);
- g_SAM = the gradient of the same update's loss at w + e: the same 32,768-review windows, the same carried input states (every window reads the carry table as of the update start), the same replay KD targets, the same dropout seed (`item.seed`);
- the weights are restored to w bit for bit, g_SAM goes through the trainer's own clip (0.5), then the stock optimizer steps (`expertium-iter53`: Muon + AdamW, the release model's optimizer state).

What stays the first pass's (at w): the carried states written back after the update, the logged loss and gradient norm. This is SAM as published (two forward / backward passes per update, the ascent step normalised to radius rho), with nothing else added: no adaptive (per-weight) scaling, no m-sharpness, no momentum on e.

**Placement (no run-tree edit), `sm172_sam.py`.** `install` patches two methods of the run tree's `scripts.expertium_options.CatchUp`, the object the trainer already calls once per update: `loss` (the outer call of an update clones the carry table's slots and keeps the call's arguments; the first pass then commits in place, as stock) and `optimizer` (the optimizer it returns gets an instance `step` that runs the second pass, then the stock step). In the second pass the table is the clone, so it reads the states of the update start; its commits go to the clone, which is dropped; the first pass's table is put back. The trainer keeps fp32 master weights and a bfloat16 child (selectively cast): e is added to the master, the child is downcast from it (one rounding, as `MasterChildSync.downcast`), the child's gradient is moved to the master in fp32 (with the stock `+ 0.0`), the master is restored, the child downcast again. The trainer reaches SAM through the existing allow-listed `state_vq` track option: `{"recipe": "sm172-sam", "lambda": 0.0, "rho": R}`; `sm172_run.py` installs the SM171 hook at lambda 0 (the control's exact code path: stock arithmetic plus its no-grad dropped-share diagnostic) and SAM. The SM171 hook is switched off during the second pass, so its diagnostic samples the same first-pass calls as in the control. Nothing from `scratchpad/row17-bundle` or `row55-bundle` is used or opened.

## Proofs (CPU, before the screen)

`sm172_sam.py cpu --users 4745,2308,7112` -> `check-cpu-sm172.json` (run tree `cea0f931`, clean; SM171 hook unchanged, LF sha256 388d83e0...): **ALL OK**.

- **S1** the real trainer path on CPU with a separate master and child (`MasterChildSync`, as on the GPU), release weights and Muon state, the release cool-down's options (strided-lane carry, show-time inputs, replay KD targets), 256-review windows so that users carry state across windows, 9 updates: control (SM171 hook at lambda 0), SAM rho 0 and SAM rho 0.05 from the same start. SAM rho 0 equals the control bit for bit on every update (loss, first and stepped gradients, weights, carry table), which proves the replay of the update (seed, carried input states, table restore, gradient transfer, clip, step). With rho 0.05: the first update's loss, first gradient and written-back carry table equal the control's; the weights are restored bit for bit before every inner step; the second-pass gradient of the first update equals an independent reference (a fresh model moved to w + e, a fresh carry table, the same seed) bit for bit; later weights differ; everything finite; the SM171 hook sees exactly the control's calls. The loss rise L(w + e) - L(w) is 1.2 to 1.9 times its first-order value rho ||g|| (positive curvature along g).
- **S2 (report)** bfloat16 survival: on real first gradients (users 4745, 2308 and 7112, windows up to 2,048 reviews), the downcast perturbation cast(w + e) - cast(w) keeps e itself (its part along e is 0.98 to 1.07) but adds rounding of norm 4.0-4.5 x rho at rho 0.01, 2.8-3.2 x at 0.02, 1.6-1.8 x at 0.05, 1.1-1.25 x at 0.1, 0.74-0.81 x at 0.2. 183,376 of the 185,057 parameters are bfloat16 in the child.

## Radii (fixed now): rho = 0.2 and rho = 0.05

Published SAM radii with Adam-type optimizers sit between 0.02 and 0.2 (0.05 the default of the original SAM paper and the usual choice for fine-tuning; 0.05 to 0.2 for vision transformers trained with AdamW; 0.02 to 0.15 for language-model fine-tuning). Our child weights are bfloat16, and S2 shows that below 0.05 the second pass would mostly see rounding (at 0.01 the rounding is 4 x the step), so 0.01 would not test SAM. **Arms: `sm172-tiny-r20` (rho 0.2, the top of the published range; rounding 0.8 x the step) and `sm172-tiny-r05` (rho 0.05, the default; rounding 1.7 x).** For scale: the weights' global norm is 176.8, so rho 0.2 moves them by 0.11%.

## Recipe

Both arms: exactly the SM171 control's recipe (`sm172_prepare.py pair` proves each arm's `decay.json` and `protocol.json` equal the control's except the `state_vq` option): start = the release model's final `master.pth` AND `optimizer.pth` (Muon state); the release folder's options, loss and targets unchanged (show-time inputs, strided-lane carry, 32,768-review windows, replay KD targets of the show-time Big model (SM91s2) at alpha 0.5, deleted cards neighbour); 944 users, 67,310,673 reviews in the control's order (`random.Random(171)`, training users only, disjoint from the 732 set); learning rate 0 -> 7e-5 over 64 updates, then one_minus_sin to 0 (about 2,570 updates); snapshots every 100 users, stop / resume through the resumable decay (SM69) machinery.

Diagnostics: `<arm>/sam.jsonl` in the run tree, one line per 64 updates (the clipped gradient norm, L(w) and L(w + e), the second-pass gradient norm, and every 16th update the bfloat16 survival of e for rho 0.01 to 0.2), and `<arm>/energy.jsonl` (the SM171 dropped share, comparable with the control's).

## Scoring

Exactly as SM171: the run tree's `scripts/small_model_heldout.py`, `--pool 732 --users-file <sm72s eval-users-732-fit.json> --inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows --prior-buckets 0,1,3,5,9,16,32,128,300,1000`, `PYTORCH_JIT=0`, GPU cap 0.8 (0.92 for a retry pass, memory only). Labels for each SAM arm: REF `w8+s8hx8`, T24 `w8+lr2.16.4q8x8`, Q2 `w8+lr2q8x8`, all on show-time inputs, curve and instant. The control and the original model are not rescored; an integrity ticket rescores the original model's REF on the 3 SM148 smoke users and must equal the SM148 rows to 1e-6 (this checks the new run tree).

## Rules (pre-registered; `sm172_report.py`)

Statistic as SM148 / SM171: % = 100 x (sum of the label's LogLoss / sum of the reference's LogLoss - 1), review-weighted, paired user bootstrap, 2,000 draws, numpy `default_rng(0)`, 95% CI, on the users common to the six SAM labels, the control's three labels and the original model's rows (rule users).

- **Q2 cost** of an arm = its Q2 vs its own REF, per head. **Effect** = the SAM arm's Q2 cost minus the control's Q2 cost, both ratios on the same bootstrap draw.
- **REF guard** = (the SAM arm's REF vs the original model's REF) minus (the control's REF vs the original model's REF), on the same draw, per head.
- **Go for a SAM arm:** on at least one head (curve or instant), effect <= -0.05% with the CI upper bound below 0 (the usual minimum effect), AND on both heads the REF guard's CI upper bound <= +0.05%.
- **Verdict:** GO if at least one SAM arm passes; otherwise NO GO. Coverage / finiteness: at least 720 rule users for every label and no non-finite row, else `pending` (exit 2) and the main session decides.
- **Reported, no verdict:** the T24 cost of each arm and its change vs the control; each arm at REF / T24 / Q2 vs the original model at REF; SAM Q2 vs control Q2 and vs control T24; SAM REF vs control REF; the instant Q2 cost per answer-count bucket and its change vs the control; the dropped-share curves (`energy.jsonl`) vs the control's; the SAM diagnostics (`sam.jsonl`); training reviews/s and GPU memory.

`sm172_report.py selftest` (the original model's SM148 rows as every arm) gives zero effects and NO GO on 723 users (`gpu/selftest-result.md`); with the control's own rows as the SAM arms it reproduces the control's numbers (Q2 +0.084% / +0.135%, REF vs the original -0.008% / +0.008%).

## Follow-up if GO (not part of this screen)

The passing arm with the better REF guard is the candidate. Next: (1) the strict rules of the low-rank int8 states experiment (SM148) for that model; (2) the same SAM cool-down on six-layer Tiny Instant-Only (SM130), then Small two-head (SM118); (3) whether T24 then becomes cheap enough to drop to Q2 for the next family. If NO GO: T24 stays the chosen format and SAM is closed for state compression.

## Estimates and budget

| step | GPU minutes |
|---|---|
| trainer smoke, two arms | about 3 |
| `sm172-tiny-r20`, `sm172-tiny-r05` (67.3M reviews; the control ran 55.7k reviews/s, SAM doubles the forward and backward: about 28k reviews/s) | about 40 to 45 each |
| integrity ticket | under 1 |
| scoring 6 labels on the 732 set (SM171: 9 labels in 70 min) | about 47, plus a retry pass |
| **total** | **about 2.3 h** (brief: at most about 4 GPU hours) |

Memory: cap 0.8 of 16.3 GB for every ticket (the control peaked at 4.2 GB allocated, 8.4 GB reserved; SAM adds one clone of the user's carry table and of the weights). CPU: the CPU proofs and the report stay under 3 GB; the trainer's process tree is the control's. No 8k / 2k 10-epoch ETA: this is a short cool-down, not a training run.
