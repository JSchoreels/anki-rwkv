# Smaller model, card stream 1 layer (SM73 stage 2): early check, design addendum (2026-11-02)

Addendum to the smaller model (SM73) design (`wt/sm90/artifacts/small-model/sm73/DESIGN.md`, sections 2 B, 4 "Stage 2", 5). Stage 1 result: `wt/sm73/artifacts/small-model/sm73/sm73-formats.md` (int4 card + int8 note state passes, int4 everywhere diverges on one user).

## Deviation from the design's precondition (main-session decision)

The design runs stage 2 only after a buy-back lever brings the full-size small model to <= +0.25% curve / +0.20% instant vs the big model (1 leak-free epoch). That did not happen: the better teacher (SM65) landed between bars and the later levers gave nothing. The release improved anyway: the released small model is now the leak-free small model (SM72s) 25 epochs + a 1-epoch 32k Muon cool-down at **+0.36% curve / +0.28% instant** vs the big model (1 leak-free epoch) on the 722 users of the 732 set (curve 0.30003, instant 0.27654). The room under the bar (+0.5% / +0.4%) is ~0.14 points curve and ~0.12 points instant. The main session decided to run only the cheap **early check** now, with the stop rule below fixed before launch; the finish (more epochs + the full cool-down) is decided later.

## What runs

| | Arm | Reference (reused, read only) |
|---|---|---|
| Start | SM72s constant-LR epoch 20, card stream cut to 1 layer (`w64-c1`) | SM72s epoch 20 |
| Epoch 21 | SM72s's own epoch-21 protocol, rebuilt (`sm77_warm.py epoch`): same 8,000 users, order, seeds, options, constant LR 7e-4, AdamW state of epoch 20 re-indexed (the cut layer's moments dropped) | SM72s's own epoch 21 |
| Cool-down | quarter-epoch 32k Muon cool-down, users / order / schedule from SM72s epoch 21 (1,991 users, 145,578,259 reviews, update offset 463,638), the reference's options | the same: `wt/sm77/artifacts/small-model/sm77-ref-e21-dec-q32k-muon` (the tiny architecture additions (SM77) reference) |
| Scoring | leak-free (`--split-existing --deleted-cards neighbour`), w8+s8, 732 set (723 fit users at cap 0.8 + 3 long users at 0.92) | the same, already scored (`wt/sm77/.../sm77/gpu/sm77-neighbour.jsonl`, label `sm77-ref-e21-q32k/w8+s8`) |

Both arms have 21 epochs + a quarter cool-down, so "SM72s epoch 20 -> +1 epoch" lines up with the reference in total epochs. The design said to give the reference "epoch 20" a quarter cool-down; using the existing epoch-21 reference instead matches the training exposure exactly and costs no GPU.

`sm73s2_pair.py` checks before training: the arm epoch == SM72s's epoch-21 protocol except the architecture, the start weights and paths (PASS today: 8,000 users, update offset 441,560); the arm cool-down == the reference's except the architecture, the start weights and paths (checked by the launcher after the epoch; the same script passes a w64 cool-down prepared today by this tree's tooling against the reference, so the preparation reproduces the reference's layout).

The cut (`expertium.variants` preset `w64-c1`, `scripts/sm73s2.py cut`): the card stream's second layer (`rwkv_modules.0.blocks.1.*`, its channel mixer was already stripped) is one residual branch; removing it leaves the other 12 layer steps in their round-robin order and every other tensor bit for bit. 331,511 -> 311,786 parameters (-19,725), scorer w8 weight bytes 410,459 -> 385,485.

## Stop rule (pre-registered, `sm73s2_rule.py`)

Users: the 722 users of the 732 set that the big model (1 leak-free epoch) rows have. Review-weighted benchmark LogLoss, paired arm - reference, user bootstrap 2,000, seed 0, 95% CI.

- **stop** if the point estimate of the delta is > **+0.00020 curve** or > **+0.00015 instant** (design section 4, about 3/4 of the then-room);
- otherwise **go to finish**.

Also reported, not part of the rule: each head vs the big model (1 leak-free epoch), the 725-user 732+long pool, the delta in points, and a projection "release (+0.36% / +0.28%) + delta" against the bar. Self-tests: the rule on the reference's own rows gives 0 / 0 (GO); on the SM77 arm rows it reproduces the SM77 decision (+0.000189 / +0.000208, 722 users) and says STOP on instant.

## Changes against the design

- **2,225 set not used.** The design wanted the 2,225 set for the curve head (±0.09 points). That pool adds 1,493 train users of id < 5000 that are not in the gate tier, and is invalid for any model trained on the final tier (`HANDOFF-eval-pools.md`); SM72s and the arm train on the 8,000-user final tier. So the read-out is the 732 set only. Resolution there: the SM77 pair (same recipe, same users) had 95% CIs of ±0.07 points curve and ±0.013 instant; an architecture pair is likely wider (the pruning test (SM15): ±0.19 curve on 459 users, i.e. ~±0.15 on 722). The curve stop bar (0.00020 = 0.067 points) is inside the curve noise; the rule is on point estimates as pre-registered, and the finish decision should weigh the curve CI.
- **Bytes per card + note corrected.** The fork's layout (`rslib/src/rwkv/precision.rs`, `ModuleState::compact`): 14-byte magic + 1 precision byte + u32 layer count; per layer three u32 lengths (12 B), an f16 time shift (128 B), 64 matrix rows of a 4-byte scale + 16 int8 / 8 int4 bytes, and an f16 channel shift (128 B) only where the layer has a channel mixer. It reproduces 4,554 (int8) and 3,018 (int4) for the small model. With 1 card layer: **3,134 B int8 (-31.2%), 2,110 B int4 (-53.7%)** per card + note (card 1,567 / 1,055, note unchanged). The design's 3,146 / 2,122 left out the removed layer's 12 bytes of length fields. With stage 1's int4 card + int8 note format: 1,055 + 1,567 = 2,622 B (-42%).
- No GPU kernel check ticket (unlike the SM77 GPU check): no kernel changes; the trainer and scorer run the same per-layer kernels on one layer fewer. The first GPU ticket is a 2-user trainer smoke of the arm epoch.

## CPU proofs (`scripts/sm73s2.py cpu-check` -> `check-cpu-sm73s2.json`, ALL OK)

- C0 the cut checkpoint: `w64-c1` loads it strictly; the 30 removed tensors are exactly `rwkv_modules.0.blocks.1.*` (19,725 values); the 390 kept tensors are bitwise SM72s epoch 20's; 329 AdamW states moved bitwise (25 dropped; the 5 dummy channel-mixer parameters never had one) and load into `w64-c1`'s optimizer.
- C1 layer steps: 12 = w64's 13 minus (card, 1), same order; depths [1, 4, 1, 3, 3].
- C2 the model builds scripted (TorchScript, the trainer's mode) and eager (the scorer's).
- T1 trainer path (CatchUp loss with the strided-lane state carry, train mode, seeded; users 4745 and 9037 of SM72s's epoch-21 protocol, 8 updates): `w64-c1` from the cut == w64 from SM72s epoch 20 with the layer step (card, 1) skipped: loss, outputs and every kept parameter's gradient bitwise, eager and scripted. Zero-shot (vs the full model) the loss moves by -0.015 to +0.010 per update; in the scorer (2 users, not a measurement) curve LogLoss -0.0024 / +0.0009 and instant +0.0002 / +0.0018: the cut is not a no-op.
- S1 scorer path (`small_model_heldout.py`, CPU, fp32 and w8+s8, leak-free flags, users 2388 and 4676, 4,200 reviews): `w64-c1` == w64 with the card layer's residual branch zeroed (W_o = 0): every metric and every dumped prediction bitwise.
- M1 the Muon fresh-state check passes on a `w64-c1` cool-down layout (Muon / AdamW split 291,712 / 20,074 = w64's 309,888 / 21,623 minus the removed layer's share); `scripts/check_muon_fresh_decay.py` now derives that split instead of hard-coding w64's.

## Fork / runtime (not changed)

The fork's small-model engine is specialized at compile time (`rslib/src/rwkv/architecture.rs`, feature `rwkv-sm40`): `MODULE_LAYERS = [2, 4, 1, 3, 3]` and the hard-coded 13-entry `LAYER_STEPS` would become `[1, 4, 1, 3, 3]` without `(0, 1)`; `channel_mixer_present` stays right ((0, 0) keeps its channel mixer). The weights loader (`RwkvModule::load` per `MODULE_LAYERS`), the export spec test (`sm40_tests.rs`: `n_layers` per stream) and the bench constants follow from those. The state format already allows 1 to 4 layers per entity, but stored 2-layer card states are not valid for a 1-layer model: a model switch must drop and rebuild the card state cache (the usual full rebuild). No state-format change is needed.

## Cost and timing

GPU: smoke ~15 min, epoch ~125 min, cool-down ~150 min, scoring ~12 + 8 min: ~5.2 h if the queue is free. Launched 2026-11-02 morning; see `HANDOFF.md`.
