# SM15 handoff: first pruning step, depth cut of the w64 context streams (2026-10-19)

Branch `prune` (worktree `C:/Users/jonat/workspace/wt/prune`, from `sm21v2`). Design: `docs/SMALL_MODEL_PRUNING.md` (small-model worktree).
Nothing was enqueued or run on the GPU. The launcher is written and NOT started.

## What was built
- `expertium/variants.py`: presets `w64-d{D}p{P}u{U}[-nocmix]` (D 1-4, P 1-3, U 1-3) keep the leading deck / preset / user layers; `w64`
  itself is unchanged (331,511). Asserted counts: d2p2u2 244,232; d2p1u2 and d2p2u1 224,507; d2p1u1 204,782; d2p1u1-nocmix 179,645.
- `scripts/small_model_prune.py`: `depth`, `make-run`, `ft-prepare`, `promote`, `decide` (design 2.3 helper), `screen-report`, `--self-test`.
- `artifacts/small-model/runbook-prune-depth.sh` (stages `screen`, `e1`; `DRYRUN=1` stubs the queue) and
  `artifacts/small-model/launch-prune-after-chain.sh` (not started).
- `scripts/eval_pools.py` and `scripts/small_model_heldout.py` are verbatim copies of the small-model worktree's versions (the `--pool` support).
  Deviation from the brief: that worktree's scorer cannot build the pruned presets (it imports its own `expertium/`), so the 732-pool scoring
  runs the prune worktree's copy. Same code; when `prune` is merged the files are identical, so it merges cleanly.
- The parent experiment is only read. `make-run` writes a synthetic experiment `artifacts/accuracy/experiments/prune-<arch>-sc-lanes-e12/`
  (run.json/experiment.json = parent's with `options.architecture`, `pruned_from`; epoch-12 = pruned master + fresh AdamW + complete result.json).

## Self-test (`artifacts/small-model/check-prune.json`): ALL OK (31 s on CPU)
1. strict loads and parameter counts: all five asserted presets match; `w64` = 331,511.
2. child forward == parent forward with the dropped blocks skipped: max abs diff 0.0 (d2p1u1 and d2p2u2, user 948, 2,803 reviews, fp32, eval).
3. fresh `optimizer.pth` (empty state, 5 param groups) loads; first step bit-equal to a fresh AdamW.
4. `ft-prepare` output has no `lr_schedule`, `CatchUp.loss` sees `decaying` false (KD alpha 0.9, replay), warmup LR then constant 7e-4, 0.5 epoch = 2,425 users / 184.9 M reviews, seed 2.
5. `zero_features == (22,)` and `features2card[0] = Linear(92, 128)` on all five presets.
6. (extra) options carried: the child run and a decay prepared from it keep `state_carry {"lanes": "strided"}`, replay targets, gpu_memory_fraction 0.8;
   2 strided-lane updates of a real user (3358) run forward+backward on CPU with the pruned arch, finite losses and gradients.

## Runbook dry run: `artifacts/small-model/runbook-prune-depth-dryrun.log`, "DRYRUN OK"
Exercises make-run, ft-prepare, the constant-LR checks, the decay prepare with the paired-user assertion against the parent's own SM40v2 decay
(same 5,000 users / 369,477,375 reviews / order / schedule; options differ only in `architecture`), the prefetch-allowlist and carry/replay validators,
and the decision helper (P_dec against itself = PASS, delta 0). The dry run cleans its scratch.

## CPU screen (`artifacts/small-model/prune-screen/report.md`): all 68 feature-screen users, all arms, nothing subset
Zero-shot, P_dec pruned with no fine-tuning, w8+s8, review-weighted, 2.43 M reviews; 2 processes x 2 threads, 85 min. Parent: curve 0.31110, instant 0.28041.

| arm | params | curve delta vs P_dec | instant delta vs P_dec |
|---|---:|---:|---:|
| D9 (d2p2u2) | 244,232 | +0.73% [+0.42, +1.02] | +0.93% [+0.75, +1.18] |
| D7 (d2p1u1) | 204,782 | +4.68% [+3.59, +6.04] | +3.47% [+3.00, +3.99] |
| D7-nocmix | 179,645 | +6.52% [+4.99, +8.61] | +5.42% [+4.74, +6.14] |
| deck cut d2p3u3 | 283,682 | +0.51% [+0.28, +0.84] | +0.21% [+0.17, +0.25] |
| preset cut d4p1u3 | 292,061 | -0.00% [-0.41, +0.24] | +0.25% [+0.19, +0.30] |
| user cut d4p3u1 | 292,061 | +1.30% [+0.92, +1.78] | +0.83% [+0.68, +1.01] |

Share of D7's zero-shot increase: deck 11% / 6%, preset 0% / 7%, user 28% / 24% (curve / instant). None exceeds 60%.

**Chosen ARCH = `w64-d2p1u1` (D7, 204,782 params, -38%, 13 -> 7 layers)** by the pre-registered rule. The single cuts add up to only about 40% of D7's
cost, so the damage is mostly interaction between the cuts (the remaining streams cannot compensate); the zero-shot number is an upper bound of the
post-fine-tune cost, not a verdict. Fallback if D7 fails the rule: D9 (`w64-d2p2u2`, zero-shot only +0.7% / +0.9%).

## GPU plan (e1; owner small-model, purposes "SM15 ...", claim gpu, gpu_memory_fraction 0.8)
Parent-matched: fine-tune P_e (undecayed epoch 12) pruned, 0.5 epoch at 2,048 x 16 (strided lanes carry, replay targets, constant 7e-4 after warmup, KD alpha 0.9),
then the SAME decay as P_dec (1 epoch at 32,768 x 1, lanes carry, replay, 7e-4 -> 0 one_minus_sin, seed 1). Delta vs P_dec = pruning cost + 0.5 epoch extra training.

| ticket | queued min | estimate |
|---|---:|---|
| fine-tune (smoke first), 184.9 M reviews | 60 | parent e12 ran 88,701 reviews/s with 13 layers (69 min/epoch); 7 layers: about 110-130k reviews/s, 25-35 min + 8 min start/smoke |
| decay 1 epoch at 32k (smoke first), 369.5 M reviews | 240 | parent: 24,500 reviews/s, 251 min pure (5.4 h wall) with 13 layers; 7 layers (recurrence-bound, 1.4-1.8x) about 140-190 min + smoke |
| 464 scoring w8+s8 + fp32 (+ resume) | 25 (+15) | about 10-20 min |
| 732-pool scoring w8+s8 (+ resume) | 40 (+20) | about 15-30 min; the 3 longest users (1534, 1601, 4625) OOM'd at the 0.8 cap for the parent: see the stop rules below |
| **Total queued** | **365 (+35)** | **expected 4.2-5.2 h wall, worst case about 7 h** |

Stop rule (design): cancel the decay if it runs below 15k reviews/s (the runbook logs the fine-tune's reviews/s and warns below 60k).
If the pool is short of 732 rows after both passes, rerun just the pool tickets with `HELDOUT_GPU_MEMORY_FRACTION=0.92` (rows resume per user), then `SUMMARY_ONLY=1`.
Outputs: `heldout-prune-w64-d2p1u1.jsonl` (464), `heldout-prune-w64-d2p1u1-pools.jsonl` (732), `prune-decision-w64-d2p1u1.json`
(PASS / EXTEND / FAIL per design 2.3: bootstrap 2,000 draws seed 0; plus vs fork, RMSE(bins), tertiles, int8 cost) and `prune-decision-w64-d2p1u1-pools.json`
(the 732 numbers vs P_dec, reported only; P_dec 732 rows: wt/sm21v2 `heldout-sm40v2-pools.jsonl` + wt/small-model `heldout-pools.jsonl`).
Do not touch scripts/ expertium/ rwkv/ of the prune worktree while tickets run (SourceWatch).

## Launch (waits for "chain done" in the small-model chain log and for HOLD-chain to disappear; checks the self-test report and a clean source tree)
```
nohup bash C:/Users/jonat/workspace/wt/prune/artifacts/small-model/launch-prune-after-chain.sh > C:/Users/jonat/workspace/wt/prune/artifacts/small-model/launch-prune-after-chain.log 2>&1 & disown
```
Direct start (GPU free): `ARCH=w64-d2p1u1 nohup bash C:/Users/jonat/workspace/wt/prune/artifacts/small-model/runbook-prune-depth.sh e1 > C:/Users/jonat/workspace/wt/prune/artifacts/small-model/runbook-prune-depth.log 2>&1 & disown`
Resume = rerun the same command.

## After the run
PASS: E2 = D7-nocmix from the D7 child (zero-shot nocmix costs another +1.8% / +1.9% on top of D7, so it needs its own fine-tune), then export parity
(`small_model_export.py package --arch w64-d2p1u1`) and CPU timing. FAIL: D9 or the screen's D8. See design section 4.
