# State-energy penalty (SM171): design

Written Wed 2026-11-25, before any screen run (only the CPU proofs and one short probe ticket, 001614, ran first; their numbers fix the penalty weights below). Branch `sm171` (from `origin/main`: design, hook, launcher, hand-off), worktree `C:/Users/jonat/workspace/wt/sm171`. Run tree: `C:/Users/jonat/workspace/wt/sm171-run`, a detached worktree at `cea0f931` (`origin/sm148` head, the tree the low-rank int8 states experiment (SM148) scored with and the rank-2 cool-down test on Small (SM149) trained in), **not edited**. Track: small model.

## Question

Our card and note states are compressed after training by low-rank truncation: rank 2 below 16 answers, then rank 4 (T24, the chosen format), or rank 2 always (Q2, `w8+lr2q8x8`). On six-layer Tiny two-head (SM130), Q2 costs +0.080% [+0.065, +0.094] curve / +0.132% [+0.120, +0.145] instant against its own int8 (REF, `w8+s8hx8`), T24 +0.040% / +0.074% (the SM148 rows, 723 users of the 732 set, recomputed by `sm171_report.py selftest`).

**Question: does a short cool-down whose gradient also penalises the share of each card / note state's energy that a rank-2 truncation would drop make rank 2 (Q2) nearly free, i.e. shrink its cost by at least 0.05% against an identical cool-down without the penalty, without making REF worse?**

The rank-2 cool-down test on Small (SM149) trained through the deployed rank-2 codec (straight-through) and removed only about a third of the rank-2 cost (ratio 0.66, "no effect" under its gate). Outside research reports the energy penalty as a pointer only; nothing of its code is used or opened here (the untrusted bundles in the scratchpad are never read or run). The penalty, the hook, the recipe and the rules are ours.

| plain name (ID) | what it is here |
|---|---|
| six-layer Tiny two-head (SM130) | the model of the screen: `w64-c1-d2p1u1`, K = 16, 1 card + 1 note WKV layer (4 heads each), release folder `wt/sm130/artifacts/small-model/sm130-th-dec2ep32k-p2` (`master.pth` sha256 48a5e724...) |
| the low-rank int8 states experiment (SM148) | the codec (`scripts/state_lowrank.py` LowRankCodec), the scorer labels REF / T24 / Q2 and the original model's rows (`artifacts/small-model/sm148/gpu/sm148-rows-732.jsonl` in the run tree) |
| the rank-2 cool-down test on Small (SM149) | the recipe this screen copies (start, subset rule, learning rate, snapshots, launcher pattern) |
| the carried-state trainer (SM21, SM72) | `scripts/state_carry_train.carry_layer`, the release cool-down's strided-lane carry |

**Why this model:** it is the cheapest meaningful setting. Tiny trains at 54.5k reviews/s with the hook installed and the penalty off (64.3k in its release cool-down), it has both heads, and its Q2 cost is the largest of the three sizes in % (+0.13% instant), so a real effect has room to show. The gap Q2 minus T24 is +0.040% curve / +0.058% instant: a shrink of 0.05% brings Q2 to about T24's cost today.

## The penalty

Per stored write of a card / note WKV head state S (K x K, after the RWKV-7 step S' = S diag(w) - (S kd)(a kd)^T + v k^T), the dropped energy share under rank-2 truncation:

d(S) = 1 - (sigma_1^2 + sigma_2^2) / ||S||_F^2, in [0, 1] (0 for a zero state).

P = the mean over the two coded streams (card, note) of the mean of d over every stored write x head of that stream in the update (all windows of the update). Training gradient = grad(loss) + lambda x grad(P). The loss value the trainer logs stays the stock loss (the penalty only enters the gradient), so the logged loss of every arm is the same function of the weights.

**Gradient without a differentiable SVD.** With the top-2 right singular subspace V (K x 2) held fixed, sigma_1^2 + sigma_2^2 = max over orthonormal V of ||S V||^2 (Ky Fan), so at the maximiser dd/dS = (2 / ||S||^2) ((||S V||^2 / ||S||^2) S - S V V^T) (Danskin). No 1 / (sigma_i^2 - sigma_j^2) terms, so ties (common in young states) are harmless.

**V without an eigen solver.** Per write one deflated power step, warm-started from the entity's previous write: q1 = unit(P V[:,0]) with P = (S^T S / tr)^16 by four trace-normalised squarings, then the same on the deflated M2 = (I - q1 q1^T) M (I - q1 q1^T) for q2 (so a small sigma_2 / sigma_1 is resolved to float rounding, not lost to the power's range). Rank below 2: any q2 orthogonal to q1 (d is then exactly 0). A carried start state gets three cold steps from a fixed seeded K x 2 direction. The tracked d is an upper bound of the exact d (it keeps at most the true top-2 energy).

**Placement (no run-tree edit).** `sm171_energy.install` wraps the module function `carry_layer` (looked up at call time by the carry adapter) to mark the card / note block that is running, and replaces `rwkv.model.wkv_state.wkv_state` (imported by `carry_layer` at each call) by a dispatcher. Any other stream, and lambda = 0, call the stock function unchanged (lambda = 0 adds a no-grad diagnostic on every 16th card / note call that touches no tensor of the graph). A card / note block with lambda > 0 calls the stock function for (out, final) unchanged, then `EnergyTap`: identity on `out` in the forward; its backward adds lambda x dP / d(k, v, w, a, kd) through the full per-step recurrence (the plain WKV backward with a zero output gradient and dd/dS' injected at every stored write; the same form as `state_vq_hooks.ste_backward`). The trainer reaches the hook through the existing `state_vq` track option (allow-listed, teacher-blind for the KD cache): `{"recipe": "sm171-energy", "lambda": L}` is installed by `sm171_run.py` instead of the state-VQ codebooks; no card / note mixer gets a `state_vq_slot`, so the carry table keeps its stock precision for every stream.

**Why not the codec itself in the loop:** that is SM149 (straight-through through the deployed codec), already measured. Why not a penalty on the window-end (carried) state only: the stored state that the next review reads is every write, and the carried state is only the last one of a window.

## Proofs (CPU, before the screen)

`sm171_energy.py cpu` -> `check-cpu-sm171.json` (run tree `cea0f931`, clean; hook sha256 388d83e0...): **ALL OK**.

- **P1** the penalty's recurrence equals the stock CPU recurrence (`wkv_with_state`): final states bitwise (zero and carried start, K = 16, 8 rows x 48 steps x 4 heads, synthetic). Tracked d vs the exact SVD d of every stored write: mean 0.1954 vs 0.1950 (zero start), 0.2179 vs 0.2176 (carried); 99th percentile of |difference| 0.0075, maximum 0.044 (random synthetic steps move the state far more per write than real ones); tracked >= exact everywhere.
- **P2** `energy_backward` equals autograd through a differentiable reference loop with the same V: 3.4e-15 / 5.6e-14 relative (float64, zero / carried start, with skipped steps). The Danskin gradient equals autograd through the exact SVD d on 64 generic matrices with a singular-value gap: 3.6e-15 relative.
- **P3** the float32 backward equals the float64 one to 8.5e-7 relative.
- **P4** six-layer Tiny two-head (SM130), the real trainer path on CPU (release weights and Muon state, the release cool-down's options, 256-review windows, users 4745 and 2308, 3 updates each): stock, hook with lambda 0 and hook with lambda 1 from the same start; the first update's loss is bitwise equal in all three arms; lambda 0 equals stock bitwise on every update (loss and every gradient, diagnostics included); lambda 1 changes the card / note gradients (+133% of their absolute sum on the first update), finite; penalty taps on both streams (25 calls), carried windows included; weights finite after the optimizer steps; logged mean d 0.030 to 0.061.

## Probe (GPU ticket 001614, 3 min; fixes the weights)

The decay smoke (2 users x 512 reviews) of a lambda 1 arm, then the first 28 users (2,134,503 reviews, 81 updates) of the screen's own user subset as two short arms (lambda 1 and lambda 0) with a 2.13M-review schedule. Raw: `gpu/probe.out`, `gpu/probe-summary.json`.

- Speed: lambda 1 19,419 reviews/s (1.34 s GPU per update), lambda 0 54,495 reviews/s (0.47 s): the torch step loop costs 2.8x. Peak allocated 4.5 GB (reserved 7.3 GB) with the penalty, 3.5 GB without.
- At the start (release weights, learning rate still warming up): mean dropped share D0 = 0.0547 (card) and 0.0596 (note), about 0.057; review-weighted training loss L0 = 0.758 (both arms equal to 4 decimals, as the hook leaves the forward alone).
- Gradient size: the median total gradient norm with lambda 1 is 1.84 x the one with lambda 0 (only the card / note parameters receive the penalty), i.e. at lambda 1 the penalty gradient is about 1.5 x the loss gradient.

**Weights (fixed now): lambda_hi = 1.0 and lambda_lo = 0.1**, a decade apart: lambda x D0 = 7.5% and 0.75% of L0; the penalty gradient about 1.5 x and 0.15 x the loss gradient. Muon orthogonalises each matrix update, so lambda sets the direction of the card / note updates rather than their size: at 1.0 the card / note directions are mostly the penalty's, at 0.1 mostly the loss's.

## Recipe

All three arms: start = the release model's final `master.pth` AND `optimizer.pth` (the end of its own cool-down, Muon state; start sha256 checked); the release folder's options, loss and targets unchanged (show-time inputs, strided-lane carry, 32,768-review windows, the replay KD targets of the show-time Big model (SM91s2) at alpha 0.5, deleted cards neighbour); plus the one track option `"state_vq": {"recipe": "sm171-energy", "lambda": L}`.

- **Arms:** `sm171-tiny-hi` (lambda 1.0), `sm171-tiny-lo` (lambda 0.1), `sm171-tiny-ctl` (lambda 0: the control, the identical cool-down with the stock arithmetic, so that the effect of extra training is separated). `sm171_prepare.py pair` proves the arms differ only by lambda and that the other options are the release folder's.
- **Data:** whole users of `random.Random(171).shuffle` of the release folder's 8,000 training rows (sorted by user id first) until 2^26 reviews: 944 users, 67,310,673 reviews (about 11.6% of one cool-down pass), the same for all arms; training users only, disjoint from the 732 set.
- **Learning rate:** 0 -> 7e-5 over 64 updates (0.1 x the cool-down peak, as SM149), then one_minus_sin to 0 over the 67.3M reviews; about 2,550 updates.
- **Snapshots** every 100 users; stop / resume through the resumable decay (SM69) machinery (`sm171_run.py stop`, then `run` again; the launcher does it on the STOP file and around quiet windows).
- **Diagnostics:** `<arm>/energy.jsonl` in the run tree, one line per 64 card / note calls: the mean dropped share per stream over those calls (every call for the penalty arms, every 16th for the control).

## Scoring

The scorer of the low-rank int8 states experiment (SM148) in the run tree, unchanged (`scripts/small_model_heldout.py`): `--pool 732 --users-file <sm72s eval-users-732-fit.json> --inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows --prior-buckets 0,1,3,5,9,16,32,128,300,1000`, `PYTORCH_JIT=0`, GPU cap 0.8 (0.92 for a retry pass without progress, memory only). Labels for each arm: REF `w8+s8hx8`, T24 `w8+lr2.16.4q8x8`, Q2 `w8+lr2q8x8`, all on show-time inputs (`@e2s-query`), curve and instant. The original model is not rescored: its SM148 rows are the reference for the report-only lines; an integrity ticket rescores its REF on the 3 SM148 smoke users and must equal those rows to 1e-6.

## Rules (pre-registered; `sm171_report.py`)

Statistic as SM148 / SM149: % = 100 x (sum of the label's LogLoss / sum of the reference's LogLoss - 1), review-weighted, paired user bootstrap, 2,000 draws, numpy `default_rng(0)`, 95% CI, on the users common to all nine labels and the original model's rows (rule users).

- **Q2 cost** of an arm = its Q2 vs its own REF, per head. **Effect** of a penalty arm = its Q2 cost minus the control's Q2 cost, both ratios on the same bootstrap draw.
- **Go for a penalty arm**, on BOTH heads (curve and instant): effect <= -0.05% with the CI upper bound below 0 (the usual minimum effect), AND the arm's REF vs the control's REF has a CI upper bound <= +0.05% (REF not worse by more than 0.05%).
- **Verdict:** GO if at least one penalty arm passes (follow-up below); PARTIAL if an arm passes on one head only (the main session decides); otherwise NO GO.
- **Coverage / finiteness:** at least 720 rule users for every label and no non-finite row, else `pending` (exit 2) and the main session decides.
- **Reported, no verdict:** T24 cost of each arm and its change vs the control; each arm at REF / T24 / Q2 vs the original model at REF; penalty Q2 vs control Q2 and vs control T24 (is Q2 now as cheap as today's T24?); the instant Q2 cost per answer-count bucket and its change vs the control; the dropped-share curves of `energy.jsonl`; training reviews/s and GPU memory.

## Follow-up if GO (not part of this screen)

The passing arm with the smaller REF cost is the candidate. Next, in this order: (1) the strict rules of the low-rank int8 states experiment (SM148) for that model (R2 / R2T vs the original model at int8, long-card users on the CPU, the per-card guard), (2) the same recipe on six-layer Tiny Instant-Only (SM130), then Small two-head (SM118), (3) a fused kernel for the penalty loop (track SE01 in `docs/GPU_OPTIMIZATIONS.md`) before any Small or Big run. If NO GO: the rank-2 format stays out; T24 stays the chosen format.

## Estimates and budget

| step | GPU minutes |
|---|---|
| trainer smoke, three arms | about 2 |
| `sm171-tiny-hi`, `sm171-tiny-lo` (67.3M reviews at 19.4k reviews/s + start-up) | about 61 each |
| `sm171-tiny-ctl` (67.3M at 54.5k) | about 24 |
| integrity ticket | about 2 |
| scoring 9 labels on the 732 set (SM148: Tiny REF 2.4 min, a low-rank label about 10 min) | about 67, plus a retry pass |
| **total** | **about 3.7 h** (within the 6 to 8 GPU hours of the brief) |

Memory: cap 0.8 of 16.3 GB for every ticket; the probe peaked at 4.5 GB allocated (7.3 GB reserved). CPU: the launcher and the report stay far under 6 GB. No 8k / 2k 10-epoch ETA: this is a short cool-down, not a training run.
