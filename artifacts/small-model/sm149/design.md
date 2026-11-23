# Rank-2 card and note states through a short cool-down (SM149): design

Written Mon 2026-11-23, before any training at scale. Branch `sm149` (from `origin/main`, design, proofs and hand-off
only), worktree `C:/Users/jonat/workspace/wt/sm149`. Run tree: branch `sm149-run`, worktree
`C:/Users/jonat/workspace/wt/sm149-run`, cut from `origin/sm148` (cea0f931, the tree the low-rank int8 states experiment
(SM148) scored with) and **not edited**. Track: small model. Status: design; the main session reviews it and launches.

GPU used by this design: short queue tickets only (owner sm149): 001505 micro-benchmark (36 s), 001509 trainer
smoke of both arms (22 s), 001510 to 001512 real-data throughput of both arms on scratch copies of the prepared folders
(3 to 5 min each, deleted after the read-out); 001504 and 001506 to 001508 failed within seconds while the scripts were
fixed (timestep limit of the benchmark, interpreter path, an import side effect, the scripted-model dispatch).

## Question

The low-rank int8 states experiment (SM148) stored every card and note state matrix as rank-r factors after training:
rank 4 (Q4) passes everywhere (+0.005% to +0.02% dataset-wide), rank 2 (Q2) fails everywhere (+0.05% to +0.16%, rising
with the answer count), rank 2 below 16 answers then rank 4 (T24) fails the instant heads of Big and Small (+0.05% to
+0.06%). Rank 2 would save -82.8% of the state bytes against int8 for Big (Q4: -74.4%), -68.2% for Small (Q4: -52.3%),
-66.7% for Tiny (Q4: -51.1%).

**Question: does a short cool-down with the rank-2 codec in the training loop (quantization-aware training, QAT) make
each release model robust enough that rank 2 passes the rules rank 4 passes today?**

Outside research reports, as a pointer only, that a 124-step QAT decay with a straight-through estimator on card rank-2
/ note rank-1 int8 factors cost +0.016% (ahead head) / +0.107% (immediate head) relative to the same decay with QAT off,
on 750 users. Nothing of that code is used or opened here: the codec, the hook, the recipe and the rules are ours.

| plain name (ID) | what it is here |
|---|---|
| the low-rank int8 states experiment (SM148) | the post-training codec (`scripts/state_lowrank.py` LowRankCodec, fused CUDA encode `scripts/state_lowrank_cuda.py`), its scorer labels, rules and rows: the reference of this design |
| the state-VQ QAT of the final decay (SM32b), learned catalogs with carried state (SM32c) | the existing QAT plumbing reused: the mixers' `state_vq_slot` switch, `expertium/model/state_vq_hooks.py` (straight-through WKV backward `ste_backward`, the fused K = 16 backward kernel), the carried-state entry of `scripts/state_carry_train.py` |
| the resumable decay (SM69) | stop / resume of a cool-down from its snapshots (`scripts/expertium_decay.py stop` / `run`), used for breaks |
| Big two-head (SM91s2), Big Instant-Only (SM99) | the Big release models (the fork architecture, K = 32, 3 card + 2 note WKV layers, AdamW) |
| Small two-head (SM118), Small Instant-Only (SM113) | the Small release models (w64, K = 16, 2 card + 1 note layers, Muon, KD replay targets) |
| six-layer Tiny two-head (SM130), six-layer Tiny Instant-Only (SM130) | the Tiny release models (w64-c1-d2p1u1, K = 16, 1 card + 1 note layer) |

## Gradient path

**Chosen: straight-through estimator (STE) through the exact codec of the low-rank int8 states experiment (SM148), at every stored write.**

- **Forward = the deployed codec, bit for bit.** Per card / note review the answer's output uses the float update
  S' = S G + v k^T (the RWKV-7 step); the stored state, i.e. what the entity's next review starts from in this window or
  in the next one through the state carry, is the codec's reconstruction: the rank min(n, 2) truncation of S' through
  the low-rank int8 states experiment (SM148)'s factored solve (orthonormal basis of span(left factors, v), small eigen problems, no K x K SVD), then int8
  factor vectors with a float16 scale each, the product of the decoded factors stored. Token-shift vectors read from an
  earlier review are REF's int8 vectors (`w8+s8hx8`). This is `scripts/state_lowrank.lowrank_loop`, the scorer's own
  loop: proof P1 shows the training forward equals it **bitwise** (K = 16 and 32).
- **Backward = identity through the codec.** The stored state's gradient passes to S' unchanged, i.e. the plain WKV
  backward along the coded trajectory (`state_vq_hooks.ste_backward`; on CUDA with K = 16 the fused backward kernel of the state-VQ QAT (SM32b),
  equal to it to 2.2e-7 relative on the GPU). Proof P2: equal to autograd through a straight-through loop to
  2.6e-7 relative, finite, non-zero, with and without a carried state.
- **Where: every stored write.** In our placement every stored write is exactly the state the entity's next review
  reads (or the state carried to the next window), so "every write" and "only the state read by the next review" are
  the same set; only an entity's very last write in a user history is never read (coding it costs a few steps, no
  effect). Coding only the carried state at window ends would be almost free but trains a different model from the one
  deployed (within a window the next review would read a float state): rejected.
- **Carried windows.** The training carry holds only the stored matrix, not the factors. At a window start the left
  factors are rebuilt from the carried (rank <= 2) matrix by a seeded randomised range finder in float64 (Y = S Omega
  with Omega a fixed K x 2 Gaussian, then the Householder QR basis of Y); only their span matters and it holds col(S)
  (rank(S Omega) = rank(S) for a generic Omega), so the next
  write is the scorer's truncation. Proof P3: a history cut into three carried windows gives the uncut outputs
  bitwise (K = 16 and 32). A carried entity has had at least one write, so its next write has n >= 2 and rank
  min(n, 2) = 2: the update count is not needed from the carry for rank 2 (it would be for rank 4 training).
  The carry table keeps coded streams in float32 already (`CarryTable.fp32_streams`, from learned catalogs with carried state (SM32c)), as the phone stores the
  decoded factors exactly.
- **One pitfall found by the proofs:** the straight-through value must be written `coded + (x - x.detach())` (exactly
  the codec's value), not `x + (coded - x).detach()`: the latter differs by float rounding, and the rank-2 truncation
  is discontinuous at singular-value ties, so the trajectories drift apart (P2 failed on K = 32 with the second form:
  0.023 forward difference). The hook itself stores the codec output directly.

**Alternatives considered.**
- *Differentiable truncation* (gradients through the eigen / singular vectors): the derivative has 1 / (sigma_i^2 -
  sigma_j^2) terms that blow up at ties, which are common (young states, sigma_2 close to sigma_3), and the int8 factor
  rounding needs an STE anyway. Not used.
- *Projected STE* (the stored gradient projected on the tangent space of rank-2 matrices at the stored point): a
  follow-up only if the identity STE does not close enough of the gap.
- *The vendored per-matrix-scale low-rank QAT helper of the old upstream port* (`fake_lowrank_state`, never used here):
  a different codec (one scale per factor matrix, full K x K SVD per step, no young-state rank, no carry) and the
  inexact STE form above. Not used: train = score = deploy needs the low-rank int8 states experiment (SM148)'s codec.

**Training-time cost (measured on the GPU, the real trainer).** The coded layers run a per-step loop (one fused
encode kernel per step plus the float step) instead of the WKV kernel; deck, preset and user streams keep their
kernels. Real-data throughput of Small two-head (SM118) on its prepared folders (tickets 001510 and 001512, the first
80 updates paired: same users, same windows, `gpu/timing-sm149.json`): control 29,369 reviews/s (0.97 s per 28.4k-review
update, the release cool-down's speed), QAT 9,344 reviews/s (3.19 s per update): **3.08 x the control's GPU time**, about
+7.1e-5 s per review over 3 coded layers. The QAT arm's training loss over those 80 updates is +0.15% above the
control's (0.68571 vs 0.68467), the codec's cost before any adaptation. The micro-benchmark on the split shapes of 12
sampled 32k updates (`gpu/bench-sm149.json`) gives extra seconds per update of about +0.45 (Tiny, 1 card + 1 note
layer), +0.68 (Small, 2 + 1) and +1.0 (Big, 3 + 2 layers at K = 32 with the torch backward, per 16k update); real
updates have longer entity sequences than that synthetic sample, so the real overhead (+2.2 s per Small update) is about
3 x the benchmark's. The loop is host-bound (GPU utilisation 19% to 79% during the QAT arm): a sync-free step (LR01 in
`docs/GPU_OPTIMIZATIONS.md`) is the first speed-up to try, not needed for stage 1.

## Code (no source change in the run tree)

- `sm149_qat.py`: the hook (`TrainCodec`, `LowRankSTEWKV`, `carried_left`, `install`), the CPU proofs (`cpu`) and the GPU
  micro-benchmark (`bench`). `install` sets the card / note mixers' `state_vq_slot` (the state-VQ QAT (SM32b) switch) and replaces the
  three autograd class names of `state_vq_hooks` with dispatching shims: the hook functions and the scripted mixers
  look them up at call time, so the scripted trainer model reaches the hook (the first smoke showed that replacing the
  hook functions themselves misses the scripted path). Any other codec goes to the original classes unchanged.
- `sm149_run.py`: the trainer entry = the run tree's `scripts/expertium_decay.py` with one difference: a `"state_vq"`
  track option whose recipe is a low-rank label (`{"recipe": "lr2q8x8"}`) is installed by the hook instead of the
  state-VQ codebooks. The option key `state_vq` is already in the trainer's option allow-list and teacher-blind for
  the KD cache, so nothing else of the trainer changes. The trainer's source watch hashes the unedited run tree.
- `sm149_prepare.py`: the two arms of one release model (`arm`), and `pair` (the arms differ only by the option; the
  control's options are the release folder's own).
- A port into the source (the class dispatch in `state_vq_hooks.py`, a low-rank branch in `state_vq.install_training`,
  the hook in `state_lowrank.py`) is only needed if the format ships with a QAT model; the run itself does not need it.

## Recipe

**Both arms per model:** start = the release model's final `master.pth` AND `optimizer.pth` (the end of its own
cool-down: Muon state for Small and Tiny, AdamW for Big; sha256 of the start checked against the low-rank int8 states experiment (SM148)'s `models.sh`);
the release folder's options, loss and targets unchanged (show-time inputs, strided-lane carry, the same window sizes,
for Small and Tiny the same KD replay targets and alpha 0.5 of the cool-down, curve loss weight 0 for the Instant-Only
models); deleted cards neighbour (`RWKV_DELETED_CARDS=neighbour`, as the release cool-downs).

- **QAT arm** `sm149-<model>-qat`: plus `"state_vq": {"recipe": "lr2q8x8"}` (rank 2 on every card and note write, int8
  factors, REF's int8 shift vectors).
- **Control arm** `sm149-<model>-ctl`: the identical folder without it. The control matters: re-warming and decaying
  the learning rate on new data moves LogLoss by itself.
- **Data:** whole users of `random.Random(149).shuffle` of the release folder's own training rows (sorted by user id
  first), until the review budget R: Small and Tiny R = 2^26 = 67.1M reviews of the 8,000-user final tier (Small
  two-head (SM118): 794 users, 67,256,032 reviews, about 11.6% of one cool-down pass); Big R = 2^25 = 33.6M reviews of
  its 5,001 training users (users 5000-10000). The same users for both arms and for both models of a size; all are
  training users, disjoint from the 732 set and the long-card users.
- **Learning rate:** 0 -> peak over 64 updates (update offset 0), then one_minus_sin to 0 over the R reviews. Peak =
  0.1 x the model's own cool-down peak: 7e-5 for Small and Tiny (Muon via its lr_scale), 3.5e-5 for Big (AdamW). At
  0.1 x peak the release cool-down was at about 71% of its 2R schedule, i.e. still moving; a short decay from there
  lets the card and note writes adapt to the codec and ends at a learning rate of 0 again.
- **Steps:** about 2,400 updates per model (Small two-head: 67.3M reviews at 28.4k reviews per update measured =
  about 2,370 updates; Big at about 14.8k reviews per update = about 2,270). Outside research's 124 steps are a pointer
  for "short"; our updates are one user each (noisier), so we take about 20 times more.
- **Snapshots** every 100 users; stop / resume through the resumable decay (SM69) machinery (`sm149_run.py stop`, then
  `run` again).
- **Order:** stage 1 = Small two-head (SM118), both arms. It is the cheapest decisive model: the strict +0.05% rule
  applies (the Tiny models are judged against Small with a +1.00% budget, so they cannot decide alone), it has both
  heads, and its rank-2 cost after training is the smallest (+0.047% [+0.038, +0.056] curve, +0.079% [+0.072, +0.087]
  instant): QAT needs to remove about half of the instant cost.

## Scoring

- **Scorer:** the scorer of the low-rank int8 states experiment (SM148) of the run tree (unchanged, `scripts/small_model_heldout.py`), the same flags:
  `--pool 732 --users-file <sm72s eval-users-732-fit.json> --inputs e2s-query --split-existing --deleted-cards neighbour
  --nonfinite-rows --prior-buckets 0,1,3,5,9,16,32,128,300,1000`, `PYTORCH_JIT=0`, GPU cap 0.8 (0.92 for a retry pass
  without progress, memory only); the long-card users on the CPU (`--device cpu --workers 1`, `users-longcard.json`,
  `--dump-scope all`, one fresh process per user and size, memory guard 5.9 GB). Model specs as the low-rank int8 states experiment (SM148)'s `models.sh`
  (`fork:` for Big, `expertium:w64` for Small, `expertium:w64-c1-d2p1u1` for Tiny) with the new `master.pth` files.
- **Labels:** QAT arm: REF (`w8+s8hx8`), Q2 (`w8+lr2q8x8`), T24 (`w8+lr2.16.4q8x8`); control arm: REF and Q2. The
  original model's rows are the low-rank int8 states experiment (SM148)'s committed rows (`artifacts/small-model/sm148/gpu/sm148-rows-732.jsonl`,
  `cpu/sm148-longcard-rows.jsonl` on `sm148`), not rescored: same scorer commit; the first scoring ticket re-scores
  REF of the original model on the 3 smoke users and must equal the low-rank int8 states experiment (SM148)'s rows to 1e-6 (as the low-rank int8 states experiment (SM148) checked the int3 young tier (SM145)'s).
- **Per-review read-out:** the low-rank int8 states experiment (SM148)'s `sm148_dr.py` on the dumps (new Q2 vs original REF; also new Q2 vs new REF).

## Pre-registered rules (the low-rank int8 states experiment (SM148) rules, the reference changed to the original model)

Statistic as the low-rank int8 states experiment (SM148): % = 100 x (sum of the label's LogLoss / sum of the reference's LogLoss - 1), review-weighted;
paired user bootstrap, 2,000 draws, seed 0, 95% CI (`showtime_report.compare`). Heads: curve and instant for the
two-head models, instant for the Instant-Only models. "Original" = the release model scored by the low-rank int8 states experiment (SM148); "new" = the
QAT arm's final model.

- **R1 finite:** no non-finite row and no non-finite LogLoss for any label (732 set on the GPU, long-card users on the
  CPU).
- **R2 dataset-wide:** new model at the label vs the ORIGINAL model at int8 (REF), 732 set rule users, CI upper bound
  <= **+0.05%** per head. Reported next to it, no verdict: vs the control arm at int8, vs the new model's own int8.
- **R2T (six-layer Tiny models only, the user's Tiny rule):** new Tiny at the label vs the matching original Small
  model at int8 (Small two-head (SM118) for Tiny two-head, Small Instant-Only (SM113) for Tiny Instant-Only), CI upper
  bound <= **+1.00%** per head; it replaces R2 for Tiny (the strict R2 verdict is reported too).
- **R3 long cards:** on the long-card users, reviews whose card had 128 or more earlier answers, new at the label vs
  the original REF: point <= **+0.5%**, CI upper <= **+1.0%** per head.
- **G per-card guard:** share of long-card users' cards whose worst review has |dR| > 0.10 (dR = p(new, label) -
  p(original, REF)) <= **1%** per head.
- **R6 the new model at int8:** new REF vs original REF, CI upper bound <= **+0.05%** per head (R2's statistic). The
  new model replaces the original only if R6 passes; otherwise no go for that model whatever R2 says.
- **R5 coverage:** as the low-rank int8 states experiment (SM148), with its known shortfall accepted in advance: on the GPU at least 720 of the 722 rule
  users per label for Small and Tiny, and the same 718 users for Big that the low-rank int8 states experiment (SM148) covered (the five users with 424k to
  532k reviews do not fit the scorer on this GPU); on the CPU at least 48 of 50 long-card users for Small and Tiny and
  the same 37 for Big (13 Big users exceed 6 GB per CPU process). Pairs use only users both sides have. Otherwise the
  report is `pending` (exit 2) and the main session decides.
- **Go per size:** a label passes for a model when R1, R2 (R2T for Tiny), R3, G and R6 pass; for a size when it passes
  for both of its models. The size takes the passing label with the fewest bytes (Q2 < T24); the QAT models then
  replace the originals for that size. If no rank-2 label passes, the size keeps the original models at rank 4 (Q4,
  the low-rank int8 states experiment (SM148)'s go), and the QAT models are not used.

**Reported, no verdict:** control REF vs original REF (the decay alone); control Q2 vs control REF (rank-2 cost
without QAT after the same decay); new Q2 vs control Q2 (the QAT effect at rank 2); new Q2 vs new REF; the cost per
answer-count bucket (1-2, ..., 128+); the per-review |dR| tables; training reviews/s, GPU memory; bytes unchanged from
the low-rank int8 states experiment (SM148)'s `sm148-bytes.json`.

## Stage gates (fixed now)

1. **Stage 1, Small two-head (SM118), both arms.** After scoring:
   - **pass** (Q2 or T24 passes for this model) -> stage 2: the other five models with the same recipe.
   - **partial**: no rank-2 label passes, but QAT at least halved the rank-2 cost on the instant head: C_qat <= 0.5 x
     C_ctl, with C_qat = new Q2 vs new REF and C_ctl = control Q2 vs control REF (point estimates, %) -> stage 1b:
     Small two-head once more with 4 x R (2^28 reviews, about 3,200 users, the same peak, a new subset seed 1149),
     with the sync-free step (LR01 in `docs/GPU_OPTIMIZATIONS.md`) first if it is ready, before any other model; then
     the same three-way read. A second partial stops too.
   - **no effect** (C_qat > 0.5 x C_ctl) -> stop: every size keeps Q4.
2. **Stage 2, by expected cost and value:** Small Instant-Only (SM113) (completes the Small size), then Big two-head
   (SM91s2) and Big Instant-Only (SM99) (largest bytes prize), then the two six-layer Tiny models (SM130). A size whose
   first model fails every rank-2 label does not run its second model.

## Estimates

GPU time per model (training of both arms plus scoring; cap 0.8, one ticket each; reviews/s of the control from the
release runs, of the QAT arm from the measured overhead):

| model | QAT arm | control arm | scoring (732 set) | total GPU | CPU (long-card users) |
|---|---|---|---|---|---|
| Small two-head (SM118), stage 1 | 1.95 h (measured ratio 3.08 x 38 min) | 0.7 h | 0.9 h | **3.6 h** (+ 0.1 h smoke and integrity) | about 2.5 h |
| Small Instant-Only (SM113) | 2.2 h | 1.0 h (its cool-down ran at 20.2k reviews/s) | 0.9 h | 4.2 h | about 2.5 h |
| six-layer Tiny two-head (SM130) | 1.2 h | 0.35 h | 0.6 h | 2.2 h | about 1.5 h |
| six-layer Tiny Instant-Only (SM130) | 1.2 h | 0.35 h | 0.6 h | 2.2 h | about 1.5 h |
| Big two-head (SM91s2) | 3.5 h (range 2.8 to 4.9 h) | 0.9 h | 1.9 h | 6.4 h (+ 0.1 h memory check) | about 6 h (37 users) |
| Big Instant-Only (SM99) | 3.5 h (range 2.8 to 4.9 h) | 0.9 h | 1.9 h | 6.4 h | about 6 h |
| **all six** | | | | **about 25 GPU h** (range about 21 to 30 h) | about 20 CPU h, alongside |

How the rows are built: control arm = R / the release run's own reviews/s (Small two-head 67.3M / 29.4k, Small
Instant-Only 67.1M / 20.2k, Tiny 67.1M / 64.3k, Big 33.6M / 11.4k) plus about 3 min of start-up; QAT arm = control +
the measured +7.1e-5 s per review for Small's 3 coded K = 16 layers, scaled per coded layer (Tiny 2 layers), and for
Big by the benchmark's K = 32 / K = 16 cost ratio (1.84) x 5 layers x 1.0 to 2.0 for its 8,192-review windows (more
windows, shorter sequences per window); scoring = the per-label minutes of the low-rank int8 states experiment (SM148)
on 723 users (Small REF 5.5 min, a low-rank label 14 min; Tiny 2.4 / 10; Big 10.5 / 27) x the 5 labels of the two arms,
plus Big retries. Stage 1b (4 x R on Small two-head), if reached: about 11 GPU h at today's speed.

No 8k / 2k 10-epoch ETA: this is a short cool-down, not a training run.

## Memory

- Cap 0.8 of 16.3 GB (13.0 GB) for every new ticket (`gpu_memory_fraction` 0.8 in the release options, kept).
- QAT keeps one stored state per coded step and row for the backward ([B, T, H, K, K] float32 per coded layer).
  Small two-head (SM118) measured: 12.0 GB in use on the card during the QAT arm (nvidia-smi, of which about 0.7 GB
  is the desktop; the release cool-down reserved 10.8 GB), i.e. it fits under the cap. Big is the risk: its release epochs already peaked at 11.1 GB allocated
  of 13.0 GB reserved, and its records are 4 x larger per step (K = 32) over 5 coded layers at half the reviews per
  update, about +2.5 to +3.5 GB by the state-VQ QAT (SM32b) scaling: over the cap. Before a Big QAT launch: a 3-minute throughput
  ticket on Big two-head (SM91s2) as done here for Small; if the peak exceeds 12.5 GB, the record keeps only the
  written steps (31% of the row steps in the state-VQ QAT (SM32b) sample, the backward gathering each row's last write) or the
  window becomes 4,096 x 2 lanes for BOTH Big arms; either change gets a CPU proof (gradients equal to the full record)
  first.
- Scoring memory is the low-rank int8 states experiment (SM148)'s: the five largest users do not fit the GPU scorer for Big (13.0 GB of 14.6 GB at cap
  0.92), and 13 Big long-card users exceed 6 GB per CPU process; R5 accepts the same coverage as the low-rank int8 states experiment (SM148).
- Batched float64 eigen problems on the GPU over a window's carried rows ran out of memory twice in the throughput
  tickets (cusolver workspace 9.8 GB for 16 x 16, 8.4 GB for 4 x 4 matrices); `carried_left` now needs one batched QR
  of the K x 2 range-finder matrix (chunks of 16,384 matrices), no eigen problem.

## Proofs (before writing this design)

CPU (`sm149_qat.py cpu` -> `check-cpu-sm149.json`, log `cpu-proof.out`, code tree `wt/sm148` at cea0f931, clean;
about 70 s, 4 threads): **ALL OK**.

- **P1** training forward == the low-rank int8 states experiment (SM148)'s scorer loop (`lowrank_loop`) **bitwise**, K = 16 and 32 (synthetic, 6 rows x 40
  steps x 4 heads); the codec is active (outputs move by up to 2.0 against a float full-rank state) and every final
  state has rank <= 2; the hook's codec (REF's bits set directly) equals the scorer's `parse_variant("w8+lr2q8x8")`.
- **P2** STE backward == autograd through a straight-through reference loop: max relative gradient difference 1.9e-7
  to 2.6e-7, forward and final state bitwise, finite, non-zero; K = 16 and 32, zero start and carried start.
- **P3** carried windows == one window: outputs and the final state bitwise equal (K = 16 and 32, three windows).
- **P4** Small two-head (SM118), real trainer path on CPU: its final weights and Muon state, the release cool-down's
  options (strided-lane carry, show-time inputs, the replay KD targets of Big two-head (SM91s2), alpha 0.5), users 4745
  and 2308 (636 and 939 reviews) in 256-review windows, 3 updates per user and arm at LR 7e-5: every loss finite, all
  79 card / note gradient tensors finite and non-zero, weights finite after the optimizer steps, the QAT loss differs
  from the control (first update +0.00054), 38 hook calls of which 26 from a carried state.
- **P4b** the same for Big two-head (SM91s2) (fork architecture in training, K = 32, AdamW, users 9037 and 6013): 144
  card / note gradient tensors finite and non-zero, 58 hook calls, 33 carried, first-update loss -0.00094 vs control.
- **P5** hook installed then removed == never hooked: loss and every gradient sha256 equal (one update).
- **P6** (report) CPU seconds per split of the QAT loop vs the plain reference loop.

GPU (queue tickets, owner sm149):
- 001505 `sm149_qat.py bench` (36 s): CUDA forward vs CPU forward max 0.033 / 0.021 on synthetic data (int8 factor
  code flips between devices, the low-rank int8 states experiment (SM148)'s discrete-codec rule; not a training concern), the K = 16 backward kernel vs the
  torch STE backward 2.2e-7 relative; per-update costs above; peak 2.8 GB.
- 001509 `expertium_decay.py smoke` of both prepared arms through `sm149_run.py` (2 users x 512 reviews each): both
  `complete`; QAT 496 reviews/s, control 1,075 (smoke scale).
- 001510 to 001512: real-data throughput of both arms (above). The QAT attempts of 001510 and 001511 stopped after 67
  hook calls with GPU out-of-memory errors in the rebuild of the carried factors (a float64 16 x 16 eigen problem over a
  window's carried rows, then a 4 x 4 one: cusolver asked for 9.8 GB and 8.4 GB); `carried_left` now uses one batched
  QR (P3 still bitwise), and 001512 ran 80 updates (961 hook calls, 931 from a carried state) until its snapshot stop.

## Files

- `design.md` (this file), `HANDOFF.md` (exact commands), `sm149_qat.py`, `sm149_run.py`, `sm149_prepare.py`.
- `check-cpu-sm149.json`, `cpu-proof.out`; `gpu/bench-sm149.json`, `gpu/bench.out`, `gpu/smoke-sm118.out`,
  `gpu/timing.out`, `gpu/timing-qat.out`, `gpu/timing-ticket.sh`, `gpu/timing-sm149.json`.
- Prepared (not committed, in the run tree): `wt/sm149-run/artifacts/small-model/sm149-sm118-qat`, `-ctl`.
