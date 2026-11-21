# Trained smaller state for Tiny (SM137): design (2026-11-21, pre-registered before any GPU ticket)

## Question

Tiny's weights and CPU cost are already small (the six-layer Tiny (SM130): 185,057 parameters, 1.06 ms per review on one CPU thread). What is left is the state the fork stores for every card and every note. With the release-candidate state format the user chose on 11-21 (int8 rows with float16 row scales and int8 shift vectors for cards and notes, after the long-card drift test (SM136) showed int4 card rows drift on long cards), the six-layer Tiny stores **1,315 B per card and 1,315 B per note**. **Can a trained Tiny keep a clearly smaller card and note state and still stay within the Tiny budget (+1.00% vs Small, 95% CI upper bound, each head), robustly on long cards?**

| plain name (ID) | role here |
|---|---|
| trained smaller state for Tiny (SM137) | this design |
| the six-layer Tiny (SM130) | the starting checkpoints: Instant-Only epoch 41 (master sha256 af16139e28bb) and two-head epoch 32 (635f1a12893b), both constant LR, before their cool-downs; its finals are the same-format comparison |
| the six-layer Tiny with longer recovery (SM135) | its Instant-Only epoch 42 (trained from SM130's epoch 41 without the cut, 0.279924 instant) is the pilot's matched control |
| the training-free state check (SM116) | the evidence that the card and note state is compressible (70% of the values for +0.015% curve / +0.032% instant, no training) |
| the long-card drift test (SM136) | the release-candidate format and the long-card users |
| the longer Small Instant-Only model (SM113) / the show-time two-head cool-down for Small (SM118) | the references of the go rule (0.275933 / curve 0.300016, instant 0.276599) |
| the show-time big model (SM91s2) / the show-time Instant-Only big model (SM99) | the cool-down teachers (unchanged) |
| the big model (1 leak-free epoch) | the constant-LR epochs' teacher (unchanged) |

## What a Tiny card or note stores today

Tiny has one card layer and one note layer, both with their channel mixer. One stored entity at the release-candidate format (`ModuleState::compact`, the layout of `sm136_bytes.py`):

| part | bytes |
|---|---|
| header: 14-byte magic, precision byte, u32 layer count | 19 |
| per layer: three u32 lengths | 12 |
| time-mixer shift vector: float16 scale + 64 int8 | 66 |
| WKV matrix: 4 heads x 16 rows x (float16 scale + 16 int8) | 1,152 |
| channel-mixer shift vector: float16 scale + 64 int8 | 66 |
| **total per card, and the same per note** | **1,315** |

The WKV matrix is 88% of the bytes. A row is one value lane of a head; its 16 entries are the head's 16 key lanes.

## Candidates

| candidate | bytes per card / note | training | fork (Rust) | risk |
|---|---|---|---|---|
| (a) fixed or learnable projection of the stored state onto an SM116-style PCA basis (store coefficients, project back before every step), at SM116's 70% | about 849 / 849 (179 int8 coefficients + one float16 scale per head, 45 + 2 per shift vector) | the projection sits between every two steps of a card's recurrence, so the existing chunked K=16 kernels cannot run it: a new fused kernel (with backward) or the PyTorch step loop, which the state-compression-in-training check (SM32b) measured 3-3.4x slower per update | a new codec plus two dense matrix products per head per stored step (8 heads x 256 x 179 x 2: about 733k multiply-adds per review, several times the model's own) and 380,416 basis values (2.1x the model's 185,057 parameters) | the state compression trained into the final decay (SM32b) was worse than compressing after training; int8 coefficient rounding after every answer is untested on long cards |
| (b) smaller heads in the card and note streams (8 heads of 8, same width), distilled from the six-layer Tiny (SM130) or a bigger model | 803 / 803 | a K=8 recurrence kernel (an NVRTC rewrite like today's K=16 one, plus the exactness checks); the warm start regroups the per-head L2 norms and group norms | `HEADS` and `HEAD_SIZE` are global compile-time constants of the fork engine (about 245 uses in `engine.rs`, `precision.rs`, `query_kernels.rs`, `bulk.rs`): a per-stream head size is a refactor of the time mixer, the state, the codec, the query kernels and the bulk path (several agent-days) | moderate; more engineering than accuracy risk |
| (b2) fewer heads (2 heads of 16, attention width 32) | 739 / 739 | the model code ties the attention width to the stream width; needs a new projection shape | the same refactor plus a second width | as (b) |
| **(c) lane cut (the pick): every card / note head keeps its first V value lanes (state rows) and its first K key lanes (state columns); the model zeroes the other k and v lanes after `W_k` / `W_v`** | **v16k8: 803 / 803**; v16k12 1,059; v12k12 835; v8k16 739; v12k8 643; v8k8 483 | none new: the same K=16 kernels, shapes and speed; the dropped state rows / columns stay exactly 0 (below); warm start = SM130's checkpoint with the lanes permuted by state energy and the dropped `W_k` / `W_v` rows zeroed, then the recipe | **no engine math change**: the trained weights carry exact-zero rows, so the fork's float recurrence keeps the same zeros and reproduces the model bit for bit at 1,315 B. The saving is a codec change only: store the kept columns of the kept rows, restore zeros on read (on top of the compact-format codec the RC already needs) | the cut may cost more than the budget allows; checked by the pilot against a matched control |
| (d) drop the card / note channel mixers | 1,249 (66 B less) | a retrain of the two layers | small | small saving; can stack with (c) later |
| (e) the exact PCA fold of the shift vectors (SM116's fold) | about 1,277 (38 B less) | none | a folded reader matrix per branch | small saving; later |

Distillation: the recipe already distils from the big teachers (the big model (1 leak-free epoch) at constant LR, the show-time big model (SM91s2) / the show-time Instant-Only big model (SM99) in the cool-down). Distilling from the six-layer Tiny (SM130) itself would need an export of its targets for the 8,000 training users and is a weaker teacher; it is not used.

### Why the lane cut is exact

Per head the update is `S' = S diag(w) - (S kd)(a kd)^T + v k^T` with `S` = [value lane, key lane], `k = kd * a` (`expertium/model/rwkv_ops.single_timestep`). If key lane j has `k_j = kd_j = 0` at every step, column j of every term is `S_ij w_j`, so a column that starts at 0 stays 0. If value lane i has `v_i = 0`, row i of every term is a function of row i, so a zero row stays 0. Zeroing a lane before the per-head L2 norm only removes it from the norm. The output `o = S r` and the bonus term `sum_j r_j bonus_j k_j v_i` then never see the dropped lanes; the group norm still runs over the head's 16 output channels (the dropped value lanes contribute zeros), exactly as the fork computes it. Float zeros stay exact zeros under multiply / add / fused multiply-add, so this holds on the CPU and in the CUDA kernels alike. The mask also zeroes the gradient of the dropped `W_k` / `W_v` rows, and their AdamW moments are zeroed at the cut, so the rows stay exactly 0 through training (checked after the epoch).

Lane order: a permutation of key lanes inside a head (rows of `W_k`, `W_r`, the decay and `a` LoRA outputs, the bonus) and of value lanes (rows of `W_v`, the gate output, the group-norm affine, columns of `W_o`) is an exact symmetry of the model up to float summation order (proof L4). Before the cut, `sm137_lanes.py` orders the lanes of every head by the energy (sum of squares) of the stored card / note states over 17 validation users outside the 732 set (SM116's direction users, 446,329 stored states), the data-driven part of the SM116 idea: the top 8 key lanes hold 82-99% of the key-side energy per head, the top 8 value lanes 69-87% of the value-side energy.

### The pick: v16k8 (all 16 value rows, 8 of 16 key lanes per head)

Zero-shot screen (no training; the lane cut of SM130's checkpoints vs the uncut checkpoint, release format w8+s8hx8, the 17 screen users of `users-screen.json`, outside the 732 set, information only; `sm137-zero-shot-screen.json`):

| lanes | bytes per card / note | saving | Instant-Only instant | two-head curve / instant |
|---|---|---|---|---|
| v16k12 | 1,059 | 19.5% | +0.143% | |
| **v16k8** | **803** | **38.9%** | **+0.464%** | **+0.348% / +0.531%** |
| v12k12 | 835 | 36.5% | +0.423% | |
| v8k16 | 739 | 43.8% | +1.122% | |
| v12k8 | 643 | 51.1% | +0.831% | |
| v8k8 | 483 | 63.3% | +1.699% | |

Cutting key lanes is much cheaper than cutting value lanes for the same bytes (v16k8 +0.46% vs v8k16 +1.12%), matching the energy profile. v16k8 saves 39% of the state (card + note 1,606 B instead of 2,630 B) and keeps every value row, so every stored row is still one int8 row with its own float16 scale, the format SM136 validated (8 codes per scale instead of 16, so the rounding is at least as fine). v12k12 costs about the same zero-shot for 32 B more, so v16k8 is preferred. The milder fallback for a second pilot is v16k12 (1,059 B). The bolder v12k8 (643 B) is a later option only if v16k8 is clearly inside the budget.

Budget: the six-layer Tiny (SM130) already uses about half of the +1% (w8+s8: Instant-Only +0.564% [+0.522, +0.607] vs SM113; two-head curve +0.467% [+0.398, +0.546], instant +0.524% [+0.484, +0.567] vs SM118). The cut may therefore cost at most about +0.35% to +0.45% on the final model, less the release-format cost (+0.01% to +0.02% on Tiny in SM136) and the wider CI. For the Instant-Only chain, SM130's cut gap did not shrink in its cool-down (+0.286% after recovery, +0.378% after the cool-down), so the pilot bar is set at the final allowance: +0.30%.

## Fork implementation (Rust) of the pick

- Weights: the export writes the trained weights as usual (the dropped `W_k` rows are exact zeros; `sm137_lanes.py verify` checks the trained master) and adds `state_key_lanes: 8` / `state_value_lanes: 16` for the card and note streams to `architecture.json`.
- Engine: no change. Proof L3 shows the lane architecture and the plain six-layer architecture give bit-identical scorer metrics and predictions on the cut checkpoint (fp32, w8+s8, w8+s8hx8), so the fork's existing six-layer engine, which has no masks, reproduces the student once the weights carry the zero rows. Today's 1,315-B format already stores the student correctly (the zero columns are int8 zeros).
- Codec (the saving): `precision.rs` `encode_matrix` / `decode_matrix` / `row_bytes` write the first K codes of each row per head (and the first V rows; all 16 for v16k8) and restore zeros on read; a precision byte or header field records the lane shape; encode refuses a state with a nonzero dropped column (a model / format mismatch). Same change in the bulk path's `compact` calls. Estimate: about 0.5-1 agent-day on top of the compact-format codec scope of the fork compact state check (SM133) (3-4 agent-days, needed for the release candidate anyway), plus tests: byte sizes (803 B), synthetic parity vectors from `sm137_lanes.py`'s reference encoder (L5), and SM133's synthetic long-card probe (20,000 answers, patterns mixed / again1d / good1d / again10m) with the lane format.
- Optional speed-up (not needed): skip the 8 zero columns in the card / note WKV loops.

## Pre-registered rules (`sm137_rule.py`, fixed now)

All 732-set numbers: show-time inputs, leak-free (neighbour), w8 weights, the 722 rule users of the 732 set (`showtime_report.py pools()`, sha256 pinned), review-weighted LogLoss, paired user bootstrap 2,000, seed 0, 95% CI; "%" = 100 x (new / reference - 1) of the reference's own LogLoss. Reference row files sha256 pinned.

1. **Pilot** (`pilot --lanes v16k8`; Instant-Only lane epoch 42 vs the matched control, the six-layer Tiny with longer recovery (SM135)'s epoch 42, instant head, both w8+s8; point estimate x): non-finite user -> stop; x > +5.00% -> stop (investigate); **x <= +0.30% -> go** (the full plan below with v16k8); otherwise -> a second pilot with v16k12 under the same rule (its x > +0.30% -> stop). Reported: the lane epoch at w8+s8hx8, and the zero-shot cut vs SM130's epoch 41 on all 722 users.
2. **Two-head early check** (`early-th`, after the two-head lane epoch 33): stop if non-finite or any head is above +1.00% vs SM130's epoch 32 (point; no matched control exists, and SM135 showed one more constant-LR epoch does not help, so the offset epoch is no hidden advantage). The two-head cool-down shrank SM130's cut gap by about half, hence the looser bar.
3. **Main rule** (`decide --chain io|th`, after the two-pass cool-down): per trained head, d = 100 x (student at the release format w8+s8hx8 / Small at its published w8+s8 - 1) vs the longer Small Instant-Only model (SM113) or the show-time two-head cool-down for Small (SM118). **Go iff every trained head's 95% CI upper bound <= +1.00%, every rule user finite, and the long-card check passes.** This is a non-inferiority rule against the user's budget (11-18); the repo's 0.05% minimum effect applies to "better than" claims and is not part of it. Reported: the same at w8+s8; vs the six-layer Tiny (SM130) final at w8+s8 and at w8+s8hx8 (rescored in the final ticket).
4. **Long-card check** (`longcard --chain io|th`): the 50 users of `users-longcard.json` (the users of the 732 set with a card of 128 or more prior answers that SM136 scored completely for both Tiny models on the CPU under the 5.9 GB guard), CPU scorer with `--prior-buckets 0,32,128,300,1000`, student and SM130 final both at w8+s8hx8. Per trained head, excess = (LogLoss cost vs SM130 on reviews of cards with 128+ prior answers) - (cost on all reviews), paired user bootstrap. **Pass iff no student user is non-finite and every trained head's excess CI upper bound <= +0.25%.** Why +0.25%: it is SM136's ceiling for a format change, applied to what long cards lose beyond the student's own average cost, so a uniform capacity cost is not counted twice; those cards are about 0.8% of reviews (SM136), so the ceiling guards robustness, not the average. The stored rows are int8 rows with float16 scales exactly as in the format SM136 validated; the new risk is less capacity on long histories, which this measures.

## Pilot (GPU; `launch-sm137.sh`, not launched)

One 16 GB RTX 5080, GPU memory cap 0.8 (the training options' `gpu_memory_fraction` 0.8; SM135's identical epoch peaked at 12.6 GB reserved; scorers `HELDOUT_GPU_MEMORY_FRACTION=0.8`, 0.92 only on a retry pass without progress, as SM135), every step queued through `scripts/r01_queue.py` (owner sm137), segments end before quiet windows and resume from snapshots.

| step | GPU minutes |
|---|---|
| checks: rule selftest, the CPU proofs report for v16k8, `sm137_epoch.py pair` (CPU) | 0 |
| GPU scorer smoke: the cut on 2 users, w8+s8 + w8+s8hx8, vs the CPU rows (|difference| <= 1e-5) | ~5 |
| trainer smoke (2 users x 512 reviews) | ~3 |
| zero-shot score of the cut, 723 users, w8+s8 (report only) | ~10 |
| **Instant-Only lane epoch 42**: SM135's epoch 42 with the architecture `w64-c1-d2p1u1-v16k8` (581,942,724 reviews; SM135's identical epoch: 54.1 min at 179,160 reviews/s) | ~55 |
| dropped-row check of the trained master (CPU), then scoring w8+s8 + w8+s8hx8, 723 users | ~20 |
| rule `pilot` (CPU) | 0 |
| **total** | **about 1.5 GPU h** (about 2.8 h if the epoch ran at the cool-down speed of ~65k reviews/s) |

A second pilot (v16k12, only on the rule's `pilot_v16k12`) costs the same.

## Full plan if the pilot goes (prepared after the pilot; not in this launcher)

The same recipe as the six-layer Tiny (SM130), with one constant-LR lane epoch per chain instead of three (SM135 showed more constant-LR epochs make the six-layer Instant-Only model worse):

| step | reviews | GPU h at the measured six-layer speeds | GPU h at 65k reviews/s throughout |
|---|---|---|---|
| Instant-Only cool-down from the lane epoch 42: SM115's / SM130's two passes (one one_minus_sin decay 7e-4 -> 0 over 2 x 581.9M reviews, fresh Muon, 32k windows, SM99's targets, alpha 0.5) | 2 x 581.9M | 4.9 (65.7-66.0k reviews/s) | 5.0 |
| Instant-Only final scoring: student w8+s8 + w8+s8hx8, SM130 final w8+s8hx8 | | 0.5 | 0.5 |
| Instant-Only long-card check (CPU, 50 users x 2 models, < 6 GB per process) | | 0 (about 1-2 CPU h) | 0 |
| rule `decide --chain io`; the two-head chain runs only if it is go | | | |
| two-head: lane cut of SM130's epoch 32 (CPU), smokes, lane epoch 33 (SM135's two-head epoch-33 step), early check | 581.9M | 1.2 (~180k reviews/s) | 2.7 |
| two-head cool-down: SM124's / SM130's two passes with SM91s2's targets | 2 x 581.9M | 5.0 (64.3-64.5k reviews/s) | 5.0 |
| two-head final scoring + long-card check (CPU) + `decide --chain th` | | 0.5 | 0.5 |
| **full plan total** | | **about 12 GPU h** | **about 14 GPU h** |

With the pilot: about 13.5 GPU h (16-17 h at 65k reviews/s throughout); with a second pilot add 1.5 h.

## Stop rules

- Pilot: the rule above (go / second pilot / stop). A pilot stop ends SM137 with a written result; no cool-down runs.
- Any ticket: STOP on a failed smoke, a GPU vs CPU disagreement above 1e-5, a non-finite loss, a trained master whose dropped rows are not exactly 0, a cool-down pass that ends at the wrong learning rate, or the STOP file; nothing is relaunched blindly.
- Two-head: the early check above; the two-head chain starts only after the Instant-Only chain's go (or an explicit OK from the main session).
- Final: per chain the main rule plus the long-card check. A chain that fails stays experimental; the release choice is the user's.

## CPU proofs (`sm137_lanes.py cpu-check`, done 11-21 before any GPU ticket; `check-cpu-sm137-v16k8.json`)

**ALL OK** (108 s, CPU only, < 3 GB RAM). Checks:

| check | what it shows | result |
|---|---|---|
| L1 default unchanged (eager + TorchScript) | trainer path (CatchUp loss + strided-lane state carry, 2 users, 7 windows) of `w64-c1-d2p1u1` from SM130's epoch 41 in this branch vs the branch's code before the change (sm135 tip c131b3ae, git archive): loss, predictions and every gradient bitwise | PASS |
| L2 lane trainer path (eager + TorchScript) | `w64-c1-d2p1u1-v16k8` from the cut: finite losses; the dropped `W_k` / `W_v` rows get exactly zero gradient, the kept rows do not | PASS (the first run failed: `scripts/state_carry_train.py` has its own copy of the time-mixer forward; it now applies the same mask, as does `scripts/state_carry.py`) |
| L3 fork equivalence | the cut scored as the lane architecture and as the plain six-layer architecture: every metric and every dumped prediction bitwise (2 users, fp32, w8+s8, w8+s8hx8) | PASS |
| L4 permutation | lanes permuted by energy without any cut vs SM130's epoch 41: largest benchmark LogLoss difference 0.0 (2 users, 3 formats) | PASS |
| L5 stored state | 459,409 stored card states and 459,409 note states of the cut (17 screen users): no nonzero value outside the 16 x 8 block; the lane codec round trip is bitwise on 64 real and 40 synthetic states (zero, subnormal, float16-clamped rows), 640 matrix bytes, 803 B per entity | PASS |
| L6 cut checkpoint | sha256 pins, AdamW state reloads, dropped rows and both their moments are 0, 185,057 parameters | PASS |

Also checked: `sm137_epoch.py pair --chain io` (the pilot folder == SM135's trained epoch-42 folder except the architecture, the start files and their sha256, control_dir and notes), `sm137_rule.py selftest`, the long-card tooling on 2 users (`--prior-buckets` rows -> excess with bootstrap), and the launcher dry run (`gpu/launch-sm137-dryrun.log`).

## Files

- `sm137_lanes.py`: users, lane energy, the cut, the trained-master check, bytes and the lane codec, CPU proofs.
- `sm137_epoch.py`: the pilot epoch folder (`prepare`) and its proof against SM135's epoch 42 (`pair`).
- `sm137_rule.py`: the rules above, `smoke` (GPU vs CPU rows), `selftest`, `users` (long-card users).
- `launch-sm137.sh`: the pilot launcher; `memguard.ps1`: the 5.9 GB guard for its CPU steps.
- `users-lanes.json`, `users-screen.json` (17 + 17 validation users outside the 732 set), `users-smoke.json` (2 users), `users-longcard.json` (50 users).
- `sm137-bytes.json` (bytes per lane shape, codec round trip), `sm137-zero-shot-screen.json`.
- Source change (this branch): `expertium/model/rwkv_model.py` (`RWKV7TimeMixer.set_state_lanes`, guarded multiplies), `scripts/state_carry_train.py` and `scripts/state_carry.py` (the same mask in their copies of the forward), `expertium/variants.py` (`state_value_lanes` / `state_key_lanes`, preset suffix `-v{V}k{K}`), `expertium/model/variant_model.py` (installs the lanes on the card and note time mixers). Off = today's model bit for bit (proof L1).
- Not committed: the cut checkpoints `artifacts/small-model/sm137-io-v16k8-e41/` (master sha256 in its `extend.json`) and `work/`.

## Pilot result and full-plan launch (2026-11-21, no rule changed)

- Pilot (01:31-02:58): **go**. Instant-Only lane epoch 42 vs the matched control (SM135's epoch 42), w8+s8, 722 users: instant **+0.065% [+0.054, +0.077]** (0.280106 / 0.279924); at w8+s8hx8 +0.066% [+0.055, +0.078]; the zero-shot cut of SM130's epoch 41 cost +0.389% [+0.364, +0.412] before the epoch. The epoch trained at 163,493 reviews/s (59.3 min; SM135's identical epoch 179,160); the trained dropped `W_k` / `W_v` rows are exactly 0. Readout `sm137-pilot-io-v16k8.md`.
- Full plan: `launch-sm137-full.sh` (the steps and stop rules above, unchanged). Additions that are checks only: each pass 1 is paired against the reference pass 1 (SM115 / SM124) AND against SM130's own pass 1 (`sm137_pair.py`); `sm137_muon_zero.py` proves on the CPU that the cool-down's fresh Muon steps keep the dropped rows exactly 0 (Instant-Only pass 1: PASS, 5 steps); the dropped-row check runs after every pass. `scripts/check_muon_fresh_decay.py` now reads the lane preset as its base preset (same tensors, same Muon / AdamW split; it failed on `-v16k8` before: expected the plain w64 split); on the Instant-Only pass 1 it passes all 7 checks (Muon 174,464 / AdamW 10,593 parameters).
- Two-head cut made and proved before launch: `artifacts/small-model/sm137-th-v16k8-e32` (master sha256 101523b5a8b3, byte-identical to the screen cut), `check-cpu-sm137-th-v16k8.json` ALL OK (L1-L6), CPU smoke rows `cpu/smoke-cpu-th-v16k8-rows.jsonl`, `sm137_epoch.py pair --chain th` OK.
- Long-card check: one fresh CPU process per user (2 at a time, 4 threads each, memory guard 5.9 GB), the student and the SM130 final at w8+s8hx8 in the same process, rows `cpu/longcard/<chain>/u-<user>.jsonl` merged into `cpu/sm137-longcard-rows.jsonl`.
