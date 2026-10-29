# Tiny architecture additions (SM77): stage 1 design, 2026-10-29

Goal: make the leak-free small model (SM72s) more accurate with two cheap additions from the literature review (SM60,
`artifacts/small-model/sm60/LITERATURE.md`, items 1 and 2), without more card / note state bytes. The buy-back target is
the one of the better teacher (SM65) and data augmentation (SM67): the smaller model (SM73) needs about 0.1-0.3 points
back before a narrower network can fit the bar (+0.5% curve / +0.4% instant vs the big model (1 leak-free epoch), 732
set, 722 users).

**Bottom line.** Both additions are implemented behind one preset, `w64-sm77`, on branch `sm77` (cut from `sm72s`),
zero-initialised. With the flags off the trainer and the scorer are bit-identical to the `sm72s` tree; with the flags on
and the new parameters at zero, the loss and every output equal the SM72s checkpoint's bit for bit (CPU, trainer and
scorer paths). (A) needs no kernel change; (B) needs one: the WKV backward kernel must also return the gradient of the
initial state (a 2-line text rewrite of the production and row-tile kernels, plus an initial state for the scorer's
long kernel). Its GPU check is the launcher's first ticket. Stage 1 = one two-arm warm start from SM72s epoch 20 (or 25),
one 8k constant-LR epoch + a quarter-epoch 32k cool-down per arm, **~7.0 GPU h from epoch 20** (the reference epoch is
SM72s's own epoch 21, free), ~9.0 h from epoch 25. Not launched.

## 1. What changes

| | (A) key extras | (B) learned initial state |
|---|---|---|
| Where | every time mixer (13 layers) | card (2 layers) and note (1 layer) streams |
| Today | removal key `kd = k_scale * normalize(k)`, insertion key `kd * a`, `r = W_r x` | a never-seen card / note starts from WKV state 0; its first review's token shift reads its own input `x` |
| New | removal `k_scale * normalize(k * (1 + k_remove))`; insertion `k_scale * normalize(k) * (a + k_insert * (1 - a)) + k_bias`; `r = W_r x + r_bias` | WKV state `S0` [4 x 16 x 16] per layer; first-review shift `x + shift0` (time mixer and channel mixer) |
| From | RWKV-7 paper (separate removal / insertion keys; `k_insert = 1` = "write the whole key"), Mamba-3 (learnable B / C biases) | initial-state tuning for state-space models |
| Values | 4 x 64 per layer = **3,328** | 1,024 + 64 (+ 64 channel mixer) per layer = **3,392** (card layer 1 has no channel mixer) |
| Zero init | `k * 1`, `a + 0`, `+ 0`: bit-identical outputs | `S0 = 0`, `x + 0`: bit-identical outputs |
| State bytes | 0 | 0 (S0 and the offsets are weights) |
| CPU per review | one more per-head normalize and ~4 vector ops per layer | 0 (only a never-seen entity's first step reads S0) |

Parameters 331,511 -> 338,231 (+2.0%). Scorer's int8-weight size 410,459 -> 437,339 B (+26,880 B: the new tensors are
1-D / 3-D, kept in float32 by the w8 scheme). Removal keys stay normalized, so the transition keeps its eigenvalue range;
the key bias adds a constant write, so the int4 / int8 state ranges need the usual check (SM73 format table) if it ships.

Training detail for (B): with the strided-lane carry (2,048 x 16), a lane starts from zero state, also for entities that
have history in earlier lanes. S0 is used only for an entity whose **first review of the user** is in the window (the
window's lane is the entity's first lane and it has no carried state), so S0 learns from real first reviews, as the
scorer and the phone use it; the other lane-start entities keep today's zero start.

## 2. Kernel and runtime paths

| Path | (A) | (B) |
|---|---|---|
| Training forward (CUDA, exact row tiles) | elementwise before the kernel | the SM21 forward-with-state kernel (exists) |
| Training backward | none | **new**: the backward kernel also writes dL/dS0 (the register that already carries the state gradient to step 0 is stored at the end). Text rewrite of the production kernel (`rwkv/model/nvrtc.py` STATE_GRAD_REWRITES) and of the exact row-tile kernel (`scripts/gpu_row_tiles.py`), compiled into separate modules; stock modules unchanged |
| Scoring (CUDA, any length, int8 state rounding) | none | **new**: `scripts/small_model_wkv.wkv_long(..., state0)`, a separate kernel `wkv_q_s0` (state loaded from S0) |
| CPU (checks, scorer) | none | per-step references from S0 (differentiable in training) |

GPU check (launcher ticket 0, ~15 min): dS0 and all other gradients vs CPU autograd (generic and row tiles, float32 and
bf16); at S0 = 0 the new path equals the SM21 path bit for bit; the long kernel from S0 vs its reference; and the trainer
path (bf16, row tiles, carry) at zero init equals `w64` on 2 users x 512 reviews.

## 3. CPU checks (done 10-29, `artifacts/small-model/sm77/check-*.json` on branch sm77)

| Check | What | Result |
|---|---|---|
| T0 | flags off, trainer path (carry, 2 lanes x 128), 2 users, every update: loss, outputs, all gradients vs the `sm72s` tree | bitwise equal |
| S0 | flags off, full-history scorer (fp32 and w8+s8, leak-free inputs), 2 users of the 732 set (3,685 / 3,716 reviews): metrics and per-review predictions vs the `sm72s` tree | bitwise equal |
| T1 / T1j | flags on, zero init vs SM72s epoch 20, trainer path (carry, 2 lanes x 128), 2 users (636 / 1,231 reviews, 8 updates), eager and TorchScript | loss and outputs bitwise equal; old-weight gradients equal up to summation order (max 2.8e-6 of each tensor's max: the removal and insertion keys are two graph branches); every new tensor gets a gradient |
| S1 | flags on, zero init, scorer, fp32 and w8+s8 | metrics and per-review predictions bitwise equal |
| D | non-zero new parameters (normal 0.05): training carry (2 / 3 lanes, fp32 table, seeded lane starts) vs the full-history (scorer) forward, `check_state_carry_lanes.py --architecture w64-sm77` | every row within 3.4e-5 (bar 1e-4): training and scoring compute the same (A) and (B) |
| carry check, zero init | the same check at zero init | L, D, F, G pass; B (first update = the plain forward, bitwise) differs only in the 8 init-state tensors' gradients, by design (lane-start entities with history in an earlier lane do not use S0 in training); loss and all other gradients bitwise equal |
| protocol | the epoch-(K+1) protocol rebuilt from epoch K (order rule, accuracy-loop fields, trims) vs disk, epochs 2, 11, 21, 22 | equal |
| optimizer | SM72s epoch-20 AdamW state re-indexed to the new parameter order | 354 states moved, equal after load; the new parameters start with fresh moments |

## 4. What the fork runtime would need (`anki/rslib/src/rwkv`, read 10-29, not edited)

- (A): in `TimeMixer::mix_parts` (and every duplicate of it: the query kernels, bulk replay, fused packed paths), four
  optional 64-vectors per layer read by name (`...time_mixer.sm77_r_bias`, `sm77_k_bias`, `sm77_k_remove`,
  `sm77_k_insert`); `k_deformed` from a second per-head normalize of `k * (1 + k_remove)`; the insertion key as above;
  `r += r_bias`. Absent tensors = zeros = today's model, so old weight files keep working.
- (B): in the `state: None` branch of the time mixer (`x_shift = x`, matrix zeros) and of the channel mixer: `x_shift =
  x + shift0`, matrix = `S0` (f32 from the weight file, 3 x 4 KB), in the per-review, query and bulk-replay paths.
  Storage formats are unchanged: the stored state after the first review is packed (int4 / int8) as today.
- Export: `scripts/small_model_export.py` writes every state_dict tensor (the new ones as f32), but
  `architecture.json` and the NumPy reference (`scripts/small_model_numpy.py`) need the two additions before `verify`
  can pass. Not done in stage 1: only if it goes.

## 5. Stage 1: one two-arm warm start

Start: SM72s epoch K (launcher parameter `START_EPOCH`, default 20; 25 exists from Fri ~06:00).

| Step | Arm (`w64-sm77`) | Reference (`w64`) | GPU |
|---|---|---|---:|
| 0 | GPU check + trainer smoke of the arm | | ~0.25 h |
| 1 | epoch K+1 at constant LR 7e-4 from the extended epoch-K checkpoint and AdamW state: SM72s's epoch-(K+1) protocol (same 8,000 users, order, seeds, options, carry) | SM72s's own epoch K+1 (free for K <= 24); trained here for K = 25 | ~2.0 h (+2.0 h for K = 25) |
| 2 | quarter-epoch 32k Muon cool-down (SM72s's cool-down options, fraction 0.25 = ~145M reviews, the same seed-1 users for both, fresh Muon, 7e-4 -> 0) | the same | 2 x ~2.1 h |
| 3 | 732 scoring: 723 fit users + 3 long users at the 0.92 cap, leak-free, w8+s8 | the same | 2 x 0.3 h |
| | **total** | | **~7.0 h** (K = 20), ~9.0 h (K = 25) |

Why K = 20 by default: the reference epoch is then free and identical by construction (the arm's protocol is SM72s's
epoch-21 protocol rebuilt, checked equal on disk), which saves 2 h. A quarter cool-down (about 2 h) is the yardstick of
the literature review; a full 1-epoch cool-down (~8.5 h per arm) is kept for a go.

**Decision rule (pre-registered, `scripts/sm77_rule.py decide`).** 722 users (the 732 fit users in the big model's rows),
arm minus reference, review-weighted, in points of % of the big model (1 leak-free epoch) on the same users (0.20 points
~ 0.00060 curve, 0.15 ~ 0.00041 instant), paired user bootstrap 2,000, seed 0, 95% CI:
- **Go** if curve <= -0.20 or instant <= -0.15 points, that head's CI clear of zero, the other head's point <= +0.03.
- **Stop** if both heads gain less than 0.10 points.
- **Grey** in between: report and ask.
The rule also reports the 725-user 732+long pool, each arm vs the big model, and the size change.

Expectation: modest. One constant-LR epoch is the new parameters' only real training before the cool-down, so a small
gain may be under-read; a go would then be confirmed by a full cool-down and a split of (A) and (B) (presets
`w64-sm77-keys`, `w64-sm77-init`). (B) should show up on first reviews (gap by review number, SM24).

**When.** After the better teacher's (SM65) stage 1 (decision ~04:00 Sat 10-31, earlier if SM72s finished early); the
launcher waits for SM65's launcher to end. From epoch 20 and a start at ~04:00 Sat: decision ~11:00 Sat.

## 6. Files (branch `sm77` at d370cf36, worktree `wt/sm77`)

| File | Change |
|---|---|
| `expertium/model/sm77.py` (new) | the mixers with (A) and (B); `wkv_from_state` dispatch |
| `expertium/variants.py`, `expertium/model/variant_model.py` | `key_extras`, `init_state` fields, presets `w64-sm77`, `-keys`, `-init`; `Sm77Layer` only when a flag is on |
| `rwkv/model/nvrtc.py`, `scripts/gpu_row_tiles.py`, `rwkv/model/wkv_state.py` | backward with dL/dS0 (new modules), `wkv_state_grad` |
| `scripts/state_carry_train.py` | the carry layer with (A) / (B), first-lane flags |
| `scripts/small_model_wkv.py`, `scripts/state_vq.py`, `scripts/small_model_heldout.py` | scorer from S0 |
| `scripts/sm77_warm.py` (new) | extend, epoch / cool-down preparation, CPU and GPU checks |
| `scripts/sm77_rule.py` (new), `artifacts/small-model/sm77/launch-sm77.sh` | rule, launcher (dry run OK); `HANDOFF-sm77.md` |
