# The Tiny two-head model (SM106): design (2026-11-11, prepared, not launched)

## Question

Can a two-head (curve + instant) small model with the card stream cut to 1 layer (the w64-c1 architecture of the Tiny Instant-Only model (SM101)) stay inside the pruning bar on show-time inputs? The bar is new minus the big model (1 leak-free epoch): curve 95% CI upper end <= +0.5% and instant 95% CI upper end <= +0.4%.

In short: SM101's recipe, but with the curve loss kept, started from the show-time retrain (SM92) instead of the Small Instant-Only model (SM98).

Units: % = % of the big model (1 leak-free epoch)'s show-time LogLoss on the same users (curve 0.29958, instant 0.277157; 722 users of the 732 set; `artifacts/small-model/showtime/showtime-report.md`).

| plain name (ID) | what it is |
|---|---|
| the Tiny two-head model (SM106) | this design: card stream 1 layer, both heads, show-time inputs |
| the show-time retrain (SM92) | the start: the leak-free small model (SM72s) epoch 25 + epoch 26 on show-time inputs + the release's cool-down on show-time inputs (both heads) |
| the Tiny Instant-Only model (SM101) | the same cut on the instant-only chain (SM98 epoch 35 cut, 3 epochs, cool-down); verdict go |
| the big model (1 leak-free epoch) (SM72 big) | the bar's reference |
| the release (SM63) | what the fork's RWKV-Curve runs |

## Warm start: cut the constant-LR epoch 26 of the show-time retrain (SM92)

- **Source.** `wt/sm92/artifacts/small-model/sm92-e26` is SM92's only constant-LR epoch on show-time inputs (master sha256 5a629e4e932a, optimizer db5ebc37396b, update offset end 574,028). SM92's cool-down master (`sm92-e26-dec-1ep32k-muon`, 581.9M reviews at 20.4k reviews/s) is the recipe reference for our cool-down, not a start: a cool-down ends at LR 0 and has a Muon state, so more constant-LR AdamW epochs must start from the constant-LR epoch, as in SM101.
- **The cut.** `scripts/sm73s2.py cut` (the tool of the smaller-model check (SM73s2), used again by SM101): it removes the 30 tensors `rwkv_modules.0.blocks.1.*` (19,725 values) and the AdamW state of 25 of them; the other 390 tensors and 329 AdamW states are copied bit for bit. Parameters 331,511 -> 311,786. Output `artifacts/small-model/sm106-cut-e26` (master sha256 44cde2b51203, optimizer ee60829a694a; local only, not committed). Ran on CPU on 2026-11-11.

## What runs

| step | recipe | about (GPU) |
|---|---|---|
| reference score | SM92's constant-LR epoch 26 (w64), show-time inputs, 723 users (the early check's reference) | 14 min |
| **epoch 27** | from the cut; SM92 epoch 26's recipe with the epoch index advanced; the only change is `architecture` w64-c1 | ~108 min |
| **early check** | epoch 27 scored on show-time inputs; rule below | 14 min |
| epochs 28, 29 | the same step from the previous epoch (options unchanged) | ~2 x 110 min |
| **cool-down** | the release's one-epoch 32k Muon cool-down on show-time inputs from epoch 29 = SM92's cool-down except the start and the architecture | ~460-480 min |
| scoring | show-time inputs: the 732 set (723 + 3 long users), then the 1,500 pool | ~70 min |

- **The epoch recipe** (SM92 epoch 26's, `sm106_epoch.py`): 8k final tier, the leak-free small model (SM72s) run's order rule `Random(split_seed * 1000 + N)`, epoch 25's label-window trims, 2048 x 16 lanes carry, leak-free replay KD from the big model (1 leak-free epoch) (`big-leakfree-1ep-final8k-neighbour`, alpha 0.9), AdamW 7e-4 constant, `show_time_intervals` true, and **both heads at SM92's loss weights** (no `curve_loss_weight` key; the trainer's default 1.0 multiplies the curve terms by exactly 1.0, so the loss is SM92's bit for bit).
- **The cool-down** (`sm77_warm.py cooldown --epoch 25 --fraction 1.0 --arch w64-c1 --extra-options extra-options.json`, the file is SM92's): fresh Muon `expertium-iter53`, LR 7e-4 -> 0 one_minus_sin over 581.9M reviews, 32k windows, batch_chunks 1, KD replay alpha 0.9 with the same teacher targets as SM92, both heads, show-time inputs. The launcher checks it against SM92's real cool-down (`sm106_pair.py decay`: only the start and the architecture may differ) before it runs.
- **Why three constant-LR epochs.** SM101 used 3 after the same cut, and its early check showed the cut recovered almost fully in one epoch (+0.051% [-0.004, +0.109] vs its uncut source). Three epochs give about 14-15 GPU h in all, the target. A fourth epoch would add ~1.8 h (~16.8 h) for a gain SM98 measured at -0.035% over 4 uncut epochs, so it is not worth it.
- **Code.** Branch `sm106` is cut from `origin/sm101`, not from main: main lacks the chain's trainer code (no `w64-c1` preset, no `show_time_intervals`, no `scripts/sm77_warm.py` / `scripts/sm73s2.py`). The code trees (scripts/ expertium/ rwkv/) are bitwise those SM101 and SM98 trained with (1e22b62c). Versus SM92's own code they differ only by the `curve_loss_weight` knob (default 1.0, exact) and the scorer's 1,500-pool option (`check-cpu-sm106.json` K1).

## Early check (stop rule, `sm106_rule.py early`, fixed before launch)

Arm: epoch 27 of the Tiny two-head model (SM106) (w64-c1, constant LR). Reference: SM92's epoch 26 (w64, constant LR, the cut's source). Both scored on show-time inputs, both heads, 722 users, paired.

- **Stop** if the point estimate arm minus reference is above **+0.25%** on the curve head OR on the instant head. The launcher then ends after about 2.3 GPU h, with no further GPU work.
- **Continue** otherwise.
- **Why +0.25.** It is SM101's early bar, applied to each trained head: a coarse filter for a clearly bad cut, not a measurement. The two checkpoints are one epoch apart; constant-LR read-outs move by ~0.06% between epochs.

## Final rules (`sm106_rule.py decide`, pre-registered)

All on show-time inputs (`e2s-query`), leak-free w8+s8 rows (`--split-existing --deleted-cards neighbour`), 722 users of the 732 set, review-weighted LogLoss, paired user bootstrap (2,000 resamples, seed 0, 95% CI). The bootstrap and loaders are the show-time report's (`showtime_report.py`, sha256 pinned).

- **A, the pruning bar (the verdict).** The Tiny two-head model (SM106) minus the big model (1 leak-free epoch): **curve CI upper end <= +0.5% AND instant CI upper end <= +0.4%**. Go if both hold; no go otherwise.
- **B, report only (no verdict).** New minus the show-time retrain (SM92), curve and instant; new minus the Tiny Instant-Only model (SM101), instant (SM101's curve head is untrained).
- Also reported: new minus the release (SM63); the 1,500 pool on show-time inputs (% of the release's own LogLoss: new minus the release, new minus SM101 on instant); the row check on the 6 check users.

**Room under the rule** (selftest on known rows, `sm106_rule.py selftest`):

| "new" model | A curve | A instant | verdict |
|---|---|---|---|
| the show-time retrain (SM92) itself | +0.259% [+0.072, +0.463] | -0.101% [-0.214, -0.015] | go |
| the release (SM63) | +0.278% [+0.117, +0.461] | +0.202% [+0.171, +0.235] | go |
| the Tiny Instant-Only model (SM101) | curve untrained | | no go |

- **The curve bar binds, and hard.** SM92's own curve CI upper end is +0.463%, so the cut has only about **+0.04%** of curve room before A fails (assuming the CI width stays about the same). On instant the room is about +0.4%.
- **What to expect.** The two-head cut cost +0.16% curve / +0.21% instant one epoch after the cut in the smaller-model check (SM73s2, normal inputs); SM101's finished cut cost ~+0.16-0.19% instant against its uncut chain. If the curve cost is similar here, the Tiny two-head model (SM106) lands near +0.42% [+0.23, +0.62] on the curve, which **fails A**, and near +0.09% [-0.02, +0.20] on instant, which passes. So a go needs the cut to cost almost nothing on the curve head. That is unlikely, and the early bar (+0.25% per head) will not stop a cut that only fails the curve bar. **The main session may want to confirm the run is still wanted (or tighten the early curve bar, e.g. to +0.10%, before launch; the rule is pre-registered, so only before launch).**

## GPU budget and timeline (if launched ~08:00 Thu 2026-11-12, after SM99, queue empty)

| step | ends about |
|---|---|
| reference scoring, smoke, epoch 27, its scoring, early rule | 10:20 Thu (early check: ~2.3 GPU h) |
| epochs 28, 29 | ~14:00 |
| cool-down prepare + checks + smoke; first cool-down segment to the 19:58 snapshot stop | 19:58 |
| break 20:00-01:00 | |
| cool-down resume (~110-125 min left + ~5 min restart) | ~03:00 Fri |
| scoring (732 set + 3 long users) + rule, the 1,500 pool, final rule | **~04:10 Fri 2026-11-13** |

About **15 GPU h** in all (the early check is ~2.3 h of it), ~20 h wall time with the break. Speeds assumed: SM101's w64-c1 epochs 88.1-91.6k reviews/s (SM92's w64 two-head epoch 86.9k, SM98's w64 instant-only 87.4k, so the curve loss costs no visible speed); SM101's cool-down ~20.8k reviews/s (SM92's 20.4k).

10-epoch ETA at these speeds (8k tier = 581.94M reviews per epoch, 2k tier = 147.6M): 10 constant-LR epochs at ~89k reviews/s take 18.2 h (8k) / 4.6 h (2k); 10 cool-downs at ~20.8k reviews/s take 77.7 h (8k) / 19.7 h (2k).

## Files

- `launch-sm106.sh` (launcher; waits for nothing, STOPs at its preconditions if the show-time instant-only big model (SM99) has not logged `done (verdict ...)`), `sm106_epoch.py` (epoch prepare / pair), `sm106_pair.py` (cool-down pair), `sm106_rule.py` (early + final rule, selftest), `sm106_check.py` (CPU proofs), `sm106_block.py` (breaks), `extra-options.json` (= SM92's).
- `check-cpu-sm106.json`: ALL OK (K1 code, K2 options, K3 epoch pair + simulated chain + 6 negative controls, K4 the cut, K5 cool-down recipe vs SM92's cool-down + 2 negative controls, K6 rule selftest + end to end, K7 breaks, K8 CPU forward pass).
- `cpu-forward-smoke.jsonl`: CPU forward pass on 2 users (2388, 4676) on show-time inputs, w8+s8, leak-free: the cut and SM92 epoch 26 both give finite curve and instant LogLoss, within 0.001 of each other per user.
- `gpu/launch-sm106-dryrun.log` (ends `DRYRUN OK`), `gpu/launch-sm106-recipe.log` + `gpu/check-*-sm106-recipe-sim-dec-1ep32k-muon.json` (the cool-down recipe proof on CPU).
