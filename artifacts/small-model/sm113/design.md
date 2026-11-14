# The longer big instant-only teacher (SM113): design (2026-11-13)

## Question

The big instant-only teacher (SM110) re-ran the Small Instant-Only model (SM98)'s one-epoch cool-down from SM98's epoch 35 with only the teacher targets swapped for the show-time instant-only big model (SM99)'s. It was a no go: -0.023% [-0.028, -0.019] vs the Small Instant-Only model (SM98) on show-time instant, 722 users of the 732 set (493 better / 225 worse). It closed 21% of the teacher gap (the show-time instant-only big model (SM99) minus SM98 is -0.108%). At the halfway snapshot (4,000 users, learning rate 30% of its peak) the big instant-only teacher (SM110) was tied with the Small Instant-Only model (SM98)'s own snapshot (-0.001%), so the whole gain arrived in the low-learning-rate half.

This run asks: does a longer cool-down with the same SM99 targets pass the go rule against the Small Instant-Only model (SM98)?

Unit everywhere: % of the big model (1 leak-free epoch)'s normal-input instant LogLoss, 0.275757 on 722 users of the 732 set. All runs are leak-free (deleted cards: neighbour), scored w8+s8, on show-time inputs.

| plain name (ID) | what it is |
|---|---|
| the longer big instant-only teacher (SM113) | this design: the big instant-only teacher (SM110)'s cool-down stretched over two passes of the 8,000 users |
| the big instant-only teacher (SM110) | SM98's one-epoch cool-down from its epoch 35 with SM99's show-time targets; no go, -0.023% |
| the Small Instant-Only model (SM98) | 35 epochs + the one-epoch 32k Muon cool-down on show-time inputs, instant-only; the reference |
| the show-time instant-only big model (SM99) | the teacher (big architecture, 2 instant-only epochs on show-time inputs) |
| the big model (1 leak-free epoch) | the unit's denominator |

## Chosen design: a two-epoch cool-down from the Small Instant-Only model (SM98)'s epoch 35

- **One schedule, twice as long.** One one_minus_sin decay 7e-4 -> 0 over 2 x 581,942,724 reviews (the 8,000 users twice, after the same single trim of user 5599). Fresh Muon (`expertium-iter53`) at the start, as in every cool-down of the chain.
- **Everything else is the big instant-only teacher (SM110)'s:** start = SM98's epoch 35 (sha256 f7577d14ba10, read only), the same SM99 show-time replay targets (`anki-rwkv/artifacts/cache/kd-replay/sm99-big-inst-final8k-neighbour-e2s`, verified by SM110 on all 8,000 users, read only), alpha 0.9 / decay alpha 0.5 (the cool-down weight 0.5 applies to the whole decay), curve loss weight 0, show-time inputs, 32k windows, 1 lane, strided carry, the release's order and seeds, cap 0.8, snapshots every 400 users.
- **Two chained folders** (`sm113_prepare.py`), because a decay folder trains each user once:
  - **pass 1** `sm113-inst-e35st-st-dec2ep32k-t99-p1`: the folder `sm77_warm.py cooldown --epoch 25` prepares (byte-for-byte SM110's recipe), plus `lr_resume = {"total_reviews": 2R, "reviews_offset": 0, "segment": true}`. Learning rate 7e-4 -> 0.293 x 7e-4. `sm113_pair.py p1`: == SM110's cool-down except that lr_resume and the paths; == SM98's cool-down except also the teacher + replay folder.
  - **pass 2** `sm113-inst-e35st-st-dec2ep32k-t99-p2`, prepared when pass 1 is complete: pass 1's final master.pth AND optimizer.pth (the Muon state continues, the update offset continues), the same rows in a new order (`random.Random(2).shuffle` of pass 1's rows: a new epoch gets a new order, as the constant-LR epochs do), `lr_resume = {"total_reviews": 2R, "reviews_offset": R}`. Learning rate 0.293 x 7e-4 -> 0. This is exactly how a decay already resumes from a snapshot, at the boundary between the passes. `sm113_pair.py p2` proves it.
- **The one source change** (12 lines, the only difference from c99330a7, the tree SM96 / SM98 / SM99 / SM101 / SM110 trained with):
  - `scripts/expertium_options.py` (CatchUp): an lr_resume with `"segment": true` may end short of its `total_reviews` (pass 1 trains only the first half of the 2R schedule). Without the flag, the old exact add-up check is unchanged.
  - `scripts/expertium_decay.py` (resume_point): when a folder that was prepared with an lr_resume is resumed after a snapshot stop, the new lr_resume keeps its place in the longer decay (offset + done, the same total, the segment flag). Before, a resumed pass would have restarted the schedule over its own reviews. Folders prepared without an lr_resume (every earlier run) get the old dictionary, byte for byte.
  - Proof (C2): A/B against c99330a7's own two files: identical protocol.json / resume.json on fake decays, identical learning rates on SM98's and SM110's real protocols (SM110's is a resumed one), and `scripts/check_decay_resume.py` passes. C3: pass 1 + pass 2 give exactly the learning rate of one 2R decay at every probed review, also across a snapshot resume inside either pass.

Final model: the end of pass 2, label `sm113-inst-e35st-st-dec2ep32k-t99/w8+s8`.

## Why this variant

- **The gain lives in the low-learning-rate tail.** SM110 gained nothing by the halfway point (learning rate >= 30%) and -0.023% in the second half (4,000 users below 30%). In the two-epoch schedule the part below 29% is the whole of pass 2: all 8,000 users, twice SM110's low-learning-rate exposure.
- **Extra constant-LR epochs with the SM99 targets** (the other suggested variant) put the new targets where SM110 showed they do nothing (full learning rate). Each extra epoch then mostly buys more training (the Small Instant-Only model (SM98)'s 4 extra epochs gave -0.035%, ~0.009% per epoch). Two extra epochs + the usual cool-down: ~12.8 GPU h for an expected -0.03% to -0.045%, under the bar more often than not.
- **Redoing epoch 35 with the SM99 targets** (SM110's grey-zone follow-up) costs +1.9 GPU h for the same reason: little expected effect.
- **The cool-down length itself has a large measured effect.** The only clean length test, the quarter-epoch vs one-epoch 32k cool-down from the same epoch-10 weights (SM42, 732 pool), gave instant -0.00046 LogLoss (about -0.17%). That 4x stretch from a far less trained start is an upper anchor, not a prediction.

## Expected gain: about -0.06%, a go a bit more often than not

- Teacher part: -0.023% (SM110) to about -0.045% if the low-learning-rate gain scales with the low-learning-rate exposure.
- Length part (the same targets, 2x longer): -0.02% to -0.06% (from the SM42 anchor, scaled down for a 2x stretch at epoch 35 instead of 4x at epoch 10).
- Sum: about -0.04% to -0.10%, central -0.06%; chance of passing the go rule (CI below 0 and at least 0.05%) about 55-60%.
- **Attribution caveat:** a go vs the Small Instant-Only model (SM98) would mix the longer cool-down and the teacher. The read-out SM113 - SM110 gives the effect of the length with the same targets. Separating the teacher from the length needs the same two-epoch cool-down with SM98's own (old) targets (~17 GPU h): worth it only after a go, and also the natural test of a two-epoch cool-down for the two-head models.

## Pre-registered rules (`sm113_rule.py`, fixed before launch)

All numbers: 722 users of the 732 set, review-weighted instant LogLoss, show-time inputs (`e2s-query`), paired user bootstrap (2,000 resamples, seed 0, 95% CI), as % of the big model (1 leak-free epoch)'s normal-input instant LogLoss. References are read from their own committed rows (sha256 pinned), never rescored: SM96 / SM98 / SM99 rows of this branch, and SM110's final and mid-check rows copied from `wt/sm110` into `ref/` (they were never committed there).

**Early stop after pass 1 (`sm113_rule.py mid`).**
- x = SM113's pass-1 final model (learning rate 29.3% of its peak) minus the Small Instant-Only model (SM98)'s own cool-down snapshot after 4,000 of 8,000 users (learning rate 29.9%: the matching point of the one-epoch schedule; rows `sm98-dec-u4000` from SM110's mid-check, pinned).
- **Continue** to pass 2 if the point estimate x is below 0. **Stop** otherwise: verdict "no go (stopped after pass 1)", saving ~8.8 GPU h.
- Why: at the same learning rate, pass 1 has seen twice the reviews with the better targets. If it is not even ahead of SM98 there, a final gain of 0.05% would have to come entirely from pass 2.
- Read-out (no verdict): pass 1 minus SM110's snapshot after 4,000 users (the same targets at the same learning rate).

**Final go rule (`sm113_rule.py decide`).**
- **Primary:** the longer big instant-only teacher (SM113) minus the Small Instant-Only model (SM98). **Go** if its 95% CI lies entirely below 0 AND the gain is at least 0.05% (points <= -0.05). **No go** otherwise.
- **Reported with it (no verdict):** minus the big instant-only teacher (SM110) (the length effect with the same targets); the share of the teacher gap closed; minus SM99, SM96, SM92, the release (SM63) on show-time inputs; minus SM98 on normal inputs; normal inputs minus the big model (1 leak-free epoch) (the smaller-model bar: CI upper end <= +0.40%); the show-time cost; the early-stop pairs; the 1,500 validation users (minus SM98, SM110, SM99, the release, by id part, row check).
- **Selftest** (`sm113_rule.py selftest`, part of the CPU proofs): SM98 against itself 0 / no go / stop; SM96 - SM98 +0.035% no go; SM99 - SM98 -0.108% go; SM110 - SM98 -0.023% no go with gap closed 0.213; SM110's mid pair -0.001% continue; a synthetic 0.04% gain no go; the edges; an end-to-end mid + decide on relabelled rows.

## GPU hours and timeline (launch ~16:30 Fri 2026-11-13, queue empty)

Breaks (`sm113_block.py` = quiet.json + daily 20:00-01:00, Sunday from 16:00): Fri 20:00 -> Sat 01:00, Sat 20:00 -> Sun 01:00, Sun 16:00 -> Mon 01:00.

| step | GPU | ends about |
|---|---|---|
| smoke of pass 1 (2 users x 512 reviews) | 0.05 h | 16:35 Fri |
| **pass 1**, first segment (snapshot stop <= 2 min before 20:00) | 3.4 h | 19:58 Fri |
| break | | 01:00 Sat |
| **pass 1**, rest (resumes from its last snapshot, ~5 min restart) | 4.7 h | ~05:45 Sat |
| lr check; early stop: pass 1 scored on show-time (723 users, ~12 min), `mid` | 0.2 h | ~06:00 Sat |
| (if stop: done, verdict no go; ~8.3 GPU h in all) | | (~06:00 Sat) |
| pass 2 prepare + pair (CPU), smoke | 0.05 h | ~06:05 Sat |
| **pass 2** (8.0 h, no break in the way) | 8.0 h | ~14:10 Sat |
| lr check; final scoring: show-time + normal (723 + 3 long users each), the 1,500 pool, `decide` (SM110: 48 min) | 0.8 h | **~15:00 Sat 2026-11-14** |

**Total: about 17.1 GPU h** (about 8.3 GPU h if the early stop fires). If pass 2 slips past Sat 20:00 it snapshot-stops and resumes at Sun 01:00. Per pass: SM110's cool-down measured 479.3 min at ~20.2k reviews/s, nvidia-smi max 13,673 MiB, process-tree RSS ~11 GB (6 processes; the launcher holds a step while < 3 GB RAM is free).

10-epoch ETA at the measured cool-down speed (8k tier = 581.94M reviews per epoch, 2k tier = 147.6M): 10 passes at 20.2k reviews/s = 80 h (8k tier) / 20.3 h (2k tier).

## Risks

1. **The effect is below the bar.** The early stop caps the clear failures at ~8.3 GPU h.
2. **A go mixes length and teacher** (see the attribution caveat). The SM113 - SM110 read-out shows the size of the length part.
3. **The new code path** (segment lr_resume) runs for the first time on the GPU in pass 1 (the smoke drops lr_resume). Guards: CPU proofs C2/C3/C5/C6 on the real folders, and the launcher checks the last update's learning rate after each pass (pass 1 must end at 0.2925-0.2935 x peak, pass 2 at <= 0.0005 x peak) and stops otherwise.
4. **Interruptions:** every step resumes (a pass from its last 400-user snapshot, the user in flight retrained). Detached launchers die when the Claude app restarts: check after restarts.
5. **References outside this worktree:** `wt/sm98` (epoch 35, SM98's cool-down), `wt/sm110` (its cool-down folder for the p1 pair, its targets check), `wt/sm99` (the teacher), the targets cache. None may be cleaned up while this runs.

## What is built

- `sm113_prepare.py` (p1: set the segment; p2: pass 2 from pass 1's end), `sm113_pair.py` (p1, p2), `sm113_rule.py` (mid, decide, selftest), `sm113_check.py` (CPU proofs C1-C9), `sm113_block.py` (break windows, the code of `sm110_block.py` byte for byte), `launch-sm113.sh`, `extra-options.json` (SM110's), `ref/` (SM110's rows, pinned).
- CPU proofs `check-cpu-sm113.json`: ALL OK. Dry run `gpu/launch-sm113-dryrun.log`: `DRYRUN OK`.
- Pass 1 prepared: `gpu/prepare.out`, `gpu/trim.out`, `gpu/check-replay-label-windows-p1.json` (ok, trim user 5599 x 14), `gpu/check-muon-fresh-p1.json` (passed).
