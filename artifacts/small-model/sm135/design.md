# Six-layer Tiny with longer recovery (SM135): design (2026-11-20, pre-registered before any GPU ticket)

## Question

The six-layer Tiny (SM130) cut Tiny's deck / preset / user streams to 2 / 1 / 1 layers (`w64-c1-d2p1u1`, 185,057 parameters instead of 311,786, 6 sequential layer steps instead of 12). It got three constant-LR recovery epochs, then the usual two-pass cool-down, and both chains passed the Tiny-shrink budget (95% CI upper bound <= +1.00% LogLoss vs Small per trained head):

| chain | six-layer Tiny (SM130) vs Small | vs the unmodified Tiny |
|---|---|---|
| two-head, published w8+s8 | curve +0.467% [+0.398, +0.546], instant +0.524% [+0.484, +0.567] vs the show-time two-head cool-down for Small (SM118) | +0.349% / +0.340% vs the show-time two-head cool-down for Tiny (SM124) |
| Instant-Only, published w8+s8 | +0.564% [+0.522, +0.607] vs the longer Small Instant-Only model (SM113) | +0.378% vs the longer cool-down for Tiny (SM115) |
| Instant-Only, card-only w8+s4hgx4c8hx8n (the six-layer Tiny with the card-only state (SM134)) | +0.888% [+0.832, +0.943] vs the longer Small Instant-Only model (SM113) | |

With the card-only state the Instant-Only chain keeps only about 0.06% of the budget. **Do more constant-LR recovery epochs before the cool-down close part of the gap to the unmodified Tiny, so that the shipped six-layer (possibly card-only) Tiny has real headroom?**

| plain name (ID) | role here |
|---|---|
| six-layer Tiny with longer recovery (SM135) | this run: SM130's chains continued by up to four constant-LR epochs each, then SM130's cool-down and scoring |
| the six-layer Tiny (SM130) | the source (read only, `wt/sm130`): Instant-Only epoch 41 (`sm130-io-e41-st`, master sha256 af16139e28bb, update offset end 905,198) and two-head epoch 32 (`sm130-th-e32-st`, 635f1a12893b, 706,496); its final rows are the main rule's reference |
| the six-layer Tiny with the card-only state (SM134) | SM130's card-only rows (`artifacts/small-model/sm134/gpu/sm134-rows.jsonl`, merged from branch sm134 with its card-only scorer) |
| the longer Small Instant-Only model (SM113) / the show-time two-head cool-down for Small (SM118) | Small, the budget references (report only here) |
| the longer cool-down for Tiny (SM115) / the show-time two-head cool-down for Tiny (SM124) | the unmodified Tiny (report only); their pass-1 folders are the cool-down pair references |
| the show-time Instant-Only big model (SM99) / the show-time big model (SM91s2) | the cool-down teachers (their show-time targets, read only) |
| the big model (1 leak-free epoch) (SM72) | the recovery epochs' KD replay targets, as in SM130 |

## Recipe per chain (no change except the number of recovery epochs)

| step | Instant-Only (io), runs first | two-head (th), second |
|---|---|---|
| start | SM130's epoch 41 (pinned master + optimizer sha256) | SM130's epoch 32 (pinned) |
| recovery epochs (constant LR) | 42, 43, 44, 45 at most = SM130's Instant-Only epoch step (`sm135_epoch.py`: the run's order rule `random.Random(split_seed * 1000 + N)`, 8k final tier, epoch 25's trims, 2048 x 16 lanes, leak-free replay KD from the big model (1 leak-free epoch) alpha 0.9, AdamW 7e-4, curve_loss_weight 0, show-time inputs, `w64-c1-d2p1u1`); every leaf of each epoch's folder equals the previous epoch's except the order, the start and paths (`pair`) | 33, 34, 35, 36 at most = SM130's two-head step (both heads) |
| after every recovery epoch | score the constant-LR checkpoint, w8+s8, 723 users; early rule | same |
| cool-down pass 1 + 2 | SM130's (= the longer cool-down for Tiny (SM115)'s recipe): one one_minus_sin decay 7e-4 -> 0 over 2 x 581,942,724 reviews split in two folders, fresh Muon, 32k windows, the show-time Instant-Only big model (SM99)'s targets, decay alpha 0.5; pass 2 = new order `random.Random(2)`, pass 1's Muon state. Start = the last kept recovery epoch. `sm135_pair.py p1` checks pass 1 against SM115's pass 1 except the start and the architecture | SM130's (= the show-time two-head cool-down for Tiny (SM124)'s recipe, the show-time big model (SM91s2)'s targets). **Runs only after an explicit OK** (`TH_COOLDOWN=1`, budget, below) |
| final scoring | one pass, 723 users of the 732 set, show-time inputs, leak-free (neighbour): `w8+s8`, compact `w8+s4c8nhx8`, card-only `w8+s4hgx4c8hx8n`; SM130's scorer flags plus `--nonfinite-rows` (as SM131 / SM134) | same |

The tree is branch sm135 = `origin/sm130` (the six-layer preset and its Muon split check) merged with `origin/sm134` (= `origin/sm131`'s card-only scorer and the SM129 / SM131 state formats). The merge changes only scorer-side files (`small_model_heldout.py`, `small_model_wkv.py`, `compress_wkv_cuda.py`, `compress_quant.py`, `state_vq*.py`); the trainer imports `state_vq` only when a `state_vq` option is set (not here), so the training path is SM130's.

## Why at most four more epochs (from SM130's early-check curve)

| | after the first recovery epoch | after the last | change per epoch |
|---|---|---|---|
| Instant-Only, % vs the uncut epoch 38 (held-out, w8+s8) | +0.999% (epoch 39) | +0.286% (epoch 41) | -0.36% per epoch on average |
| two-head curve / instant, % vs the uncut epoch 29 | +1.378% / +0.918% (epoch 30) | +1.022% / +0.637% (epoch 32) | -0.18% / -0.14% per epoch |
| Instant-Only training loss (epoch mean, same 8,000 users) | 0.54292 (39) | 0.54148 (40), 0.54111 (41) | -0.00144, then -0.00037 |
| two-head training loss | 0.86003 (30) | 0.85741 (31), 0.85673 (32) | -0.00262, then -0.00068 |
| unmodified Tiny at the same stage (SM101 epochs 36-38; SM106 epochs 27-29) | | | -0.00020 / -0.00010; -0.00042 / -0.00018 |

The held-out gap was still shrinking fast at the last check, and the training loss of the second-to-third recovery epoch still fell about 2-4 times faster than the unmodified Tiny's normal constant-LR drift, so recovery was not finished. The fall is also slowing (the training-loss step shrank about 4 times from the first to the second pair, part of which is the cut's own first-epoch transient). A geometric extrapolation of the training loss gives most of the remaining gain within one or two epochs; a floor at the normal drift gives a slow tail. Four epochs cover both shapes, and the early rules below end the recovery as soon as an epoch stops helping, so the average cost is lower than the maximum. Prediction (not a gate): a gain of about 0.05% to 0.2% at the end for the Instant-Only chain; the cool-down did not shrink SM130's Instant-Only constant-LR gap (+0.286% before, +0.378% after vs the unmodified Tiny), so the transfer from constant-LR gains to the final model is the main uncertainty.

## Pre-registered rules (`sm135_rule.py`, fixed now; selftest passes)

All: show-time inputs, leak-free (neighbour), w8 weights, the 722 rule users of the 732 set (`showtime_report.py pools()`, sha256 pinned), review-weighted LogLoss, paired user bootstrap 2,000, seed 0, 95% CI; "%" = 100 x (new / reference - 1) of the reference's own LogLoss. Trained heads: Instant-Only instant; two-head curve and instant. Reference rows sha256 pinned: SM130's early rows (f22c2f85ecab) and final rows (f4b38086f872), SM134's card-only rows (7b286b434f53), SM118, SM113, SM124, SM115.

**Early rule after every recovery epoch N (constant-LR checkpoint, w8+s8).** x = epoch N vs epoch N - 1 (for the first, SM130's last recovery epoch); y = epoch N vs SM130's last recovery epoch. In this order:
1. Non-finite rows -> stop the chain.
2. **Worse:** if any trained head's 95% CI lower bound of x is above 0 (the epoch made that head significantly worse): after the first new epoch (42 / 33) the chain ends, "no go (early stop)", no cool-down; later, the recovery ends and the cool-down starts from epoch N - 1 (but after the second new epoch, only if the first new epoch had gained at least 0.05% on every trained head; otherwise the chain ends).
3. **No gain:** after the second new epoch (43 / 34), if any trained head's y point estimate is above -0.05% (two more epochs gained less than the minimum effect on a head the main rule needs), the chain ends, "no go (early stop)". This saves the ~5 GPU h cool-down when the extra recovery visibly does nothing.
4. After the last allowed epoch (45 / 36): cool-down from it. Otherwise: next epoch.

**Main rule (`decide`, per chain, after the cool-down):** d = 100 x (SM135 final / SM130 final - 1) per trained head, both published w8+s8. **Go iff for every trained head the 95% CI upper bound of d is < 0 AND the point estimate is <= -0.05%** (the repo's minimum-effect rule), on all 722 rule users with finite rows; a non-finite rule user is a no go, a missing one keeps `decide` pending (exit 2).

**Reported, no verdict:** for each of w8+s8, compact w8+s4c8nhx8 and card-only w8+s4hgx4c8hx8n: SM135 vs Small's published w8+s8 (the show-time two-head cool-down for Small (SM118) / the longer Small Instant-Only model (SM113)) with the budget check (CI upper bound <= +1.00%), and SM135 vs SM130 in the same format (compact: SM130's rows; card-only: SM134's rows); SM135 vs the unmodified Tiny (SM124 / SM115) at w8+s8; every early check; reviews/s per training folder.

Selftest (CPU, passes): SM130's epoch 41 relabelled as epoch 42 continues (x = 0); SM130's epoch 39 as epoch 42 stops (worse); scaled rows give continue at -0.10%, stop at -0.02% (no gain), cool-down from 43 when 44 is worse, from 45 at the end, stop when 43 is worse and 42 gained only -0.01%; two-head: SM130's epoch 30 as 33 stops, a gain on curve only stops at 34. Final: SM130's rows relabelled as SM135's give no go (d = 0) and reproduce SM130's numbers vs Small (+0.564% [+0.522, +0.607]; +0.467% [+0.398, +0.546] / +0.524% [+0.484, +0.567]; compact +0.603%, +0.483%), SM134's card-only numbers (+0.888% [+0.832, +0.943]; +0.659% [+0.585, +0.743] / +0.780% [+0.730, +0.833]) and the gaps to the unmodified Tiny (+0.378%, +0.349%); x 0.995 goes (-0.500%), x 0.9997 is a no go (-0.03%, CI below 0, below the minimum effect), two-head with only instant improved is a no go, one non-finite rule user is a no go, a dropped user is pending.

## Budget, order and ETA (at most about 14 GPU h)

Speed basis (SM130, same architecture and recipe): recovery epochs 53-55 min (177,800-182,500 reviews/s), constant-LR scoring 10-12 min per checkpoint, cool-down passes 147-151 min (64,300-66,000 reviews/s), final scoring of two formats ~11 min (three here: ~15 min).

| step | GPU (central) |
|---|---|
| Instant-Only: 4 x (epoch + scoring) | 4.4 h |
| Instant-Only: cool-down passes 1 + 2 (+ 2 smokes, prepares on CPU) | 5.0 h |
| Instant-Only: final scoring, 3 formats | 0.25 h |
| **Instant-Only chain** | **~9.7 h** (1.1 h if the first early stop fires, 2.2 h if the no-gain stop fires) |
| two-head: 4 x (epoch + scoring), only if the Instant-Only chain goes | 4.4 h |
| **total within the budget** | **~14.1 h** |
| two-head cool-down + final scoring (held: needs the main session's OK, `TH_COOLDOWN=1`) | +5.3 h (total ~19.4 h) |

Both full chains would need ~19.4 GPU h, above the ~14 h budget. So: the Instant-Only chain (the tightest budget) runs completely first; the two-head recovery epochs run only if the Instant-Only chain goes on the main rule (if more recovery does not help there, where the epoch-to-epoch gains were largest, it is not worth ~10 GPU h on the two-head chain); the two-head cool-down then waits for an explicit OK, with its start epoch already decided by the early rule, so the OK costs only the 5.3 h remainder.

| stage (launch ~02:10 Fri 20 Nov, GPU free, nothing queued) | central |
|---|---|
| Instant-Only epoch 42 + early check | ~03:15 Fri |
| epochs 43 / 44 / 45 + early checks | ~04:20 / ~05:25 / ~06:35 Fri |
| Instant-Only cool-down passes 1 + 2 | ~09:10 / ~11:35 Fri |
| **Instant-Only result** (`rule decide (Instant-Only)`) | **~11:50 Friday 20 November 2026** (range ~11:00-13:30; ~03:15 / ~04:20 if an early stop fires) |
| two-head epochs 33-36 + early checks (if the Instant-Only chain goes) | ~16:15 Fri, then `done (...)` with the two-head verdict "held" |
| two-head result if the cool-down is approved right away | ~21:30 Fri 20 Nov |

Breaks: quiet.json windows of scope all / gpu only, re-read before every ticket and every 30 s while a folder trains (`sm135_block.py`, no daily or Sunday window in the code; its selftest checks that). A training folder snapshot-stops at most 2 min before a window and resumes after it; scoring tickets start only when they fit.

## Memory and GPU

GPU cap 0.8 (training options `gpu_memory_fraction` 0.8; scorer `HELDOUT_GPU_MEMORY_FRACTION` 0.8, 0.92 only for a retry pass without progress, memory only, same numerics), every GPU step through `scripts/r01_queue.py`. CPU-only commands of the launcher (rule, epoch / pair / prepare scripts, block helper, Muon and label-window checks, sm77 prepare) run under `memguard.ps1` for the launcher's whole life (kill above 5.9 GB, then STOP; it ends with the launcher's Windows PID). The training process trees reach ~11 GB RSS (SM130, unchanged trainer; GPU tickets are not matched by the guard).

## Risks

1. **More constant-LR epochs may not transfer through the cool-down** (SM130's Instant-Only gap grew from +0.286% to +0.378% across it). The main rule judges the final model only; the early rules only stop chains that do not improve at constant LR.
2. **Noise in the early comparisons:** paired, so the CIs are narrow on instant (~±0.04%) but wider on curve; "worse" needs a CI above 0, the no-gain stop uses the point estimate (a noisy two-head curve can end that chain early; it is report-only within this budget).
3. **The cut's long-run capacity:** the remaining gap may be capacity, not under-training; then the answer is a clean no go and the shipped six-layer Tiny keeps SM130's numbers.
4. **Read-only inputs outside this worktree** (must not be cleaned up while this runs): `wt/sm130` (start epochs 41 / 32 and their optimizer states), `wt/sm115` and `wt/sm124` (reference pass-1 folders), `wt/sm118`, `wt/sm113` (rows), `wt/sm91s2`, `wt/sm99` (teachers), `wt/sm72` / `wt/sm72s` (run config, label DB, eval users), the KD replay caches `big-leakfree-1ep-final8k-neighbour`, `sm91s2-big-showtime-final8k-neighbour-e2s`, `sm99-big-inst-final8k-neighbour-e2s`.
5. **Interruptions:** every step resumes (folders from their last snapshot, finished steps skipped, decisions kept in `gpu/cool-start-*`, `gpu/verdict-*`, `sm135-early-*.json`). Detached launchers can die at a session restart or a power event: check after any gap and relaunch from HANDOFF.md.

## What is built and checked (CPU, before the launch)

`sm135_epoch.py` (prepare / pair / selfcheck: SM130's real epochs 41 vs 40 and 32 vs 31 pass the same leaf rule; the pinned starts match), `sm135_pair.py` and `sm135_prepare.py` (SM130's with the note key `sm135`; selfcheck passes), `sm135_rule.py` (early, decide, selftest), `sm135_block.py` (quiet.json only, selftest), `memguard.ps1` (SM130's plus `-ParentPid`; checked to exit when its parent ends), `extra-options-{io,th}.json` (byte copies of SM130's), `launch-sm135.sh` (`DRYRUN=1` OK for the default run and for `CHAINS=th TH_FORCE=1 TH_COOLDOWN=1`). Prepared (local): `sm135-io-e42-st` (pair OK: SM130's epoch 41 step, start af16139e28bb at offset 905,198). Dry pass-1 prepares of both chains from SM130's epochs 41 / 32 in this tree passed the label-window check, the Muon fresh-state check (7 PASS lines each) and `sm135_pair.py p1` against SM115's / SM124's pass 1 (folders deleted; logs in `cpu/`).
