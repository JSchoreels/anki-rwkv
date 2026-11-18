# Six-layer Tiny (SM130): design (2026-11-18, pre-registered before any GPU ticket)

## Question

The user wants Tiny clearly smaller and accepts up to +1% LogLoss against Small on each head, judged on the 95% CI upper bound (direction approved Wed 11-18 ~22:30). The outside review (`artifacts/small-model/tiny-smaller-codex/analysis.md` on main) ranks the context-stream depth cut first: keep Tiny's one card layer and one note layer, cut the deck / preset / user streams to 2 / 1 / 1 layers. **Can a six-layer Tiny, retrained with the current recovery and cool-down recipe, stay within +1% of Small on every trained head, for the two-head and the Instant-Only model?**

| plain name (ID) | what it is |
|---|---|
| six-layer Tiny (SM130) | this design: `w64-c1-d2p1u1`, 185,057 parameters, 6 sequential layer steps (Tiny: 311,786 and 12) |
| the context-stream pruning (SM15) | the same cut on the older 13-layer Small (`w64-d2p1u1`, 204,782 parameters): zero-shot +4.68% curve / +3.47% instant (68 users); after a short fine-tune + long-window cool-down +0.36% / +0.42% (459 users), +0.44% / +0.43% (724 users) vs its parent |
| the Tiny two-head model (SM106) | the two-head chain: its constant-LR epoch 29 is the cut's source (`wt/sm106/.../sm106-c1-e29-st`, master sha256 b3a694ca14cb, update offset end 640,262) |
| the Tiny Instant-Only model (SM101) | the Instant-Only chain: its constant-LR epoch 38 is the cut's source (`wt/sm101/.../sm101-c1-e38-st`, e6ea69124af0, offset 838,964) |
| the show-time two-head cool-down for Tiny (SM124) | the unmodified Tiny two-head candidate (from SM106 epoch 29); its two-pass cool-down recipe is reused |
| the longer cool-down for Tiny (SM115) | the unmodified Tiny Instant-Only candidate (from SM101 epoch 38); its recipe is reused |
| the show-time big model (SM91s2) / the show-time Instant-Only big model (SM99) | the cool-down teachers (their exported show-time targets for the 8,000 users, reused read only) |
| the show-time two-head cool-down for Small (SM118) / the longer Small Instant-Only model (SM113) | the references of the rule |
| compact state for Tiny (SM129) | the companion training-free test; if its compact arm goes, the six-layer students are also scored in that format |

The epochs the review suggested (29 and 38) are confirmed: SM124's pass 1 starts from SM106's epoch 29 (`sm124/design.md`), SM115's pass 1 from SM101's epoch 38 (`sm115/design.md`).

## The cut (`sm130_cut.py`, done on CPU 11-18 22:3x; `expertium/variants.py` preset `w64-c1-d2p1u1`)

- Preset grammar `w64-c{C}-d{D}p{P}u{U}`: the leading C card / D deck / P preset / U user layers of w64 (note 1), as SM15's `w64-d{D}p{P}u{U}`; the interleaved schedule runs layer r of every stream in round r, so the survivors are an exact subsequence of Tiny's order. Layer steps: Tiny (0,0) (1,0) (2,0) (3,0) (4,0) (1,1) (3,1) (4,1) (1,2) (3,2) (4,2) (1,3) -> six-layer (0,0) (1,0) (2,0) (3,0) (4,0) (1,1). Kept channel mixers: card:0, deck:0, note:0 (as Tiny).
- Removed: `rwkv_modules.1.blocks.2-3.*`, `rwkv_modules.3.blocks.1-2.*`, `rwkv_modules.4.blocks.1-2.*` (126,729 values); every kept tensor and its AdamW state copied bit for bit. Two cuts: `artifacts/small-model/sm130-th-cut-e29` (master 57cb724e3116) and `sm130-io-cut-e38` (09f77d31abaf) (local, not committed).
- Checked now on CPU (seconds, `sm130_cut.py quick`): 185,057 parameters (the outside review's figure), 6 layer steps in Tiny's order, strict loads, one trainer-path forward + backward of each cut finite (loss 1.23 / 1.04 / 0.74 on a 636-review user). The full CPU proofs (`sm130_cut.py cpu-check`: trainer path cut == Tiny with the six layer steps skipped, bitwise, eager and TorchScript; scorer path cut == Tiny with the dropped branches zeroed, bitwise; Muon fresh-state check on the cut's layout) run at 01:00 under the memory guard, before any GPU ticket.
- Source change (committed before the launch): the preset (27 lines in `expertium/variants.py`, no existing preset changes) and the Muon / AdamW split derivation in `scripts/check_muon_fresh_decay.py` for the new preset (the SM73 derivation; w64-c1's path unchanged). Dry-prepared cool-downs of both chains passed the pair proofs and the Muon check (folders deleted).

## Recipe per chain (no other change than the architecture)

| step | two-head (th) | Instant-Only (io) |
|---|---|---|
| uncut reference for the early checks (scored first) | SM106 epoch 29 (constant LR) | SM101 epoch 38 (constant LR) |
| recovery epochs (constant LR) | 30, 31, 32 = SM106's epoch step (`sm130_epoch.py`: the run's order rule, 8k final tier, epoch 25's trims, 2048 x 16 lanes, leak-free replay KD from the big model (1 leak-free epoch) alpha 0.9, AdamW 7e-4, both heads, show-time inputs) | 39, 40, 41 = SM101's epoch step (curve_loss_weight 0) |
| cool-down pass 1 | SM124's pass 1 except the start (epoch 32) and the architecture (`sm130_pair.py p1`): one one_minus_sin decay 7e-4 -> 0 over 2 x 581,942,724 reviews, first half, fresh Muon, 32k windows, the show-time big model (SM91s2)'s targets, decay alpha 0.5 | SM115's pass 1 except the start (epoch 41) and the architecture: the show-time Instant-Only big model (SM99)'s targets |
| cool-down pass 2 | pass 1's end, new order (`random.Random(2)`), second half of the decay (`sm130_pair.py p2`) | same |
| final scoring | show-time, 723 users, `w8+s8` (+ `w8+s4c8nhx8` if SM129's compact arm passed for the two-head model) | same for the Instant-Only model |

Three recovery epochs, as SM106 and SM101 after their card cut; the cut here removes 6 of 12 layer steps (SM15's zero-shot cost +3.5% to +4.7%), so recovery matters more. The recovery epochs keep the chain's own targets (the big model (1 leak-free epoch)) exactly as SM106 / SM101 trained epochs 27-29 / 36-38; SM91s2's and SM99's targets enter in the cool-down, exactly where SM124 and SM115 used them. So the six-layer students differ from the unmodified Tiny candidates only by the cut and three more constant-LR epochs.

## Pre-registered rules (`sm130_rule.py`, fixed now)

All: show-time inputs, leak-free (neighbour), w8 weights, 722 rule users of the 732 set (`showtime_report.py pools()`, sha256 pinned), review-weighted LogLoss, paired user bootstrap 2,000, seed 0, 95% CI; "%" = 100 x (new / reference - 1) of the reference's own LogLoss. Trained heads: two-head curve + instant, Instant-Only instant.

**Early stops (cheap, for clearly hopeless runs; constant-LR checkpoints, w8+s8):** x = six-layer recovery epoch minus the uncut start epoch (SM106 epoch 29 / SM101 epoch 38), per trained head.
- After the first recovery epoch (30 / 39): **stop the chain if any trained head's point estimate is above +1.50%**.
- After the last recovery epoch (32 / 41), before the ~8 GPU h cool-down: **stop if any is above +1.20%**.
- Why: at the end the budget left after Tiny's own gap is about +0.82% to +0.88% against Small; the card cut of the Tiny two-head model (SM106) shrank from +0.26% curve at its first epoch to +0.11% after the cool-down (instant +0.19% -> +0.18%), so the bars leave room for a 30-45% shrink and stop only runs that would need more. A stop ends that chain ("no go (early stop)"); the other chain still runs.

**Final rule (`decide`), per chain:**
- **Go if every trained head's 95% CI upper bound of 100 x (six-layer / Small - 1) is <= +1.00%** in the published w8+s8 format, all 722 users with finite rows (Small = SM118 for two-head, SM113 for Instant-Only, committed rows, sha256 pinned). Ceilings: curve <= 0.30302, instant <= 0.27936 (two-head), instant <= 0.27869 (Instant-Only), as the user stated.
- Compact format (only if SM129's compact arm passed for that model): the same rule on the `w8+s4c8nhx8` rows -> "go in the compact format", reported beside the verdict.
- Reported, no verdict: six-layer minus the unmodified Tiny (SM124 / SM115), % of Tiny's LogLoss with CI; the early pairs; parameters, weight bytes, state bytes, reviews/s per training segment, CPU latency.
- Selftest (CPU, passes now): SM106's epoch 27 vs SM92's epoch 26 as an early pair continues (+0.26%); instant scaled +1.25% stops at the last bar; SM124 relabelled as the six-layer model reproduces the outside review's +0.118% [+0.057, +0.174] / +0.183% [+0.164, +0.204] vs SM118 and goes; SM115 relabelled gives +0.186% [+0.166, +0.208] and goes; compact rows scaled by 1.009 fail.

## Size (report)

| | Tiny (w64-c1) | six-layer Tiny | change |
|---|---:|---:|---:|
| parameters | 311,786 | 185,057 | -41% |
| sequential layer steps | 12 | 6 | -50% |
| scorer w8 weight bytes (row field) | 385,485 | measured by the scorer rows | |
| int8 release file (outside review's estimate from the tensor headers) | 409,483 (released SM106) | ~238,585 | ~-42% |
| state per card / note, fork default | 1,055 / 1,567 B | same | 0 |
| state per card / note, compact (SM129) | 803 / 1,315 B | same | -24% card, -16% note |
| CPU review step, PyTorch eager, 1 thread (`sm130_timing.py`, after the chains; a 20-step smoke on the untrained cut gave 1.06 vs 2.03 ms, not a measurement) | | | |

The CPU number is an indication of the layer-count effect only (the fork runs its own Rust engine; a native benchmark needs the fork and is not part of this run, no installs).

## GPU hours and ETA

Speed basis: Tiny's constant-LR epochs ran at ~91k reviews/s (107 min for 581.9M reviews), its cool-down passes at 21-29k reviews/s (336-463 min). The 32k cool-down is recurrence-bound (S06), so half the sequential layers should run ~1.6-1.9x faster there; the 2048-window epochs gain less. Planned: epoch 60-107 min, pass 200-460 min (central 75 min / 260 min).

| step (per chain) | GPU (central, range) |
|---|---|
| uncut reference scoring (723 users) | 0.2 h |
| smoke + epoch 1 + scoring + early check | 1.5 h (1.2-2.1) |
| epochs 2-3 + scoring + early check | 2.7 h (2.2-3.8) |
| cool-down pass 1 + pass 2 (+ 2 smokes) | 8.7 h (6.7-15.4) |
| final scoring (1-2 formats) | 0.4 h |
| **per chain** | **~13.5 h (10.5-21.5)**; ~1.7 h if the first early stop fires, ~4.4 h if the second does |

**Both chains: about 27 GPU h (21-43)**, plus < 1 CPU h (CPU proofs ~10-20 min, timing ~5 min). Breaks: daily 20:00-01:00 (Sunday from 16:00); every training folder snapshot-stops before a break and resumes after it (snapshots every 400 users); scoring tickets start only when they fit.

| stage | central | range |
|---|---|---|
| CPU proofs (beside SM129) | 01:20 Thu 19 Nov | |
| SM129 done, SM130 GPU starts | ~02:45 Thu 19 Nov | 02:15-03:15 |
| two-head early check 1 (after epoch 30) | ~04:30 Thu 19 Nov | 04:00-05:30 |
| two-head early check 2 (after epoch 32) | ~07:15 Thu 19 Nov | 06:15-09:15 |
| **two-head result** | **~16:30 Thursday 19 November 2026** | 13:30 Thu - 02:30 Fri 20 Nov (a slow cool-down crosses the Thursday break) |
| Instant-Only early checks | ~18:30 Thu / ~02:15 Fri 20 Nov (epoch 41 pauses at the Thursday break) | |
| **Instant-Only result** | **~11:45 Friday 20 November 2026** | 07:00 Fri - ~10:00 Sat 21 Nov |

Both results land before the RC checks (~11-25) for the Saturday 11-28 release decision; anything here is held for that decision (no release, tag, fork or Hugging Face change).

## Risks

1. **The cut costs more than the budget.** The SM15 transfer is +0.56% curve / +0.61% / +0.62% vs Small (estimate); the curve head may degrade more (user context removed) and CIs are ~±0.06-0.1% on curve. The early stops cap a hopeless chain at ~1.7-4.4 GPU h.
2. **Heavy users and long histories:** the rule pool is the 722 users; the 3 long users of the 732 set are not scored (as in every published pair).
3. **Recipe interaction:** three constant-LR epochs at 2048 windows after a cut this large may under-recover before the long cool-down; the second early check sits exactly there.
4. **Interruptions:** every step resumes (folders from their last snapshot, finished steps skipped). Detached launchers die at some session restarts: check after restarts and relaunch from HANDOFF.md.
5. **Read-only inputs outside this worktree** (none may be cleaned up while this runs): `wt/sm106` and `wt/sm101` (start epochs), `wt/sm124` and `wt/sm115` (reference pass-1 folders, final models for the timing), `wt/sm118`, `wt/sm113`, `wt/sm91s2`, `wt/sm99`, `wt/sm72s`, `wt/sm55`, the KD replay caches `big-leakfree-1ep-final8k-neighbour`, `sm91s2-big-showtime-final8k-neighbour-e2s`, `sm99-big-inst-final8k-neighbour-e2s`.
6. **Memory:** GPU cap 0.8 (scorers 0.92 only for a retry without progress); the training process trees reach ~11 GB RSS (as SM106 / SM124, unchanged trainer); CPU-only steps run under `memguard.ps1` (kill above 5.9 GB).

## What is built

`sm130_cut.py` (cut, quick, cpu-check), `sm130_epoch.py` (epoch prepare / pair, both chains; selfcheck on SM106's and SM101's real epochs passes), `sm130_prepare.py` (SM124's prepare, names only), `sm130_pair.py` (cool-down pairs; selfcheck passes), `sm130_rule.py` (early, decide, selftest), `sm130_timing.py`, `sm130_block.py` (byte copy of SM124's), `memguard.ps1`, `extra-options-th.json` (= SM124's), `extra-options-io.json` (= SM115's), `launch-sm130.sh` (dry run OK). Prepared (local): `sm130-th-e30-st`, `sm130-io-e39-st` (pair OK).

Branch `sm130` is cut from `origin/sm124` (7cbf3d6f: the trainer tree of SM115 / SM118 / SM124; main lacks the chain's trainer code, as for SM106 and SM124).
