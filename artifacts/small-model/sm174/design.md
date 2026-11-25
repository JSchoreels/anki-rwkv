# How many users for MIT weights (SM174): design

Written Thu 2026-11-26 ~01:00, before any GPU ticket (pre-registered; only CPU checks ran: `sm174_chain.py check`
ALL OK, `sm174_report.py selftest` OK). Branch `sm174` (from `origin/main`: design, chain, report, launcher, hand-off),
worktree `C:/Users/jonat/workspace/wt/sm174`. Run tree: `C:/Users/jonat/workspace/wt/sm174-run`, a detached worktree
at `cea0f931` (`origin/sm148` head, the trainer and scorer tree that the state-energy penalty (SM171) also trained and
scored with; it has the six-layer preset, which `main` lacks), **not edited**. Track: small model.

## Question

The goal is RWKV weights under the MIT license, trained only on review histories donated to us (donation campaign:
14 donors, about 3.15M reviews so far). Those weights must be trained **from scratch with no teacher**, because every
teacher we use (the srs-benchmark RWKV, our big model) was trained on the anki-revlogs-10k dataset, whose license blocks
redistribution.

**Question: how many users does a from-scratch, no-teacher six-layer Tiny need to come within +1% and +0.5% LogLoss
of the six-layer Tiny two-head release model (SM130), on both heads?** That number sets the campaign goal.

The anki-revlogs-10k users stand in for donated users here. **The weights of this test are experiment-only and are
never shipped** (they are trained on the restricted dataset).

| plain name (ID) | what it is here |
|---|---|
| the six-layer Tiny two-head release model (SM130) | the reference: `w64-c1-d2p1u1`, 185,057 parameters, `sm130-th-dec2ep32k-p2` (two-pass 32k cool-down), its SM148 rows on the 732 set (`w8+s8hx8`, show-time inputs, scorer tree cea0f931) |
| the leak-free small model run (SM72s) | the 8,000 training users, their review counts and the per-epoch recipe of the chain the release model comes from |
| the big model (1 leak-free epoch) (SM72) | the usual constant-LR replay teacher (alpha 0.9, decay_alpha 0.5), used only by the control arm |
| data diversity vs epochs (SM43) | the earlier from-scratch test (12-layer w64 with a teacher): 125 users x 110 epochs was +0.77% curve / +0.55% instant worse than 1,250 users x 11 epochs at equal reviews; it is the only earlier evidence on user counts, and it had a teacher |
| this test's arms (SM174 n0500 ... n8000, n8000-kd) | "500 users, no teacher (SM174 n0500)" etc.; "8,000 users, with the usual teacher (SM174 n8000-kd)" is the control |

No existing model serves as a no-teacher reference: the old screen-tier w64 runs without a teacher are another
architecture and an older recipe. The release model (SM130) cannot serve as the teacher control either, because it
had far more training (about 30 epochs of the 8,000 users plus a 32k-window cool-down) than any arm here; the
control is trained under exactly the arms' budget.

## Data

`users.json` (`sm174_chain.py init`, deterministic, checked by `check`):
- **Nested random subsets** of the SM72s run's 8,000 training users: sorted ids shuffled with `random.Random(174)`,
  the first 500 / 1,000 / 2,000 / 4,000 / 8,000. Reviews per epoch: 38,746,390 / 75,792,571 / 146,143,476 /
  288,404,346 / 581,942,724 (user 5599's label-window trim of the template, 14 reviews, applies when it is in).
- **Held-out slice for early stopping**: 190 validation users of the same split with id >= 5000 (never trained on by
  any arm, not in the 732 set), at most 120,000 reviews each, 6,002,062 reviews. It is used only to pick the stopping
  epoch, never for the result.
- **Scoring**: the 732 set (the 723 users of `eval-users-732-fit.json`), disjoint from every training subset.

## Recipe (every arm)

Everything is the six-layer Tiny two-head (SM130) epoch-30 folder (`template.json`, copied once from
`wt/sm130/artifacts/small-model/sm130-th-e30-st`, sha256 pinned) except what is listed here:
- **Fresh init** (`CatchUp.fresh_checkpoint`, training_seed 2025), update offset 0, so the R01 5-update warmup to 7e-4.
- **No teacher**: the `distillation` option removed; the loss is the trainer's plain two-head LogLoss on the real
  outcomes (curve + instant). The control arm n8000-kd keeps the template's distillation unchanged (replay targets of
  the big model (1 leak-free epoch), alpha 0.9; decay_alpha 0.5 during the decay, as every scheduled run does).
- Kept from the template: `w64-c1-d2p1u1`, show-time inputs (`show_time_intervals`), leak-free
  (`RWKV_DELETED_CARDS=neighbour`), 2048-review windows x 16 strided state-carry lanes, AdamW 7e-4, gpu fraction 0.8,
  snapshots every 400 users.
- Rows: the arm's subset, order `random.Random(174000 + epoch)` per epoch; decay segments continue the epoch index.
- One folder per epoch, trained by the run tree's `scripts/expertium_decay.py run` (resumable from snapshots).

**What is not reproduced:** the release recipe's 32k-window two-pass cool-down with the show-time big model's
targets. The decay here is a 2048-window one_minus_sin decay to 0. The same for every arm, including the control.

## Steps / epochs rule (fixed now; `sm174_chain.py`, `plan`)

- **Review budget B = 4 x 581,942,724 = 2.33B reviews per arm** (four 8k epochs), so smaller sets get more epochs:
  E = round(B / R_n) epochs, R_n the subset's reviews per epoch.
- Decay share: D_max = max(1, round(0.15 E)) decay epochs; constant-LR epochs at most C_max = E - D_max.
- **Early stopping on the held-out slice**: score the slice every k = max(1, round(290M / R_n)) constant epochs and at
  C_max. Slice loss = mean of the review-weighted curve and instant LogLoss. The constant phase stops at the first eval
  point whose loss is not at least 0.05% below the best earlier point, or at C_max.
- **Decay** from the best eval point b (its weights, AdamW state and update offset): D = max(1, round(D_max b / C_max))
  epochs, one_minus_sin 7e-4 -> 0 over exactly D x R_n reviews (chained one-epoch segments, `lr_resume` segment).
  The last segment is the arm's model.

| arm | users | epochs E (C_max + D_max) | slice every | eval points | training (upper bound) |
|---|---:|---|---:|---|---:|
| 500 users, no teacher (SM174 n0500) | 500 | 60 (51 + 9) | 7 | 7, 14, ..., 49, 51 | ~4.1 h |
| 1,000 users, no teacher (SM174 n1000) | 1,000 | 31 (26 + 5) | 4 | 4, 8, ..., 24, 26 | ~3.9 h |
| 2,000 users, no teacher (SM174 n2000) | 2,000 | 16 (14 + 2) | 2 | 2, 4, ..., 14 | ~3.8 h |
| 4,000 users, no teacher (SM174 n4000) | 4,000 | 8 (7 + 1) | 1 | 1 ... 7 | ~3.7 h |
| 8,000 users, no teacher (SM174 n8000) | 8,000 | 4 (3 + 1) | 1 | 1, 2, 3 | ~3.7 h |
| 8,000 users, with the usual teacher (SM174 n8000-kd) | 8,000 | 4 (3 + 1) | 1 | 1, 2, 3 | ~3.7 h |

Speed basis: the six-layer Tiny's 2048-window epochs ran at ~178k reviews/s (SM130 epochs 30-31, 54.5 min per 8k
epoch), plus ~30 s per folder. Order: smallest first, so partial results come early; the control runs last.

**Why 8,000 and a control at 8,000:** both fit the budget (about 27 GPU h in all). Small (SM118 architecture) is
not added: at about half Tiny's speed it would not fit the 1-2 day budget; it is a follow-up if the curve asks for it.

## Scoring

Each arm's final model, the 732 set (723 users), show-time inputs (`--inputs e2s-query --split-existing
--deleted-cards neighbour --nonfinite-rows --prior-buckets ...`, exactly SM171's flags), `w8+s8hx8`, GPU cap 0.8
(0.92 only for a retry pass without progress), rows `gpu/sm174-rows-732.jsonl`. An integrity ticket first scores the
release model on SM148's 3 smoke users: the rows must equal SM148's (max |LogLoss difference| <= 1e-6), so the
reference rows and ours come from the same scorer.

## Read-out (`sm174_report.py report`, after each arm and at the end)

- Per arm and head: % = 100 x (review-weighted LogLoss / the release model's (SM130) - 1), paired user bootstrap
  2,000 draws, numpy default_rng(0), 95% CI, on the users both have (at least 720). Gap = the worse head.
- **Users needed** for +1.0% and +0.5%: interpolation of the gap in log2(users) over the no-teacher arms, first
  crossing from above, on the point estimate and on the CI upper bound. Not reached at 8,000: an indicative power-law
  fit gap(n) = a + b n^-c; its floor a is the cost of the budget and recipe, not of the data.
- **Decomposition** (same bootstrap): teacher cost at equal compute = 8,000 no teacher vs 8,000 with the teacher;
  budget and recipe cost = 8,000 with the teacher vs the release model (SM130); data cost = each arm vs 8,000 no
  teacher. A second "users needed" column uses the control (equal compute) as the reference.
- Also reported: the slice curve per arm (eval point losses, the stop reason), epochs, reviews, GPU hours, reviews/s.
- No go / no-go verdict: this measures a curve. Its use is the campaign goal.

## Budget and ETA

| step | GPU |
|---|---:|
| smoke (two arms) + integrity | ~0.2 h |
| six arms, training (upper bound, no early stop) | ~22.8 h |
| slice evals (35 x ~4 min) | ~2.3 h |
| 732 scoring (6 x ~15 min) | ~1.5 h |
| **total** | **~27 h (upper bound; early stops shorten it)** |

Start ~01:15 Thu 26 Nov. Planned arm ends (upper bound): n0500 ~06:30 Thu, n1000 ~11:15 Thu, n2000 ~15:50 Thu,
n4000 ~20:20 Thu, n8000 ~00:30 Fri 27 Nov, n8000-kd **~04:45 Fri 27 Nov** (the end).

## Risks

1. **The budget confound.** Every arm trains much less than the release model (SM130). If even the 8,000-user arms
   are above +1%, the "users needed" answer is "> 8,000 at this budget", and the decomposition tells how much of the gap
   is budget and recipe (the control) and how much is data and the missing teacher.
2. **Overfitting** of the small arms (60 epochs of 500 users): the slice decides when to stop; the slice is small
   (190 users), but it only picks the stopping point.
3. **Interruptions**: every folder resumes from its last snapshot; the launcher re-derives the next step from disk.
   Detached launchers die at some session restarts: check after restarts and relaunch from HANDOFF.md.
4. **Read-only inputs outside this worktree** (must not be cleaned up while this runs): `wt/sm130` (template, release
   checkpoint), `wt/sm72s` (run.json, label-db, the 732 users file), `wt/sm72` (teacher weights), the replay cache
   `anki-rwkv/artifacts/cache/kd-replay/big-leakfree-1ep-final8k-neighbour`.
5. **Memory**: GPU cap 0.8 (training ~12.5 GB reserved as SM130's epochs); CPU steps are seconds and small.

## What is built

`sm174_chain.py` (init, plan, next, slice, check), `sm174_report.py` (report, selftest), `launch-sm174.sh` (dry run
OK), `users.json`, `template.json`.
