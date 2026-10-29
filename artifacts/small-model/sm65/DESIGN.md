# Better teacher (SM65): design, 2026-10-29

Goal: make the leak-free small model (SM72s) more accurate by distilling from a better teacher than the big model
(1 leak-free epoch). The smaller-model design (SM73, `artifacts/small-model/sm73/DESIGN.md` section 4) needs about
0.1-0.3 points bought back before a narrower network can fit the bar (+0.5% curve / +0.4% instant vs the big model
(1 leak-free epoch), 732 set, 722 users). The bar stays fixed: a better teacher is not the new reference.

**Bottom line.** Of the cheap options, only one is likely to give a teacher gain of the right size: **the big model
trained on more users**. That means one more leak-free epoch over the 3,969 final-tier training users it has never
seen. Ensembles of what we have were measured on CPU today and give 0.03-0.09 points. More epochs on the big model's
own users should give less. Expected student gain: **~0.03-0.13 points per head (central ~0.05-0.07)**. That is the
lower half of the 0.1-0.2 hoped for, so stage 1 is built to stop early and cheaply.
Stage 1 = that teacher epoch + its 732-set score, **~7.3 GPU h**. It needs a small code change (user list and start
checkpoint for the big-model epoch script), so it is a plan only: no launcher yet.

## 1. Users: who trained on what

From `artifacts/full-run/run-8k-20261013/run.json` (the 8k split) and the big model's users 5000-10000:

| Set | Users | Reviews | Big model trained on it? |
|---|---:|---:|---|
| Big model's training users (5000-10000) | 5,001 | 370.3M | yes (4,031 final-tier train + 768 validation + 202 final-test users) |
| Student's 8,000 final-tier training users | 8,000 | 581.9M | 4,031 yes, **3,969 no** (ids < 5000, 285.9M reviews) |
| 732 set (validation users with id < 5000) and 464 set (a subset of it) | 732 / 464 | 49.2M | no: must stay so |
| Final-test users with id < 5000 (sealed) | 298 | | no: must stay so |
| 2,225 set (732 + 1,493 gate-unseen training users < 5000) | 2,225 | | already invalid for any final-tier model; also invalid for the new teacher |

So the **only users a teacher can still gain** are the 3,969 final-tier training users below 5000. They are
disjoint from the 732 / 464 sets and from the final test. After this lever there are no further users in the
10k dataset: every other user is either already in the big model or held out.

## 2. Candidates, per point gained

"Teacher gain" = vs the big model (1 leak-free epoch) on the 732 set, in points of %. Student gain assumes about 1/3
transfer (section 3). Stage costs are GPU hours at the measured rates: big-model epoch 11,460 reviews/s (the big model's leak-free epoch (SM72 step
1): 370.3M reviews in 558 min), teacher export 4.0 h for 8,000 users (10-26 05:57 -> 09:56), student cool-down 470-520
min, 732 scoring 15-18 min.

| | Teacher | Teacher GPU | Expected teacher gain (curve / instant) | Student GPU after | Expected student gain | Verdict |
|---|---|---:|---|---:|---|---|
| (a) | 2nd leak-free epoch on the same 5,001 users | 9.3 h | <= 0.05-0.10 / <= 0.03 | 12.5 h | <= 0.03 | worst per point |
| **(b1)** | **1 epoch on the 3,969 new users only**, from the big model (1 leak-free epoch) | **7.0 h** | **0.1-0.4 / 0.05-0.25** | 12.5 h | **0.03-0.13 / 0.02-0.15** | **recommended** |
| (b2) | 1 epoch on all 8,970 users (5,001 + 3,969) | 16 h | (b1) + a little | 12.5 h | about (b1) | fallback only (stage 4) |
| (c1) | ensemble: shipped big model (fork RWKV) + big model (1 leak-free epoch) | 0 (both target caches exist) | **measured: 0.03-0.09 / 0.06**, partly from the leak | 8.5 h | <= 0.03 | dropped (measured) |
| (c2) | ensemble: big model + the small model itself | - | n/a | - | - | not a teacher (see below) |
| (d1) | distil only on users the big model never trained on (SM05) | 0 (needs a mask in the code) | n/a | 8.5 h | unknown sign | idea, after stage 3 |
| (d2) | a new big teacher from scratch on 8,970 users | days | unknown | | | out of budget |

Reasoning:
- **(a) Was epoch 1 converged?** One leak-free epoch gained 0.23 / 0.16 points, but most of that undoes the leak:
  removing the leak inputs costs the shipped model +0.08 / +0.13 points. Against the shipped model on its own (leak)
  inputs, the epoch gained only 0.13 [-0.06, +0.29] / 0.06 [0.05, 0.07] points (the big model's leak-free epoch (SM72) results row). That part is one
  more low-LR pass with a cool-down over users the model already fits. A second pass should give at most half of it:
  <= 0.06 / 0.03 points of teacher, <= 0.03 for the student. It costs more than (b1).
- **(b) More users is the strongest lever we know.** On the student, 5k -> 8k users gave 0.16 / 0.09 points with the
  new users only in the decay (8k model (SM55)), and 0.3 / 0.15 in all with a full retrain (full retrain (SM56)). The
  big model has 8 times the parameters, so it should be at least as hungry for data. One epoch from fitted weights
  sees each new review once, hence the range 0.1-0.4 / 0.05-0.25.
  - **New users only (b1), not the union (b2).** The 5,001 old users are already fitted (the argument of (a)), so
    the new information is in the 3,969 new users. (b1) carries it at 43% of (b2)'s cost. Risk: drift on users >= 5000.
    It does not matter for the readout, since every held-out user is below 5000, the same population. Teacher targets
    on the 4,031 old users then come from a model whose last epoch did not see them, which is closer to the idea of distilling only on teacher-unseen users (SM05).
  - **In-sample targets.** After (b1) the 3,969 users are in-sample for the teacher. That is a single pass at a low LR
    (3.5e-4 -> 0), so it is close to out-of-sample prediction. The 4,031 old users have been in-sample (multi-epoch)
    since the shipped model.
- **(c1) Ensemble of what we have: free to test, and tested.** Both teachers' targets exist for all 8,000 training
  users (`kd-replay/fork-fp32-final8k`, leak inputs, and `kd-replay/big-leakfree-1ep-final8k-neighbour`). The
  CPU proxy `artifacts/small-model/sm65/teacher_proxy.py` (output `teacher-proxy.json`) uses 2 x 250 users per group,
  binary recall labels, and only the unseen users (out-of-sample for both models). The probability average beats the
  big model (1 leak-free epoch) by 0.00027 / 0.00008 curve (0.09 / 0.03 points) and 0.00015 instant (0.06 points)
  in both samples. Part of it is the shipped model's leak inputs, which a leak-free student cannot use. The two
  models are one low-LR epoch apart, hence highly correlated. At 1/3 transfer that is <= 0.03 points for the student:
  not worth an export or a cool-down. Scoring the pair on the 732 set would need per-review dumps (~1 h GPU) to
  confirm a number already too small, so it is not proposed.
- **(c2) Big + small ensemble.** The small model is a different architecture, so the pair could look good on the
  732 set. But on its own 8,000 training users the small model's predictions are in-sample after 20-25 epochs. A
  teacher built from them is self-distillation, and the 732-set ensemble score would not measure what it teaches.
- **(d)** Whole-curve distillation (SM59 step 2) stays shelved: step 1 (more teacher weight) did not help.
  Per-head weights (SM53) were dropped. Distilling only on teacher-unseen users (SM05) is the one teacher-side idea left.
  It needs a per-user mask and a cool-down pair (~8.5 h), and its sign is unknown. It is better tested after stage 3,
  on whichever targets win.

## 3. How much of a teacher gain reaches the student

The only past teacher change is the leak-free teacher: big model (1 leak-free epoch) vs the shipped one, +0.23 / +0.16
points. The paired comparison is the leak-free small model (SM72s) vs the full retrain (SM56) with leak inputs. Both
are cool-downs from epoch 15, on 722 users with leak-free scoring: -0.00010 [-0.00042, +0.00029] curve (0.03 points,
a tie) and -0.00027 [-0.00035, -0.00020] instant (0.10 points).
- That pair also changed the **student's own inputs** (trained leak-free). The big model's own leak-removal penalty
  was +0.08 / +0.13 points, so the student's input match alone could explain the whole instant gain. Transfer is
  therefore somewhere between 0 and ~0.6 on instant and unresolved on curve. The smaller-model design's (SM73) "a third to half" is
  the optimistic end.
- More teacher weight did not help (SM59: 0.9 and 1.0 worse than 0.5 by 0.00025-0.00032 curve). Early on,
  distillation was worth -0.0056 / -0.0051 (screens (SM02), 1 epoch, 2,000 users), but no measurement exists at
  convergence.
- Working figure: **~1/3 transfer (range 0-0.6)**. Buying 0.1 student points needs ~0.3 teacher points, and 0.2 needs
  ~0.6. Only (b) can plausibly reach that, so the stage-1 bar below is set at 0.2 / 0.15 points.

Stage 3 measures the transfer directly: the same cool-down with only the targets swapped.

## 4. Recommended path

### Stage 1: the teacher (big model, more users) and its 732-set score: ~7.3 GPU h

Code (new branch `sm65` cut from `sm72s`, which has `scripts/sm72_big_epoch.py`, the budgeted carry table, the
label-window pre-flight and the cool-down tooling; ~1-2 h agent work, CPU only):
1. `scripts/sm72_big_epoch.py prepare`: `--users-run <run.json> --users-below 5000` takes the final-tier training
   users with id < 5000 (3,969 users, 285,886,387 reviews). The script refuses any validation or final-test user of the
   8k split and writes the disjointness check into decay.json. `--init <checkpoint>` sets the start weights; the
   default is the shipped file, so behaviour is unchanged. Start = `wt/sm72/.../big-leakfree-1ep/master.pth`
   (sha256 `ac7f0d8a...`).
   Recipe exactly as the big model's leak-free epoch (SM72 step 1): carried state 2 lanes x 8,192, peak LR 3.5e-4 -> 0, warmup 200 updates, fresh AdamW,
   snapshots every 400 users, cap 0.8, `RWKV_DELETED_CARDS=neighbour`, shuffle seed 1.
2. `scripts/sm72_label_windows.py --trim` on the same user list (the label-window pre-flight).
3. CPU checks: with no new flags, `prepare` writes the leak-free epoch's (SM72) decay.json (minus timestamps). The init checkpoint loads bit
   for bit (as `check-big-load-sm72.json`). The user list is disjoint from the 732 / 464 / final-test users.
4. Launcher `artifacts/small-model/sm65/launch-sm65-teacher.sh`, a copy of `launch-sm72-big.sh`. It trains in segments
   that stop 2 min before a break and resume after it (`RATE=11400`). It then scores `big-moreusers/w8+s8` with leak-free
   inputs (`--split-existing`, `--deleted-cards neighbour`) on the 732 pool: 723 fit users, then the 3 long users at
   the 0.92 cap. It does **not export**: export is stage 2, after the decision. Then `scripts/sm65_rule.py decide`
   pairs it against `wt/sm72/.../gpu-big/bigleak-neighbour.jsonl`. DRYRUN, `HANDOFF-sm65.md`, hidden WMI launch.

GPU: 285.9M reviews / ~11,400 per s = **~7.0 h** + 15-18 min scoring. Queued behind the leak-free small model's (SM72s) run (epochs 21-25 and
the cool-down from epoch 25, done ~13:00-15:00 Fri 10-30). The epoch would run ~15:00-21:00 Fri, pause for the
break, and resume at 02:00 Sat 10-31, giving **a decision ~04:00 Sat 10-31**.

**Decision rule (pre-registered).** Users: the 722 of the 732 set that the leak-free epoch's (SM72) rows use (reference big model (1
leak-free epoch) 0.29897 / 0.27576). Paired user bootstrap, 2,000 resamples, seed 0, review-weighted, 95% CI.
- **Go** (stages 2-3) if curve <= -0.20% or instant <= -0.15%, with that head's CI clear of zero, and the other head's
  point estimate <= +0.03%.
- **Stop the better teacher (SM65)** if both heads gain less than 0.10 points. The expected student gain would then be < 0.04 points, and
  no cheap teacher is left (all remaining users are held out; ensembles measured at <= 0.09).
- **Grey zone** (between those): report and ask. Stages 2-3 cost 12.5 h for an expected ~0.03-0.05 points.

### Stage 2: teacher targets on the 8,000 training users: 4.0 GPU h

`scripts/sm11_replay_export.py export` with the new `master.pth` into a new folder
`artifacts/cache/kd-replay/big-moreusers-final8k-neighbour` (`RWKV_DELETED_CARDS=neighbour`, the same run file
`wt/sm55/.../sm55-final8k-run.json`, tickets cut at breaks with `--time-limit`, then `status` and the CPU `verify`).
Disk: 5.8 GB (2.0 TB free). No new code.

### Stage 3: paired student cool-down: ~8.5 GPU h

The 1-epoch 32k Muon cool-down of the leak-free small model (SM72s) from the **same start checkpoint as the cool-down
the SM72s run already has** (epoch 25, or epoch 20 if the epoch-25 cool-down loses). Options are identical except
`distillation.teacher` / `distillation.replay` -> the new ones. The existing cool-down from that epoch, with the old
targets, is the control: it costs nothing. The label-window check runs against the new replay (CPU 10-20 min).
Then the 732 score (723 + 3 long users) and the pair on 722 users. No new code: the leak-free small model's (SM72s) `expertium_decay.py prepare`
with a changed `$EXTRA`. Resolution: cool-down pairs gave ±0.0001 curve / ±0.00003 instant (e20 vs e15), i.e. ±0.035
/ ±0.01 points, enough for a 0.05-point student gain.
- **Ship** (replaces the release) if both heads' point estimates are better and at least one CI is clear of zero.
  Then the fork's default state check (int4 cards + int8 notes, ~13 min) as in SM73.
- **Stop the better teacher (SM65)** if both point estimates are above -0.00010 (a gain under 0.03 points). Keep the release.
- In between: a tie, no ship. Record the measured transfer ratio.

### Stage 4: only if stage 3 gains >= 0.1 points on a head

Either the union epoch for the teacher ((b2), 16 h + 4 h export + 8.5 h cool-down), or student constant-LR epochs on
the new targets before a cool-down (~2 h per epoch + 8.5 h). If the full-size small model reaches <= +0.25% /
+0.20%, the smaller model's stage 2 (card stream 1 layer, SM73) is unlocked.

## 5. Budget

| Stage | GPU | Decides |
|---|---:|---|
| 1. Teacher epoch on 3,969 new users + 732 score | ~7.3 h | whether a better teacher exists at all (>= 0.2 / 0.15 points) |
| 2. Teacher targets, 8,000 users | 4.0 h | (no decision) |
| 3. Student cool-down with new targets vs the existing one | ~8.5 h | ship or stop; measures the transfer |
| Total to a ship / stop answer | **~20 h** (~1.3 days with breaks) | |
| If stage 1 stops | 7.3 h | |
