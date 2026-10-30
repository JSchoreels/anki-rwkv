# Data augmentation (SM67): design, 2026-10-29

Goal: make the leak-free small model (SM72s) more accurate with more training data, without waiting for new users
(more data (SM61) waits for the 100k-user dataset). The buy-back target is the same as for the better teacher (SM65):
the smaller model (SM73) needs about 0.1-0.3 points bought back before a narrower network can fit the bar of +0.5%
curve / +0.4% instant vs the big model (1 leak-free epoch) on the 732 set (722 users). The bar stays fixed.

**Bottom line.** Only one augmentation keeps every input and every label true for this model: **splitting a user into
two pseudo-users along a random half of its notes**. Each card keeps its whole review sequence: its intervals,
ratings, durations, states and labels are bit-identical (25 training users checked on CPU, 1.31M reviews). Only the
cross-card context changes: the "reviews today" and "other reviews since" inputs, and what the deck, preset and user
streams see. The other candidates either do nothing (time shift, prefixes) or teach the model a wrong forgetting curve
(review drop, interval scaling, start crop). Teacher targets: **keep the original ones**. The big model's
full-history prediction is still an unbiased target for a student that sees half the context, and it is free.
Recomputing costs 4.0 GPU h per augmented copy. Stage 1 = **the 1-epoch cool-down from the same start checkpoint as
the leak-free small model's (SM72s) last cool-down, with half of the large users split, ~9 GPU h**. The control is
that cool-down, which already exists (free). The split leaves every card's own sequence unchanged, so the expected
gain is modest: **~0-0.05 points per head**. Stage 1 is built to stop cheaply, and its output is a release candidate
if it wins. Code: ~400 lines on a new branch `sm67` cut from `sm72s` (CPU only, ~0.5 day of agent work). Stage 1 is
implemented and dry-run checked (section 8, 10-30); not launched.

Side finding (section 1, (0)): the training inputs are **bit-identical in every epoch**. `training_seed` is 2025 in
every epoch. Each window's seed is `training_seed + user + offset`, so the random id codes, the day baseline and the
dropout masks repeat every epoch. Only the user order changes. A per-epoch re-seed is a free augmentation of its own.
It is listed as a stage-2 arm.

## 1. Which augmentations fit this data and model

What the model reads (`rwkv/data_processing.get_rwkv_data` on the leak-free frame, `RWKV_DELETED_CARDS=neighbour`):
- **Per card:** elapsed days / seconds since the card's last review, plus their cumulative and sin/cos versions;
  duration; rating one-hot; state; the three deleted-card flags, which are 0 in leak-free inputs.
- **Cross-card context:** `diff_new_cards`, `diff_reviews` (other reviews since this card's last one),
  `cum_new_cards_today`, `cum_reviews_today`, `day_offset_diff` (days since the user's previous review),
  `day_of_week` (relative to the section start).
- **Streams:** card, note, deck, preset and user.
- **Labels:** the card's next review (`label_y`, `label_elapsed_*`). The curve head is trained on (elapsed time,
  outcome) pairs. The instant head is trained on the query row at the next review's time.
- **Absolute time:** the dataset has none. `day_offset` is re-based to the section start, and there is no clock time
  of day.

Measured columns come from `augment_check.py` (output `augment-check.json`, section 6).

| | Augmentation | What it does to the inputs | Labels | Verdict |
|---|---|---|---|---|
| (0) | **Per-epoch re-seed** (new id codes, day baseline and dropout masks per epoch) | none to the data; new random codes per entity | unchanged | **valid, free**; stage-2 arm |
| **(1)** | **Note split**: each user -> 2 pseudo-users, each note (or each deleted card, which has a note of its own) in one half by a hash | kept cards: **own inputs, ids and labels identical on 100% of rows**. Changed: `cum_reviews_today` on 98.7% of rows (mean \|Δ\| 0.49 sd, about a halving), `diff_reviews` 83.5% (0.21 sd), `diff_new_cards` 69.2% (0.21), `cum_new_cards_today` 56.5% (0.22), `day_of_week` 8.3% (a half can start on another day), `day_offset_diff` 0.6%. The deck / preset / user streams see half the reviews | **all kept**: the halves' labelled rows sum to the user's exactly (25 / 25 users) | **recommended** |
| (2) | Deck split (users studying a subset of their decks) | like (1), but deck and preset streams are intact for the kept decks | all kept | valid, but lopsided: 11 of 25 users have at most one deck with >= 10% of their reviews (median 16 decks); variant for later |
| (3) | Note subsample (keep a random share p of notes, one copy) | like (1) | drops (1 - p) of the labels | valid only across many epochs; in a 1-epoch test it is pure data loss |
| (4) | Time shift (day_offset + k) | **none**: every input column identical (25 / 25 users). `day_offset` is re-based to the section start and there is no absolute time | unchanged | no-op |
| (5) | Prefix copies (history cut at an earlier review) | **none** on the kept rows: the model is causal and every input row is identical. Only the last review of each card loses its label (7.7% of rows at a 60% cut) | fewer | no new information |
| (6) | Review drop (10% of reviews at random) | 8.5% of the remaining labelled rows get a longer interval input. Recomputing it is possible (`elapsed_days` = the card's day gap on 100% of rows) | 8.5% of the labelled rows get another label, the review after the dropped one | **breaks the curve**: the outcome after the hidden review was produced by a memory that includes it. The model would learn "recall after longer gaps is higher" |
| (7) | Interval scaling / jitter (elapsed x k) | all interval inputs, and the label times of the curve head | the label time moves, the outcome does not | **breaks the curve** (label noise on the very axis the curve head models) |
| (8) | Start crop (drop the user's first 30% of reviews) | **27.4%** of the visible cards start with a non-new review. In the dataset that never happens (0.0% of cards; every card starts with a new-card review) | kept, but on out-of-distribution inputs | out of distribution: rejected |
| (9) | User mixing (two users merged into one) | context of two learners in one user stream | kept | unrealistic users: rejected |

Why (1) is the only data augmentation that survives:
- A half of a real collection is itself a plausible collection. A user who never added those notes would see exactly
  these per-card sequences. The kept cards' outcomes are real.
- **Leaks.** Nothing from the future enters the inputs. Note membership is static metadata, and the split hash
  reads no review. Deleted cards are split by card id, with the same 50/50 odds as notes, so the share of
  deleted-card reviews stays the same in expectation. The neighbour transform takes its deck and preset from the
  full cards table, so it is unchanged. A split that favoured or avoided deleted cards would bring the deletion leak
  back as a selection signal; the hash rules that out.
- **The one real bias.** The outcomes of the kept cards happened under the user's full workload. The halves show
  about half the daily load (`cum_reviews_today` -0.49 sd). Splitting only part of the users each epoch keeps the
  real load relation in most of the data (stage 1: about 38% of the reviews in split form).
- **The pseudo-users keep the real user's id-code seed.** Every card, note, deck and preset then gets the same random
  code as in the whole user. The stage-1 arm differs from its control only in which reviews share a sequence; it
  is not also a re-seed.
- **Size rule.** Split only users with >= 65,536 reviews (2,355 of the 8,000 users, 76.9% of the reviews), so that
  every half has at least one full 32k window. Splitting small users adds cold starts and partial windows and
  gives little.
  - Updates per epoch: 22,079 plain; 23,457 (+6%) with every eligible user split; 27,578 (+25%) with all users split.
  - Pseudo-user starts (zero state): 8,000 plain, 10,355 with every eligible user split.
  - Share of reviews in windows that start from zero state (2,048 x 16 lanes; the same figure for the 32k
    cool-down's first window): 32.6% plain, 45.8% with every eligible user split. This is the price of the split.

How much it can give. Real new users are the strongest lever we know: 5k -> 8k users gave 0.16 / 0.09 points in the
decay only (the 8k model (SM55)) and about 0.3 / 0.15 with a full retrain (the full retrain (SM56)). Data diversity
beat more passes by +0.77% / +0.55% (data diversity vs epochs (SM43)). A split carries far less new information than
a new user: the card stream sees the same sequences. The card stream holds most of the gap, since 72% of the curve gap
is on learning-step reviews (the smaller model's design (SM73)). The split only varies the context inputs and the
context streams. Expected gain: **~0-0.05 points per head**, and the sign is not certain (the load bias).

## 2. Teacher targets

The KD targets (`ReplayTargets`, `distillation.kd_source = replay`) are the big model's (1 leak-free epoch)
predictions with the full history. Each labelled row stores the curve logit of the ahead row and the query row's 4
rating probabilities, indexed by the original `label_review_th`. Cool-down weight: `decay_alpha` 0.5.

| Option | Cost | Validity | Verdict |
|---|---|---|---|
| Recompute on each augmented copy | 4.0 GPU h per copy of the 8,000 users (the stage-2 export of the better teacher (SM65)); per-epoch re-drawn splits would cost 4 h per epoch, more than the epoch itself (1.8 h); 5.8 GB per copy | exact for the half's context | too expensive for what it changes |
| Labels only on augmented rows | 0 | valid | drops KD on ~38% of the reviews: changes the loss mix, so the arm would differ from its control in two ways |
| **Keep the original targets** | **0** (one index remap) | **valid**: privileged-information distillation | **recommended** |

Why the original targets are right, not just cheap:
- The student with half the context minimises E[CE(q_full, p) | half context]. With soft targets that minimum is
  p* = E[q_full | half context] = E[y | half context] (tower rule): the same optimum as with labels, at lower
  variance.
- The teacher's target uses only the past (other cards' earlier reviews), never the card's future, so it is not a
  label leak. What the student cannot see raises its KD loss floor a little; it does not bias the optimum.

Mechanics: get_rwkv_data renumbers `review_th` for the half (the context features need it), so the replay lookup goes
through the half's kept original review numbers (`kept[label_review_th - 1]`). These are recomputed from the
deterministic hash and cached per user. The CPU check (code step 7) asserts the remapped targets equal the direct
lookup for every labelled row.

A CPU proxy for the size of the context effect is part of the pre-flight. On 5 training users, score the released
small model on CPU on the whole user vs its two halves, same labels (~20 min CPU). If the halves cost almost nothing,
the split is a mild perturbation, and the original targets are close to exact.

## 3. Held-out safety

- **Eligible users.** Augmentation applies only to the rows of the cool-down's training list, the 8,000 final-tier
  training users (`wt/sm55/.../sm55-final8k-run.json`).
- **Disjointness check.** `prepare` asserts the real user ids are disjoint from:
  - the 8k split's 1,500 validation users, which contain the 732 and 464 sets;
  - its 500 + 100 final-test users;
  - the 732 and 464 fit lists.
  It writes the check into decay.json. Measured today: 0 overlaps against the 2,000 held-out ids
  (`augment-check.json`).
- **The 2,225 set.** Its 1,493 extra users are final-tier training users. It is already invalid for any final-tier
  model, so it is never used for a data augmentation (SM67) decision.
- **Pseudo-user ids.** They are `10,000,000 + 4 x user + part`, so they never collide with a real id (< 10,001). The
  decoder refuses any id whose user is not in the training list.
- **The scorer never augments.** The option lives in the trainer's options (`"augment": {...}`), not in an
  environment variable. `get_rwkv_data(..., keep=None)` is the default and stays bit-identical (checked).
  `small_model_heldout.py` has no code path that reads it.
- **Teacher targets.** They come from a teacher that never saw the 732, 464 or final-test users.

## 4. Recommended path

### Stage 1: augmented cool-down vs the existing one, matched compute: ~9 GPU h

**Start.** The same constant-LR checkpoint as the cool-down that the leak-free small model's (SM72s) run releases or
keeps: epoch 25, or epoch 20 if the epoch-25 cool-down loses. The control is that cool-down, already trained and
scored on the 732 set (free).

**Arm.** The same 1-epoch 32k Muon cool-down (LR 7e-4 -> 0). Options are identical (`$EXTRA` of `launch-sm72s.sh`,
same replay targets, same user order, `RWKV_DELETED_CARDS=neighbour`, cap 0.8) plus
`"augment": {"kind": "note_split", "min_reviews": 65536, "share": 0.5, "seed": 67}`:
- 1,178 of the 2,355 eligible users (a hash of seed and user) are split into two halves; the rest stay whole.
- **Matched compute.** The same 581.9M reviews, the same labels and the same targets; the updates are +3%
  (22,079 -> ~22,770).
- The pseudo-users keep the real user's id-code seed.

The arm tests one thing: does the cool-down, where the fit is made, gain from seeing the same reviews in varied
context? The constant-LR read-outs cannot answer it: they swing by 0.27 points (curve) from one epoch to the next,
while the cool-down turns +1.73% at epoch 20 into +0.46% (464 set). If a context-diversity effect is large enough to
matter, it has to show up here.

Resolution:
- Cool-down pairs from one start gave ±0.0001 curve / ±0.00003 instant (e20 vs e15), i.e. ±0.035 / ±0.01 points.
- The bootstrap does not include run-to-run training noise. Same start, same order and same codes keep that noise
  small, but a result right at a threshold is read as grey.

GPU:
- Cool-down: 581.9M reviews at ~20.2k reviews/s ≈ 8.0 h, +3% updates, plus setup and smoke: **~8.7 h**
  (`MIN_DECAY` 520 min, segmented at breaks as in the resumable cool-down (SM69)).
- 732 scoring: 723 fit users + the 3 long users at the 0.92 cap, **15-18 min**.
- Total **~9 h**.
- Memory: unchanged (cap 0.8 of 16.3 GB). The pseudo-users are smaller than their source users, and the peak is set
  by the largest whole user, as today.
- CPU side: split users are read twice (one parquet read per half). The cool-down is GPU-bound at 20k reviews/s.

**When.** After the better teacher's (SM65) stage 1, which decides at ~04:00 Sat 10-31. If the better teacher (SM65)
goes on to its stages 2-3 (12.5 h), it keeps the GPU first, because its expected gain is larger, and data
augmentation (SM67) follows.
- Directly after the better teacher's stage 1: ~04:00 -> ~13:00 Sat.
- After its stages 2-3: Sun 10-31 -> 11-01. Mind the Sunday break from 16:00; the segmented cool-down pauses at
  breaks and resumes.

**Decision rule (pre-registered).**
- **Users:** the 722 of the 732 set that every series of the leak-free small model (SM72s) has. Reference: the big model (1 leak-free epoch),
  0.29897 / 0.27576.
- **Comparison:** augmented minus plain cool-down, paired user bootstrap, 2,000 resamples, seed 0, review-weighted,
  95% CI. In points of % vs the big model (1 leak-free epoch): 0.20 points ≈ 0.00060 curve, 0.15 ≈ 0.00041 instant,
  0.10 ≈ 0.00030 / 0.00028.
- **Go** (stage 2) if curve <= -0.20 points or instant <= -0.15 points, with that head's CI clear of zero, and the
  other head's point estimate <= +0.03 points.
- **Stop data augmentation (SM67)** if both heads gain less than 0.10 points. The split changes too little to buy
  back what the smaller model (SM73) needs, and no other augmentation is valid.
- **Grey zone** (between those): report and ask. Stage 2 costs ~18 h.
- **Ship check (independent of go):** the arm is a full release-recipe model. If both point estimates are better and
  at least one CI is clear of zero, it replaces the release, after the fork's default-state check (int4 cards +
  int8 notes, ~13 min, as in the smaller model (SM73)).

### Stage 2 (only on go): augmentation in the constant-LR epochs: ~18 GPU h

From the epoch-20 checkpoint, 5 constant-LR epochs (21'-25') follow, then the cool-down from 25' on plain data.
- **Data:** each epoch re-draws the split (seed + epoch) and re-seeds the codes and dropout (arm (0)).
- **Control:** the existing chain (epochs 21-25 + the cool-down from 25), free.
- **Cost:** 5 x ~1.8 h (92k reviews/s, 106 min per epoch) + ~8.7 h + scoring ≈ **18 h**.
- If it wins, a re-seed-only arm (~18 h) tells which part did the work. The re-seed alone is the simpler recipe.
- If the full-size small model reaches <= +0.25% / +0.20%, the smaller model's (SM73) stage 2 is unlocked.

## 5. Code changes (new worktree `wt/sm67`, branch `sm67` cut from `sm72s`; CPU only)

Never edit `wt/sm72s` or `wt/sm65` while their GPU tickets run.

| File | Change | Size |
|---|---|---:|
| `rwkv/data_processing.py` | `get_rwkv_data(..., keep=None)`: optional boolean mask over the raw revlog rows, applied right after the parquet read and before every feature; adds `orig_review_th`. `None` = today's frame, bit-identical | ~10 |
| `scripts/augment.py` (new) | spec parse / validate; note-split hash (seed, user, note; deleted cards by card); pseudo-row expansion (`user_id` = pseudo id, `source_user`, `part`, `available_reviews`, `train_range`, `id_seed_user`); id encode / decode; cached `kept_review_ths` | ~150 |
| `scripts/gpu_batch1.py` (`frame_stream`), `scripts/gpu_preprocessing_stage.py` (`worker_history`) | load `source_user` with the half's `keep` mask | ~15 |
| `scripts/gpu_r01_prefetch.py` | item seed from `id_seed_user` (the real user), so codes and day baseline match the whole user | ~5 |
| `scripts/expertium_options.py` (`ReplayTargets`) | decode the pseudo id -> the real user's npz; remap `label_review_th` through `kept` | ~25 |
| `scripts/expertium_decay.py prepare` | `"augment"` in the options: expand rows, record the spec, the counts and the disjointness check in decay.json; the LR schedule from the expanded rows | ~30 |
| `scripts/check_augment.py` (new) | (a) no `augment` -> frames and batches bit-identical; (b) 3 users: kept-card inputs and labels identical, labels conserved; (c) remapped replay targets = the direct lookup on every labelled row; (d) held-out ids refused; (e) id codes identical between the whole user and its halves; (f) the CPU proxy of section 2 | ~150 |
| `artifacts/small-model/sm67/launch-sm67.sh`, `sm67_rule.py` | a copy of the leak-free small model's (SM72s) cool-down section and of `sm72s_rule.py`: segmented cool-down, 723 + 3 long-user scoring, `decide` with the rule above; DRYRUN, `HANDOFF-sm67.md`, hidden WMI launch | ~250 (mostly copied) |

Pre-flight before the ticket:
- the label-window check (`sm72_label_windows.py` on the expanded rows);
- a 20-user smoke cool-down (CPU part);
- `DRYRUN=1` of the launcher.

## 6. CPU sanity check (done 10-29, ~15 s, no GPU)

`augment_check.py` reads the dataset and the leak-free feature code of `wt/sm72s` (imported read-only, no bytecode
written). It samples 25 training users stratified by size (2k-250k reviews, 1,314,491 reviews) and writes
`augment-check.json`. The numbers quoted in sections 1 and 3 come from it:
- **Held-out:** 0 of 2,000 held-out ids in the training list.
- **Note split:**
  - all 25 users keep every label;
  - the smaller half holds 48.8% of the reviews on average;
  - kept cards' own inputs, ids and labels are unchanged; the 6 context columns above change.
- **Time shift (+3 days):** no input column changes for any user.
- **Prefix (60%):** only the label columns of each card's last kept review change.
- **Review drop (10%):** 8.5% of the remaining labelled rows change their interval input, and 8.5% change their label.
  `elapsed_days` equals the card's day gap on 100% of rows.
- **Start crop (30% of reviews):** 27.4% of the visible cards start with a non-new review, vs 0.0% naturally.
- **Decks:** 11 of 25 users have at most one deck with >= 10% of their reviews.
- **review_counts of all 8,000 users:** the update and cold-start counts of section 1.

## 7. Budget

| Stage | GPU | Decides |
|---|---:|---|
| Code + CPU checks + pre-flight | 0 | (no decision) |
| 1. Augmented cool-down vs the existing one + 732 score | ~9 h | whether context augmentation helps at all (>= 0.20 / 0.15 points); a release candidate if it wins |
| 2. Augmented constant-LR epochs (re-drawn splits + re-seed) + cool-down | ~18 h | whether augmentation in training buys the smaller model's (SM73) room |
| If stage 1 stops | ~9 h | the augmentation axis is closed (only the split is valid) |

## 8. Implementation (10-30, branch `sm67` c414bc83, worktree `wt/sm67`; CPU only, not launched)

Code (flag off = today's training, bit for bit):
- `scripts/augment.py` (new): spec check, pseudo ids `10,000,000 + 4 x user + part`, note hash
  (seed, user, note; a deleted card by its card id), selection, row expansion, the kept review numbers, the GPU-smoke
  folder.
- `rwkv/data_processing.get_rwkv_data(..., keep=None)`: the mask over the raw revlog rows, applied before every
  feature, adds `orig_review_th`.
- Loaders (`scripts/gpu_batch1.py`: `row_frame`, `seed_user`; `gpu_r01_prefetch.py`, `gpu_preprocessing_stage.py`, the
  trainer's primer):
  - a pseudo row reads its real user's half;
  - its item seed (id codes, day baseline, dropout) is the real user's;
  - pseudo rows need the option, on the plain ordered-prefetch path.
- `scripts/expertium_options.ReplayTargets`: a pseudo batch reads the real user's file. `label_review_th` is mapped
  through the half's kept original review numbers.
- `scripts/expertium_decay.py prepare`: `"augment"` in `--extra-options` expands the rows in place in the user order
  and records the counts in decay.json (`augment`, `training_rows`).
- `scripts/check_replay_label_windows.py`: pseudo rows checked on their half of the real user's file. The two halves'
  label counts must sum to the export's record.
- `scripts/check_augment.py` (new; `cpu`, `pair`), `scripts/sm67_rule.py` (new),
  `artifacts/small-model/sm67/launch-sm67.sh`, `HANDOFF-sm67.md` (exact hidden launch command).

One change from section 3: **the 2,225 pool's users are never split.** Its 1,493 extra users are final-tier training
users, so they stay in training whole, exactly as in the control. No pseudo-user is ever built from a user of any
evaluation set. That leaves 1,918 candidates (437 large users of the 2,225 pool excluded), of which **959 are split**:
- 8,959 rows, 187.1M reviews (32.2%) in halves (section 4 assumed 38%);
- smallest half 31,826 reviews (13 halves under 32,768);
- 32k updates 22,079 -> 22,642 (+2.5%).

CPU checks, **ALL OK** (`check-augment-cpu.json` on the branch, 87 s):

| Check | Result |
|---|---|
| Flag off, loader | the frames of 2 users are bitwise equal to the `wt/sm72s` tree |
| Flag off, trainer path (the cool-down options, replay KD, lanes carry, epoch-25 weights, 8 updates) | prepared batches, teacher targets, loss, outputs and every gradient bitwise equal to the `wt/sm72s` tree |
| Option on, users that are not split | the same bits as the control |
| 3 split users (65,560 / 115,853 / 248,965 reviews) | kept cards' own inputs, ids and labels identical on 100% of rows; each review in exactly one half; labelled reviews conserved; label review numbers map back to the whole user's |
| Context columns (share of rows changed) | `cum_reviews_today` 97-100%, `diff_reviews` 82-89%, `diff_new_cards` 62-71%, `cum_new_cards_today` 45-71%, `day_offset_diff` 0.2-1.3% |
| Teacher targets | trainer path: the remapped targets equal the direct lookup of the real user's file on every labelled row; frame level: the index remap is exact on every labelled row of the 3 users |
| Seeds and id codes | pseudo item seeds are the real user's; a half's id codes equal the whole user's |
| Held-out | 0 training rows in the 464 / 732 / validation / final-test sets; 0 pseudo-users from any evaluation user; a held-out row is refused; the scorer never reads `augment` |

Dry run of the launcher (START_EPOCH=25) **OK**:
- the real prepare, the label-window trims (only user 5599, as in the control) and the Muon fresh-state check;
- the pair check against the leak-free small model's (SM72s) epoch-25 cool-down: identical options except
  `augment`; the same start, schedule, users, order and reviews; labelled reviews 505,739,554 in both.

Launcher steps:
1. Wait for the tiny architecture additions' (SM77) launcher to end (its log's last line; `gpu/NO-WAIT` skips it).
2. CPU prepare and checks.
3. Ticket 0, ~15 min: GPU smoke of the pseudo-user path (the two halves of one split user).
4. The cool-down in segments at the breaks (quiet.json plus the daily 21:00-02:00).
5. 732 scoring: 723 users plus the 3 long users at the 0.92 cap.
6. Wait for the control's 732 rows, then `sm67_rule.py decide` with the rule of section 4 (722 users).

`START_EPOCH` (default 25) picks the start; the control is the leak-free small model's (SM72s) existing cool-down from
that epoch. The rule script reproduces the leak-free small model's (SM72s) summary when given its rows (the e20
cool-down vs the e15 one: 0.30019 / 0.30038, +0.41% / +0.47% vs the big model (1 leak-free epoch)).

**GPU for stage 1: ~8.7 h** (smoke 0.2 h, cool-down ~8.2 h at ~20.2k reviews/s with +2.5% updates, scoring 0.3 h).
The CPU proxy (section 2: the released model on whole users vs their halves) was not run. It does not gate the
run.
