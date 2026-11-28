# Tiny cool-down with session inputs (SM183): design

Written Sat 2026-11-28, before any GPU ticket (only the CPU proofs ran first). Branch `sm183` (from `origin/sm181`, `8d406426`), worktree `C:/Users/jonat/workspace/wt/sm183`. Run tree: `C:/Users/jonat/workspace/wt/sm179-run`, branch `sm179-run` = `cd220759` (the session-input feature code of SM179), reused unedited and clean as SM179 and SM181 did; it is not edited while this runs. Track: small model.

## Question

The session inputs screen (SM179) and its confirmation (SM181) measured, after a short fine-tune of about 2,550 updates, an instant LogLoss change of about -0.065% for six-layer Tiny two-head (SM130) (732 set and decisive users) and for Small two-head (SM118), curve unchanged. **Question: with the six session inputs trained through a full cool-down, the same length and recipe as the release cool-down of six-layer Tiny (SM130), how large is the gain, and does the result make a deployable candidate?** The full cool-down may learn more from the inputs than the short fine-tune did.

| plain name (ID) | what it is here |
|---|---|
| six-layer Tiny two-head (SM130) | release folder `wt/sm130/artifacts/small-model/sm130-th-dec2ep32k-p2` (`master.pth` sha256 48a5e724...); its cool-down started from recovery epoch 32 (`sm130-th-e32-st`, master sha256 635f1a12...) |
| six-layer Tiny Instant-Only (SM130) | release folder `sm130-io-dec2ep32k-p2`; cool-down from recovery epoch 41 (`sm130-io-e41-st`) |
| the session inputs screen (SM179) | the feature code (`SM179_INPUTS=session`, 6 columns, zero-initialised weights) and the scorer of the run tree |
| the session inputs confirmation (SM181) | the decisive user list (765 users with id >= 5000 of the 1,500 pool, a copy in `users-decisive.json`), the report statistic and the AUC code |

## The inputs (unchanged from SM179)

`sm179_gap`, `sm179_n10`, `sm179_n60`, `sm179_sess_min`, `sm179_sess_pos`, `sm179_sess_again` (`docs/SMALL_MODEL_INTEGRATION.md`, `wt/sm179/.../sm179/design.md` "Inputs"): reviews in the last 10 and 60 minutes, minutes since the session start (30-minute gap rule), position in the session, the session's recent Again share, and the gap since the previous review. All causal, standardised with SM179's means and standard deviations, inserted at input index 24.

## Recipe (SM130's release cool-down; only the start and the input width change)

SM130's cool-down (`wt/sm130/artifacts/small-model/sm130/design.md` "Recipe", `sm130_prepare.py`, `launch-sm130.sh`), per chain: one one_minus_sin decay 7e-4 -> 0 over 2 x 581,942,724 reviews, split at its middle into two passes over the 8,000 final-tier training users; pass 1 from the recovery epoch with a fresh Muon optimizer (`expertium-iter53`), first half of the decay; pass 2 from pass 1's end (Muon state continued, update offset continued), `random.Random(2)` shuffle of the rows, second half. 32,768-review windows, strided-lane state carry, show-time inputs, deleted cards neighbour, training seed 2025, replay KD at alpha 0.9 / decay alpha 0.5 from the show-time teachers: the show-time big model (SM91s2) for two-head, the show-time Instant-Only big model (SM99) for Instant-Only (curve_loss_weight 0). SM130's measured passes: 150.3 / 150.8 min (64,547 / 64,303 reviews/s) two-head, 147.0 / 147.6 min (65,967 / 65,706 reviews/s) Instant-Only.

`sm183_prepare.py` copies SM130's pass-1 `protocol.json` and `decay.json` (same rows, order, schedule, `lr_resume`, options, teacher targets) into the run tree and changes only the start: SM130's recovery epoch with 6 zero columns inserted at index 24 of `features2card.0.weight` (every other weight bit for bit; fresh optimizer as in SM130). Pass 2 is prepared by `sm130_prepare.py p2`'s code from the arm's pass-1 end. `sm183_prepare.py pair` proves every folder equals SM130's folder of the same pass except the start, the control directory and (pass 2) the update offset, which must also be equal unless pass 1 was resumed from a snapshot.

**Arms:**

1. `sm183-th-ses` (two-head, session inputs), pass 1 `sm183-th-ses-p1`, pass 2 `sm183-th-ses-p2`. First priority.
2. `sm183-io-ses` (Instant-Only, session inputs), the same two passes from epoch 41, if the budget allows (below).

## Control

**The release models are the controls**, if the reproduction check G0 passes: the release model of each chain is exactly this cool-down without the session inputs. G0 trains SM130's own pass-1 folder (copied unchanged, `sm183-th-ctlcheck`, flag unset) in the run tree up to its first periodic snapshot (400 users, 1,189 updates) and compares it with SM130's pass 1 at the same snapshot, bit for bit: every `master.pth` tensor, every `optimizer.pth` state tensor, the snapshot counters and finished users, and the per-update log (user, offset, reviews, seed, loss, gradient norm, labels, learning rate) of all 1,189 updates. With the same data order, seeds and code this shows that the run tree reproduces SM130's training, so the only difference between an arm and its release model is the session inputs (proof K2a below also shows the flag-off run tree equals SM130's tree bit for bit on CPU). The Instant-Only chain uses the same trainer, options builder and protocol copy (its pair check is the same), so G0 on the two-head pass covers it.

**If G0 fails:** the control arm `sm183-th-ctl` (the same two passes from SM130's start, flag unset, run tree) is trained after the two-head session arm and is the two-head control; the Instant-Only chain is then not run (budget).

## Proofs (CPU, before the GPU; `sm183_check.py cpu` -> `check-cpu-sm183.json`)

- **K1** the show-time data of every user read with the flag on: the 8,000 training users (the same rows for both chains) and the 732 set pass `rwkv/extra_inputs.py`'s own alignment (shared unique parquet row index, rating and duration equal row by row), all session values finite; answer-time inversions reported. The 765 decisive users passed the same scan in SM181.
- **K2** the real trainer path on CPU (SM130's pass-1 options, 256-review windows, users 4745 and 2308, 3 updates each, fresh optimizer): (a) SM130's pass-1 start, flag off, run tree = SM130's own tree (`wt/sm130`, `2ce266f0`) bit for bit (loss, model input, every gradient); (b) per chain the padded start with the flag on: first-update loss equal to the flag-off loss within 1e-6 relative, input width 98, extra columns and their gradients non-zero, finite.
- **G0** on the GPU, above (`sm183_check.py ctl` -> `check-ctl-sm183.json`).
- Trainer smoke of each session arm (2 users x 512 reviews, scratch copy).

## Scoring

The run tree's scorer (`scripts/small_model_heldout.py`), one ticket per model so that each process has its own `SM179_INPUTS` (release models unset, arms `session`): label REF `w8+s8hx8` only (int8 weights, int8 card / note state, int8 shift vectors), show-time inputs (`--inputs e2s-query`), `--split-existing --deleted-cards neighbour --nonfinite-rows --prior-buckets 0,1,3,5,9,16,32,128,300,1000 --dump`, `PYTORCH_JIT=0`, cap 0.8, then retry passes at cap 0.92 for missing users while a pass makes progress (memory only).

- **732 set:** `--pool 732 --users-file wt/sm72s/.../eval-users-732-fit.json` (723 scored users).
- **Decisive users:** `--pool 1500 --users-file users-decisive.json` (765 users, SM181 scored 761 of them).
- Order: two-head control, two-head arm on the 732 set, then both on the decisive users; then the Instant-Only chain; its decisive scoring runs only if the budget allows (below).
- Dumps (per review `y`, `p_ahead`, `p_instant`): `wt/sm183/work/sm183-dump/{732,decisive}/`, never committed.

## Rules (pre-registered; `sm183_report.py`)

Statistic as SM179 / SM181: % = 100 x (sum over users of LogLoss x reviews of the arm / the same of the control - 1), per head, paired user bootstrap, 2,000 draws, numpy `default_rng(0)`, 95% CI, on the rule users of the set (a finite row for both models, equal benchmark sizes). The selftest on SM181's rows relabelled reproduces SM181's numbers exactly (732 set -0.065% instant, decisive -0.063%).

- **Passes on a set:** instant change <= -0.05% with the CI upper bound below 0, and (two-head) the curve point estimate <= +0.05% (not worse by more than 0.05%). The Instant-Only model's curve head is not trained: reported only.
- **Verdict per chain:** GO if it passes on the 732 set and on the decisive users; PARTIAL if on one of them (the main session decides); NO GO if on neither. If the decisive users were not scored (Instant-Only, budget), the verdict is the 732 set's, marked "732 set only".
- **Pending** (exit 2): fewer than 720 rule users on the 732 set or 743 on the decisive users, a non-finite row, or an integrity check fails (the control's benchmark sizes equal the reference rows of the same model: SM130's own rows on the 732 set, SM164's REF rows on the decisive users, at least 600 common users).
- **Reported, no verdict:** both heads with CI on both sets; the change per answer-count bucket; overall and within-user AUC per head (from the dumps, arm minus control, paired bootstrap); training reviews/s, minutes, GPU memory per pass vs SM130's passes; the gain vs the short fine-tune's -0.065%.

## Budget and ETA

| step | GPU (estimate) |
|---|---|
| two-head smoke + G0 (400 users of pass 1, plus start-up) | 0.3 h |
| two-head session arm, 2 passes (SM130: 150 + 151 min) | 5.0-5.5 h |
| two-head scoring: 2 models x (732 set ~15 min, decisive ~25 min + retries) | 1.3 h |
| **two-head subtotal (the useful partial result, log line `SM183 TH`)** | **about 6.7 h** |
| Instant-Only smoke + 2 passes (SM130: 147 + 148 min) | 5.0-5.4 h |
| Instant-Only scoring on the 732 set, 2 models | 0.5 h |
| **total** | **about 12.2 h** (cap about 12 GPU hours) |

Budget gates (measured GPU minutes from each ticket's own start, `gpu/gpu-minutes.jsonl`): the Instant-Only chain starts only if the two-head steps used at most 400 minutes; its decisive scoring runs only if all steps so far used at most 680 minutes (normally skipped, as a follow-up if Instant-Only goes). Every training folder resumes from its last snapshot (every 400 users) after a stop or a restart; finished scoring users are kept.

Launch Sat 11-28 ~06:00: two-head read-out (`SM183 TH`) about 13:00 Sat 11-28 (12:30-14:30), Instant-Only and `SM183 DONE` about 19:00 Sat 11-28 (18:00-21:00), later if other tickets interleave (the queue serialises them).

Memory: GPU cap 0.8 of 16.3 GB for every ticket (SM130's passes peaked at 4.2 GB allocated; the training process tree reaches about 10-11 GB RSS, as SM130 and SM179, unchanged trainer). CPU: the launcher, the proofs and the report stay far under 6 GB.

## Follow-up (not part of this run)

If two-head goes: the fork's feature builder (`features_for`) for the six columns with test vectors, then the same cool-down for Small two-head (SM118) and the Instant-Only decisive scoring. If NO GO while the short fine-tune passed: the gain does not survive a full cool-down; the 92 inputs stay for the release.
