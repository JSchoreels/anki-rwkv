# Small cool-down with session inputs (SM185): design

Written Sat 2026-11-28, before any GPU ticket (only the CPU proofs ran first). Branch `sm185` (from `origin/sm183`, `9c75b66a`), worktree `C:/Users/jonat/workspace/wt/sm185`. Run tree: `C:/Users/jonat/workspace/wt/sm179-run`, branch `sm179-run` = `cd220759` (the session-input feature code of SM179), reused unedited and clean as SM179, SM181 and SM183 did; it is not edited while this runs. Track: small model.

## Question

The Tiny cool-down with session inputs (SM183) repeated the release cool-down of six-layer Tiny (SM130) with the six session inputs and went GO: two-head instant -0.499% (732 set) and -0.474% (decisive users), curve -0.03%; Instant-Only instant -0.589% (732 set); within-user instant AUC about +0.25%. **Question: does the same hold for Small?** This run repeats the exact release cool-down of Small two-head (SM118) with the six session inputs, then (only if the budget allows, below) the one of Small Instant-Only (SM113).

| plain name (ID) | what it is here |
|---|---|
| Small two-head (SM118) | "the show-time two-head cool-down for Small": release folder `wt/sm118/artifacts/small-model/sm118-e26-dec2ep32k-t91s2-p2` (`master.pth` sha256 b4fe1175...); its cool-down started from the show-time retrain of the release (SM92)'s constant-LR epoch 26 (`wt/sm92/.../sm92-e26`, master sha256 5a629e4e...) |
| Small Instant-Only (SM113) | "the longer big instant-only teacher" = the longer Small Instant-Only cool-down: release folder `wt/sm113/.../sm113-inst-e35st-st-dec2ep32k-t99-p2` (sha256 e4574e2a...); its cool-down started from the Small Instant-Only model (SM98)'s epoch 35 (sha256 f7577d14...) |
| the show-time big model (SM91s2) | SM118's teacher (curve and grade replay targets) |
| the show-time instant-only big model (SM99) | SM113's teacher (replay targets; curve loss weight 0) |
| the Tiny cool-down with session inputs (SM183) | the design copied here (scripts, G0, scoring, rules) |
| the session inputs screen (SM179) / confirmation (SM181) | the feature code (`SM179_INPUTS=session`, 6 columns, zero-initialised weights), the decisive user list (765 users with id >= 5000 of the 1,500 pool, `users-decisive.json`), the statistic and the AUC code |

## The inputs (unchanged from SM179)

`sm179_gap`, `sm179_n10`, `sm179_n60`, `sm179_sess_min`, `sm179_sess_pos`, `sm179_sess_again`: reviews in the last 10 and 60 minutes, minutes since the session start (30-minute gap rule), position in the session, the session's recent Again share, and the gap since the previous review. All causal, standardised with SM179's means and standard deviations, inserted at input index 24.

## Recipes reproduced (only the start and the input width change)

Both release cool-downs are the same two-pass schedule ("dec2ep32k"), per chain: one one_minus_sin decay 7e-4 -> 0 over 2 x 581,942,724 reviews (the 8,000 final-tier users after the trim of user 5599 by 14 reviews), split at its middle into two passes; pass 1 from the start checkpoint with a fresh Muon optimizer (`expertium-iter53`), first half of the decay (7e-4 -> 0.2929 x 7e-4), `lr_resume = {total 2R, offset 0, segment}`; pass 2 from pass 1's end (master and Muon state continued, update offset continued), `random.Random(2)` shuffle of the rows, second half (`lr_resume = {total 2R, offset R}`). 32,768-review windows, 1 lane, strided-lane state carry, show-time inputs, deleted cards neighbour, training seed 2025, release order, snapshots every 400 users, replay KD at alpha 0.9 / decay alpha 0.5 from the teacher's show-time targets:

- **Small two-head (SM118)** (`wt/sm118/.../sm118/design.md`, `launch-sm118.sh`, `sm118_prepare.py`): start = SM92's epoch 26 (update offset 574,028), w64, both heads trained, teacher targets of the show-time big model (SM91s2) (`anki-rwkv/artifacts/cache/kd-replay/sm91s2-big-showtime-final8k-neighbour-e2s`, curve logit + 4 grade probabilities). Trained in `wt/sm118`'s own tree. Measured: pass 1 335.4 min training (28,915 reviews/s, peak allocated 7,428 MiB, process tree RSS 11.1 GB); pass 2 338.5 min in two segments (snapshot stop for a break after 5,327 of 8,000 users, resumed after it; 28,690 reviews/s).
- **Small Instant-Only (SM113)** (`wt/sm113/.../sm113/design.md`, `launch-sm113.sh`, `sm113_prepare.py`): start = SM98's epoch 35 (update offset 772,730), w64, curve loss weight 0, teacher targets of the show-time instant-only big model (SM99) (`.../kd-replay/sm99-big-inst-final8k-neighbour-e2s`). Trained in `wt/sm113`'s own tree. Pass 1 was snapshot-stopped once for a break (attempt 1 to 2,000+ users, then resumed); pass 2 unbroken. Measured at 20.2k reviews/s in that older tree (about 480 min per pass); the run tree trains Small at about 28.8k reviews/s (SM181).
- The two chains train the same 8,000 users with the same rows in the same order (checked: `protocol.json` rows equal).

`sm185_prepare.py` copies each source's pass-1 prepared protocol (`protocol.prepared.json` for SM113, whose `protocol.json` was rewritten by the resume) and `decay.json` (same rows, order, schedule, `lr_resume`, options, teacher targets, trim) into the run tree and changes only the start: the source's start with 6 zero columns inserted at index 24 of `features2card.0.weight` (128 x 92 -> 128 x 98; every other weight bit for bit; fresh optimizer as in the source). Pass 2 is prepared by `sm118_prepare.py p2`'s code (identical to `sm113_prepare.py p2` and `sm183_prepare.py p2`) from the arm's pass-1 end. `sm185_prepare.py pair` proves every folder equals the source folder of the same pass except the start, the control directory and (pass 2) the update offset, which must also be equal unless pass 1 was resumed. Prepared and checked: `sm185-th-ses-p1`, `sm185-th-ctlcheck`, `sm185-io-ses-p1`, `sm185-io-ctlcheck` (pair: ok).

**Arms:**

1. `sm185-th-ses` (Small two-head, session inputs), pass 1 `sm185-th-ses-p1`, pass 2 `sm185-th-ses-p2`. The run's main arm.
2. `sm185-io-ses` (Small Instant-Only, session inputs), the same two passes from SM98's epoch 35, only if the budget allows (below; by the estimate it does not, so it is prepared and proven but normally left as a follow-up).

## Control

**The release models are the controls**, if the reproduction check G0 of the chain passes: each release model is exactly this cool-down without the session inputs. G0 trains the source's own pass-1 folder (copied unchanged, `sm185-<chain>-ctlcheck`, flag unset) in the run tree up to its first periodic snapshot (400 users; for SM118 1,189 updates, 31,962,943 reviews) and compares it with the source's pass 1 at the same snapshot bit for bit: every `master.pth` tensor, every `optimizer.pth` state tensor, the snapshot counters and finished users, and the per-update log (step, user, offset, reviews, seed, loss, gradient norm, labels, learning rate) of every update. The source snapshots: SM118's pass 1 (never resumed); SM113's pass 1 attempt 1 (its first segment, from the start). With the same data order, seeds and code this shows that the run tree reproduces the release training, so the only difference between an arm and its release model is the session inputs. Unlike SM183, the two chains were trained by two different trees (`wt/sm118` and `wt/sm113`), so each chain gets its own G0 (two-head before its arm; Instant-Only at the start of its chain), and proof K2a below checks both trees on CPU.

Known difference that G0 does not cover: the release pass 2 of SM118 was snapshot-stopped once for a break and resumed (stop snapshot after 5,327 users; its `resume.json` says `exact: false`, because the user in flight at the stop is retrained from its start); the arm's pass 2 normally runs unbroken. The resume restores the weights, the Muon state, the update offset and the learning-rate position at a user boundary (the resumable decay, SM69), so the difference is at most one user's partial updates replayed; it is reported, not corrected. Likewise SM113's pass 1 was resumed once (its G0 compares attempt 1, from the start).

**If the two-head G0 fails:** the launcher stops (`SM185 STOP`) before the session arm. A control arm `sm185-th-ctl` (the same two passes from SM92's epoch 26, flag unset, run tree) would add about 11.5 GPU hours and break the 16-hour budget, so it runs only with the main session's OK (relaunch with `CONTROL_ARM=1`: session arm, then control arm). If the Instant-Only G0 fails, the Instant-Only chain is not run.

## Proofs (CPU, before the GPU; `sm185_check.py cpu` -> `check-cpu-sm185.json`)

- **K1** the show-time data of every user read with the flag on: the 8,000 training users (the same rows for both chains) and the 732 set pass `rwkv/extra_inputs.py`'s own alignment (shared unique parquet row index, rating and duration equal row by row), all session values finite; answer-time inversions reported. The 765 decisive users passed the same scan in SM181.
- **K2** the real trainer path on CPU (each source's pass-1 options, 256-review windows, users 4745 and 2308, 3 updates each, fresh optimizer): (a) per chain, the source's start, flag off: run tree = the tree that trained the release (`wt/sm118` at `c8f5abf7`, `wt/sm113` at `f561e1c0`, both clean) bit for bit (loss, model input, every gradient); (b) per chain the padded start with the flag on: first-update loss equal to the flag-off loss within 1e-6 relative, input width 98, extra columns and their gradients non-zero, finite.
- **G0** on the GPU, above (`sm185_check.py ctl --chain th` -> `check-ctl-sm185.json`; `--chain io` -> `check-ctl-io-sm185.json`).
- Trainer smoke of each session arm (2 users x 512 reviews, scratch copy), the first ticket of each chain.
- Report selftest (`sm185_report.py selftest`): SM183's committed rows relabelled through this report's statistic reproduce `sm183-result.json`'s six LogLoss changes exactly.

## Scoring

The run tree's scorer (`scripts/small_model_heldout.py`), one ticket per model so that each process has its own `SM179_INPUTS` (release models unset, arms `session`): label REF `w8+s8hx8` only (int8 weights, int8 card / note state, int8 shift vectors), architecture `w64`, show-time inputs (`--inputs e2s-query`), `--split-existing --deleted-cards neighbour --nonfinite-rows --prior-buckets 0,1,3,5,9,16,32,128,300,1000 --dump`, `PYTORCH_JIT=0`, cap 0.8, then retry passes at cap 0.92 for missing users while a pass makes progress (memory only).

- **732 set:** `--pool 732 --users-file wt/sm72s/.../eval-users-732-fit.json` (723 scored users).
- **Decisive users:** `--pool 1500 --users-file users-decisive.json` (765 users).
- Order: two-head control (`sm118-th-release`), two-head arm on the 732 set, then both on the decisive users; then (budget) the Instant-Only chain on the 732 set, its decisive users only if the budget still allows.
- Dumps (per review `y`, `p_ahead`, `p_instant`): `wt/sm185/work/sm185-dump/{732,decisive}/`, never committed.

## Rules (pre-registered; `sm185_report.py`, the same as SM183's)

Statistic as SM179 / SM181 / SM183: % = 100 x (sum over users of LogLoss x reviews of the arm / the same of the control - 1), per head, paired user bootstrap, 2,000 draws, numpy `default_rng(0)`, 95% CI, on the rule users of the set (a finite row for both models, equal benchmark sizes).

- **Passes on a set:** instant change <= -0.05% with the CI upper bound below 0, and (two-head) the curve point estimate <= +0.05% (not worse by more than 0.05%). The Instant-Only model's curve head is not trained: reported only.
- **Verdict per chain:** GO if it passes on the 732 set and on the decisive users; PARTIAL if on one of them (the main session decides); NO GO if on neither. If the decisive users were not scored (Instant-Only, budget), the verdict is the 732 set's, marked "732 set only".
- **Pending** (exit 2): fewer than 720 rule users on the 732 set or 743 on the decisive users, a non-finite row, or an integrity check fails (the control's benchmark sizes equal the reference rows of the same model: SM118's / SM113's own rows on the 732 set, SM164's REF rows of the same model and label on the decisive users, at least 600 common users; for the decisive users the LogLoss difference to SM164's rows of the same label is reported).
- **Reported, no verdict:** both heads with CI on both sets; the change per answer-count bucket; overall and within-user AUC per head (from the dumps, arm minus control, paired bootstrap); training reviews/s, minutes, GPU memory per pass vs the release passes; the gain vs the short fine-tune of Small two-head (SM118) in SM181 (-0.065% instant) and vs the Tiny result of SM183.

## Budget and ETA

| step | GPU (estimate) |
|---|---|
| two-head smoke + G0 (400 users of pass 1, plus start-up) | 0.5 h |
| two-head session arm, 2 passes (SM118: 336 + 339 min; run tree ~28.8k reviews/s) | 11.5 h |
| two-head scoring: 2 models x (732 set ~12-15 min, decisive ~17-20 min + retries) | 1.2 h |
| **two-head total (log line `SM185 TH`)** | **about 13.2 h** |
| Instant-Only: smoke + G0 + 2 passes + 732-set scoring (2 models) | about 12.3 h (`IO_EST` = 740 min) |
| Instant-Only decisive scoring | about 1.2 h |

Budget gate (measured GPU minutes from each ticket's own start, `gpu/gpu-minutes.jsonl`; cap `BUDGET_TOTAL` = 960 minutes = 16 GPU hours): the Instant-Only chain starts only if the minutes used + 740 fit in the cap, i.e. only if the two-head steps took at most 220 minutes. With the two-head steps at about 790 minutes it will not start: the launcher then logs "Instant-Only chain not run (budget ...)" and ends with `SM185 DONE`. The Instant-Only chain is prepared and proven on CPU (K1, K2 for both chains, pair ok) so that a follow-up is a relaunch with `BUDGET_TOTAL=<minutes>` (main session's OK); finished two-head steps are skipped. Its decisive scoring runs only if the minutes used + 70 still fit. Every training folder resumes from its last snapshot (every 400 users) after a stop or a restart; finished scoring users are kept.

Launch Sat 11-28 ~18:15 (queue empty, no quiet window): G0 result about 18:50, pass 1 ends about 00:50 Sun, pass 2 about 06:50 Sun, **`SM185 TH` and `SM185 DONE` about 08:00 Sun 11-29** (07:30-09:30), later if other tickets interleave (the queue serialises them).

Memory: GPU cap 0.8 of 16.3 GB for every ticket (SM118's passes peaked at 7.4 GB allocated / 12.8 GB reserved; the training process tree reaches about 11 GB RSS as in SM118 and SM113, unchanged trainer). CPU: the launcher, the proofs and the report stay far under 6 GB.

## Follow-up (not part of this run)

If two-head goes: the Instant-Only chain of Small (SM113) with the main session's OK (about 13.5 GPU h with its decisive scoring), the fork's feature builder for the six columns with test vectors, and the shipped T24 / 6-bit state formats checked with the new models. If NO GO for Small while Tiny went: the inputs go only into Tiny's recipe.
