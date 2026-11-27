# Session inputs (SM179): design

Written Fri 2026-11-27, before any GPU ticket (only the CPU statistics and proofs ran first). Branch `sm179` (from `origin/main`: design, scripts, launcher, hand-off), worktree `C:/Users/jonat/workspace/wt/sm179`. Run tree: `C:/Users/jonat/workspace/wt/sm179-run`, branch `sm179-run` = `cea0f931` (`origin/sm148`, the tree of the state-energy penalty screen (SM171) and of its control) plus one commit, `cd220759`, with the gated feature code; it is not edited while the screen runs. Track: small model. The user (11-27) asked to keep exploring inputs that lower LogLoss or shrink the models.

## Question

Today the model sees 92 inputs per review (`docs/SMALL_MODEL_INTEGRATION.md` 6.4 / 6.5). About the user's study session it only sees "reviews today" and "new cards today" (inputs 20, 21) and "days since any review" (16), all on a day scale. **Question: do session and fatigue inputs (A), or the card's review kind (B, the information of input 22), lower the LogLoss of six-layer Tiny two-head (SM130) by at least 0.05% on one head, when they are added to a short cool-down from the release model?**

| plain name (ID) | what it is here |
|---|---|
| six-layer Tiny two-head (SM130) | the model of the screen: `w64-c1-d2p1u1`, release folder `wt/sm130/artifacts/small-model/sm130-th-dec2ep32k-p2` (`master.pth` sha256 48a5e724...) |
| the state-energy penalty screen (SM171) | its control arm `sm171-tiny-ctl` (lambda 0 = the stock trainer, proven bitwise) is this screen's control: the same 944 users, 67,310,673 reviews, learning rate, seed and options |
| the low-rank int8 states experiment (SM148) | the scorer of the run tree and the REF label `w8+s8hx8` (int8 weights, int8 card / note state, int8 shift vectors) |

## Inputs

All inputs describe the moment the card is shown; they are computed only from earlier reviews of the same user (review k reads reviews j < k) and from the show time of review k. In the fork every one of them is computable from the revlog (answer time = revlog id, show time = id minus the answer time) and the card's current queue. The query row and the update row of a review carry the same values.

**Where the time comes from.** The training dataset (`anki-revlogs-10k-original-20261011`, answer-time layout) has no clock. The show-time export of the same revlogs (`data/anki-revlogs-10k`) has `review_time` (epoch ms of the show, = revlog id minus `taken_millis`). The two exports share the parquet row index: on all 1,667 users of the screen (944 training, 723 scored) the rows match one to one by that index (same rating, same duration, card ids a bijection; `scan` in the CPU statistics), and the answer times `review_time + duration` are non-decreasing in the training order (0 inversions over 67.3M training rows). `rwkv/extra_inputs.py` reads the review_time by that index and checks rating and duration of every row (any mismatch stops the process). The true time of day and weekday are still unknown (time zone): donated data has them; testing them is a future idea (below).

**(A) session, 6 columns** (`SM179_INPUTS=session`), each `(value - mean) / std` with the mean / std of the 944 training users' rows (`input-stats.json`):

| column | value | mean, std |
|---|---|---|
| `sm179_gap` | ln(1 + seconds from the previous review's answer, any card, to this show), 0 on the first review | 0.5174, 1.6065 |
| `sm179_n10` | ln(1 + reviews answered in the 10 minutes before this show) | 3.3827, 1.2234 |
| `sm179_n60` | ln(1 + reviews answered in the 60 minutes before this show) | 4.1561, 1.3916 |
| `sm179_sess_min` | ln(1 + minutes since the session's first show); a session starts at a review whose gap is >= 30 minutes | 2.9301, 1.3054 |
| `sm179_sess_pos` | ln(1 + earlier reviews of this session) | 4.2047, 1.5089 |
| `sm179_sess_again` | Again share of the last <= 10 earlier reviews of this session (0 if none) | 0.1575, 0.2165 |

The brief's five inputs plus the gap itself (the basis of the session rule).

**(B) card state, 4 columns** (`SM179_INPUTS=state`): one-hot of the review kind of this review (dataset `state` = Anki revlog type + 1): learning (state 1, 22.0% of training rows), review (2, 53.8%), relearning (3, 10.3%), filtered or other (>= 4, 1.2%); all 0 on a card's first review (state 0, 12.8%, already flagged by elapsed = -1).

**Why B is tested (input 22).** Input 22 (`scaled_state = state - 2`) is zeroed in the model (`zero_features: [22]`) because the outside research's iteration-15 preset does so; our docs (`docs/EXPERTIUM_CATCH_UP.md` 4b, `docs/ACCURACY_IDEAS.md` A06, `git log`) give no other reason and no measurement. The add_queries code already blanks it on query rows as a precaution ("TODO state can be kept(?)"). Neither concern holds as a leak: the revlog type is fixed before the answer (an Again on a review card is logged as a review), so it carries nothing about this review's outcome, on the update row or on the query row. Fork parity is not a blocker: the fork knows the card's queue (learning, review, relearning, filtered) when it asks the instant question. So B is tested, as new one-hot columns (one scalar `state - 2` cannot express a category); input 22 itself stays zeroed and unchanged.

**Placement.** The extra columns are inserted after the 24 deterministic inputs (indices 24 .. 24 + n - 1); codes and calendar move up by n. The first Linear (`features2card.0.weight`, 128 x 92) gets n zero columns at index 24, and its Muon momentum too (`sm179_prepare.py`), so step 0 computes what the release model computes. Every other weight and optimizer state is the release model's end state.

**Gate.** `SM179_INPUTS` unset (the default) = no extra column and width 92 everywhere (`CARD_FEATURE_COLUMNS`, `add_queries`, `gpu_r01_samples.KEEP`, the model width): today's code bit for bit (proof C1). The variable is read when `rwkv/extra_inputs.py` is imported, so the trainer's preparation workers and the scorer inherit it; a wrong mode cannot load the start (width mismatch) and `sm179_run.py` refuses an environment that differs from the arm folder's `decay.json`.

## Proofs (CPU, before the screen)

`sm179_check.py cpu` -> `check-cpu-sm179.json` (stock tree `wt/sm171-run` at `cea0f931`, run tree `cd220759`, clean). Results are in the file and in the hand-off.

- **C1** flag off = stock, bitwise: 3 real updates on each of users 4745 and 2308 (release weights and Muon state, release options with 256-review windows so that windows are carried): loss, model input and every gradient equal bit for bit between the two trees.
- **C2** flag on, step 0 = release: the same updates from the zero-padded starts with `session` / `state`: first-update loss equal to the stock loss within 1e-6 relative, extra columns and their gradients non-zero, everything finite.
- **C3** the session values equal a plain per-review loop; recomputed on 40 prefixes of each history they equal the full-history values (causal); changing review k's own rating or duration leaves review k's values unchanged; in the scorer's samples (show-time inputs) every query row carries its update row's extra columns, the other 92 columns are bitwise the stock tree's, and flag off the samples equal the stock tree's.
- **C4** the scorer: the release model (flag off) and the padded session start (flag on) give the same benchmark LogLoss (1e-6) on 2 small users of the 732 set (fp32 and REF).

## Recipe

Both arms: SM171's control cool-down with only the inputs changed. Start = the release model's final `master.pth` and `optimizer.pth` (end of its own cool-down, Muon state), widened as above; the release folder's options, loss and targets unchanged (show-time inputs, strided-lane carry, 32,768-review windows, replay KD targets of the show-time Big model (SM91s2) at alpha 0.5, deleted cards neighbour); no `state_vq` option (SM171's control had the lambda-0 hook, bitwise equal to stock, P4 of SM171).

- **Arms:** `sm179-tiny-ses` (A, 6 columns), `sm179-tiny-st` (B, 4 columns). No A+B arm (at most 2 new arms; A+B only if both pass).
- **Data / schedule:** SM171's 944 users, 67,310,673 reviews (`random.Random(171)` shuffle of the release rows), 0 -> 7e-5 over 64 updates then one_minus_sin to 0, about 2,550 updates, snapshots every 100 users, training seed 2025. `sm179_prepare.py pair` proves the rows, schedule, protocol and options equal the control's (options minus `state_vq`).
- **Control:** `wt/sm171-run/artifacts/small-model/sm171-tiny-ctl` (20.1 min, 55,723 reviews/s), not retrained.

## Scoring

The run tree's scorer (`scripts/small_model_heldout.py`), one ticket per model so that each process has its own `SM179_INPUTS`: the control with the flag unset, `sm179-tiny-ses` with `session`, `sm179-tiny-st` with `state`. `--pool 732 --users-file <sm72s eval-users-732-fit.json> --inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows --prior-buckets 0,1,3,5,9,16,32,128,300,1000`, `PYTORCH_JIT=0`, label REF `w8+s8hx8` only (show-time inputs, curve and instant), GPU cap 0.8 (0.92 for a retry pass, memory only). Integrity: the control's rows from this scorer must equal SM171's rows of the same model (max |LogLoss difference| <= 1e-6 on every common user), which also re-checks the flag-off scorer on the full set.

## Rules (pre-registered; `sm179_report.py`)

Statistic as SM148 / SM171: % = 100 x (sum over users of LogLoss x reviews of the arm / the same of the control - 1), per head, paired user bootstrap, 2,000 draws, numpy `default_rng(0)`, 95% CI, on the rule users (a finite row for all three models, equal benchmark sizes).

- **Go for a follow-up (per arm):** on one head (curve or instant) the change is <= -0.05% with the CI upper bound below 0, and on the other head the point estimate is <= +0.05% (not worse by more than 0.05%).
- **Verdict:** GO if an arm passes; otherwise NO GO. **Pending** (exit 2, the main session decides) if fewer than 720 rule users, a non-finite row, or the integrity check fails.
- **Reported, no verdict:** both heads with CI for both arms; the change per answer-count bucket (all predicted reviews); training reviews/s, minutes and GPU memory vs the control.

## Follow-up if GO (not part of this screen)

In this order: (1) A+B together if both pass; (2) the passing inputs in a full cool-down of Tiny (the release recipe, all 8,000 users), then Small; (3) the fork's feature builder (`features_for`) for the new columns, with test vectors. If the session inputs help, the true time of day and weekday from donated data are the next inputs to test (not possible on the 10k dataset). If NO GO: the 92 inputs stay; input 22 stays zeroed with the reason now measured.

## Estimates and budget

| step | GPU minutes |
|---|---|
| trainer smoke, two arms | about 2 |
| `sm179-tiny-ses`, `sm179-tiny-st` (67.3M reviews at about 50k reviews/s + start-up; the control ran 20.1 min) | about 25 each |
| scoring 3 models x REF on the 732 set (SM148: Tiny REF about 2.4 min; plus start-up) | about 15 |
| **total** | **about 1.2 h** (budget 4 GPU hours) |

Memory: cap 0.8 of 16.3 GB for every ticket (the control peaked at 4.2 GB allocated). CPU: the launcher, the report and the CPU proofs stay far under 6 GB. No 8k / 2k 10-epoch ETA: this is a short cool-down, not a training run.
